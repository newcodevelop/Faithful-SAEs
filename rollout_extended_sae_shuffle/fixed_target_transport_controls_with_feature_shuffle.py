#!/usr/bin/env python3
"""
Fixed-target transport + matched-norm feature-shuffled and random controls for
horizon-conditioned SAE rollout analysis.

This script extends the original horizon-conditioned rollout experiment.

Original experiment
-------------------
For a patch layer l and rollout horizon h, the original analysis patches the SAE
reconstruction at l and measures the base-vs-proxy divergence after h downstream
blocks or at the final output. This shows whether SAE reconstruction errors are
local, amplified, persistent, or stable under rollout.

Reviewer concern
----------------
Even at matched rollout horizons, a late-layer SAE patch may look good partly
because the downstream map at late layers is generically less sensitive to
perturbations, not only because the SAE reconstruction is more faithful.

This script adds two targeted diagnostics.

Experiment 1: Fixed-target-layer transport
------------------------------------------
Patch the SAE reconstruction at layer l, but measure the transported divergence
at fixed absolute later layers r. This produces matrices

    D^{SAE}_{l -> r}

that show whether errors introduced at early layers persist, amplify, or are
corrected as they travel through the model stack.

Experiment 2: Matched-norm feature-shuffled SAE control
-------------------------------------------------------
For each layer l and example/token position, compute the SAE perturbation

    Delta_l^SAE(x) = SAE_recon_l(x) - h_l(x).

Then construct a feature-shuffled SAE reconstruction by preserving top-k coefficients but permuting decoder identities, and rescale its perturbation to the same per-token L2 norm as the true SAE error. Optionally, the script can also sample an isotropic random perturbation with the same per-token L2 norm:

    Delta_l^rand(x) = ||Delta_l^SAE(x)||_2 * z / ||z||_2.

Patch h_l(x) + Delta_l^rand(x) and measure the same fixed-target divergences.
If the SAE is only benefiting from generic downstream robustness, SAE and random
matched-norm perturbations should behave similarly. If the SAE reconstruction is
representation-aligned, then

    D^{SAE}_{l -> r} << D^{RAND}_{l -> r}

at the same patch layer l and target layer r.

Outputs
-------
The script saves:

1. detailed per-batch CSV:
   final_ops_transport/fixed_target_transport_detailed.csv

2. summary CSV:
   final_ops_transport/fixed_target_transport_summary.csv

3. matrix CSVs:
   final_ops_transport/matrices/<model>_<metric>_<perturbation>.csv

4. figures:
   - SAE transport heatmaps
   - random transport heatmaps
   - SAE/random ratio heatmaps
   - representation-distance heatmaps
   - optional horizon-style line plots reconstructed from fixed targets

Usage
-----
Fast Llama diagnostic:
    python fixed_target_transport_controls.py \
        --models "Llama-3.1-8B" \
        --eval-tokens 8192 \
        --patch-layers 4,8,12,16,20,24,28,30 \
        --target-layers 8,12,16,20,24,28,30,final

Full Llama diagnostic:
    python fixed_target_transport_controls.py \
        --models "Llama-3.1-8B" \
        --eval-tokens 32000 \
        --random-repeats 3

Run cross-model fixed SAE layers:
    python fixed_target_transport_controls.py \
        --models "GPT-2 Small,Gemma-2B,Llama-3.1-8B"

Notes
-----
- The default target-layer list is "auto": all configured patch layers plus final.
- The primary metric is KL in bits after applying the logit lens at the target
  residual stream. For target=final, this is the actual final output KL.
- The secondary representation metric is normalized L2 distance at target layer.
- For target=final, representation metrics are reported as NaN because there is
  no residual target activation.
"""

import argparse
import gc
import math
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens import SAE
from sparsify import Sae


# =============================================================================
# 1. Configuration
# =============================================================================

CONFIG = {
    "EVAL_TOKENS": 32_000,
    "SEQ_LEN": 32,
    "TOKENIZER_BUFFER_SIZE": 32_000,
    "TOP_K": 64,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "SEED": 42,
    "RANDOM_REPEATS": 1,
    "OUTPUT_DIR": "./final_ops_transport",
    "MODELS": {
        "GPT-2 Small": {
            "name": "gpt2-small",
            "sae_backend": "sae_lens",
            "sae_release": "gpt2-small-res-jb",
            "sae_id": "blocks.6.hook_resid_pre",
            "batch_size": 16,
            "color": "#1f77b4",
            "patch_layers": [6],
            "hook_suffix": "hook_resid_pre",
        },
        "Gemma-2B": {
            "name": "gemma-2b",
            "sae_backend": "sae_lens",
            "sae_release": "gemma-2b-res-jb",
            "sae_id": "blocks.12.hook_resid_post",
            "batch_size": 16,
            "color": "#d62728",
            "patch_layers": [12],
            "hook_suffix": "hook_resid_post",
        },
        "Llama-3.1-8B": {
            "name": "meta-llama/Meta-Llama-3-8B",
            "sae_backend": "sparsify",
            "sae_release": "EleutherAI/sae-llama-3-8b-32x",
            "batch_size": 16,
            "color": "#26ba15",
            "dtype": torch.bfloat16,
            "patch_layers": [4, 8, 12, 16, 20, 24, 28, 30],
            "hook_suffix": "hook_resid_post",
        },
    },
}


# =============================================================================
# 2. Generic helpers
# =============================================================================

def parse_csv_strings(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str) -> List[int]:
    if text is None or text.strip() == "":
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_target_layers(text: str, auto_layers: List[int]) -> List[Union[int, str]]:
    """
    Parse target layers. "auto" means all configured patch layers plus final.
    Integer targets are absolute layer indices. "final" means final output.
    """
    if text is None or text.strip().lower() == "auto":
        return sorted(set(auto_layers)) + ["final"]

    out: List[Union[int, str]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() == "final":
            out.append("final")
        else:
            out.append(int(item))
    # Keep order but remove duplicates.
    seen = set()
    dedup = []
    for x in out:
        key = str(x)
        if key not in seen:
            dedup.append(x)
            seen.add(key)
    return dedup


def safe_model_name(name: str) -> str:
    return name.replace(" ", "_").replace(".", "_").replace("/", "_").replace(":", "_")


def get_token_batches(model, batch_size: int, target_tokens: int, seq_len: int, buffer_size: int, seed: int):
    """
    Materialize token batches once per model so the same data is reused across
    patch layers and controls.
    """
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    iterator = iter(dataset)

    batches = []
    total_tokens = 0

    while total_tokens < target_tokens:
        batch_texts = []
        try:
            for _ in range(batch_size):
                item = next(iterator)
                text = item["text"] if "text" in item else item["content"]
                batch_texts.append(text)
        except StopIteration:
            if not batch_texts:
                break

        if not batch_texts:
            break

        tokens = model.to_tokens(batch_texts)
        if tokens.shape[1] < seq_len:
            continue

        tokens = tokens[:, :seq_len].cpu()
        batches.append(tokens)
        total_tokens += int(tokens.numel())

    print(f"Cached {len(batches)} batches ({total_tokens:,} tokens).")
    return batches, total_tokens


def extract_pre_acts(encode_out):
    if torch.is_tensor(encode_out):
        return encode_out
    if hasattr(encode_out, "pre_acts"):
        return encode_out.pre_acts
    if isinstance(encode_out, dict) and "pre_acts" in encode_out:
        return encode_out["pre_acts"]
    if isinstance(encode_out, (list, tuple)) and len(encode_out) > 0:
        return encode_out[0]
    raise TypeError(f"Unsupported SAE encode output type: {type(encode_out)}")


def get_topk_sparse(activations: torch.Tensor, k: int):
    top_acts, top_indices = torch.topk(activations, k=k, dim=-1)
    top_acts = torch.relu(top_acts)
    return top_acts, top_indices


def apply_topk_dense(activations: torch.Tensor, k: int):
    topk_vals, topk_inds = torch.topk(activations, k=k, dim=-1)
    topk_vals = torch.relu(topk_vals)
    sparse = torch.zeros_like(activations)
    sparse.scatter_(-1, topk_inds, topk_vals)
    return sparse


class SAEAdapter:
    def __init__(self, sae, backend: str, device: str):
        self.sae = sae
        self.backend = backend
        self.device = device

    def encode_pre_acts(self, flat_act: torch.Tensor) -> torch.Tensor:
        return extract_pre_acts(self.sae.encode(flat_act))

    def get_latent_dim(self, d_model: int, dtype: torch.dtype) -> int:
        """
        Infer SAE dictionary size m. For sae_lens this is stored in cfg.d_sae.
        For sparsify we infer it by encoding a dummy residual vector.
        """
        if self.backend == "sae_lens":
            return int(self.sae.cfg.d_sae)

        dummy_act = torch.zeros(1, d_model, device=self.device, dtype=dtype)
        return int(self.encode_pre_acts(dummy_act).shape[-1])

    def topk_code(self, act: torch.Tensor, k: int):
        """
        Encode a residual tensor and return the top-k sparse code in flattened
        token space.

        Returns:
            top_acts:    [batch*seq, k]
            top_indices: [batch*seq, k]
            flat_shape:  original flattened activation shape metadata
        """
        flat_act = act.reshape(-1, act.shape[-1])
        feature_acts_raw = self.encode_pre_acts(flat_act)
        top_acts, top_indices = get_topk_sparse(feature_acts_raw, k)
        return top_acts, top_indices

    def decode_topk_code(self, top_acts: torch.Tensor, top_indices: torch.Tensor, d_sae: int) -> torch.Tensor:
        """
        Decode top-k coefficients using the underlying SAE decoder.
        """
        if self.backend == "sparsify":
            return self.sae.decode(top_acts, top_indices)

        sparse_acts = torch.zeros(
            top_acts.shape[0],
            d_sae,
            device=top_acts.device,
            dtype=top_acts.dtype,
        )
        sparse_acts.scatter_(-1, top_indices, top_acts)
        return self.sae.decode(sparse_acts)

    def reconstruct_tensor(self, act: torch.Tensor, k: int, d_sae: int = None) -> torch.Tensor:
        """
        Standard top-k SAE reconstruction:
            \hat h = sum_j c_j d_j.
        """
        if d_sae is None:
            # Backward-compatible path for sae_lens only.
            if self.backend == "sae_lens":
                d_sae = int(self.sae.cfg.d_sae)
            else:
                raise ValueError("d_sae must be provided for sparsify reconstruction.")

        top_acts, top_indices = self.topk_code(act, k)
        recons = self.decode_topk_code(top_acts, top_indices, d_sae)
        return recons.reshape(act.shape)

    def reconstruct_feature_shuffled_tensor(
        self,
        act: torch.Tensor,
        k: int,
        d_sae: int,
        permutation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Feature-shuffled SAE reconstruction.

        Original reconstruction:
            \hat h(x) = sum_{j in A(x)} c_j(x) d_j.

        Shuffled reconstruction:
            \hat h_shuffle(x) = sum_{j in A(x)} c_j(x) d_{pi(j)}.

        Operationally, this is implemented by keeping the same top-k
        coefficients c_j but replacing each top-k index j by pi(j) before
        decoding. This preserves sparsity and coefficient magnitudes, while
        destroying the learned feature-to-decoder identity.

        The caller usually turns this into a matched-norm perturbation by
        rescaling
            \Delta_shuffle = \hat h_shuffle - h
        to have the same per-token norm as
            \Delta_SAE = \hat h - h.
        """
        top_acts, top_indices = self.topk_code(act, k)
        shuffled_indices = permutation[top_indices.to(torch.long)]
        recons = self.decode_topk_code(top_acts, shuffled_indices, d_sae)
        return recons.reshape(act.shape)


def parse_hook_name(hook_name: str):
    m = re.match(r"blocks\.(\d+)\.(hook_resid_(?:pre|post))$", hook_name)
    if not m:
        raise ValueError(f"Unsupported hook format: {hook_name}")
    return int(m.group(1)), m.group(2)


def make_hook_name(layer_idx: int, suffix: str) -> str:
    return f"blocks.{layer_idx}.{suffix}"


def residual_to_logits(model, resid: torch.Tensor) -> torch.Tensor:
    """
    Map residual-stream state at any layer to a vocabulary distribution using the
    final layer norm and unembedding. This is the same logit-lens style metric
    used in the original horizon-conditioned code.
    """
    return model.unembed(model.ln_final(resid))


def mean_next_token_kl_bits(base_logits: torch.Tensor, proxy_logits: torch.Tensor) -> torch.Tensor:
    """
    KL(base || proxy), averaged over batch and sequence positions, in bits.
    """
    base_lp = torch.log_softmax(base_logits[:, :-1, :].float(), dim=-1)
    proxy_lp = torch.log_softmax(proxy_logits[:, :-1, :].float(), dim=-1)
    base_p = base_lp.exp()
    kl_nats = (base_p * (base_lp - proxy_lp)).sum(dim=-1)
    kl_bits = kl_nats / math.log(2.0)
    return kl_bits.mean()


def normalized_l2(base: torch.Tensor, proxy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-token normalized L2 distance averaged over batch and sequence:
        ||base - proxy||_2 / (||base||_2 + eps).
    """
    base_f = base.float()
    proxy_f = proxy.float()
    num = torch.linalg.vector_norm(base_f - proxy_f, dim=-1)
    den = torch.linalg.vector_norm(base_f, dim=-1).clamp_min(eps)
    return (num / den).mean()


def cosine_distance(base: torch.Tensor, proxy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Mean 1-cosine similarity over token positions.
    """
    base_f = base.float()
    proxy_f = proxy.float()
    dot = (base_f * proxy_f).sum(dim=-1)
    den = torch.linalg.vector_norm(base_f, dim=-1).clamp_min(eps) * torch.linalg.vector_norm(proxy_f, dim=-1).clamp_min(eps)
    return (1.0 - dot / den).mean()


def make_matched_norm_random_delta(delta: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """
    Create Gaussian random perturbation with the same per-token L2 norm as delta.
    Shape is [batch, seq, d_model]. Norm is matched along the last dimension.
    """
    z = torch.randn(
        delta.shape,
        dtype=torch.float32,
        device=delta.device,
        generator=generator,
    )
    delta_norm = torch.linalg.vector_norm(delta.float(), dim=-1, keepdim=True)
    z_norm = torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(1e-8)
    rand = z * (delta_norm / z_norm)
    return rand.to(delta.dtype)


def match_perturbation_norm(source_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    """
    Rescale target_delta so that each token-position has the same L2 norm as
    source_delta.

    This is used for the feature-shuffled SAE control:
        source_delta = true SAE reconstruction error
        target_delta = shuffled-feature reconstruction error

    The resulting perturbation preserves the direction of target_delta but
    matches the per-token magnitude of source_delta.
    """
    source_norm = torch.linalg.vector_norm(source_delta.float(), dim=-1, keepdim=True)
    target_norm = torch.linalg.vector_norm(target_delta.float(), dim=-1, keepdim=True).clamp_min(1e-8)
    matched = target_delta.float() * (source_norm / target_norm)
    return matched.to(source_delta.dtype)


def load_model(config: Dict, device: str):
    dtype = config.get("dtype", torch.float32)
    model = HookedTransformer.from_pretrained(
        config["name"],
        device=device,
        dtype=dtype,
    )
    model.eval()
    return model, dtype


def load_sae_for_layer(config: Dict, patch_layer: int, device: str):
    """
    Load an SAE and return adapter + patch hook.

    For sae_lens configurations in this script, the SAE id is fixed. For
    sparsify Llama, the hookpoint is constructed from patch_layer.
    """
    if config["sae_backend"] == "sae_lens":
        sae, _, _ = SAE.from_pretrained(
            release=config["sae_release"],
            sae_id=config["sae_id"],
            device=device,
        )
        hook_name = config["sae_id"]
    elif config["sae_backend"] == "sparsify":
        sae_hookpoint = f"layers.{patch_layer}"
        sae = Sae.load_from_hub(config["sae_release"], hookpoint=sae_hookpoint)
        sae = sae.to(device)
        suffix = config.get("hook_suffix", "hook_resid_post")
        hook_name = make_hook_name(patch_layer, suffix)
    else:
        raise ValueError(f"Unknown sae_backend: {config['sae_backend']}")

    sae.eval()
    return SAEAdapter(sae=sae, backend=config["sae_backend"], device=device), hook_name


def valid_targets_for_patch(patch_hook: str, target_layers: List[Union[int, str]], n_layers: int):
    """
    Keep target layers r >= patch_layer and r < n_layers. Always keep "final".
    We include r == patch_layer to measure local transported divergence.
    """
    patch_layer, suffix = parse_hook_name(patch_hook)
    targets = []
    for t in target_layers:
        if t == "final":
            targets.append(t)
        else:
            if isinstance(t, int) and patch_layer <= t < n_layers:
                targets.append(t)
    return targets


# =============================================================================
# 3. Core measurement
# =============================================================================

def compute_metrics_from_targets(
    model,
    base_logits: torch.Tensor,
    base_cache: Dict[str, torch.Tensor],
    proxy_logits: torch.Tensor,
    proxy_cache: Dict[str, torch.Tensor],
    target_layers: List[Union[int, str]],
    suffix: str,
):
    """
    Compute KL and representation metrics for each target.

    For target layer r, KL is computed by applying the logit lens to h_r.
    For final, KL is computed from the final model logits.
    """
    rows = {}

    for target in target_layers:
        if target == "final":
            kl_bits = mean_next_token_kl_bits(base_logits, proxy_logits).item()
            rows[target] = {
                "kl_bits": kl_bits,
                "norm_l2": np.nan,
                "cosine_distance": np.nan,
            }
        else:
            hook = make_hook_name(int(target), suffix)
            base_t = base_cache[hook]
            proxy_t = proxy_cache[hook]
            base_t_logits = residual_to_logits(model, base_t)
            proxy_t_logits = residual_to_logits(model, proxy_t)

            rows[target] = {
                "kl_bits": mean_next_token_kl_bits(base_t_logits, proxy_t_logits).item(),
                "norm_l2": normalized_l2(base_t, proxy_t).item(),
                "cosine_distance": cosine_distance(base_t, proxy_t).item(),
            }

    return rows


def evaluate_batch_fixed_targets(
    model,
    sae_adapter: SAEAdapter,
    tokens: torch.Tensor,
    patch_hook: str,
    target_layers: List[Union[int, str]],
    top_k: int,
    random_repeats: int,
    seed: int,
    batch_index: int,
    d_sae: int,
    feature_permutation: torch.Tensor,
    controls: List[str],
):
    """
    Evaluate SAE patch and matched-norm random controls for one batch and one
    patch layer.

    Returns a list of detailed rows, one row per perturbation x target x repeat.
    """
    patch_layer, suffix = parse_hook_name(patch_hook)
    target_hooks = [make_hook_name(int(t), suffix) for t in target_layers if t != "final"]
    names_filter = sorted(set([patch_hook] + target_hooks))

    out_rows = []

    with torch.no_grad():
        base_logits, base_cache = model.run_with_cache(tokens, names_filter=names_filter)
        patch_act = base_cache[patch_hook]
        patch_recons = sae_adapter.reconstruct_tensor(patch_act, k=top_k, d_sae=d_sae)
        sae_delta = patch_recons - patch_act

        # Useful diagnostic: the actual reconstruction perturbation size.
        mean_delta_norm = torch.linalg.vector_norm(sae_delta.float(), dim=-1).mean().item()
        rel_delta_norm = (
            torch.linalg.vector_norm(sae_delta.float(), dim=-1)
            / torch.linalg.vector_norm(patch_act.float(), dim=-1).clamp_min(1e-8)
        ).mean().item()

        # --------------------------
        # Perturbation 1: true SAE reconstruction.
        # --------------------------
        def patch_sae_fn(activations, hook):
            return patch_recons.to(activations.dtype)

        with model.hooks(fwd_hooks=[(patch_hook, patch_sae_fn)]):
            proxy_logits_sae, proxy_cache_sae = model.run_with_cache(tokens, names_filter=target_hooks)

        sae_metrics = compute_metrics_from_targets(
            model=model,
            base_logits=base_logits,
            base_cache=base_cache,
            proxy_logits=proxy_logits_sae,
            proxy_cache=proxy_cache_sae,
            target_layers=target_layers,
            suffix=suffix,
        )

        for target, vals in sae_metrics.items():
            out_rows.append({
                "perturbation": "sae",
                "random_repeat": -1,
                "target_layer": str(target),
                "target_sort": 9999 if target == "final" else int(target),
                "horizon": "full" if target == "final" else int(target) - patch_layer,
                "kl_bits": vals["kl_bits"],
                "norm_l2": vals["norm_l2"],
                "cosine_distance": vals["cosine_distance"],
                "mean_delta_norm": mean_delta_norm,
                "relative_delta_norm": rel_delta_norm,
            })

        del proxy_logits_sae, proxy_cache_sae

        # --------------------------
        # Perturbation 2: feature-shuffled SAE control.
        # --------------------------
        # This control preserves the SAE top-k coefficient values and sparsity,
        # but destroys decoder-feature identity by replacing every decoder
        # feature j with d_{pi(j)}. We then match the per-token perturbation
        # norm to the true SAE perturbation so that any excess divergence is not
        # simply due to larger perturbation magnitude.
        if "feature_shuffle" in controls:
            patch_recons_shuffle_raw = sae_adapter.reconstruct_feature_shuffled_tensor(
                patch_act,
                k=top_k,
                d_sae=d_sae,
                permutation=feature_permutation,
            )
            shuffle_delta_raw = patch_recons_shuffle_raw - patch_act
            shuffle_delta = match_perturbation_norm(
                source_delta=sae_delta,
                target_delta=shuffle_delta_raw,
            )
            shuffle_patch = patch_act + shuffle_delta

            def patch_feature_shuffle_fn(activations, hook, shuffle_patch=shuffle_patch):
                return shuffle_patch.to(activations.dtype)

            with model.hooks(fwd_hooks=[(patch_hook, patch_feature_shuffle_fn)]):
                proxy_logits_shuffle, proxy_cache_shuffle = model.run_with_cache(tokens, names_filter=target_hooks)

            shuffle_metrics = compute_metrics_from_targets(
                model=model,
                base_logits=base_logits,
                base_cache=base_cache,
                proxy_logits=proxy_logits_shuffle,
                proxy_cache=proxy_cache_shuffle,
                target_layers=target_layers,
                suffix=suffix,
            )

            for target, vals in shuffle_metrics.items():
                out_rows.append({
                    "perturbation": "feature_shuffled_matched_norm",
                    "random_repeat": -1,
                    "target_layer": str(target),
                    "target_sort": 9999 if target == "final" else int(target),
                    "horizon": "full" if target == "final" else int(target) - patch_layer,
                    "kl_bits": vals["kl_bits"],
                    "norm_l2": vals["norm_l2"],
                    "cosine_distance": vals["cosine_distance"],
                    "mean_delta_norm": mean_delta_norm,
                    "relative_delta_norm": rel_delta_norm,
                })

            del patch_recons_shuffle_raw, shuffle_delta_raw, shuffle_delta, shuffle_patch
            del proxy_logits_shuffle, proxy_cache_shuffle

        # --------------------------
        # Perturbation 3: matched-norm random controls.
        # --------------------------
        if "random" in controls:
            for rep in range(random_repeats):
                gen = torch.Generator(device=tokens.device)
                gen.manual_seed(seed + 10_000 * batch_index + 997 * patch_layer + rep)

                random_delta = make_matched_norm_random_delta(sae_delta, generator=gen)
                random_patch = patch_act + random_delta

                def patch_random_fn(activations, hook, random_patch=random_patch):
                    return random_patch.to(activations.dtype)

                with model.hooks(fwd_hooks=[(patch_hook, patch_random_fn)]):
                    proxy_logits_rand, proxy_cache_rand = model.run_with_cache(tokens, names_filter=target_hooks)

                rand_metrics = compute_metrics_from_targets(
                    model=model,
                    base_logits=base_logits,
                    base_cache=base_cache,
                    proxy_logits=proxy_logits_rand,
                    proxy_cache=proxy_cache_rand,
                    target_layers=target_layers,
                    suffix=suffix,
                )

                for target, vals in rand_metrics.items():
                    out_rows.append({
                        "perturbation": "random_matched_norm",
                        "random_repeat": rep,
                        "target_layer": str(target),
                        "target_sort": 9999 if target == "final" else int(target),
                        "horizon": "full" if target == "final" else int(target) - patch_layer,
                        "kl_bits": vals["kl_bits"],
                        "norm_l2": vals["norm_l2"],
                        "cosine_distance": vals["cosine_distance"],
                        "mean_delta_norm": mean_delta_norm,
                        "relative_delta_norm": rel_delta_norm,
                    })

                del random_delta, random_patch, proxy_logits_rand, proxy_cache_rand

        del base_logits, base_cache, patch_act, patch_recons, sae_delta

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_rows


def run_model_experiment(model_key: str, config: Dict, args: argparse.Namespace):
    print(f"\n🚀 STARTING FIXED-TARGET TRANSPORT CONTROLS: {model_key}")

    torch.cuda.empty_cache()
    gc.collect()

    model, dtype = load_model(config, args.device)
    n_layers = model.cfg.n_layers

    patch_layers = args.patch_layers if args.patch_layers else config["patch_layers"]
    target_layers = parse_target_layers(args.target_layers, auto_layers=patch_layers)

    print(f"Patch layers: {patch_layers}")
    print(f"Target layers: {target_layers}")
    print(f"Random repeats: {args.random_repeats}")

    token_batches, total_cached_tokens = get_token_batches(
        model=model,
        batch_size=config["batch_size"],
        target_tokens=args.eval_tokens,
        seq_len=args.seq_len,
        buffer_size=args.tokenizer_buffer_size,
        seed=args.seed,
    )

    detailed_rows = []
    total_work_units = len(token_batches) * len(patch_layers)
    pbar = tqdm(total=total_work_units, desc=f"Eval {model_key}")

    for patch_layer in patch_layers:
        print(f"Loading SAE for patch layer {patch_layer}...")
        sae_adapter, patch_hook = load_sae_for_layer(config, patch_layer, args.device)
        d_sae = sae_adapter.get_latent_dim(d_model=model.cfg.d_model, dtype=dtype)

        # One fixed decoder-feature permutation per model/layer. This makes the
        # feature-shuffled control deterministic and comparable across batches.
        perm_gen = torch.Generator(device=args.device)
        perm_gen.manual_seed(args.seed + 123_457 * int(patch_layer))
        feature_permutation = torch.randperm(d_sae, device=args.device, generator=perm_gen)

        valid_targets = valid_targets_for_patch(patch_hook, target_layers, n_layers)

        if len(valid_targets) == 0:
            print(f"No valid target layers for patch layer {patch_layer}; skipping.")
            continue

        print(f"Patch hook: {patch_hook}; valid targets: {valid_targets}")

        try:
            for batch_idx, tokens_cpu in enumerate(token_batches):
                tokens = tokens_cpu.to(args.device, non_blocking=True)

                rows = evaluate_batch_fixed_targets(
                    model=model,
                    sae_adapter=sae_adapter,
                    tokens=tokens,
                    patch_hook=patch_hook,
                    target_layers=valid_targets,
                    top_k=args.top_k,
                    random_repeats=args.random_repeats,
                    seed=args.seed,
                    batch_index=batch_idx,
                    d_sae=d_sae,
                    feature_permutation=feature_permutation,
                    controls=args.controls,
                )

                for row in rows:
                    row.update({
                        "Model": model_key,
                        "patch_layer": patch_layer,
                        "patch_hook": patch_hook,
                        "batch_index": batch_idx,
                        "tokens_in_batch": int(tokens.numel()),
                        "total_cached_tokens": int(total_cached_tokens),
                        "top_k": int(args.top_k),
                    })
                detailed_rows.extend(rows)

                del tokens
                pbar.update(1)

        finally:
            del sae_adapter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pbar.close()

    del token_batches, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return detailed_rows


# =============================================================================
# 4. Aggregation and matrix construction
# =============================================================================

def summarize_detailed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate over batches and random repeats.
    """
    if df.empty:
        return df

    group_cols = [
        "Model",
        "patch_layer",
        "patch_hook",
        "target_layer",
        "target_sort",
        "horizon",
        "perturbation",
        "top_k",
    ]

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            mean_kl_bits=("kl_bits", "mean"),
            std_kl_bits=("kl_bits", "std"),
            mean_norm_l2=("norm_l2", "mean"),
            std_norm_l2=("norm_l2", "std"),
            mean_cosine_distance=("cosine_distance", "mean"),
            std_cosine_distance=("cosine_distance", "std"),
            mean_delta_norm=("mean_delta_norm", "mean"),
            mean_relative_delta_norm=("relative_delta_norm", "mean"),
            n_measurements=("kl_bits", "count"),
            total_cached_tokens=("total_cached_tokens", "max"),
        )
        .reset_index()
    )

    summary["std_kl_bits"] = summary["std_kl_bits"].fillna(0.0)
    summary["std_norm_l2"] = summary["std_norm_l2"].fillna(0.0)
    summary["std_cosine_distance"] = summary["std_cosine_distance"].fillna(0.0)

    # Add SAE/control ratio rows. These are the most useful reviewer-facing
    # diagnostics because they normalize away generic sensitivity of the same
    # layer-to-target map.
    sae = summary[summary["perturbation"] == "sae"].copy()
    key_cols = ["Model", "patch_layer", "target_layer", "target_sort", "horizon", "top_k"]

    control_specs = [
        ("random_matched_norm", "sae_over_random_ratio", "_rand"),
        ("feature_shuffled_matched_norm", "sae_over_feature_shuffle_ratio", "_ctrl"),
    ]

    ratio_rows = []
    for control_name, ratio_name, suffix_control in control_specs:
        ctrl = summary[summary["perturbation"] == control_name].copy()
        if ctrl.empty:
            continue

        merged = sae.merge(
            ctrl,
            on=key_cols,
            suffixes=("_sae", suffix_control),
            how="inner",
        )

        for _, row in merged.iterrows():
            ratio_rows.append({
                "Model": row["Model"],
                "patch_layer": row["patch_layer"],
                "patch_hook": row["patch_hook_sae"],
                "target_layer": row["target_layer"],
                "target_sort": row["target_sort"],
                "horizon": row["horizon"],
                "perturbation": ratio_name,
                "top_k": row["top_k"],
                "mean_kl_bits": row["mean_kl_bits_sae"] / max(row[f"mean_kl_bits{suffix_control}"], 1e-12),
                "std_kl_bits": np.nan,
                "mean_norm_l2": (
                    row["mean_norm_l2_sae"] / max(row[f"mean_norm_l2{suffix_control}"], 1e-12)
                    if not pd.isna(row["mean_norm_l2_sae"]) else np.nan
                ),
                "std_norm_l2": np.nan,
                "mean_cosine_distance": (
                    row["mean_cosine_distance_sae"] / max(row[f"mean_cosine_distance{suffix_control}"], 1e-12)
                    if not pd.isna(row["mean_cosine_distance_sae"]) else np.nan
                ),
                "std_cosine_distance": np.nan,
                "mean_delta_norm": row["mean_delta_norm_sae"],
                "mean_relative_delta_norm": row["mean_relative_delta_norm_sae"],
                "n_measurements": min(row["n_measurements_sae"], row[f"n_measurements{suffix_control}"]),
                "total_cached_tokens": row["total_cached_tokens_sae"],
            })

    if ratio_rows:
        summary = pd.concat([summary, pd.DataFrame(ratio_rows)], ignore_index=True)

    return summary.sort_values(["Model", "patch_layer", "target_sort", "perturbation"]).reset_index(drop=True)


def save_matrices(summary: pd.DataFrame, output_dir: str):
    """
    Save pivot matrices for later paper/table analysis.

    Rows: patch_layer
    Columns: target_layer
    Values: metric mean
    """
    matrix_dir = os.path.join(output_dir, "matrices")
    os.makedirs(matrix_dir, exist_ok=True)

    metrics = {
        "kl_bits": "mean_kl_bits",
        "norm_l2": "mean_norm_l2",
        "cosine_distance": "mean_cosine_distance",
    }

    for model_name in summary["Model"].unique():
        sub_model = summary[summary["Model"] == model_name].copy()
        model_safe = safe_model_name(model_name)

        for perturbation in sub_model["perturbation"].unique():
            sub = sub_model[sub_model["perturbation"] == perturbation].copy()
            for metric_name, col in metrics.items():
                if col not in sub.columns:
                    continue
                mat = sub.pivot_table(
                    index="patch_layer",
                    columns="target_layer",
                    values=col,
                    aggfunc="mean",
                )
                # Sort columns by numeric target then final.
                sorted_cols = sorted(mat.columns, key=lambda x: 9999 if str(x) == "final" else int(x))
                mat = mat[sorted_cols]
                path = os.path.join(matrix_dir, f"{model_safe}_{perturbation}_{metric_name}_matrix.csv")
                mat.to_csv(path)
                print(f"Saved matrix: {path}")


# =============================================================================
# 5. Plotting
# =============================================================================

def plot_heatmap_from_matrix(
    matrix: pd.DataFrame,
    title: str,
    cbar_label: str,
    out_path_prefix: str,
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
):
    """
    Matplotlib heatmap with numeric values. Avoids seaborn dependency and makes
    paper-ready matrices.
    """
    if matrix.empty:
        return

    values = matrix.to_numpy(dtype=float)
    fig_w = max(6.0, 0.85 * len(matrix.columns) + 2.2)
    fig_h = max(4.5, 0.55 * len(matrix.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels([str(c) for c in matrix.columns], rotation=45, ha="right")
    ax.set_yticklabels([str(i) for i in matrix.index])

    ax.set_xlabel("Target layer r")
    ax.set_ylabel("Patch layer ℓ")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # Annotate values.
    finite_vals = values[np.isfinite(values)]
    threshold = np.nanmedian(finite_vals) if finite_vals.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                color = "white" if val > threshold else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    plt.savefig(out_path_prefix + ".png", dpi=450, bbox_inches="tight")
    plt.savefig(out_path_prefix + ".pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_path_prefix}.png")
    print(f"Saved {out_path_prefix}.pdf")


def plot_all_heatmaps(summary: pd.DataFrame, output_dir: str):
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    metrics = {
        "kl_bits": ("mean_kl_bits", "Mean KL (bits)"),
        "norm_l2": ("mean_norm_l2", "Normalized L2"),
    }

    for model_name in summary["Model"].unique():
        model_safe = safe_model_name(model_name)
        sub_model = summary[summary["Model"] == model_name].copy()

        for perturbation, pretty in [
            ("sae", "SAE reconstruction"),
            ("feature_shuffled_matched_norm", "Feature-shuffled matched-norm"),
            ("sae_over_feature_shuffle_ratio", "SAE / feature-shuffled ratio"),
            ("random_matched_norm", "Matched-norm random"),
            ("sae_over_random_ratio", "SAE / random ratio"),
        ]:
            sub = sub_model[sub_model["perturbation"] == perturbation].copy()
            if sub.empty:
                continue

            for metric_name, (col, label) in metrics.items():
                mat = sub.pivot_table(index="patch_layer", columns="target_layer", values=col, aggfunc="mean")
                if mat.empty:
                    continue
                sorted_cols = sorted(mat.columns, key=lambda x: 9999 if str(x) == "final" else int(x))
                mat = mat[sorted_cols]

                if perturbation in {"sae_over_random_ratio", "sae_over_feature_shuffle_ratio"}:
                    cmap = "coolwarm"
                    vmin, vmax = 0.0, 1.0 if metric_name != "norm_l2" else None
                    cbar_label = f"{label} ratio"
                else:
                    cmap = "viridis"
                    vmin, vmax = None, None
                    cbar_label = label

                title = f"{model_name}: {pretty} ({label})"
                prefix = os.path.join(fig_dir, f"{model_safe}_{perturbation}_{metric_name}_heatmap")
                plot_heatmap_from_matrix(
                    matrix=mat,
                    title=title,
                    cbar_label=cbar_label,
                    out_path_prefix=prefix,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )


def plot_horizon_lines_from_summary(summary: pd.DataFrame, output_dir: str):
    """
    Reconstruct horizon-style line plots from fixed targets:
        horizon = target_layer - patch_layer.
    This is useful for comparing against the original Figure 3, but now with
    SAE and matched-random curves on the same axes.
    """
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    for model_name in summary["Model"].unique():
        sub_model = summary[
            (summary["Model"] == model_name)
            & (summary["target_layer"] != "final")
            & (summary["perturbation"].isin(["sae", "feature_shuffled_matched_norm", "random_matched_norm"]))
        ].copy()
        if sub_model.empty:
            continue

        model_safe = safe_model_name(model_name)
        patch_layers = sorted(sub_model["patch_layer"].unique())

        fig, axes = plt.subplots(
            1,
            len(patch_layers),
            figsize=(4.2 * len(patch_layers), 3.7),
            squeeze=False,
            sharey=True,
        )
        axes = axes[0]

        for ax, layer in zip(axes, patch_layers):
            sub_layer = sub_model[sub_model["patch_layer"] == layer].copy()
            for perturbation, linestyle, label in [
                ("sae", "-", "SAE"),
                ("feature_shuffled_matched_norm", "-.", "feature-shuffled"),
                ("random_matched_norm", "--", "matched random"),
            ]:
                g = sub_layer[sub_layer["perturbation"] == perturbation].copy()
                g = g[g["horizon"] != "full"]
                if g.empty:
                    continue
                g["horizon_num"] = g["horizon"].astype(int)
                g = g.sort_values("horizon_num")
                ax.plot(
                    g["horizon_num"],
                    g["mean_kl_bits"],
                    marker="o",
                    linewidth=2.0,
                    linestyle=linestyle,
                    label=label,
                )
            ax.set_title(f"Patch L{layer}")
            ax.set_xlabel("Absolute target horizon r-ℓ")
            ax.grid(True, alpha=0.25)

        axes[0].set_ylabel("Mean KL (bits)")
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            axes[-1].legend(frameon=True)

        fig.suptitle(f"{model_name}: fixed-target transport by horizon", fontsize=14, fontweight="bold")
        plt.tight_layout()
        prefix = os.path.join(fig_dir, f"{model_safe}_fixed_target_horizon_lines")
        plt.savefig(prefix + ".png", dpi=450, bbox_inches="tight")
        plt.savefig(prefix + ".pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {prefix}.png")
        print(f"Saved {prefix}.pdf")


# =============================================================================
# 6. CLI and runner
# =============================================================================

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Fixed-target transport and matched-norm random controls for SAE rollout analysis."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="Llama-3.1-8B",
        help="Comma-separated model keys. Available: GPT-2 Small,Gemma-2B,Llama-3.1-8B",
    )
    parser.add_argument(
        "--patch-layers",
        type=str,
        default="",
        help="Optional comma-separated patch layers overriding the model config.",
    )
    parser.add_argument(
        "--target-layers",
        type=str,
        default="auto",
        help="Comma-separated absolute target layers plus optional final. Use auto for patch layers plus final.",
    )
    parser.add_argument("--eval-tokens", type=int, default=CONFIG["EVAL_TOKENS"])
    parser.add_argument("--seq-len", type=int, default=CONFIG["SEQ_LEN"])
    parser.add_argument("--tokenizer-buffer-size", type=int, default=CONFIG["TOKENIZER_BUFFER_SIZE"])
    parser.add_argument("--top-k", type=int, default=CONFIG["TOP_K"])
    parser.add_argument(
        "--controls",
        type=str,
        default="sae,feature_shuffle",
        help=(
            "Comma-separated controls to run. Always includes sae. "
            "Options: feature_shuffle, random. "
            "Default runs true SAE plus feature-shuffled matched-norm control. "
            "Use sae,feature_shuffle,random to also include isotropic Gaussian noise."
        ),
    )
    parser.add_argument("--random-repeats", type=int, default=CONFIG["RANDOM_REPEATS"])
    parser.add_argument("--seed", type=int, default=CONFIG["SEED"])
    parser.add_argument("--device", type=str, default=CONFIG["DEVICE"])
    parser.add_argument("--output-dir", type=str, default=CONFIG["OUTPUT_DIR"])
    return parser


def main():
    args = build_arg_parser().parse_args()
    args.models = parse_csv_strings(args.models)
    args.patch_layers = parse_csv_ints(args.patch_layers)
    args.controls = sorted(set(parse_csv_strings(args.controls)))
    if "sae" not in args.controls:
        args.controls.insert(0, "sae")

    valid_controls = {"sae", "feature_shuffle", "random"}
    invalid_controls = [c for c in args.controls if c not in valid_controls]
    if invalid_controls:
        raise ValueError(f"Unknown controls {invalid_controls}. Valid controls: {sorted(valid_controls)}")

    missing = [m for m in args.models if m not in CONFIG["MODELS"]]
    if missing:
        raise ValueError(f"Unknown models {missing}. Available: {list(CONFIG['MODELS'].keys())}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 88)
    print("FIXED-TARGET TRANSPORT + MATCHED-NORM RANDOM CONTROL")
    print("=" * 88)
    print(f"Models          : {args.models}")
    print(f"Patch override  : {args.patch_layers if args.patch_layers else 'model defaults'}")
    print(f"Target layers   : {args.target_layers}")
    print(f"Eval tokens     : {args.eval_tokens:,}")
    print(f"Top-k           : {args.top_k}")
    print(f"Controls        : {args.controls}")
    print(f"Random repeats  : {args.random_repeats}")
    print(f"Device          : {args.device}")
    print(f"Output dir      : {args.output_dir}")
    print("=" * 88)

    all_rows = []
    for model_key in args.models:
        rows = run_model_experiment(model_key, CONFIG["MODELS"][model_key], args)
        all_rows.extend(rows)

        # Save partial results after every model.
        pd.DataFrame(all_rows).to_csv(
            os.path.join(args.output_dir, "fixed_target_transport_detailed_partial.csv"),
            index=False,
        )

    detailed = pd.DataFrame(all_rows)
    detailed_path = os.path.join(args.output_dir, "fixed_target_transport_detailed.csv")
    detailed.to_csv(detailed_path, index=False)
    print(f"Saved {detailed_path}")

    summary = summarize_detailed(detailed)
    summary_path = os.path.join(args.output_dir, "fixed_target_transport_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    if not summary.empty:
        save_matrices(summary, args.output_dir)
        plot_all_heatmaps(summary, args.output_dir)
        plot_horizon_lines_from_summary(summary, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
