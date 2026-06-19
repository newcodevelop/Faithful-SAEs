#!/usr/bin/env python3
"""
Feature-fragmentation / aliasing control for SAE sparse-proxy certificates.

This script is designed as a controlled empirical test of the claim that the
SAE certificate is not merely a reconstruction-quality score.

High-level idea
---------------
Given a trained SAE feature j with decoder direction d_j, create r aliases
(j,0), ..., (j,r-1), all of which decode to exactly the same direction d_j.
For each token/context, route the activation of feature j to exactly one alias.
Therefore, the unrestricted reconstruction is unchanged:

    c_j(x) d_j  ==  c_{j,a(x)}(x) d_{j,a(x)}.

However, the identity of the active support is fragmented across examples.
The calibration pool G* is now a pool of aliases rather than original SAE
features. If the alias identity is unstable between calibration and evaluation,
then P = |G*| and/or eta increase even though reconstruction is preserved.

This implements the control without explicitly materializing an expanded SAE
decoder of size m*r. Instead, it keeps the original SAE decoder and simulates
alias identities only in the support statistics and pool restriction.

What the script measures
------------------------
For each model and each alias factor r, the script computes:

    P_r                  : alias-pool size observed on calibration data
    eta_hat_r            : evaluation probability that unrestricted top-k alias
                           support is not contained in the calibration alias pool
    eps_loss_hat         : loss-level gap between base LM and unrestricted SAE proxy
                           (unchanged across alias factors except numerical noise)
    R_hat_hG_r           : empirical risk of the alias-pool-restricted proxy
    total certificate    : Eq. 15-style certificate with m replaced by m*r and
                           P replaced by P_r

Expected empirical signature
----------------------------
If the certificate only measured autoencoding quality, aliasing should not affect
it because unrestricted reconstruction is unchanged. If the certificate also
requires reusable sparse support identities, increasing r should worsen P_r,
eta_hat_r, and the total bound. This is the desired behavior: it empirically
instantiates the Appendix J.1 phenomenon that perfect reconstruction is
insufficient when support mismatch is high.

The implementation follows the structure of the user's Figure-1 code:
    - load model and SAE
    - calibrate active support pool
    - evaluate bound terms over increasing sample sizes
    - save CSVs and plots

Usage examples
--------------
Run GPT-2 Small only:
    python feature_fragmentation_alias_control.py --models "GPT-2 Small"

Run all configured models and alias factors:
    python feature_fragmentation_alias_control.py \
        --models "GPT-2 Small,Gemma-2B,Llama-3.1-8B" \
        --alias-factors 1,2,4,8,16,32

Use the stronger adversarial split aliasing mode:
    python feature_fragmentation_alias_control.py --alias-mode split

Notes
-----
1. alias_factor=1 recovers the original SAE-support certificate.
2. token_hash aliasing is the default and is the most natural control.
3. split aliasing intentionally routes calibration and evaluation to disjoint
   alias subsets. It is useful as a stress test but should be reported as an
   adversarial diagnostic rather than the main naturalistic control.
"""

import argparse
import gc
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

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
# 1. Default experiment configuration
# =============================================================================

DEFAULT_CONFIG = {
    "CALIBRATION_TOKENS": 2_240_000,
    "N_STEPS": [
        32_000,
        64_000,
        96_000,
        32_000 * 4,
        32_000 * 5,
        32_000 * 6,
        32_000 * 7,
        32_000 * 8,
        32_000 * 9,
        32_000 * 10,
    ],
    "ALPHA": 0.5,
    "DELTA": 0.05,
    "TOP_K": 64,
    "SEQ_LEN": 32,
    "TOKENIZER_BUFFER_SIZE": 32_000,
    "SEED": 42,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "ALIAS_FACTORS": [1, 2, 4, 8, 16, 32],
    "ALIAS_MODE": "token_hash",  # one of {"token_hash", "split"}
    "OUTPUT_DIR": "./feature_fragmentation_ops",
    "MODELS": {
        "GPT-2 Small": {
            "name": "gpt2-small",
            "sae_backend": "sae_lens",
            "sae_release": "gpt2-small-res-jb",
            "sae_id": "blocks.6.hook_resid_pre",
            "hook_name": "blocks.6.hook_resid_pre",
            "batch_size": 16,
            "color": "#1f77b4",
        },
        "Gemma-2B": {
            "name": "gemma-2b",
            "sae_backend": "sae_lens",
            "sae_release": "gemma-2b-res-jb",
            "sae_id": "blocks.12.hook_resid_post",
            "hook_name": "blocks.12.hook_resid_post",
            "batch_size": 16,
            "color": "#d62728",
        },
        "Llama-3.1-8B": {
            "name": "meta-llama/Meta-Llama-3-8B",
            "sae_backend": "sparsify",
            "sae_release": "EleutherAI/sae-llama-3-8b-32x",
            "sae_hookpoint": "layers.30",
            "hook_name": "blocks.30.hook_resid_post",
            "batch_size": 16,
            "color": "#26ba15",
            "dtype": torch.bfloat16,
        },
    },
}


# =============================================================================
# 2. Helpers
# =============================================================================

def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def get_topk_sparse(activations: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return top-k feature values and indices after ReLU-style thresholding.
    This matches the sparse SAE usage in the original code, where negative
    pre-activations are suppressed after selecting the top-k coordinates.
    """
    top_acts, top_indices = torch.topk(activations, k=k, dim=-1)
    top_acts = torch.relu(top_acts)
    return top_acts, top_indices


def apply_topk_dense(activations: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only top-k activations in a dense tensor and zero out the rest."""
    topk_vals, topk_inds = torch.topk(activations, k=k, dim=-1)
    topk_vals = torch.relu(topk_vals)
    sparse = torch.zeros_like(activations)
    sparse.scatter_(-1, topk_inds, topk_vals)
    return sparse


def smoothed_bpd_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    alpha: float,
    vocab_size: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Smoothed next-token BPD loss.

    The smoothing follows the bounded-loss construction used in the original
    certificate:
        p_tilde(y|x) = (1-alpha) p(y|x) + alpha / |V|.
    """
    probs = torch.softmax(logits, dim=-1)
    probs_shifted = probs[:, :-1, :]
    tokens_shifted = tokens[:, 1:]
    true_probs = torch.gather(
        probs_shifted,
        -1,
        tokens_shifted.unsqueeze(-1),
    ).squeeze(-1)
    smoothed_probs = (1.0 - alpha) * true_probs + (alpha / vocab_size)
    log_probs = -torch.log2(smoothed_probs)
    loss_per_seq = log_probs.mean(dim=-1)
    if reduction == "mean":
        return loss_per_seq.mean()
    return loss_per_seq


def mean_next_token_kl_bits(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Mean KL(p || q) in bits over next-token distributions and sequence positions.

    This is not a term in Eq. 15, but it is a useful behavioral-fidelity
    diagnostic for the downstream-transfer theorem.
    """
    logp = torch.log_softmax(logits_p[:, :-1, :], dim=-1)
    logq = torch.log_softmax(logits_q[:, :-1, :], dim=-1)
    p = torch.exp(logp)
    kl_nats = (p * (logp - logq)).sum(dim=-1)
    kl_bits = kl_nats / math.log(2.0)
    kl_per_seq = kl_bits.mean(dim=-1)
    if reduction == "mean":
        return kl_per_seq.mean()
    return kl_per_seq


def stable_alias_slots(
    flat_token_ids: torch.Tensor,
    alias_factor: int,
    seed: int,
    mode: str,
    phase: str,
) -> torch.Tensor:
    """
    Deterministically assign each token/context to an alias slot in {0,...,r-1}.

    token_hash mode:
        Calibration and evaluation use the same hash family. This is the main
        control: each feature is fragmented into r aliases, and feature reuse
        across held-out examples must be rediscovered through calibration.

    split mode:
        Calibration uses the first half of aliases; evaluation uses the second
        half. This is an adversarial stress test that deliberately creates
        out-of-distribution support identity mismatch.
    """
    if alias_factor <= 1:
        return torch.zeros_like(flat_token_ids, dtype=torch.long)

    if mode not in {"token_hash", "split"}:
        raise ValueError(f"Unknown alias mode: {mode}")

    # Simple deterministic integer hash. We avoid Python's built-in hash because
    # it is process-randomized. The constants are standard LCG-style constants.
    hashed = flat_token_ids.to(torch.long) * 1_103_515_245 + 12_345 + int(seed)
    hashed = torch.remainder(torch.abs(hashed), 2_147_483_647)

    if mode == "token_hash":
        return torch.remainder(hashed, alias_factor).to(torch.long)

    # Adversarial split mode.
    left = max(1, alias_factor // 2)
    right = alias_factor - left
    if phase == "calibration":
        return torch.remainder(hashed, left).to(torch.long)
    if right <= 0:
        return torch.zeros_like(flat_token_ids, dtype=torch.long)
    return (left + torch.remainder(hashed, right)).to(torch.long)


def batch_flat_token_ids(token_offset: int, num_flat_tokens: int, device: str) -> torch.Tensor:
    """Return global token ids for all token positions in the current batch."""
    return torch.arange(
        token_offset,
        token_offset + num_flat_tokens,
        device=device,
        dtype=torch.long,
    )


def get_tokens_generator(
    model: HookedTransformer,
    batch_size: int,
    device: str,
    mode: str,
    calibration_limit: int = 0,
    seq_len: int = 32,
    tokenizer_buffer_size: int = 32_000,
    seed: int = 42,
) -> Iterable[Tuple[torch.Tensor, int]]:
    """
    Yield disjoint batches of tokens and a global token offset.

    For evaluation, the generator skips approximately calibration_limit tokens
    before yielding. This mirrors the original Figure-1 code and keeps
    calibration/evaluation streams disjoint.
    """
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=tokenizer_buffer_size)

    iterator = iter(dataset)
    tokens_processed_global = 0
    skip_needed = mode == "evaluation"

    while True:
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

        tokens = tokens[:, :seq_len]
        num_tok = tokens.numel()
        current_offset = tokens_processed_global

        if skip_needed:
            tokens_processed_global += num_tok
            if tokens_processed_global < calibration_limit:
                continue
            skip_needed = False
            current_offset = tokens_processed_global - num_tok

        elif mode == "calibration":
            if tokens_processed_global >= calibration_limit:
                break
            tokens_processed_global += num_tok

        else:
            tokens_processed_global += num_tok

        yield tokens.to(device), current_offset


def extract_pre_acts(encode_out):
    """
    Normalize encode outputs across SAE backends.

    sae_lens generally returns a dense tensor directly.
    sparsify may return an object or dict that stores pre_acts.
    """
    if torch.is_tensor(encode_out):
        return encode_out
    if hasattr(encode_out, "pre_acts"):
        return encode_out.pre_acts
    if isinstance(encode_out, dict) and "pre_acts" in encode_out:
        return encode_out["pre_acts"]
    if isinstance(encode_out, (list, tuple)) and len(encode_out) > 0:
        return encode_out[0]
    raise TypeError(f"Unsupported SAE encode output type: {type(encode_out)}")


class SAEAdapter:
    """
    Thin wrapper that makes sae_lens and sparsify SAEs look the same.
    """

    def __init__(self, sae, backend: str, device: str):
        self.sae = sae
        self.backend = backend
        self.device = device

    def encode_pre_acts(self, flat_act: torch.Tensor) -> torch.Tensor:
        return extract_pre_acts(self.sae.encode(flat_act))

    def topk_code(
        self,
        feature_acts_raw: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return top-k values, top-k indices, and dense active mask.
        """
        top_acts, top_indices = get_topk_sparse(feature_acts_raw, k)
        active_mask = torch.zeros_like(feature_acts_raw, dtype=torch.bool)
        active_mask.scatter_(-1, top_indices, top_acts > 0)
        return top_acts, top_indices, active_mask

    def decode_from_topk(
        self,
        top_acts: torch.Tensor,
        top_indices: torch.Tensor,
        d_sae: int,
    ) -> torch.Tensor:
        """
        Decode a sparse top-k code using the underlying SAE decoder.
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

    def decode_topk(
        self,
        feature_acts_raw: torch.Tensor,
        k: int,
        d_sae: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode the unrestricted top-k code.

        Returns:
            reconstruction, active_mask, top_acts, top_indices
        """
        top_acts, top_indices, active_mask = self.topk_code(feature_acts_raw, k)
        recons = self.decode_from_topk(top_acts, top_indices, d_sae)
        return recons, active_mask, top_acts, top_indices

    def get_latent_dim(self, d_model: int, dtype: torch.dtype) -> int:
        if self.backend == "sae_lens":
            return int(self.sae.cfg.d_sae)
        dummy_act = torch.zeros(1, d_model, device=self.device, dtype=dtype)
        return int(self.encode_pre_acts(dummy_act).shape[-1])


def load_model_and_sae(config: Dict, device: str):
    dtype = config.get("dtype", torch.float32)
    model = HookedTransformer.from_pretrained(
        config["name"],
        device=device,
        dtype=dtype,
    )

    if config["sae_backend"] == "sae_lens":
        sae, _, _ = SAE.from_pretrained(
            release=config["sae_release"],
            sae_id=config["sae_id"],
            device=device,
        )
        hook_name = config["hook_name"]
    elif config["sae_backend"] == "sparsify":
        sae = Sae.load_from_hub(
            config["sae_release"],
            hookpoint=config["sae_hookpoint"],
        )
        sae = sae.to(device)
        hook_name = config["hook_name"]
    else:
        raise ValueError(f"Unknown sae_backend: {config['sae_backend']}")

    sae.eval()
    adapter = SAEAdapter(sae=sae, backend=config["sae_backend"], device=device)
    return model, adapter, hook_name, dtype


# =============================================================================
# 3. Calibration: build alias pools
# =============================================================================

def calibrate_alias_pools(
    model: HookedTransformer,
    sae_adapter: SAEAdapter,
    token_gen: Iterable[Tuple[torch.Tensor, int]],
    hook_name: str,
    target_tokens: int,
    alias_factors: List[int],
    alias_mode: str,
    seed: int,
    top_k: int,
    device: str,
    d_sae: int,
) -> Dict[int, torch.Tensor]:
    """
    Build a calibration pool for every alias factor in one pass.

    For alias factor r, the alias pool is a boolean vector of length m*r.
    The alias id for original feature j at token/context x is:

        alias_id(j,x) = j*r + a(x),

    where a(x) is the token/context alias slot.

    The decoder is not expanded: this is support bookkeeping only.
    """
    print(f"  --> [Calibration] Building alias pools for r={alias_factors}")

    pools = {
        r: torch.zeros(d_sae * r, dtype=torch.bool, device=device)
        for r in alias_factors
    }

    total_tokens = 0
    pbar = tqdm(total=target_tokens, desc="Calibration")

    for tokens, token_offset in token_gen:
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            act = cache[hook_name]
            flat_act = act.reshape(-1, act.shape[-1])
            num_flat = flat_act.shape[0]

            feature_acts_raw = sae_adapter.encode_pre_acts(flat_act)
            top_acts, top_indices, _ = sae_adapter.topk_code(feature_acts_raw, top_k)
            active_topk = top_acts > 0

            flat_ids = batch_flat_token_ids(token_offset, num_flat, device)

            for r in alias_factors:
                slots = stable_alias_slots(
                    flat_ids,
                    alias_factor=r,
                    seed=seed,
                    mode=alias_mode,
                    phase="calibration",
                )
                alias_ids = top_indices.to(torch.long) * r + slots.unsqueeze(-1)
                alias_ids = alias_ids[active_topk]
                if alias_ids.numel() > 0:
                    pools[r][alias_ids.reshape(-1)] = True

        num_tok = tokens.numel()
        total_tokens += num_tok
        pbar.update(num_tok)
        if total_tokens >= target_tokens:
            break

    pbar.close()

    for r in alias_factors:
        P = int(pools[r].sum().item())
        print(f"  <-- [Calibration] r={r:>2}: alias pool size P_r={P:,} / m_r={d_sae*r:,}")

    return pools


# =============================================================================
# 4. Evaluation: measure bound terms under alias fragmentation
# =============================================================================

@dataclass
class RunningAliasResults:
    loss_hG: List[float]
    eps_loss: List[float]
    pool_violation: List[float]
    kl_M_Som: List[float]

    @staticmethod
    def empty() -> "RunningAliasResults":
        return RunningAliasResults([], [], [], [])


def alias_restricted_reconstruction(
    feature_acts_raw: torch.Tensor,
    pool_flat: torch.Tensor,
    alias_slots: torch.Tensor,
    alias_factor: int,
    sae_adapter: SAEAdapter,
    top_k: int,
    d_sae: int,
) -> torch.Tensor:
    """
    Construct the pool-restricted alias proxy h_{G*} without materializing the
    expanded alias dictionary.

    For token/context i and original feature j, the corresponding alias is
        j*r + alias_slots[i].

    The alias is allowed iff it belongs to the calibration pool. This induces
    an allowed mask over the original m features for each token/context. We mask
    the original pre-activations by this allowed set and then apply TopK+decode.

    This is equivalent to applying the theorem's pool restriction in the
    expanded alias support space and then decoding through tied alias directions.
    """
    m = d_sae
    pool_matrix = pool_flat.view(m, alias_factor)

    # allowed_mask[i,j] = True iff alias (j, alias_slots[i]) is in G*
    # Shape: [num_flat_tokens, m]
    allowed_mask = pool_matrix[:, alias_slots].transpose(0, 1).contiguous()

    masked_acts = feature_acts_raw * allowed_mask.to(feature_acts_raw.dtype)
    recons, _, _, _ = sae_adapter.decode_topk(masked_acts, top_k, d_sae)
    return recons


def evaluate_alias_control_for_model(
    model_key: str,
    config: Dict,
    args: argparse.Namespace,
) -> Tuple[List[Dict], List[Dict]]:
    print(f"\n🚀 STARTING FEATURE-FRAGMENTATION CONTROL: {model_key}")

    torch.cuda.empty_cache()
    gc.collect()

    try:
        model, sae_adapter, hook_name, dtype = load_model_and_sae(config, args.device)
    except Exception as exc:
        print(f"Error loading {model_key}: {exc}")
        return [], []

    d_sae = sae_adapter.get_latent_dim(d_model=model.cfg.d_model, dtype=dtype)
    vocab_size = model.cfg.d_vocab

    print(f"  SAE dictionary size m={d_sae:,}")
    print(f"  hook_name={hook_name}")
    print(f"  top_k={args.top_k}, alias_factors={args.alias_factors}, alias_mode={args.alias_mode}")

    B = math.log2(vocab_size / args.alpha)
    random_baseline = math.log2(vocab_size)

    cal_gen = get_tokens_generator(
        model=model,
        batch_size=config["batch_size"],
        device=args.device,
        mode="calibration",
        calibration_limit=args.calibration_tokens,
        seq_len=args.seq_len,
        tokenizer_buffer_size=args.tokenizer_buffer_size,
        seed=args.seed,
    )

    alias_pools = calibrate_alias_pools(
        model=model,
        sae_adapter=sae_adapter,
        token_gen=cal_gen,
        hook_name=hook_name,
        target_tokens=args.calibration_tokens,
        alias_factors=args.alias_factors,
        alias_mode=args.alias_mode,
        seed=args.seed,
        top_k=args.top_k,
        device=args.device,
        d_sae=d_sae,
    )

    pool_summary = []
    for r, pool in alias_pools.items():
        pool_summary.append({
            "Model": model_key,
            "alias_factor": r,
            "m_original": d_sae,
            "m_alias": d_sae * r,
            "P_alias": int(pool.sum().item()),
            "P_fraction": float(pool.sum().item()) / float(d_sae * r),
            "alias_mode": args.alias_mode,
            "calibration_tokens": args.calibration_tokens,
            "top_k": args.top_k,
        })

    eval_gen = get_tokens_generator(
        model=model,
        batch_size=config["batch_size"],
        device=args.device,
        mode="evaluation",
        calibration_limit=args.calibration_tokens,
        seq_len=args.seq_len,
        tokenizer_buffer_size=args.tokenizer_buffer_size,
        seed=args.seed,
    )

    running = {r: RunningAliasResults.empty() for r in args.alias_factors}
    plot_points = []
    total_tokens = 0
    current_step_idx = 0
    max_steps = max(args.n_steps)

    pbar = tqdm(total=max_steps, desc="Evaluation")

    for tokens, token_offset in eval_gen:
        with torch.no_grad():
            orig_logits, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            loss_M = smoothed_bpd_loss(
                orig_logits,
                tokens,
                alpha=args.alpha,
                vocab_size=vocab_size,
                reduction="none",
            )

            act = cache[hook_name]
            flat_act = act.reshape(-1, act.shape[-1])
            num_flat = flat_act.shape[0]
            flat_ids = batch_flat_token_ids(token_offset, num_flat, args.device)

            feature_acts_raw = sae_adapter.encode_pre_acts(flat_act)

            # Unrestricted SAE proxy: identical for every alias factor because
            # alias fragmentation ties decoder directions and only changes
            # feature identities.
            recons_unrestricted, active_mask_unrestricted, top_acts, top_indices = (
                sae_adapter.decode_topk(feature_acts_raw, args.top_k, d_sae)
            )
            recons_unrestricted = recons_unrestricted.reshape(act.shape)

            def hook_unrestricted(activations, hook):
                return recons_unrestricted

            logits_Som = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, hook_unrestricted)],
            )
            loss_Som = smoothed_bpd_loss(
                logits_Som,
                tokens,
                alpha=args.alpha,
                vocab_size=vocab_size,
                reduction="none",
            )
            eps_loss_batch = torch.abs(loss_M - loss_Som)
            kl_batch = mean_next_token_kl_bits(orig_logits, logits_Som, reduction="none")

            active_topk = top_acts > 0

            for r in args.alias_factors:
                pool_flat = alias_pools[r]
                slots = stable_alias_slots(
                    flat_ids,
                    alias_factor=r,
                    seed=args.seed,
                    mode=args.alias_mode,
                    phase="evaluation",
                )

                # Pool violation in alias space:
                # The unrestricted active alias support is {j*r + slot(x)}.
                alias_ids_topk = top_indices.to(torch.long) * r + slots.unsqueeze(-1)
                unseen_alias = ~pool_flat[alias_ids_topk]
                violation_flat = (active_topk & unseen_alias).any(dim=-1)
                seq_has_violation = violation_flat.reshape(tokens.shape[0], -1).any(dim=-1).float()

                recons_restricted = alias_restricted_reconstruction(
                    feature_acts_raw=feature_acts_raw,
                    pool_flat=pool_flat,
                    alias_slots=slots,
                    alias_factor=r,
                    sae_adapter=sae_adapter,
                    top_k=args.top_k,
                    d_sae=d_sae,
                ).reshape(act.shape)

                def hook_restricted(activations, hook, recons=recons_restricted):
                    return recons

                logits_hG = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, hook_restricted)],
                )
                loss_hG = smoothed_bpd_loss(
                    logits_hG,
                    tokens,
                    alpha=args.alpha,
                    vocab_size=vocab_size,
                    reduction="none",
                )

                running[r].loss_hG.extend(loss_hG.detach().cpu().float().numpy())
                running[r].eps_loss.extend(eps_loss_batch.detach().cpu().float().numpy())
                running[r].pool_violation.extend(seq_has_violation.detach().cpu().float().numpy())
                running[r].kl_M_Som.extend(kl_batch.detach().cpu().float().numpy())

                # Clear per-r tensors aggressively on small GPUs.
                del recons_restricted, logits_hG, loss_hG, seq_has_violation

            num_tok = tokens.numel()
            total_tokens += num_tok
            pbar.update(num_tok)

            if current_step_idx < len(args.n_steps) and total_tokens >= args.n_steps[current_step_idx]:
                N = total_tokens / args.seq_len
                delta = args.delta

                for r in args.alias_factors:
                    P_r = int(alias_pools[r].sum().item())
                    m_r = d_sae * r

                    R_hat_hG = float(np.mean(running[r].loss_hG))
                    eps_loss_hat = float(np.mean(running[r].eps_loss))
                    eta_hat = float(np.mean(running[r].pool_violation))
                    kl_hat = float(np.mean(running[r].kl_M_Som))

                    # Eq. 15-style terms with alias-space dictionary size m_r and
                    # alias-pool size P_r.
                    t1 = R_hat_hG
                    t2 = eps_loss_hat
                    t3 = eta_hat * B
                    ssd = P_r * math.log((math.e * m_r) / P_r) if P_r > 0 else 0.0
                    t4 = B * math.sqrt((ssd + math.log(2.0 / delta)) / (2.0 * N))
                    t5 = B * math.sqrt(math.log(4.0 / delta) / (2.0 * N))
                    total_bound = t1 + t2 + t3 + t4 + t5

                    asymptotic_floor = t1 + t2 + t3

                    plot_points.append({
                        "Model": model_key,
                        "alias_factor": r,
                        "alias_mode": args.alias_mode,
                        "N": N,
                        "Total Bound": total_bound,
                        "Random Baseline": random_baseline,
                        "B": B,
                        "P": P_r,
                        "m_alias": m_r,
                        "R_hat_hG": R_hat_hG,
                        "eps_loss_hat": eps_loss_hat,
                        "eta_hat": eta_hat,
                        "KL_M_Som_bits": kl_hat,
                        "t_proxy_risk": t1,
                        "t_reconstruction_gap": t2,
                        "t_mismatch": t3,
                        "t_sparse_complexity": t4,
                        "t_concentration": t5,
                        "asymptotic_floor": asymptotic_floor,
                        "top_k": args.top_k,
                        "alpha": args.alpha,
                        "delta": args.delta,
                        "calibration_tokens": args.calibration_tokens,
                    })

                    print(
                        f"  [N={N:,.0f} | r={r:>2}] "
                        f"Bound={total_bound:.3f} Base={random_baseline:.2f} "
                        f"| P={P_r:,}/{m_r:,} "
                        f"| Risk={R_hat_hG:.3f} Gap={eps_loss_hat:.3f} "
                        f"| Eta={eta_hat:.4f} KL={kl_hat:.4f}"
                    )

                current_step_idx += 1
                if current_step_idx >= len(args.n_steps):
                    break

        if current_step_idx >= len(args.n_steps):
            break

    pbar.close()

    # Optional smooth extrapolation of the finite-sample terms.
    if not args.no_extrapolation and plot_points:
        plot_points = add_smooth_extrapolation(plot_points, args.alias_factors, random_baseline)

    # Free model memory before next model.
    del model, sae_adapter
    torch.cuda.empty_cache()
    gc.collect()

    return plot_points, pool_summary


def add_smooth_extrapolation(
    plot_points: List[Dict],
    alias_factors: List[int],
    random_baseline: float,
) -> List[Dict]:
    """
    Add smooth extrapolated curves using the last observed estimates.

    For a fixed alias factor, the bound has form:
        C + A / sqrt(N)
    after empirical estimates R_hat, eps_hat, eta_hat and P are fixed.
    This mirrors the extrapolation in the user's Figure-1 code.
    """
    df = pd.DataFrame(plot_points)
    extra = []

    for (model_name, r), group in df.groupby(["Model", "alias_factor"]):
        group = group.sort_values("N")
        last = group.iloc[-1].to_dict()

        C = (
            float(last["t_proxy_risk"])
            + float(last["t_reconstruction_gap"])
            + float(last["t_mismatch"])
        )
        B = float(last["B"])
        delta = float(last["delta"])
        P = float(last["P"])
        m_alias = float(last["m_alias"])
        N_last = float(last["N"])

        ssd = P * math.log((math.e * m_alias) / P) if P > 0 else 0.0
        A = (
            B * math.sqrt((ssd + math.log(2.0 / delta)) / 2.0)
            + B * math.sqrt(math.log(4.0 / delta) / 2.0)
        )

        if C >= random_baseline:
            # No finite crossing; still extrapolate moderately for visualization.
            N_end = N_last * 4.0
            N_cross = np.nan
        else:
            N_cross = (A / (random_baseline - C)) ** 2
            N_end = max(N_cross * 1.15, N_last * 1.05)

        N_grid = np.unique(np.ceil(np.geomspace(max(N_last, 1.0), N_end, 80)).astype(int))
        for N_i in N_grid[1:]:
            t4 = B * math.sqrt((ssd + math.log(2.0 / delta)) / (2.0 * N_i))
            t5 = B * math.sqrt(math.log(4.0 / delta) / (2.0 * N_i))
            total_bound = C + t4 + t5

            row = last.copy()
            row["N"] = float(N_i)
            row["Total Bound"] = float(total_bound)
            row["t_sparse_complexity"] = float(t4)
            row["t_concentration"] = float(t5)
            row["Crossing N"] = float(N_cross) if not np.isnan(N_cross) else np.nan
            row["extrapolated"] = True
            extra.append(row)

    for row in plot_points:
        row["extrapolated"] = False

    return plot_points + extra


# =============================================================================
# 5. Plotting
# =============================================================================

def plot_alias_curves(df: pd.DataFrame, output_dir: str):
    if df.empty:
        print("No results to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for model_name, group in df.groupby("Model"):
        group = group.sort_values(["alias_factor", "N"])

        fig, ax = plt.subplots(figsize=(10, 6))

        for r, sub in group.groupby("alias_factor"):
            sub = sub.sort_values("N")
            label = f"r={r}"
            ax.plot(sub["N"], sub["Total Bound"], linewidth=2.2, label=label)
            observed = sub[~sub["extrapolated"].astype(bool)]
            if not observed.empty:
                ax.scatter(observed["N"], observed["Total Bound"], s=25)

        baseline = float(group["Random Baseline"].iloc[0])
        ax.axhline(baseline, linestyle="--", linewidth=1.6, label="random baseline")

        ax.set_xscale("log")
        ax.set_xlabel("Number of evaluation samples N")
        ax.set_ylabel("Certificate value (bits)")
        ax.set_title(f"Feature-fragmentation control: {model_name}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(title="Alias factor")

        safe_name = model_name.replace(" ", "_").replace(".", "_").replace("/", "_")
        png_path = os.path.join(output_dir, f"{safe_name}_feature_fragmentation_alias_curves.png")
        pdf_path = os.path.join(output_dir, f"{safe_name}_feature_fragmentation_alias_curves.pdf")

        plt.tight_layout()
        plt.savefig(png_path, dpi=400, bbox_inches="tight")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {png_path}")
        print(f"Saved {pdf_path}")


def plot_decomposition_at_last_N(df: pd.DataFrame, output_dir: str):
    """
    Bar plot of bound terms at the largest observed non-extrapolated N.
    """
    if df.empty:
        return

    os.makedirs(output_dir, exist_ok=True)

    term_cols = [
        "t_proxy_risk",
        "t_reconstruction_gap",
        "t_mismatch",
        "t_sparse_complexity",
        "t_concentration",
    ]
    term_labels = [
        "proxy risk",
        "reconstruction gap",
        "mismatch",
        "sparse complexity",
        "concentration",
    ]

    observed = df[~df["extrapolated"].astype(bool)].copy()

    for model_name, group in observed.groupby("Model"):
        last_N = group["N"].max()
        sub = group[group["N"] == last_N].sort_values("alias_factor")
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        bottoms = np.zeros(len(sub))
        x = np.arange(len(sub))
        for col, label in zip(term_cols, term_labels):
            vals = sub[col].to_numpy(dtype=float)
            ax.bar(x, vals, bottom=bottoms, label=label)
            bottoms += vals

        ax.axhline(float(sub["Random Baseline"].iloc[0]), linestyle="--", linewidth=1.5, label="random baseline")
        ax.set_xticks(x)
        ax.set_xticklabels([f"r={int(v)}" for v in sub["alias_factor"]])
        ax.set_ylabel("Certificate contribution (bits)")
        ax.set_xlabel("Alias factor")
        ax.set_title(f"Bound decomposition at largest observed N: {model_name}")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)

        safe_name = model_name.replace(" ", "_").replace(".", "_").replace("/", "_")
        png_path = os.path.join(output_dir, f"{safe_name}_alias_decomposition_lastN.png")
        pdf_path = os.path.join(output_dir, f"{safe_name}_alias_decomposition_lastN.pdf")

        plt.tight_layout()
        plt.savefig(png_path, dpi=400, bbox_inches="tight")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {png_path}")
        print(f"Saved {pdf_path}")


# =============================================================================
# 6. CLI and runner
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature-fragmentation / aliasing control for SAE sparse-proxy certificates."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="GPT-2 Small,Gemma-2B,Llama-3.1-8B",
        help="Comma-separated model keys to run. Must match keys in DEFAULT_CONFIG['MODELS'].",
    )
    parser.add_argument(
        "--alias-factors",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated alias factors r. r=1 is the original SAE support.",
    )
    parser.add_argument("--alias-mode", type=str, default="token_hash", choices=["token_hash", "split"])
    parser.add_argument("--calibration-tokens", type=int, default=DEFAULT_CONFIG["CALIBRATION_TOKENS"])
    parser.add_argument(
        "--n-steps",
        type=str,
        default=",".join(str(x) for x in DEFAULT_CONFIG["N_STEPS"]),
        help="Comma-separated evaluation-token checkpoints, not sequence counts.",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG["ALPHA"])
    parser.add_argument("--delta", type=float, default=DEFAULT_CONFIG["DELTA"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_CONFIG["TOP_K"])
    parser.add_argument("--seq-len", type=int, default=DEFAULT_CONFIG["SEQ_LEN"])
    parser.add_argument("--tokenizer-buffer-size", type=int, default=DEFAULT_CONFIG["TOKENIZER_BUFFER_SIZE"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["SEED"])
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["DEVICE"])
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG["OUTPUT_DIR"])
    parser.add_argument("--no-extrapolation", action="store_true")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.models = parse_csv_strings(args.models)
    args.alias_factors = sorted(set(parse_csv_ints(args.alias_factors)))
    args.n_steps = parse_csv_ints(args.n_steps)

    missing = [m for m in args.models if m not in DEFAULT_CONFIG["MODELS"]]
    if missing:
        raise ValueError(f"Unknown model keys: {missing}. Available: {list(DEFAULT_CONFIG['MODELS'].keys())}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("FEATURE-FRAGMENTATION / ALIASING CONTROL")
    print("=" * 80)
    print(f"Models              : {args.models}")
    print(f"Alias factors       : {args.alias_factors}")
    print(f"Alias mode          : {args.alias_mode}")
    print(f"Calibration tokens  : {args.calibration_tokens:,}")
    print(f"Evaluation steps    : {args.n_steps}")
    print(f"Top-k               : {args.top_k}")
    print(f"Seq len             : {args.seq_len}")
    print(f"Device              : {args.device}")
    print(f"Output dir          : {args.output_dir}")
    print("=" * 80)

    all_rows = []
    all_pool_rows = []

    for model_key in args.models:
        rows, pool_rows = evaluate_alias_control_for_model(
            model_key=model_key,
            config=DEFAULT_CONFIG["MODELS"][model_key],
            args=args,
        )
        all_rows.extend(rows)
        all_pool_rows.extend(pool_rows)

        # Save after every model so partial results survive failures/OOM.
        pd.DataFrame(all_rows).to_csv(
            os.path.join(args.output_dir, "feature_fragmentation_bound_results_partial.csv"),
            index=False,
        )
        pd.DataFrame(all_pool_rows).to_csv(
            os.path.join(args.output_dir, "feature_fragmentation_pool_summary_partial.csv"),
            index=False,
        )

    df = pd.DataFrame(all_rows)
    df_pool = pd.DataFrame(all_pool_rows)

    bound_csv = os.path.join(args.output_dir, "feature_fragmentation_bound_results.csv")
    pool_csv = os.path.join(args.output_dir, "feature_fragmentation_pool_summary.csv")

    df.to_csv(bound_csv, index=False)
    df_pool.to_csv(pool_csv, index=False)

    print(f"Saved {bound_csv}")
    print(f"Saved {pool_csv}")

    plot_alias_curves(df, args.output_dir)
    plot_decomposition_at_last_N(df, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
