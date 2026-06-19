
import gc
import math
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens import SAE
from sparsify import Sae

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
CONFIG = {
    "EVAL_TOKENS": 32000,
    "SEQ_LEN": 32,
    "TOKENIZER_BUFFER_SIZE": 32000,
    "TOP_K": 64,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    # Matched rollout horizons. "full" always means all remaining blocks.
    "HORIZONS": [0, 1, 2, 4, 8, "full"],
    "MODELS": {
        "GPT-2 Small": {
            "name": "gpt2-small",
            "sae_backend": "sae_lens",
            "sae_release": "gpt2-small-res-jb",
            "sae_id": "blocks.6.hook_resid_pre",
            "batch_size": 16,
            "color": "#1f77b4",
            # fixed-layer cross-model sanity check
            "patch_layers": [6],
        },
        "Gemma-2B": {
            "name": "gemma-2b",
            "sae_backend": "sae_lens",
            "sae_release": "gemma-2b-res-jb",
            "sae_id": "blocks.12.hook_resid_post",
            "batch_size": 16,
            "color": "#d62728",
            # fixed-layer cross-model sanity check
            "patch_layers": [12],
        },
        "Llama-3.1-8B": {
            "name": "meta-llama/Meta-Llama-3-8B",
            "sae_backend": "sparsify",
            "sae_release": "EleutherAI/sae-llama-3-8b-32x",
            "batch_size": 16,
            "color": "#26ba15",
            "dtype": torch.bfloat16,
            # layer sweep to deconfound local fidelity vs downstream amplification
            "patch_layers": [4, 8, 12, 16, 20, 24, 28, 30],
            "hook_suffix": "hook_resid_post",
        },
    },
}


# ==========================================
# 2. HELPERS
# ==========================================
def get_token_batches(model, batch_size, target_tokens):
    """Materialize token batches once per model so they can be reused across patch layers."""
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=CONFIG["TOKENIZER_BUFFER_SIZE"])
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
        if tokens.shape[1] < CONFIG["SEQ_LEN"]:
            continue

        tokens = tokens[:, : CONFIG["SEQ_LEN"]].cpu()
        total_tokens += tokens.numel()
        batches.append(tokens)

    print(f"Cached {len(batches)} token batches ({total_tokens} tokens) for reuse across layers.")
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


def get_topk_sparse(activations, k):
    top_acts, top_indices = torch.topk(activations, k=k, dim=-1)
    top_acts = torch.relu(top_acts)
    return top_acts, top_indices


def apply_topk_dense(activations, k):
    topk_vals, topk_inds = torch.topk(activations, k=k, dim=-1)
    mask = torch.zeros_like(activations, dtype=torch.bool)
    mask.scatter_(-1, topk_inds, True)
    return activations * mask


class SAEAdapter:
    def __init__(self, sae, backend, device):
        self.sae = sae
        self.backend = backend
        self.device = device

    def encode_pre_acts(self, flat_act):
        return extract_pre_acts(self.sae.encode(flat_act))

    def reconstruct_tensor(self, act, k):
        flat_act = act.reshape(-1, act.shape[-1])
        feature_acts_raw = self.encode_pre_acts(flat_act)

        if self.backend == "sparsify":
            top_acts, top_indices = get_topk_sparse(feature_acts_raw, k)
            recons = self.sae.decode(top_acts, top_indices)
        else:
            sparse_acts = apply_topk_dense(feature_acts_raw, k)
            recons = self.sae.decode(sparse_acts)

        return recons.reshape(act.shape)


def parse_hook_name(hook_name):
    m = re.match(r"blocks\.(\d+)\.(hook_resid_(?:pre|post))$", hook_name)
    if not m:
        raise ValueError(f"Unsupported hook format: {hook_name}")
    return int(m.group(1)), m.group(2)


def make_hook_name(layer_idx, suffix):
    return f"blocks.{layer_idx}.{suffix}"


def valid_horizons_for_hook(hook_name, n_layers, configured_horizons):
    layer_idx, suffix = parse_hook_name(hook_name)
    valid = []
    for h in configured_horizons:
        if h == "full":
            valid.append(h)
        else:
            target_layer = layer_idx + h
            if target_layer < n_layers:
                valid.append(h)
    return valid


def residual_to_logits(model, resid):
    # Read intermediate residual-stream states directly into vocabulary space.
    return model.unembed(model.ln_final(resid))


def mean_next_token_kl_bits(base_logits, proxy_logits):
    base_lp = torch.log_softmax(base_logits[:, :-1, :].float(), dim=-1)
    proxy_lp = torch.log_softmax(proxy_logits[:, :-1, :].float(), dim=-1)
    base_p = base_lp.exp()
    kl_nats = (base_p * (base_lp - proxy_lp)).sum(dim=-1)  # [B, S-1]
    kl_bits = kl_nats / math.log(2.0)
    return kl_bits.mean()


def load_model(config):
    dtype = config.get("dtype", torch.float32)
    model = HookedTransformer.from_pretrained(
        config["name"],
        device=CONFIG["DEVICE"],
        dtype=dtype,
    )
    model.eval()
    return model, dtype


def load_sae_for_layer(config, patch_layer):
    if config["sae_backend"] == "sae_lens":
        sae, _, _ = SAE.from_pretrained(
            release=config["sae_release"],
            sae_id=config["sae_id"],
            device=CONFIG["DEVICE"],
        )
        hook_name = config["sae_id"]
    elif config["sae_backend"] == "sparsify":
        sae_hookpoint = f"layers.{patch_layer}"
        sae = Sae.load_from_hub(config["sae_release"], hookpoint=sae_hookpoint)
        sae = sae.to(CONFIG["DEVICE"])
        suffix = config.get("hook_suffix", "hook_resid_post")
        hook_name = f"blocks.{patch_layer}.{suffix}"
    else:
        raise ValueError(f"Unknown sae_backend: {config['sae_backend']}")

    sae.eval()
    return SAEAdapter(sae=sae, backend=config["sae_backend"], device=CONFIG["DEVICE"]), hook_name


def evaluate_batch_horizons(model, sae_adapter, tokens, patch_hook, horizons, top_k):
    out = {}

    # Step 1: get the native activation at the patch hook.
    with torch.no_grad():
        _, patch_cache = model.run_with_cache(tokens, names_filter=[patch_hook])
        patch_act = patch_cache[patch_hook]
        patch_recons = sae_adapter.reconstruct_tensor(patch_act, k=top_k)
        del patch_cache

    def patch_fn(activations, hook):
        return patch_recons.to(activations.dtype)

    # Step 2: evaluate matched rollout horizons.
    for h in horizons:
        if h == "full":
            with torch.no_grad():
                base_logits = model(tokens)
                proxy_logits = model.run_with_hooks(tokens, fwd_hooks=[(patch_hook, patch_fn)])
                out[h] = mean_next_token_kl_bits(base_logits, proxy_logits).item()
            continue

        target_layer, suffix = parse_hook_name(patch_hook)
        target_hook = make_hook_name(target_layer + h, suffix)

        # Local horizon h=0 can be read directly without an additional forward pass.
        if h == 0:
            base_logits = residual_to_logits(model, patch_act)
            proxy_logits = residual_to_logits(model, patch_recons)
            out[h] = mean_next_token_kl_bits(base_logits, proxy_logits).item()
            continue

        with torch.no_grad():
            _, base_cache = model.run_with_cache(tokens, names_filter=[target_hook])
            base_target = base_cache[target_hook]
            base_logits = residual_to_logits(model, base_target)
            del base_cache

            with model.hooks(fwd_hooks=[(patch_hook, patch_fn)]):
                _, proxy_cache = model.run_with_cache(
                    tokens,
                    names_filter=[target_hook],
                )
            proxy_target = proxy_cache[target_hook]
            proxy_logits = residual_to_logits(model, proxy_target)
            del proxy_cache

            out[h] = mean_next_token_kl_bits(base_logits, proxy_logits).item()

    del patch_act, patch_recons
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def run_model_experiment(model_key, config):
    print(f"\n🚀 STARTING HORIZON-CONDITIONED KL EXPERIMENT: {model_key}")

    torch.cuda.empty_cache()
    gc.collect()

    model, dtype = load_model(config)
    n_layers = model.cfg.n_layers
    patch_layers = config["patch_layers"]
    top_k = CONFIG["TOP_K"]

    # Materialize tokens once so they can be reused across patch layers.
    token_batches, total_cached_tokens = get_token_batches(
        model,
        config["batch_size"],
        target_tokens=CONFIG["EVAL_TOKENS"],
    )

    # Aggregate KL values over batches for each patch layer and horizon.
    agg = defaultdict(list)

    total_work_units = len(token_batches) * len(patch_layers)
    pbar = tqdm(total=total_work_units, desc=f"Eval {model_key}")

    for patch_layer in patch_layers:
        print(f"Loading SAE for patch layer {patch_layer} once and reusing across all batches...")
        sae_adapter, patch_hook = load_sae_for_layer(config, patch_layer)
        horizons = valid_horizons_for_hook(patch_hook, n_layers, CONFIG["HORIZONS"])

        try:
            for tokens_cpu in token_batches:
                tokens = tokens_cpu.to(CONFIG["DEVICE"], non_blocking=True)
                batch_res = evaluate_batch_horizons(
                    model=model,
                    sae_adapter=sae_adapter,
                    tokens=tokens,
                    patch_hook=patch_hook,
                    horizons=horizons,
                    top_k=top_k,
                )
                for h, v in batch_res.items():
                    agg[(patch_layer, h)].append(v)

                del tokens
                pbar.update(1)
        finally:
            del sae_adapter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pbar.close()

    rows = []
    for (patch_layer, horizon), values in agg.items():
        arr = np.array(values, dtype=float)
        rows.append(
            {
                "Model": model_key,
                "Patch Layer": patch_layer,
                "Horizon": str(horizon),
                "Horizon Sort": 999 if horizon == "full" else int(horizon),
                "Mean KL (bits)": float(arr.mean()),
                "Std KL (bits)": float(arr.std(ddof=0)),
                "Num Batches": int(len(arr)),
                "Tokens Cached": int(total_cached_tokens),
            }
        )

    # Free model-specific caches before returning.
    del token_batches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows


# ==========================================
# 3. RUNNER
# ==========================================
all_rows = []
for model_key, cfg in CONFIG["MODELS"].items():
    all_rows.extend(run_model_experiment(model_key, cfg))

os.makedirs("./final_ops", exist_ok=True)
df = pd.DataFrame(all_rows)
df = df.sort_values(["Model", "Patch Layer", "Horizon Sort"]).reset_index(drop=True)
df.to_csv("./final_ops/horizon_conditioned_proxy_kl_results.csv", index=False)
print("Saved ./final_ops/horizon_conditioned_proxy_kl_results.csv")
print(df)


# ==========================================
# 4. PLOTTING
# ==========================================
if not df.empty:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
    })

    model_names = list(df["Model"].unique())
    fig, axes = plt.subplots(1, len(model_names), figsize=(5.6 * len(model_names), 4.8), squeeze=False)
    axes = axes[0]

    horizon_order = [str(h) for h in CONFIG["HORIZONS"]]
    horizon_to_x = {h: i for i, h in enumerate(horizon_order)}

    for ax, model_name in zip(axes, model_names):
        sub = df[df["Model"] == model_name].copy()
        base_color = CONFIG["MODELS"][model_name]["color"]
        patch_layers = sorted(sub["Patch Layer"].unique())

        # create a light-to-dark color ramp over patch layers
        cmap = plt.cm.get_cmap("viridis", max(3, len(patch_layers)))

        for idx, layer in enumerate(patch_layers):
            g = sub[sub["Patch Layer"] == layer].sort_values("Horizon Sort")
            xs = [horizon_to_x[h] for h in g["Horizon"].tolist() if h in horizon_to_x]
            ys = g["Mean KL (bits)"].tolist()
            yerr = g["Std KL (bits)"].tolist()
            color = cmap(idx / max(1, len(patch_layers) - 1)) if len(patch_layers) > 1 else base_color

            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker="o",
                linewidth=2.2,
                markersize=6,
                capsize=3,
                color=color,
                label=f"L{layer}",
            )

        ax.set_xticks(range(len(horizon_order)))
        ax.set_xticklabels(horizon_order)
        ax.set_xlabel("Rollout Horizon $h$", fontweight="bold")
        ax.set_title(model_name, fontsize=14, fontweight="bold", pad=10)
        ax.grid(True, linestyle="-", alpha=0.18)

    axes[0].set_ylabel("Mean Base–Proxy KL (bits)", fontweight="bold")

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(title="Patch Layer", frameon=True)

    fig.suptitle(
        "Horizon-Conditioned Base–Proxy KL: Local Fidelity vs Downstream Error Accumulation",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig("./final_ops/horizon_conditioned_proxy_kl_plot.png", dpi=600, bbox_inches="tight")
    plt.savefig("./final_ops/horizon_conditioned_proxy_kl_plot.pdf", bbox_inches="tight")
    print("Saved ./final_ops/horizon_conditioned_proxy_kl_plot.png")
