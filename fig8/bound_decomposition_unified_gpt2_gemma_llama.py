import gc
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens import SAE
from sparsify import Sae


# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "CALIBRATION_TOKENS": 200_000,
    "TEST_TOKENS": 2_240_000,
    "BATCH_SIZE_DEFAULT": 16,
    "CONTEXT_LEN": 32,
    "ALPHA": 0.5,
    "DELTA": 0.05,
    "TOP_K": 64,
    "SLIGHT_OOD_CORRUPTION_RATE": 0.15,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "TOKENIZER_BUFFER_SIZE": 10_000,
    "OUTPUT_DIR": "./final_ops_final",
    "CONDITIONS": [
        {
            "name": "English (IID)",
            "short_name": "IID",
            "mode": "english",
            "seed": 42,
        },
        {
            "name": "Corrupted English (Slight-OOD)",
            "short_name": "Slight-OOD",
            "mode": "corrupted_english",
            "seed": 43,
        },
        {
            "name": "Random Noise (Far-OOD)",
            "short_name": "Far-OOD",
            "mode": "random_noise",
            "seed": 44,
        },
    ],
    "MODELS": {
        "GPT-2 Small": {
            "name": "gpt2-small",
            "sae_backend": "sae_lens",
            "sae_release": "gpt2-small-res-jb",
            "sae_id": "blocks.6.hook_resid_pre",
            "batch_size": 16,
            "color": "#1f77b4",
        },
        "Gemma-2B": {
            "name": "gemma-2b",
            "sae_backend": "sae_lens",
            "sae_release": "gemma-2b-res-jb",
            "sae_id": "blocks.12.hook_resid_post",
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
            "dtype": torch.bfloat16,
            "color": "#2ca02c",
        },
    },
}


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def smoothed_bpd_loss(logits, tokens, alpha, vocab_size, reduction="mean"):
    probs = torch.softmax(logits, dim=-1)
    probs_shifted = probs[:, :-1, :]
    tokens_shifted = tokens[:, 1:]
    true_probs = torch.gather(probs_shifted, -1, tokens_shifted.unsqueeze(-1)).squeeze(-1)
    smoothed_probs = (1 - alpha) * true_probs + (alpha / vocab_size)
    log_probs = -torch.log2(smoothed_probs)
    loss_per_seq = log_probs.mean(dim=-1)

    if reduction == "mean":
        return loss_per_seq.mean()
    return loss_per_seq


def build_text_iterator(seed: int):
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=CONFIG["TOKENIZER_BUFFER_SIZE"])
    return iter(dataset)


def mildly_corrupt_tokens(tokens: torch.Tensor, vocab_size: int, corruption_rate: float) -> torch.Tensor:
    """
    Slight-OOD regime: replace a small fraction of token identities with random ids,
    while keeping sequence length and rough structure intact.
    """
    if corruption_rate <= 0:
        return tokens

    corrupted = tokens.clone()
    mask = torch.rand_like(corrupted.float()) < corruption_rate
    if corrupted.shape[1] > 0:
        mask[:, 0] = False  # keep the first token / BOS untouched

    random_tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=corrupted.shape,
        device=corrupted.device,
        dtype=corrupted.dtype,
    )
    corrupted[mask] = random_tokens[mask]
    return corrupted


def get_condition_batch(
    condition_cfg: Dict,
    model,
    batch_size: int,
    iterator=None,
) -> Optional[torch.Tensor]:
    mode = condition_cfg["mode"]

    if mode == "random_noise":
        vocab = model.cfg.d_vocab
        return torch.randint(0, vocab, (batch_size, CONFIG["CONTEXT_LEN"]), device=CONFIG["DEVICE"])

    batch_tokens = []
    try:
        while len(batch_tokens) < batch_size:
            item = next(iterator)
            text = item.get("text", item.get("content", ""))
            tokens = model.to_tokens(text)[:, : CONFIG["CONTEXT_LEN"]]
            if tokens.shape[1] < CONFIG["CONTEXT_LEN"]:
                continue
            batch_tokens.append(tokens)
    except StopIteration:
        if not batch_tokens:
            return None

    if not batch_tokens:
        return None

    tokens = torch.cat(batch_tokens, dim=0).to(CONFIG["DEVICE"])

    if mode == "corrupted_english":
        tokens = mildly_corrupt_tokens(
            tokens,
            vocab_size=model.cfg.d_vocab,
            corruption_rate=CONFIG["SLIGHT_OOD_CORRUPTION_RATE"],
        )

    return tokens


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


def apply_topk_dense(activations: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    k = min(k, activations.shape[-1])
    topk_vals, topk_inds = torch.topk(activations, k=k, dim=-1)
    mask = torch.zeros_like(activations, dtype=torch.bool)
    mask.scatter_(-1, topk_inds, True)
    sparse_acts = activations * mask
    active_mask = sparse_acts > 0
    return sparse_acts, active_mask


def get_topk_sparse(activations: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = min(k, activations.shape[-1])
    top_acts, top_indices = torch.topk(activations, k=k, dim=-1)
    top_acts = torch.relu(top_acts)
    active_mask = torch.zeros_like(activations, dtype=torch.bool)
    active_mask.scatter_(-1, top_indices, top_acts > 0)
    return top_acts, top_indices, active_mask


class SAEAdapter:
    def __init__(self, sae, backend: str, device: str):
        self.sae = sae
        self.backend = backend
        self.device = device

    def encode_pre_acts(self, flat_act: torch.Tensor) -> torch.Tensor:
        return extract_pre_acts(self.sae.encode(flat_act))

    def decode_topk(self, feature_acts: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.backend == "sparsify":
            top_acts, top_indices, active_mask = get_topk_sparse(feature_acts, k)
            recons = self.sae.decode(top_acts, top_indices)
            return recons, active_mask

        sparse_acts, active_mask = apply_topk_dense(feature_acts, k)
        recons = self.sae.decode(sparse_acts)
        return recons, active_mask

    def get_latent_dim(self, d_model: int, dtype: torch.dtype) -> int:
        if self.backend == "sae_lens":
            return self.sae.cfg.d_sae

        dummy_act = torch.zeros(1, d_model, device=self.device, dtype=dtype)
        return self.encode_pre_acts(dummy_act).shape[-1]


def load_model_and_sae(config: Dict):
    dtype = config.get("dtype", torch.float32)
    model = HookedTransformer.from_pretrained(
        config["name"],
        device=CONFIG["DEVICE"],
        dtype=dtype,
    )

    if config["sae_backend"] == "sae_lens":
        sae, _, _ = SAE.from_pretrained(
            release=config["sae_release"],
            sae_id=config["sae_id"],
            device=CONFIG["DEVICE"],
        )
        hook_name = config["sae_id"]
    elif config["sae_backend"] == "sparsify":
        sae = Sae.load_from_hub(
            config["sae_release"],
            hookpoint=config["sae_hookpoint"],
        )
        sae = sae.to(CONFIG["DEVICE"])
        hook_name = config["hook_name"]
    else:
        raise ValueError(f"Unknown SAE backend: {config['sae_backend']}")

    sae.eval()
    adapter = SAEAdapter(sae=sae, backend=config["sae_backend"], device=CONFIG["DEVICE"])
    return model, adapter, hook_name, dtype


# ==========================================
# 3. MEASUREMENT LOOP
# ==========================================
def run_decomposition_for_model(model_key: str, model_cfg: Dict) -> pd.DataFrame:
    print(f"\n🚀 Running bound decomposition for {model_key}")

    torch.cuda.empty_cache()
    gc.collect()

    model, sae_adapter, hook_name, dtype = load_model_and_sae(model_cfg)
    vocab_size = model.cfg.d_vocab
    bound_B = math.log2(vocab_size / CONFIG["ALPHA"])
    d_model = model.cfg.d_model
    d_sae = sae_adapter.get_latent_dim(d_model=d_model, dtype=dtype)

    results_rows: List[Dict] = []
    batch_size = model_cfg.get("batch_size", CONFIG["BATCH_SIZE_DEFAULT"])

    for cond_cfg in CONFIG["CONDITIONS"]:
        cond_name = cond_cfg["name"]
        print(f"\nCondition: {cond_name}")

        iterator = None if cond_cfg["mode"] == "random_noise" else build_text_iterator(cond_cfg["seed"])

        # --- Phase 1: Calibration (learn P) ---
        pool_mask = torch.zeros(d_sae, dtype=torch.bool, device=CONFIG["DEVICE"])
        calib_tokens_processed = 0
        pbar_calib = tqdm(total=CONFIG["CALIBRATION_TOKENS"], desc=f"{model_key} | {cond_cfg['short_name']} | calib")

        while calib_tokens_processed < CONFIG["CALIBRATION_TOKENS"]:
            tokens = get_condition_batch(cond_cfg, model, batch_size, iterator)
            if tokens is None:
                break

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
                act = cache[hook_name]
                flat_act = act.reshape(-1, act.shape[-1])
                feature_acts = sae_adapter.encode_pre_acts(flat_act)
                _, active_mask = sae_adapter.decode_topk(feature_acts, CONFIG["TOP_K"])
                pool_mask |= active_mask.any(dim=0)

            n_tok = tokens.numel()
            calib_tokens_processed += n_tok
            pbar_calib.update(n_tok)
        pbar_calib.close()

        P = int(pool_mask.sum().item())
        print(f"  --> Active Concept Pool P = {P}")

        # Fresh iterator for test phase on text-backed conditions.
        iterator = None if cond_cfg["mode"] == "random_noise" else build_text_iterator(cond_cfg["seed"] + 1000)

        # --- Phase 2: Testing (measure Risk / Gap / Eta) ---
        metrics = {
            "loss_h_G": [],
            "epsilon_loss": [],
            "pool_violation": [],
        }

        test_tokens_processed = 0
        pbar_test = tqdm(total=CONFIG["TEST_TOKENS"], desc=f"{model_key} | {cond_cfg['short_name']} | test")

        while test_tokens_processed < CONFIG["TEST_TOKENS"]:
            tokens = get_condition_batch(cond_cfg, model, batch_size, iterator)
            if tokens is None:
                break

            with torch.no_grad():
                # Base model forward.
                orig_logits, cache = model.run_with_cache(tokens, names_filter=[hook_name])
                loss_M = smoothed_bpd_loss(orig_logits, tokens, CONFIG["ALPHA"], vocab_size, reduction="none")

                # SAE encode.
                act = cache[hook_name]
                flat_act = act.reshape(-1, act.shape[-1])
                feature_acts_raw = sae_adapter.encode_pre_acts(flat_act)

                # Unrestricted proxy: top-k only.
                recons_unrestricted, active_mask_unrestricted = sae_adapter.decode_topk(
                    feature_acts_raw,
                    CONFIG["TOP_K"],
                )
                recons_unrestricted = recons_unrestricted.reshape(act.shape).to(act.dtype)

                # Restricted proxy: mask by learned pool, then top-k.
                masked_acts = feature_acts_raw * pool_mask.unsqueeze(0).to(feature_acts_raw.dtype)
                recons_restricted, _ = sae_adapter.decode_topk(masked_acts, CONFIG["TOP_K"])
                recons_restricted = recons_restricted.reshape(act.shape).to(act.dtype)

                # Pool mismatch / eta.
                violation_mask = active_mask_unrestricted & (~pool_mask.unsqueeze(0))
                seq_has_violation = violation_mask.any(dim=-1).reshape(tokens.shape[0], -1).any(dim=-1).float()

                # Proxy forward passes.
                def hook_unrestricted(activations, hook):
                    return recons_unrestricted

                logits_Som = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_unrestricted)])
                loss_Som = smoothed_bpd_loss(logits_Som, tokens, CONFIG["ALPHA"], vocab_size, reduction="none")

                def hook_restricted(activations, hook):
                    return recons_restricted

                logits_hG = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_restricted)])
                loss_hG = smoothed_bpd_loss(logits_hG, tokens, CONFIG["ALPHA"], vocab_size, reduction="none")

                gap = torch.abs(loss_M - loss_Som)

                metrics["loss_h_G"].extend(loss_hG.detach().cpu().float().numpy())
                metrics["epsilon_loss"].extend(gap.detach().cpu().float().numpy())
                metrics["pool_violation"].extend(seq_has_violation.detach().cpu().float().numpy())

            n_tok = tokens.numel()
            test_tokens_processed += n_tok
            pbar_test.update(n_tok)
        pbar_test.close()

        N = max(1, int(test_tokens_processed / CONFIG["CONTEXT_LEN"]))
        m = d_sae
        delta = CONFIG["DELTA"]
        B = bound_B

        R_hat_hG = float(np.mean(metrics["loss_h_G"]))
        eps_loss_hat = float(np.mean(metrics["epsilon_loss"]))
        eta_hat = float(np.mean(metrics["pool_violation"]))

        # Bound components.
        t1 = R_hat_hG
        t2 = eps_loss_hat
        t3 = eta_hat * B
        ssd = P * math.log((math.e * m) / P) if P > 0 else 0.0
        t4 = B * math.sqrt((ssd + math.log(2 / delta)) / (2 * N))
        t5 = B * math.sqrt(math.log(4 / delta) / (2 * N))
        complexity_total = t4 + t5
        empirical_subtotal = t1 + t2 + t3
        total_bound = empirical_subtotal + complexity_total
        random_baseline = math.log2(vocab_size)

        print(
            f"  [Result] Risk: {t1:.3f} | Gap: {t2:.3f} | Eta*B: {t3:.3f} | "
            f"Complexity: {complexity_total:.3f} | Total: {total_bound:.3f}"
        )

        results_rows.append(
            {
                "Model": model_key,
                "Condition": cond_name,
                "Condition Short": cond_cfg["short_name"],
                "Empirical Risk (R)": t1,
                "Reconstruction Gap (ε)": t2,
                "Pool Mismatch (ηB)": t3,
                "Complexity + Concentration": complexity_total,
                "Empirical Subtotal": empirical_subtotal,
                "Total Bound": total_bound,
                "Random Baseline": random_baseline,
                "P": P,
                "N": N,
                "B": B,
                "eta": eta_hat,
                "epsilon": eps_loss_hat,
                "risk": R_hat_hG,
            }
        )

    del model, sae_adapter
    gc.collect()
    torch.cuda.empty_cache()
    return pd.DataFrame(results_rows)


# ==========================================
# 4. PLOTTING
# ==========================================
def plot_decomposition(df: pd.DataFrame):
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    component_order = [
        "Empirical Risk (R)",
        "Reconstruction Gap (ε)",
        "Pool Mismatch (ηB)",
    ]
    component_colors = {
        "Empirical Risk (R)": "#4C78A8",
        "Reconstruction Gap (ε)": "#F58518",
        "Pool Mismatch (ηB)": "#E45756",
    }
    condition_order = [cfg["name"] for cfg in CONFIG["CONDITIONS"]]
    short_names = {cfg["name"]: cfg["short_name"] for cfg in CONFIG["CONDITIONS"]}
    model_order = [m for m in CONFIG["MODELS"] if m in df["Model"].unique()]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, len(model_order), figsize=(5.2 * len(model_order), 4.9), sharey=False)
    if len(model_order) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_order):
        sub = df[df["Model"] == model_name].copy()
        sub["Condition"] = pd.Categorical(sub["Condition"], categories=condition_order, ordered=True)
        sub = sub.sort_values("Condition")

        x = np.arange(len(sub))
        width = 0.58
        bottoms = np.zeros(len(sub))

        for comp in component_order:
            vals = sub[comp].to_numpy()
            ax.bar(
                x,
                vals,
                bottom=bottoms,
                width=width,
                color=component_colors[comp],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            bottoms += vals

        # Full bound marker, while keeping the complexity term off the stacked bars.
        ax.plot(
            x,
            sub["Total Bound"].to_numpy(),
            linestyle="none",
            marker="D",
            markersize=6.5,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            zorder=5,
        )

        baseline = float(sub["Random Baseline"].iloc[0])
        ax.axhline(baseline, color="black", linestyle=(0, (5, 2)), linewidth=1.4, alpha=0.9, zorder=4)

        for xi, empirical, total, P in zip(x, sub["Empirical Subtotal"], sub["Total Bound"], sub["P"]):
            ax.text(xi, empirical + 0.18, f"{empirical:.1f}", ha="center", va="bottom", fontsize=9)
            ax.text(xi, total + 0.2, f"⋄ {total:.1f}", ha="center", va="bottom", fontsize=8.8, color="black")
            ax.text(xi, 0.02, f"P={int(P)}", ha="center", va="bottom", fontsize=8.2,
                    transform=ax.get_xaxis_transform(), color="#444444")

        ax.set_xticks(x)
        ax.set_xticklabels([short_names[c] for c in sub["Condition"].tolist()])
        ax.set_title(model_name, fontweight="bold")
        ax.grid(axis="y", linestyle="-", alpha=0.16, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)

        ymax = max(float(sub["Total Bound"].max()), baseline) * 1.18
        ax.set_ylim(0, ymax)

    axes[0].set_ylabel("Bits per Dimension (BPD)", fontweight="bold")
    fig.supxlabel("Data distribution", y=0.03, fontweight="bold")
    fig.suptitle(
        f"Generalization-Bound Decomposition (Top-K={CONFIG['TOP_K']})\n"
        f"Empirical components shown as stacks; full bound marked with ⋄",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    legend_handles = [Patch(facecolor=component_colors[c], label=c) for c in component_order]
    legend_handles += [
        Line2D([0], [0], marker="D", color="black", markerfacecolor="white", linestyle="none", label="Full bound"),
        Line2D([0], [0], color="black", linestyle=(0, (5, 2)), label="Random baseline"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=5,
        frameon=False,
    )

    plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.86])
    png_path = os.path.join(CONFIG["OUTPUT_DIR"], "bound_decomposition_unified_gpt2_gemma_llama.png")
    pdf_path = os.path.join(CONFIG["OUTPUT_DIR"], "bound_decomposition_unified_gpt2_gemma_llama.pdf")
    plt.savefig(png_path, dpi=500, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"✅ Plot saved: {png_path}")
    print(f"✅ Plot saved: {pdf_path}")


# ==========================================
# 5. MAIN
# ==========================================
def main():
    all_results = []
    for model_key, model_cfg in CONFIG["MODELS"].items():
        try:
            df_model = run_decomposition_for_model(model_key, model_cfg)
            all_results.append(df_model)
        except Exception as exc:
            print(f"❌ Skipping {model_key} due to error: {exc}")
            gc.collect()
            torch.cuda.empty_cache()

    if not all_results:
        raise RuntimeError("No model completed successfully.")

    df = pd.concat(all_results, ignore_index=True)
    print("\nFinal results:\n", df)

    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    csv_path = os.path.join(CONFIG["OUTPUT_DIR"], "bound_decomposition_unified_gpt2_gemma_llama.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Results saved: {csv_path}")

    plot_decomposition(df)


if __name__ == "__main__":
    main()
