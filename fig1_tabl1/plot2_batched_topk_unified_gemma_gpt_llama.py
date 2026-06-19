import gc
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens import SAE
from sparsify import Sae

print('here')

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
CONFIG = {
    "CALIBRATION_TOKENS": 2240000,
    "N_STEPS": [32000, 64000, 96000, 32000*4, 32000*5, 32000*6, 32000*7, 32000*8, 32000*9, 32000*10],
    "ALPHA": 0.5,
    "DELTA": 0.05,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "TOP_K": 64,
    "SEQ_LEN": 32,
    "TOKENIZER_BUFFER_SIZE": 32000,
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
            "batch_size": 16,
            "color": "#d62728",
        },
        # "Gemma-2B": {
        #     "name": "google/gemma-3-1b-pt",
        #     "sae_backend": "sae_lens",
        #     "sae_release": "gemma-scope-2-1b-pt-res",
        #     "sae_id": "layer_22_width_65k_l0_medium",
        #     "hook_name": "blocks.22.hook_resid_post",
        #     "batch_size": 16,
        #     "color": "#d62728",
        # },
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


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_topk_sparse(activations, k):
    """
    Extract the top-k activations and their indices. ReLU is applied to match
    sparse SAE usage where negative values are suppressed.
    """
    top_acts, top_indices = torch.topk(activations, k=k, dim=-1)
    top_acts = torch.relu(top_acts)
    return top_acts, top_indices


def apply_topk_dense(activations, k):
    """
    Keep only top-k activations in a dense tensor and zero out the rest.
    """
    topk_vals, topk_inds = torch.topk(activations, k=k, dim=-1)
    mask = torch.zeros_like(activations, dtype=torch.bool)
    mask.scatter_(-1, topk_inds, True)
    return activations * mask


def smoothed_bpd_loss(logits, tokens, alpha, vocab_size, reduction='mean'):
    probs = torch.softmax(logits, dim=-1)
    probs_shifted = probs[:, :-1, :]
    tokens_shifted = tokens[:, 1:]
    true_probs = torch.gather(probs_shifted, -1, tokens_shifted.unsqueeze(-1)).squeeze(-1)
    smoothed_probs = (1 - alpha) * true_probs + (alpha / vocab_size)
    log_probs = -torch.log2(smoothed_probs)
    loss_per_seq = log_probs.mean(dim=-1)

    if reduction == 'mean':
        return loss_per_seq.mean()
    return loss_per_seq


def get_tokens_generator(model, batch_size, device, mode, calibration_limit=0):
    """
    Yields disjoint batches of tokens for calibration/evaluation.
    Uses the already-loaded model tokenizer so all model families are handled
    uniformly, including Llama.
    """
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=CONFIG["TOKENIZER_BUFFER_SIZE"])

    iterator = iter(dataset)
    tokens_processed_global = 0
    skip_needed = (mode == "evaluation")

    while True:
        batch_texts = []
        try:
            for _ in range(batch_size):
                item = next(iterator)
                text = item['text'] if 'text' in item else item['content']
                batch_texts.append(text)
        except StopIteration:
            if not batch_texts:
                break

        if not batch_texts:
            break

        tokens = model.to_tokens(batch_texts)
        if tokens.shape[1] < CONFIG["SEQ_LEN"]:
            continue

        tokens = tokens[:, : CONFIG["SEQ_LEN"]]
        num_tok = tokens.numel()

        if skip_needed:
            tokens_processed_global += num_tok
            if tokens_processed_global < calibration_limit:
                continue
            skip_needed = False

        elif mode == "calibration":
            if tokens_processed_global >= calibration_limit:
                break
            tokens_processed_global += num_tok

        yield tokens.to(device)


def extract_pre_acts(encode_out):
    """
    Normalize encode outputs across backends.
    - sae_lens: returns dense tensor directly.
    - sparsify: returns an object/dict that stores pre_acts.
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
    def __init__(self, sae, backend, device):
        self.sae = sae
        self.backend = backend
        self.device = device

    def encode_pre_acts(self, flat_act):
        return extract_pre_acts(self.sae.encode(flat_act))

    def decode_topk(self, feature_acts_raw, k):
        if self.backend == "sparsify":
            top_acts, top_indices = get_topk_sparse(feature_acts_raw, k)
            recons = self.sae.decode(top_acts, top_indices)
            active_mask = torch.zeros_like(feature_acts_raw, dtype=torch.bool)
            active_mask.scatter_(-1, top_indices, top_acts > 0)
            return recons, active_mask

        sparse_acts = apply_topk_dense(feature_acts_raw, k)
        recons = self.sae.decode(sparse_acts)
        active_mask = sparse_acts > 0
        return recons, active_mask

    def get_latent_dim(self, d_model, dtype):
        if self.backend == "sae_lens":
            return self.sae.cfg.d_sae

        dummy_act = torch.zeros(1, d_model, device=self.device, dtype=dtype)
        return self.encode_pre_acts(dummy_act).shape[-1]


def load_model_and_sae(config):
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
        hook_name = config["hook_name"]
    elif config["sae_backend"] == "sparsify":
        sae = Sae.load_from_hub(
            config["sae_release"],
            hookpoint=config["sae_hookpoint"],
        )
        sae = sae.to(CONFIG["DEVICE"])
        hook_name = config["hook_name"]
    else:
        raise ValueError(f"Unknown sae_backend: {config['sae_backend']}")

    sae.eval()
    adapter = SAEAdapter(sae=sae, backend=config["sae_backend"], device=CONFIG["DEVICE"])
    return model, adapter, hook_name, dtype


def measure_pool_and_p(model, sae_adapter, token_gen, hook_name, target_tokens, k, device, d_model, dtype):
    print(f"  --> [Calibration] Measuring Pool Size (P) on {target_tokens} tokens with k={k}...")

    d_sae = sae_adapter.get_latent_dim(d_model=d_model, dtype=dtype)
    active_indices = torch.zeros(d_sae, dtype=torch.bool, device=device)
    total_tokens = 0

    pbar = tqdm(total=target_tokens, desc="Calibration")

    for tokens in token_gen:
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            act = cache[hook_name]
            flat_act = act.reshape(-1, act.shape[-1])

            feature_acts_raw = sae_adapter.encode_pre_acts(flat_act)
            _, active_mask = sae_adapter.decode_topk(feature_acts_raw, k)
            batch_active = active_mask.any(dim=0)
            active_indices = active_indices | batch_active

        num_tok = tokens.numel()
        total_tokens += num_tok
        pbar.update(num_tok)

        if total_tokens >= target_tokens:
            break

    pbar.close()

    P = active_indices.sum().item()
    print(f"  <-- [Calibration] Complete. Active Pool Size P = {P}")
    return active_indices, P, d_sae


def run_experiment_for_model(model_key, config):
    print(f"\n🚀 STARTING EXPERIMENT: {model_key}")

    torch.cuda.empty_cache()
    gc.collect()

    try:
        model, sae_adapter, hook_name, dtype = load_model_and_sae(config)
    except Exception as e:
        print(f"Error loading {model_key}: {e}")
        return []

    m = sae_adapter.get_latent_dim(d_model=model.cfg.d_model, dtype=dtype)
    print(f"SAE Dictionary Size (m): {m}")

    vocab_size = model.cfg.d_vocab
    bounded_loss_cap = math.log2(vocab_size / CONFIG["ALPHA"])
    random_baseline = math.log2(vocab_size)
    top_k = CONFIG["TOP_K"]
    cal_tokens_limit = CONFIG["CALIBRATION_TOKENS"]

    cal_gen = get_tokens_generator(
        model,
        config["batch_size"],
        CONFIG["DEVICE"],
        mode="calibration",
        calibration_limit=cal_tokens_limit,
    )

    pool_mask, P, _ = measure_pool_and_p(
        model=model,
        sae_adapter=sae_adapter,
        token_gen=cal_gen,
        hook_name=hook_name,
        target_tokens=cal_tokens_limit,
        k=top_k,
        device=CONFIG["DEVICE"],
        d_model=model.cfg.d_model,
        dtype=dtype,
    )

    print("  --> [Evaluation] Running bound measurement...")

    eval_gen = get_tokens_generator(
        model,
        config["batch_size"],
        CONFIG["DEVICE"],
        mode="evaluation",
        calibration_limit=cal_tokens_limit,
    )

    results = {
        "loss_h_G": [],
        "epsilon_loss": [],
        "pool_violation": [],
    }

    plot_points = []
    total_tokens = 0
    current_step_idx = 0
    max_steps = max(CONFIG["N_STEPS"])

    pbar = tqdm(total=max_steps, desc="Evaluation")

    df_bound_params = {
        "R_hat_hG": [],
        "eps_loss_hat": [],
        "eta_hat": [],
        "B": [],
        "delta": [],
        "m": [],
        "N": [],
        "P": [],
        "Model": [],
    }

    total_bound = None
    ssd = 0.0
    t1 = t2 = t3 = B = delta = N = R_hat_hG = eps_loss_hat = eta_hat = 0.0

    for tokens in eval_gen:
        with torch.no_grad():
            orig_logits, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            loss_M = smoothed_bpd_loss(orig_logits, tokens, CONFIG["ALPHA"], vocab_size, reduction='none')

            act = cache[hook_name]
            flat_act = act.reshape(-1, act.shape[-1])
            feature_acts_raw = sae_adapter.encode_pre_acts(flat_act)

            # --- C. Unrestricted Proxy ---
            recons_unrestricted, active_mask_unrestricted = sae_adapter.decode_topk(feature_acts_raw, top_k)
            recons_unrestricted = recons_unrestricted.reshape(act.shape)

            # --- D. Restricted Proxy ---
            masked_acts = feature_acts_raw * pool_mask.unsqueeze(0)
            recons_restricted, _ = sae_adapter.decode_topk(masked_acts, top_k)
            recons_restricted = recons_restricted.reshape(act.shape)

            # --- E. Pool violation ---
            violation_mask = active_mask_unrestricted & (~pool_mask.unsqueeze(0))
            seq_has_violation = violation_mask.any(dim=-1).reshape(tokens.shape[0], -1).any(dim=-1).float()

            def hook_unrestricted(activations, hook):
                return recons_unrestricted

            logits_Som = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_unrestricted)])
            loss_Som = smoothed_bpd_loss(logits_Som, tokens, CONFIG["ALPHA"], vocab_size, reduction='none')

            def hook_restricted(activations, hook):
                return recons_restricted

            logits_hG = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_restricted)])
            loss_hG = smoothed_bpd_loss(logits_hG, tokens, CONFIG["ALPHA"], vocab_size, reduction='none')

            gap = torch.abs(loss_M - loss_Som)

            results["loss_h_G"].extend(loss_hG.detach().cpu().float().numpy())
            results["epsilon_loss"].extend(gap.detach().cpu().float().numpy())
            results["pool_violation"].extend(seq_has_violation.detach().cpu().float().numpy())

            num_tok = tokens.numel()
            total_tokens += num_tok
            pbar.update(num_tok)

            target_N = CONFIG["N_STEPS"][current_step_idx]
            if total_tokens >= target_N:
                N = total_tokens / CONFIG["SEQ_LEN"]
                delta = CONFIG["DELTA"]
                B = bounded_loss_cap

                R_hat_hG = float(np.mean(results["loss_h_G"]))
                eps_loss_hat = float(np.mean(results["epsilon_loss"]))
                eta_hat = float(np.mean(results["pool_violation"]))

                t1 = R_hat_hG
                t2 = eps_loss_hat
                t3 = eta_hat * B
                ssd = P * math.log((math.e * m) / P) if P > 0 else 0.0
                t4 = B * math.sqrt((ssd + math.log(2 / delta)) / (2 * N))
                t5 = B * math.sqrt(math.log(4 / delta) / (2 * N))

                total_bound = t1 + t2 + t3 + t4 + t5

                df_bound_params["R_hat_hG"].append(R_hat_hG)
                df_bound_params["eps_loss_hat"].append(eps_loss_hat)
                df_bound_params["eta_hat"].append(eta_hat)
                df_bound_params["B"].append(B)
                df_bound_params["delta"].append(delta)
                df_bound_params["m"].append(m)
                df_bound_params["N"].append(N)
                df_bound_params["P"].append(P)
                df_bound_params["Model"].append(model_key)

                plot_points.append({
                    "Model": model_key,
                    "N": N,
                    "Total Bound": total_bound,
                    "Random Baseline": random_baseline,
                    "P": P,
                    "Risk": R_hat_hG,
                    "Color": config["color"],
                })

                print(
                    f"  [N={N}] Bound: {total_bound:.3f} (Base: {random_baseline:.2f}) "
                    f"| P={P} | Risk={R_hat_hG:.3f} | Gap={eps_loss_hat:.3f} | Eta={eta_hat:.3f}"
                )

                current_step_idx += 1
                if current_step_idx >= len(CONFIG["N_STEPS"]):
                    break

    os.makedirs("./final_ops", exist_ok=True)
    # pd.DataFrame(df_bound_params).to_csv(
    #     f"./final_ops/{model_key.replace(' ', '_').replace('.', '_')}_bound_params30.csv",
    #     index=False,
    # )

    if total_bound is None:
        pbar.close()
        return plot_points

    if total_bound < random_baseline:
        pbar.close()
        return plot_points

    # 
    # while total_bound >= random_baseline:
    #     N += 1000
    #     t4 = B * math.sqrt((ssd + math.log(2 / delta)) / (2 * N))
    #     t5 = B * math.sqrt(math.log(4 / delta) / (2 * N))
    #     total_bound = t1 + t2 + t3 + t4 + t5
    #     plot_points.append({
    #         "Model": model_key,
    #         "N": N,
    #         "Total Bound": total_bound,
    #         "Random Baseline": random_baseline,
    #         "P": P,
    #         "Risk": R_hat_hG,
    #         "Color": config["color"],
    #     })
    #     print("In loop:", N, total_bound, random_baseline)

    # ----- Exact / smooth extrapolation for plotting -----
    # total_bound(N) = C + A / sqrt(N)
    C = t1 + t2 + t3
    A = (
        B * math.sqrt((ssd + math.log(2 / delta)) / 2.0)
        + B * math.sqrt(math.log(4 / delta) / 2.0)
    )

    # If asymptotic floor is already above the random baseline, crossing never happens
    if C >= random_baseline:
        print(
            f"{model_key}: asymptotic floor C={C:.4f} >= random baseline={random_baseline:.4f}. "
            "No finite crossing."
        )
        pbar.close()
        return plot_points

    # Exact crossing point
    N_cross = (A / (random_baseline - C)) ** 2

    print(f"{model_key}: exact crossing at N ≈ {N_cross:.2f}")

    # Generate smooth extrapolation on a geometric grid (much better for log-scale plots)
    N_start = max(float(N), 1.0)
    N_end = max(N_cross * 1.15, N_start * 1.05)

    N_grid = np.unique(np.ceil(np.geomspace(N_start, N_end, 120)).astype(int))

    for N_i in N_grid[1:]:
        t4_i = B * math.sqrt((ssd + math.log(2 / delta)) / (2 * N_i))
        t5_i = B * math.sqrt(math.log(4 / delta) / (2 * N_i))
        total_bound_i = t1 + t2 + t3 + t4_i + t5_i

        plot_points.append({
            "Model": model_key,
            "N": N_i,
            "Total Bound": total_bound_i,
            "Random Baseline": random_baseline,
            "P": P,
            "Risk": R_hat_hG,
            "Color": config["color"],
            "Crossing N": N_cross,
        })

    pbar.close()
    return plot_points



# ==========================================
# 3. RUNNER
# ==========================================
all_data = []
for key, cfg in CONFIG["MODELS"].items():
    all_data.extend(run_experiment_for_model(key, cfg))

df = pd.DataFrame(all_data)
os.makedirs("./final_ops", exist_ok=True)
df.to_csv("./final_ops/bound_results_v2_unified30_k=64_final.csv", index=False)
print("Saved ./final_ops/bound_results_v2_unified30_k=64_final.csv")


# ==========================================
# 4. PLOTTING
# ==========================================

# ==========================================
# 4. PLOTTING
# ==========================================
if not df.empty:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
    })

    colors = {m: c["color"] for m, c in CONFIG["MODELS"].items() if m in df["Model"].unique()}

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.35], hspace=0.06)

    ax = fig.add_subplot(gs[0])
    ax_zoom = fig.add_subplot(gs[1], sharex=ax)

    for model_name, group in df.groupby("Model"):
        group = group.sort_values("N").reset_index(drop=True)

        x = group["N"].to_numpy()
        y = group["Total Bound"].to_numpy()
        baseline = float(group["Random Baseline"].iloc[0])
        color = colors[model_name]

        # Main curve
        ax.plot(x, y, color=color, linewidth=2.5, label=model_name)
        ax_zoom.plot(x, y, color=color, linewidth=2.5)

        # Mark only a few representative points
        n_marks = min(10, len(x))
        mark_idx = np.linspace(0, len(x) - 1, n_marks, dtype=int)
        ax.scatter(x[mark_idx], y[mark_idx], color=color, s=30, zorder=3)
        ax_zoom.scatter(x[mark_idx], y[mark_idx], color=color, s=30, zorder=3)

        # Random baseline
        ax.axhline(y=baseline, color=color, linestyle="--", linewidth=1.6, alpha=0.75)
        ax_zoom.axhline(y=baseline, color=color, linestyle="--", linewidth=1.6, alpha=0.75)

        # Exact crossing point if available
        if "Crossing N" in group.columns and group["Crossing N"].notna().any():
            N_cross = float(group["Crossing N"].dropna().iloc[0])
            ax_zoom.scatter(
                [N_cross], [baseline],
                color=color, s=80, marker="D", edgecolors="black", linewidths=0.7, zorder=5
            )
            ax_zoom.annotate(
                f"{model_name}: N≈{int(round(N_cross)):,}",
                xy=(N_cross, baseline),
                xytext=(8, 6),
                textcoords="offset points",
                color=color,
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xscale("log")
    ax_zoom.set_xscale("log")

    # Full panel
    ax.set_ylabel("Generalization Bound (Bits)", fontweight="bold")
    ax.set_title("Sparse Semantic Generalization Bound (Concept Pool)", fontsize=15, pad=12)
    ax.grid(True, which="both", linestyle="-", alpha=0.18)
    ax.legend(title="Model", frameon=True)

    # Zoom panel near threshold crossing
    zoom_bottom = min(df["Total Bound"].min(), df["Random Baseline"].min()) - 0.6
    zoom_top = df["Random Baseline"].max() + 8.0
    ax_zoom.set_ylim(zoom_bottom, zoom_top)

    ax_zoom.set_xlabel("Number of Samples (N)", fontweight="bold")
    ax_zoom.set_ylabel("Zoom", fontweight="bold")
    ax_zoom.grid(True, which="both", linestyle="-", alpha=0.18)

    # Keep full panel auto-scaled on top
    ax.set_ylim(0, df["Total Bound"].max() * 1.05)

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.tight_layout()
    plt.savefig("./final_ops/concept_pool_bound_plot_unified_better30_k=64_final.png", dpi=600, bbox_inches="tight")
    plt.savefig("./final_ops/concept_pool_bound_plot_unified_better30_k=64_final.pdf", bbox_inches="tight")
    print("Plot saved as ./final_ops/concept_pool_bound_plot_unified_better30_k=64_final.png")



# if not df.empty:
#     plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
#     fig, ax = plt.subplots(figsize=(10, 6))

#     colors = {m: c["color"] for m, c in CONFIG["MODELS"].items() if m in df["Model"].unique()}

#     sns.lineplot(
#         data=df,
#         x="N",
#         y="Total Bound",
#         hue="Model",
#         palette=colors,
#         style="Model",
#         markers=True,
#         markersize=8,
#         linewidth=2.5,
#         ax=ax,
#     )

#     for model_name, group in df.groupby("Model"):
#         baseline = group["Random Baseline"].iloc[0]
#         color = colors[model_name]
#         ax.axhline(y=baseline, color=color, linestyle='--', alpha=0.6)
#         ax.text(
#             max(CONFIG["N_STEPS"]) / CONFIG["SEQ_LEN"],
#             baseline + 0.1,
#             f"Random ({model_name})",
#             color=color,
#             va="bottom",
#             ha="right",
#             fontsize=9,
#             fontweight='bold',
#         )

#     ax.set_xscale("log")
#     ax.set_xlabel("Number of Samples (N)", fontweight='bold')
#     ax.set_ylabel("Generalization Bound (Bits)", fontweight='bold')
#     ax.set_title("Sparse Semantic Generalization Bound (Concept Pool)", fontsize=14, pad=15)
#     ax.grid(True, which="both", linestyle='-', alpha=0.2)
#     ax.set_ylim(bottom=0)

#     plt.tight_layout()
#     plt.savefig("./final_ops/concept_pool_bound_plot_unified.png", dpi=300)
#     print("Plot saved as ./final_ops/concept_pool_bound_plot_unified.png")
