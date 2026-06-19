import argparse
import gc
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from sparsify import Sae
from tqdm import tqdm
from transformer_lens import HookedTransformer

print('here')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=24, help="Transformer layer index for the SAE and hook.")
    return parser.parse_args()


ARGS = parse_args()
SELECTED_LAYER = ARGS.layer

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
CONFIG = {
    "CALIBRATION_TOKENS": 2240000,
    "N_STEPS": [320000, 960000, 1600000, 2240000],
    "ALPHA": 0.5,
    "DELTA": 0.05,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "MODELS": {
        "Llama-3.1-8B": {
            "name": "meta-llama/Meta-Llama-3-8B",
            "sae_release": "EleutherAI/sae-llama-3-8b-32x",
            "sae_hookpoint": f"layers.{SELECTED_LAYER}",
            "hook_name": f"blocks.{SELECTED_LAYER}.hook_resid_post",
            "batch_size": 16,
            "color": "#1f77b4",
            "dtype": torch.bfloat16,
        }
    },
}
print(CONFIG["MODELS"]["Llama-3.1-8B"]["sae_hookpoint"])
# [320000, 960000, 1600000, 2240000, 320000*10, 320000*15, 320000*20, 320000*25, 320000*30]


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_topk_sparse(activations, k):
    """
    Extracts the sparse top-k elements directly from dense pre-activations.
    Returns the top values (after ReLU) and their indices to feed into sae.decode().
    """
    top_acts, top_indices = torch.topk(activations, k=k, dim=-1)
    # Most SAEs apply ReLU on the top-k elements to enforce non-negativity
    top_acts = torch.relu(top_acts)
    return top_acts, top_indices



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
    else:
        return loss_per_seq



def get_tokens_generator(model, batch_size, device, mode, calibration_limit=0):
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=32000)

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
        if tokens.shape[1] < 32:
            continue

        tokens = tokens[:, :32]
        num_tok = tokens.numel()

        if skip_needed:
            tokens_processed_global += num_tok
            if tokens_processed_global < calibration_limit:
                continue
            else:
                skip_needed = False

        elif mode == "calibration":
            if tokens_processed_global >= calibration_limit:
                break
            tokens_processed_global += num_tok

        yield tokens.to(device)



def extract_pre_acts(encode_out):
    """Safely extracts the dense pre-activations from the sparsify output object."""
    if hasattr(encode_out, "pre_acts"):
        return encode_out.pre_acts
    elif isinstance(encode_out, dict) and "pre_acts" in encode_out:
        return encode_out["pre_acts"]
    else:
        # Fallback if structure changes
        return encode_out[0]



def measure_pool_and_p(model, sae, token_gen, hook_name, target_tokens, device):
    k = 64
    print(f"  --> [Calibration] Measuring Pool Size (P) on {target_tokens} tokens with k={k}...")

    # Determine the latent dimension size dynamically
    dummy_act = torch.zeros(1, model.cfg.d_model, device=device, dtype=torch.bfloat16)
    d_sae = extract_pre_acts(sae.encode(dummy_act)).shape[-1]

    active_indices = torch.zeros(d_sae, dtype=torch.bool, device=device)
    total_tokens = 0

    pbar = tqdm(total=target_tokens, desc="Calibration")

    for tokens in token_gen:
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            act = cache[hook_name]
            flat_act = act.reshape(-1, act.shape[-1])

            # Extract dense pre-activations
            feature_acts_raw = extract_pre_acts(sae.encode(flat_act))
            top_acts, top_indices = get_topk_sparse(feature_acts_raw, k)

            # Scatter only the active indices to build the pool
            batch_active = torch.zeros_like(feature_acts_raw, dtype=torch.bool)
            batch_active.scatter_(-1, top_indices, top_acts > 0)
            batch_active = batch_active.any(dim=0)
            active_indices = active_indices | batch_active

        num_tok = tokens.numel()
        total_tokens += num_tok
        pbar.update(num_tok)

        if total_tokens >= target_tokens:
            break

    pbar.close()

    P = active_indices.sum().item()
    print(f"  <-- [Calibration] Complete. Active Pool Size P = {P}")
    return active_indices, P



def run_experiment_for_model(model_key, config):
    print(f"\n🚀 STARTING EXPERIMENT: {model_key}")

    torch.cuda.empty_cache()
    gc.collect()
    try:
        dtype = config.get("dtype", torch.float32)
        print(f"Loading {config['name']} in {dtype}...")
        model = HookedTransformer.from_pretrained(config["name"], device=CONFIG["DEVICE"], dtype=dtype)

        print(f"Loading SAE from {config['sae_release']} at {config['sae_hookpoint']}...")
        sae = Sae.load_from_hub(config["sae_release"], hookpoint=config["sae_hookpoint"])
        sae = sae.to(CONFIG["DEVICE"])
        sae.eval()
    except Exception as e:
        print(f"Error loading {model_key}: {e}")
        return []

    dummy_act = torch.zeros(1, model.cfg.d_model, device=CONFIG["DEVICE"], dtype=torch.bfloat16)
    m = extract_pre_acts(sae.encode(dummy_act)).shape[-1]
    print(f"SAE Dictionary Size (m): {m}")

    vocab_size = model.cfg.d_vocab
    bounded_loss_cap = math.log2(vocab_size / CONFIG["ALPHA"])
    random_baseline = math.log2(vocab_size)

    hook_name = config.get("hook_name")
    if hook_name not in model.hook_dict:
        raise ValueError(f"Hook name not found in model.hook_dict: {hook_name}")

    top_k = 64
    cal_tokens_limit = CONFIG["CALIBRATION_TOKENS"]

    cal_gen = get_tokens_generator(
        model,
        config["batch_size"],
        CONFIG["DEVICE"],
        mode="calibration",
        calibration_limit=cal_tokens_limit,
    )

    pool_mask, P = measure_pool_and_p(model, sae, cal_gen, hook_name, cal_tokens_limit, CONFIG["DEVICE"])

    print(f"  --> [Evaluation] Running bound measurement...")

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

    df_bound_params = {"R_hat_hG": [], "eps_loss_hat": [], "eta_hat": [], "B": [], "delta": [], "m": [], "N": [], "P": []}
    for tokens in eval_gen:
        with torch.no_grad():
            orig_logits, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            loss_M = smoothed_bpd_loss(orig_logits, tokens, CONFIG["ALPHA"], vocab_size, reduction='none')

            act = cache[hook_name]
            flat_act = act.reshape(-1, act.shape[-1])

            # Extract raw pre_acts
            feature_acts_raw = extract_pre_acts(sae.encode(flat_act))

            # --- C. Unrestricted Proxy ---
            # Get sparse outputs and pass directly to sae.decode()
            top_acts_unrestricted, top_indices_unrestricted = get_topk_sparse(feature_acts_raw, top_k)
            recons_unrestricted = sae.decode(top_acts_unrestricted, top_indices_unrestricted).reshape(act.shape)

            # --- D. Restricted Proxy ---
            masked_acts = feature_acts_raw * pool_mask.unsqueeze(0)
            top_acts_restricted, top_indices_restricted = get_topk_sparse(masked_acts, top_k)
            recons_restricted = sae.decode(top_acts_restricted, top_indices_restricted).reshape(act.shape)

            # --- E. Violation Mask (Memory Efficient!) ---
            # Check if any activated index is NOT in the pool mask
            violation_mask_sparse = (top_acts_unrestricted > 0) & (~pool_mask[top_indices_unrestricted])
            seq_has_violation = violation_mask_sparse.any(dim=-1).reshape(tokens.shape[0], -1).any(dim=-1).float()

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
                N = total_tokens / 32
                delta = CONFIG["DELTA"]
                B = bounded_loss_cap

                R_hat_hG = np.mean(results["loss_h_G"])
                eps_loss_hat = np.mean(results["epsilon_loss"])
                eta_hat = np.mean(results["pool_violation"])

                t1 = R_hat_hG
                t2 = eps_loss_hat
                t3 = eta_hat * B
                if P > 0:
                    ssd = P * math.log((math.e * m) / P)
                else:
                    ssd = 0
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

                plot_points.append(
                    {
                        "Model": model_key,
                        "Layer": SELECTED_LAYER,
                        "N": N,
                        "Total Bound": total_bound,
                        "Random Baseline": random_baseline,
                        "P": P,
                        "Risk": R_hat_hG,
                        "Color": config["color"],
                    }
                )

                print(
                    f"  [N={N}] Bound: {total_bound:.3f} (Base: {random_baseline:.2f}) | "
                    f"P={P} | Risk={R_hat_hG:.3f} | Gap={eps_loss_hat:.3f} | Eta={eta_hat:.3f}"
                )

                current_step_idx += 1
                if current_step_idx >= len(CONFIG["N_STEPS"]):
                    break

    params_csv_name = f"llama3_8b_layers_{SELECTED_LAYER}.csv"
    pd.DataFrame(df_bound_params).to_csv(params_csv_name, index=False)
    print(f"Saved {params_csv_name}")
    print('random_baseline', random_baseline)

    pbar.close()
    return plot_points


# ==========================================
# 3. RUNNER
# ==========================================
all_data = []
for key, cfg in CONFIG["MODELS"].items():
    all_data.extend(run_experiment_for_model(key, cfg))

df = pd.DataFrame(all_data)
results_csv_name = f"bound_results_layer_{SELECTED_LAYER}.csv"
df.to_csv(results_csv_name, index=False)
print(f"Saved {results_csv_name}")

# ==========================================
# 4. PLOTTING
# ==========================================
if not df.empty:
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {m: c["color"] for m, c in CONFIG["MODELS"].items() if m in df["Model"].unique()}

    sns.lineplot(
        data=df,
        x="N",
        y="Total Bound",
        hue="Model",
        palette=colors,
        style="Model",
        markers=True,
        markersize=8,
        linewidth=2.5,
        ax=ax,
    )

    for model_name, group in df.groupby("Model"):
        baseline = group["Random Baseline"].iloc[0]
        color = colors[model_name]
        ax.axhline(y=baseline, color=color, linestyle='--', alpha=0.6)
        ax.text(
            max(CONFIG["N_STEPS"]) / 32,
            baseline + 0.1,
            f"Random ({model_name})",
            color=color,
            va="bottom",
            ha="right",
            fontsize=9,
            fontweight='bold',
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Samples (N)", fontweight='bold')
    ax.set_ylabel("Generalization Bound (Bits)", fontweight='bold')
    ax.set_title(
        f"Sparse Semantic Generalization Bound (Concept Pool) - Layer {SELECTED_LAYER}",
        fontsize=14,
        pad=15,
    )
    ax.grid(True, which="both", linestyle='-', alpha=0.2)
    ax.set_ylim(bottom=0)

    os.makedirs("./final_ops", exist_ok=True)
    plt.tight_layout()
    plot_path = f"./final_ops/concept_pool_bound_plot_llama3_8b_eleuther_layer_{SELECTED_LAYER}.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved as {plot_path}")
