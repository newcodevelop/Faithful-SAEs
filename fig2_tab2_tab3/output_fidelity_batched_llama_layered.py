import argparse
import gc
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from sparsify import Sae
from tqdm import tqdm
from transformer_lens import HookedTransformer


# =========================================================
# Output fidelity analysis for a fixed layer and fixed SAE pool.
#
# For each evaluation checkpoint, this script compares:
#   - M            : original model
#   - S ∘ M        : unrestricted SAE proxy
#   - h_G          : pool-restricted proxy
#
# Reported metrics:
#   1) mean KL(M || proxy)
#   2) top-1 agreement with M
#   3) mean absolute gold-token log-prob difference
#   4) smoothed loss of each system (for direct alignment with the bound code)
# =========================================================


@dataclass
class ModelConfig:
    name: str
    sae_release: str
    sae_hookpoint: str
    hook_name: str
    batch_size: int
    color: str
    dtype: torch.dtype


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--calibration_tokens", type=int, default=2_240_000)
    parser.add_argument("--n_steps", type=int, nargs="+", default=[320000, 960000, 1600000, 2240000])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--out_dir", type=str, default="./final_ops")
    return parser.parse_args()


ARGS = parse_args()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SELECTED_LAYER = ARGS.layer

MODEL_CONFIGS = {
    "Llama-3.1-8B": ModelConfig(
        name="meta-llama/Meta-Llama-3-8B",
        sae_release="EleutherAI/sae-llama-3-8b-32x",
        sae_hookpoint=f"layers.{SELECTED_LAYER}",
        hook_name=f"blocks.{SELECTED_LAYER}.hook_resid_post",
        batch_size=16,
        color="#1f77b4",
        dtype=torch.bfloat16,
    )
}


def get_topk_sparse(pre_acts: torch.Tensor, k: int):
    top_vals, top_idx = torch.topk(pre_acts, k=k, dim=-1)
    top_vals = torch.relu(top_vals)
    return top_vals, top_idx


@torch.no_grad()
def smoothed_bpd_loss(logits: torch.Tensor, tokens: torch.Tensor, alpha: float, vocab_size: int,
                      reduction: str = "mean") -> torch.Tensor:
    log_probs_nat = torch.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = tokens[:, 1:]
    tgt_log_probs_nat = torch.gather(log_probs_nat, -1, tgt.unsqueeze(-1)).squeeze(-1)
    tgt_probs = torch.exp(tgt_log_probs_nat)

    smoothed_probs = (1.0 - alpha) * tgt_probs + (alpha / vocab_size)
    smoothed_probs = torch.clamp(smoothed_probs, min=torch.finfo(smoothed_probs.dtype).tiny)
    loss_bits = -torch.log(smoothed_probs) / math.log(2.0)
    loss_per_seq = loss_bits.mean(dim=-1)

    if reduction == "none":
        return loss_per_seq
    if reduction == "mean":
        return loss_per_seq.mean()
    raise ValueError(f"Unknown reduction: {reduction}")



def get_tokens_generator(model, batch_size, device, mode, calibration_limit, seq_len):
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    # dataset = load_dataset("allenai/c4", "en", split="train")
    dataset = dataset.shuffle(seed=42, buffer_size=32000)
    iterator = iter(dataset)

    tokens_seen = 0
    in_eval = (mode == "evaluation")

    while True:
        texts = []
        try:
            for _ in range(batch_size):
                item = next(iterator)
                texts.append(item["text"] if "text" in item else item["content"])
        except StopIteration:
            if not texts:
                break

        if not texts:
            break

        tokens = model.to_tokens(texts)
        if tokens.shape[1] < seq_len:
            continue
        tokens = tokens[:, :seq_len]
        batch_token_count = tokens.numel()

        if mode == "calibration":
            if tokens_seen >= calibration_limit:
                break
            tokens_seen += batch_token_count
            yield tokens.to(device)
            continue

        if in_eval:
            tokens_seen += batch_token_count
            if tokens_seen < calibration_limit:
                continue
            in_eval = False
        yield tokens.to(device)



def extract_pre_acts(encode_out):
    if hasattr(encode_out, "pre_acts"):
        return encode_out.pre_acts
    if isinstance(encode_out, dict) and "pre_acts" in encode_out:
        return encode_out["pre_acts"]
    return encode_out[0]


@torch.no_grad()
def get_latent_dim(model, sae, dtype):
    dummy = torch.zeros(1, model.cfg.d_model, device=DEVICE, dtype=dtype)
    return extract_pre_acts(sae.encode(dummy)).shape[-1]


@torch.no_grad()
def measure_pool_and_p(model, sae, token_gen, hook_name, target_tokens, top_k, d_sae):
    active_indices = torch.zeros(d_sae, dtype=torch.bool, device=DEVICE)
    total_tokens = 0
    pbar = tqdm(total=target_tokens, desc="Calibration")

    for tokens in token_gen:
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        act = cache[hook_name]
        flat_act = act.reshape(-1, act.shape[-1])
        feature_acts_raw = extract_pre_acts(sae.encode(flat_act))
        top_vals, top_idx = get_topk_sparse(feature_acts_raw, top_k)

        batch_active = torch.zeros_like(feature_acts_raw, dtype=torch.bool)
        batch_active.scatter_(-1, top_idx, top_vals > 0)
        active_indices |= batch_active.any(dim=0)

        total_tokens += tokens.numel()
        pbar.update(tokens.numel())
        if total_tokens >= target_tokens:
            break

    pbar.close()
    return active_indices, int(active_indices.sum().item())


@torch.no_grad()
def mean_token_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """Mean KL(P || Q) across all predicted token positions."""
    log_p = torch.log_softmax(logits_p[:, :-1, :].float(), dim=-1)
    log_q = torch.log_softmax(logits_q[:, :-1, :].float(), dim=-1)
    p = torch.exp(log_p)
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return float(kl.mean().item())


@torch.no_grad()
def top1_agreement(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    pred_a = logits_a[:, :-1, :].argmax(dim=-1)
    pred_b = logits_b[:, :-1, :].argmax(dim=-1)
    return float((pred_a == pred_b).float().mean().item())


@torch.no_grad()
def mean_abs_gold_logprob_diff(logits_a: torch.Tensor, logits_b: torch.Tensor, tokens: torch.Tensor) -> float:
    tgt = tokens[:, 1:]
    logp_a = torch.log_softmax(logits_a[:, :-1, :].float(), dim=-1)
    logp_b = torch.log_softmax(logits_b[:, :-1, :].float(), dim=-1)
    gold_a = torch.gather(logp_a, -1, tgt.unsqueeze(-1)).squeeze(-1)
    gold_b = torch.gather(logp_b, -1, tgt.unsqueeze(-1)).squeeze(-1)
    return float(torch.abs(gold_a - gold_b).mean().item())


@torch.no_grad()
def run_fidelity_for_model(model_key: str, cfg: ModelConfig):
    print(f"\n🚀 STARTING OUTPUT-FIDELITY ANALYSIS: {model_key}")
    # torch.cuda.empty_cache()
    # gc.collect()

    model = HookedTransformer.from_pretrained(cfg.name, device=DEVICE, dtype=cfg.dtype)
    print('a')
    sae = Sae.load_from_hub(cfg.sae_release, hookpoint=cfg.sae_hookpoint).to(DEVICE)
    sae.eval()
    print('b')

    d_sae = get_latent_dim(model, sae, cfg.dtype)
    print('c')
    vocab_size = model.cfg.d_vocab

    cal_gen = get_tokens_generator(
        model=model,
        batch_size=cfg.batch_size,
        device=DEVICE,
        mode="calibration",
        calibration_limit=ARGS.calibration_tokens,
        seq_len=ARGS.seq_len,
    )
    print('before calibration')
    pool_mask, P = measure_pool_and_p(
        model=model,
        sae=sae,
        token_gen=cal_gen,
        hook_name=cfg.hook_name,
        target_tokens=ARGS.calibration_tokens,
        top_k=ARGS.top_k,
        d_sae=d_sae,
    )
    print(f"Calibration pool size P = {P}")

    eval_gen = get_tokens_generator(
        model=model,
        batch_size=cfg.batch_size,
        device=DEVICE,
        mode="evaluation",
        calibration_limit=ARGS.calibration_tokens,
        seq_len=ARGS.seq_len,
    )

    total_tokens = 0
    current_step_idx = 0
    max_tokens = max(ARGS.n_steps)

    running = {
        "kl_unrestricted": [],
        "kl_restricted": [],
        "top1_unrestricted": [],
        "top1_restricted": [],
        "gold_lp_diff_unrestricted": [],
        "gold_lp_diff_restricted": [],
        "loss_M": [],
        "loss_Som": [],
        "loss_hG": [],
    }
    rows = []

    pbar = tqdm(total=max_tokens, desc="Evaluation")
    for tokens in eval_gen:
        orig_logits, cache = model.run_with_cache(tokens, names_filter=[cfg.hook_name])
        act = cache[cfg.hook_name]
        flat_act = act.reshape(-1, act.shape[-1])
        feature_acts_raw = extract_pre_acts(sae.encode(flat_act))

        top_vals_u, top_idx_u = get_topk_sparse(feature_acts_raw, ARGS.top_k)
        recons_u = sae.decode(top_vals_u, top_idx_u).reshape(act.shape).to(act.dtype)

        masked_pre_acts = feature_acts_raw * pool_mask.unsqueeze(0)
        top_vals_r, top_idx_r = get_topk_sparse(masked_pre_acts, ARGS.top_k)
        recons_r = sae.decode(top_vals_r, top_idx_r).reshape(act.shape).to(act.dtype)

        def hook_unrestricted(activations, hook):
            return recons_u

        logits_som = model.run_with_hooks(tokens, fwd_hooks=[(cfg.hook_name, hook_unrestricted)])

        def hook_restricted(activations, hook):
            return recons_r

        logits_hg = model.run_with_hooks(tokens, fwd_hooks=[(cfg.hook_name, hook_restricted)])

        running["kl_unrestricted"].append(mean_token_kl(orig_logits, logits_som))
        running["kl_restricted"].append(mean_token_kl(orig_logits, logits_hg))
        running["top1_unrestricted"].append(top1_agreement(orig_logits, logits_som))
        running["top1_restricted"].append(top1_agreement(orig_logits, logits_hg))
        running["gold_lp_diff_unrestricted"].append(mean_abs_gold_logprob_diff(orig_logits, logits_som, tokens))
        running["gold_lp_diff_restricted"].append(mean_abs_gold_logprob_diff(orig_logits, logits_hg, tokens))
        running["loss_M"].extend(smoothed_bpd_loss(orig_logits, tokens, ARGS.alpha, vocab_size, reduction="none").cpu().tolist())
        running["loss_Som"].extend(smoothed_bpd_loss(logits_som, tokens, ARGS.alpha, vocab_size, reduction="none").cpu().tolist())
        running["loss_hG"].extend(smoothed_bpd_loss(logits_hg, tokens, ARGS.alpha, vocab_size, reduction="none").cpu().tolist())

        total_tokens += tokens.numel()
        pbar.update(tokens.numel())

        while current_step_idx < len(ARGS.n_steps) and total_tokens >= ARGS.n_steps[current_step_idx]:
            target_tokens = ARGS.n_steps[current_step_idx]
            N = target_tokens / ARGS.seq_len
            row = {
                "Model": model_key,
                "Layer": SELECTED_LAYER,
                "TopK": ARGS.top_k,
                "SeqLen": ARGS.seq_len,
                "Tokens": target_tokens,
                "N": N,
                "P": P,
                "KL_M_vs_SoM": float(np.mean(running["kl_unrestricted"])),
                "KL_M_vs_hG": float(np.mean(running["kl_restricted"])),
                "Top1Agree_M_vs_SoM": float(np.mean(running["top1_unrestricted"])),
                "Top1Agree_M_vs_hG": float(np.mean(running["top1_restricted"])),
                "AbsGoldLogProbDiff_M_vs_SoM": float(np.mean(running["gold_lp_diff_unrestricted"])),
                "AbsGoldLogProbDiff_M_vs_hG": float(np.mean(running["gold_lp_diff_restricted"])),
                "Loss_M": float(np.mean(running["loss_M"])),
                "Loss_SoM": float(np.mean(running["loss_Som"])),
                "Loss_hG": float(np.mean(running["loss_hG"])),
                "Color": cfg.color,
            }
            rows.append(row)
            print(
                f"  [tokens={target_tokens:,} | N={N:.0f}] "
                f"KL(M||SoM)={row['KL_M_vs_SoM']:.5f}, KL(M||hG)={row['KL_M_vs_hG']:.5f}, "
                f"Top1(M,hG)={row['Top1Agree_M_vs_hG']:.4f}, "
                f"|Δ log p_gold|(M,hG)={row['AbsGoldLogProbDiff_M_vs_hG']:.5f}"
            )
            current_step_idx += 1

        if current_step_idx >= len(ARGS.n_steps):
            break

    pbar.close()
    return rows


all_rows = []
for model_name, cfg in MODEL_CONFIGS.items():
    all_rows.extend(run_fidelity_for_model(model_name, cfg))

os.makedirs(ARGS.out_dir, exist_ok=True)
df = pd.DataFrame(all_rows)
out_csv = os.path.join(ARGS.out_dir, f"output_fidelity_layer_{SELECTED_LAYER}.csv")
df.to_csv(out_csv, index=False)
print(f"Saved {out_csv}")

if not df.empty:
    plt.rcParams.update({"font.family": "serif", "font.size": 12})

    # Plot 1: KL by N
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x="N", y="KL_M_vs_hG", marker="o", linewidth=2.5, label="KL(M || h_G)", ax=ax)
    sns.lineplot(data=df, x="N", y="KL_M_vs_SoM", marker="o", linewidth=2.5, label="KL(M || S∘M)", ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel("Number of samples (N)", fontweight="bold")
    ax.set_ylabel("Mean KL divergence", fontweight="bold")
    ax.set_title(f"Output fidelity by KL — layer {SELECTED_LAYER}", fontsize=14, pad=15)
    ax.grid(True, which="both", linestyle="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(ARGS.out_dir, f"output_fidelity_kl_layer_{SELECTED_LAYER}.png"), dpi=300)

    # Plot 2: top-1 agreement by N
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x="N", y="Top1Agree_M_vs_hG", marker="o", linewidth=2.5, label="Top-1(M, h_G)", ax=ax)
    sns.lineplot(data=df, x="N", y="Top1Agree_M_vs_SoM", marker="o", linewidth=2.5, label="Top-1(M, S∘M)", ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel("Number of samples (N)", fontweight="bold")
    ax.set_ylabel("Top-1 agreement", fontweight="bold")
    ax.set_title(f"Output fidelity by top-1 agreement — layer {SELECTED_LAYER}", fontsize=14, pad=15)
    ax.grid(True, which="both", linestyle="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(ARGS.out_dir, f"output_fidelity_top1_layer_{SELECTED_LAYER}.png"), dpi=300)
