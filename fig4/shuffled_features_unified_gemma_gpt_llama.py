import gc

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from sae_lens import SAE
from sparsify import Sae
from tqdm import tqdm
from transformer_lens import HookedTransformer

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "N_TOKENS": 2240000,
    "ALPHA": 0.5,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "SEQ_LEN": 32,
    "TOP_K": 64,
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
            "color": "#2ca02c",
            "dtype": torch.bfloat16,
            "top_k": 64,
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


def extract_pre_acts(encode_out):
    """
    Normalize encode outputs across SAE backends.
    - sae_lens: returns dense activations directly.
    - sparsify: returns an object/dict containing pre_acts.
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


def get_topk_sparse(activations, k):
    """
    Extract sparse top-k nonnegative activations for sparsify-style decode.
    """
    k = min(k, activations.shape[-1])
    top_acts, top_indices = torch.topk(activations, k=k, dim=-1)
    top_acts = torch.relu(top_acts)
    return top_acts, top_indices


def load_model_and_sae(config):
    dtype = config.get("dtype", torch.float32)
    model = HookedTransformer.from_pretrained(
        config["name"],
        device=CONFIG["DEVICE"],
        dtype=dtype,
    )

    backend = config["sae_backend"]
    if backend == "sae_lens":
        sae, _, _ = SAE.from_pretrained(
            release=config["sae_release"],
            sae_id=config["sae_id"],
            device=CONFIG["DEVICE"],
        )
        hook_name = config["hook_name"]
    elif backend == "sparsify":
        sae = Sae.load_from_hub(
            config["sae_release"],
            hookpoint=config["sae_hookpoint"],
        )
        sae = sae.to(CONFIG["DEVICE"])
        hook_name = config["hook_name"]
    else:
        raise ValueError(f"Unknown sae_backend: {backend}")

    sae.eval()
    return model, sae, backend, hook_name


def reconstruct_real_and_shuffled(sae, backend, feature_acts, device, top_k):
    """
    Build real-semantic and shuffled-semantic reconstructions.

    For sae_lens backends, we preserve the original dense behavior from the
    Gemma/GPT script.

    For sparsify backends (Llama), decode expects sparse (values, indices).
    We therefore keep the same top-k activation magnitudes and only shuffle the
    latent indices, which preserves sparsity while randomizing semantic identity.
    """
    if backend == "sae_lens":
        recons_real = sae.decode(feature_acts)
        perm_idx = torch.randperm(feature_acts.shape[1], device=device)
        feature_acts_shuffled = feature_acts[:, perm_idx]
        recons_shuffled = sae.decode(feature_acts_shuffled)
        return recons_real, recons_shuffled

    top_acts_real, top_indices_real = get_topk_sparse(feature_acts, top_k)
    recons_real = sae.decode(top_acts_real, top_indices_real)

    perm_idx = torch.randperm(feature_acts.shape[1], device=device)
    top_indices_shuffled = perm_idx[top_indices_real]
    recons_shuffled = sae.decode(top_acts_real, top_indices_shuffled)
    return recons_real, recons_shuffled


def run_shuffling_experiment(model_key, config):
    print(f"\n🧪 STARTING ABLATION (Shuffling): {model_key}")

    torch.cuda.empty_cache()
    gc.collect()

    try:
        model, sae, sae_backend, hook_name = load_model_and_sae(config)
        vocab_size = model.cfg.d_vocab
    except Exception as e:
        print(f"Skipping {model_key}: {e}")
        return []

    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    iterator = iter(dataset)

    results = []
    total_tokens = 0
    pbar = tqdm(total=CONFIG["N_TOKENS"])

    batch_size = config["batch_size"]
    top_k = config.get("top_k", CONFIG["TOP_K"])

    while total_tokens < CONFIG["N_TOKENS"]:
        batch_texts = []
        try:
            for _ in range(batch_size):
                item = next(iterator)
                text = item["text"] if "text" in item else item["content"]
                batch_texts.append(text)
        except StopIteration:
            if not batch_texts:
                break

        tokens = model.to_tokens(batch_texts)
        if tokens.shape[1] < CONFIG["SEQ_LEN"]:
            continue
        tokens = tokens[:, : CONFIG["SEQ_LEN"]]

        with torch.no_grad():
            orig_logits, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            orig_loss_vec = smoothed_bpd_loss(
                orig_logits,
                tokens,
                CONFIG["ALPHA"],
                vocab_size,
                reduction="none",
            )

            original_act = cache[hook_name]
            flat_act = original_act.reshape(-1, original_act.shape[-1])
            feature_acts = extract_pre_acts(sae.encode(flat_act))

            recons_real, recons_shuffled = reconstruct_real_and_shuffled(
                sae=sae,
                backend=sae_backend,
                feature_acts=feature_acts,
                device=CONFIG["DEVICE"],
                top_k=top_k,
            )
            recons_real = recons_real.reshape(original_act.shape)
            recons_shuffled = recons_shuffled.reshape(original_act.shape)

            def hook_real(activations, hook):
                return recons_real

            proxy_logits_real = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_real)])
            loss_real_vec = smoothed_bpd_loss(
                proxy_logits_real,
                tokens,
                CONFIG["ALPHA"],
                vocab_size,
                reduction="none",
            )

            def hook_shuffled(activations, hook):
                return recons_shuffled

            proxy_logits_shuff = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_shuffled)])
            loss_shuff_vec = smoothed_bpd_loss(
                proxy_logits_shuff,
                tokens,
                CONFIG["ALPHA"],
                vocab_size,
                reduction="none",
            )

            gap_real_vec = torch.abs(orig_loss_vec - loss_real_vec).detach().cpu().float().numpy()
            gap_shuff_vec = torch.abs(orig_loss_vec - loss_shuff_vec).detach().cpu().float().numpy()

            for g_r, g_s in zip(gap_real_vec, gap_shuff_vec):
                results.append({"Model": model_key, "Condition": "Real SAE", "Gap (Bits)": g_r})
                results.append({"Model": model_key, "Condition": "Shuffled Features", "Gap (Bits)": g_s})

        total_tokens += tokens.numel()
        pbar.update(tokens.numel())

    pbar.close()
    del model, sae
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ==========================================
# 3. RUN EXPERIMENT
# ==========================================
all_results = []
for key, cfg in CONFIG["MODELS"].items():
    res = run_shuffling_experiment(key, cfg)
    all_results.extend(res)

df = pd.DataFrame(all_results)


# ==========================================
# 4. PLOTTING
# ==========================================
print("\nGenerating Histogram Plot...")
plt.rcParams.update({"font.family": "serif", "font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42})
sns.set_theme(style="white", context="paper", font_scale=1.2)

g = sns.FacetGrid(
    df,
    col="Model",
    hue="Condition",
    height=5,
    aspect=1.3,
    palette={"Real SAE": "#2ca02c", "Shuffled Features": "#d62728"},
    sharex=False,
)
g.map(sns.kdeplot, "Gap (Bits)", fill=True, alpha=0.4, linewidth=2, clip=(0, None))

g.set_titles("{col_name}", fontweight="bold", fontsize=14)
g.set_axis_labels("Reconstruction Gap (Bits)", "Density", fontsize=12)
g.add_legend(title="Semantic Condition")

plt.subplots_adjust(top=0.85)
g.fig.suptitle(
    "Ablation B: Semantic Specificity Test\n(Same Sparsity, Random Meaning)",
    fontsize=16,
    fontweight="bold",
)

save_path_png = "./final_ops/ablation_shuffled_histogram_unified30_final.png"
save_path_pdf = "./final_ops/ablation_shuffled_histogram_unified30_final.pdf"
plt.savefig(save_path_png, dpi=600, bbox_inches="tight")
plt.savefig(save_path_pdf, bbox_inches="tight")
print(f"✅ Plot saved as {save_path_png}")
print(f"✅ Plot saved as {save_path_pdf}")
