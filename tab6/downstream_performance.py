import gc
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens import SAE
from sparsify import Sae


# ==========================================
# 1. CONFIG
# ==========================================
CONFIG = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "TOP_K": 64,
    "SEQ_LEN": 32,
    "CALIBRATION_TOKENS": 2240000,
    "TOKENIZER_BUFFER_SIZE": 32000,
    "MAX_EXAMPLES_PER_DATASET": 1000, # increase later if needed
    "OUTPUT_DIR": "./final_ops",
    "MODELS": {
        # Keep this if you want GPT/Gemma too. You can comment them out initially.
        "GPT-2 Small": {
            "name": "gpt2-small",
            "sae_backend": "sae_lens",
            "sae_release": "gpt2-small-res-jb",
            "sae_id": "blocks.6.hook_resid_pre",
            "hook_name": "blocks.6.hook_resid_pre",
            "batch_size": 16,
            "dtype": torch.float32,
            "layer_sweep": [6],
        },
        
        "Gemma-2B": {
            "name": "gemma-2b",
            "sae_backend": "sae_lens",
            "sae_release": "gemma-2b-res-jb",
            "sae_id": "blocks.12.hook_resid_post",
            "hook_name": "blocks.12.hook_resid_post",
            "batch_size": 16,
            "dtype": torch.float32,
            "layer_sweep": [12],
        },

        "Llama-3.1-8B": {
            "name": "meta-llama/Meta-Llama-3-8B",
            "sae_backend": "sparsify",
            "sae_release": "EleutherAI/sae-llama-3-8b-32x",
            "sae_hookpoint_template": "layers.{layer}",
            "hook_name_template": "blocks.{layer}.hook_resid_post",
            "batch_size": 16,
            "dtype": torch.bfloat16,
            # "layer_sweep": [30],
            # change as needed
            "layer_sweep": [4, 8, 12, 16, 20, 24, 28, 30],
        },

        
    },
}


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_topk_sparse(activations, k):
    top_acts, top_indices = torch.topk(activations, k=k, dim=-1)
    top_acts = torch.relu(top_acts)
    return top_acts, top_indices


def apply_topk_dense(activations, k):
    topk_vals, topk_inds = torch.topk(activations, k=k, dim=-1)
    mask = torch.zeros_like(activations, dtype=torch.bool)
    mask.scatter_(-1, topk_inds, True)
    return activations * mask


def get_tokens_generator(model, batch_size, device, mode, calibration_limit=0):
    """
    Same generator logic as your bound script.
    Uses C4 for calibration pool construction.
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


def load_model_and_sae(config, layer=None):
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
        assert layer is not None, "layer must be provided for sparsify backend"
        sae_hookpoint = config["sae_hookpoint_template"].format(layer=layer)
        hook_name = config["hook_name_template"].format(layer=layer)
        sae = Sae.load_from_hub(
            config["sae_release"],
            hookpoint=sae_hookpoint,
        )
        sae = sae.to(CONFIG["DEVICE"])

    else:
        raise ValueError(f"Unknown sae_backend: {config['sae_backend']}")

    sae.eval()
    adapter = SAEAdapter(sae=sae, backend=config["sae_backend"], device=CONFIG["DEVICE"])
    return model, adapter, hook_name, dtype


def measure_pool_and_p(model, sae_adapter, token_gen, hook_name, target_tokens, k, device, d_model, dtype):
    print(f" --> [Calibration] Measuring Pool Size (P) on {target_tokens} tokens with k={k}...")

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
    print(f" <-- [Calibration] Complete. Active Pool Size P = {P}")
    return active_indices, P, d_sae


# ==========================================
# 3. DATASET FORMATTERS
# ==========================================
def load_winogrande(max_examples):
    ds = load_dataset("winogrande", "winogrande_xl", split="validation")
    items = []
    for ex in ds.select(range(min(max_examples, len(ds)))):
        sentence = ex["sentence"]
        option1 = ex["option1"]
        option2 = ex["option2"]
        answer = int(ex["answer"]) - 1 # dataset uses "1"/"2"

        blank = "_"
        if blank in sentence:
            prompt1 = sentence.replace(blank, option1)
            prompt2 = sentence.replace(blank, option2)
        else:
            # fallback, though Winogrande has "_"
            prompt1 = sentence + " " + option1
            prompt2 = sentence + " " + option2

        items.append({
            "dataset": "winogrande",
            "prompt": "",
            "choices": [prompt1, prompt2],
            "label": answer,
        })
    return items


def load_piqa(max_examples):
    ds = load_dataset("piqa", split="validation")
    items = []
    for ex in ds.select(range(min(max_examples, len(ds)))):
        goal = ex["goal"].strip()
        sol1 = ex["sol1"].strip()
        sol2 = ex["sol2"].strip()
        label = int(ex["label"])
        prompt = f"Question: {goal}\nAnswer:"
        items.append({
            "dataset": "piqa",
            "prompt": prompt,
            "choices": [f" {sol1}", f" {sol2}"],
            "label": label,
        })
    return items


def load_hellaswag(max_examples):
    ds = load_dataset("hellaswag", split="validation")
    items = []
    for ex in ds.select(range(min(max_examples, len(ds)))):
        ctx = ex["ctx"].strip()
        endings = [e.strip() for e in ex["endings"]]
        label = int(ex["label"])
        prompt = ctx
        items.append({
            "dataset": "hellaswag",
            "prompt": prompt,
            "choices": [(" " + e) for e in endings],
            "label": label,
        })
    return items


def load_all_datasets(max_examples):
    return (
        load_winogrande(max_examples),
        load_piqa(max_examples),
        load_hellaswag(max_examples),
    )


# ==========================================
# 4. SCORING HELPERS
# ==========================================
def continuation_avg_logprob_from_logits(logits, full_tokens, prompt_len):
    """
    Score only the continuation tokens, using average log-probability.
    full_tokens shape: [1, T]
    logits shape: [1, T, V]
    prompt_len: number of tokens in prompt-only sequence
    """
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1) # predicts token positions 1..T-1
    target_tokens = full_tokens[:, 1:]

    # continuation token positions in target_tokens are indices >= prompt_len-1
    start = max(prompt_len - 1, 0)
    if start >= target_tokens.shape[1]:
        return -1e9

    cont_targets = target_tokens[:, start:]
    cont_log_probs = log_probs[:, start:, :]
    gathered = torch.gather(cont_log_probs, -1, cont_targets.unsqueeze(-1)).squeeze(-1)

    # average over continuation length
    return gathered.mean().item()


@dataclass
class ChoiceScores:
    base: float
    som: float
    hg: float


def score_choice_with_base_and_proxies(
    model,
    sae_adapter,
    hook_name,
    pool_mask,
    full_text,
    prompt_text,
    top_k,
):
    device = CONFIG["DEVICE"]

    full_tokens = model.to_tokens(full_text).to(device)
    prompt_tokens = model.to_tokens(prompt_text).to(device)

    prompt_len = prompt_tokens.shape[1]

    with torch.no_grad():
        base_logits, cache = model.run_with_cache(full_tokens, names_filter=[hook_name])
        act = cache[hook_name]
        flat_act = act.reshape(-1, act.shape[-1])

        feature_acts_raw = sae_adapter.encode_pre_acts(flat_act)

        # unrestricted SAE proxy: S∘M
        recons_unrestricted, _ = sae_adapter.decode_topk(feature_acts_raw, top_k)
        recons_unrestricted = recons_unrestricted.reshape(act.shape)

        # restricted proxy: h_G
        masked_acts = feature_acts_raw * pool_mask.unsqueeze(0)
        recons_restricted, _ = sae_adapter.decode_topk(masked_acts, top_k)
        recons_restricted = recons_restricted.reshape(act.shape)

        def hook_unrestricted(activations, hook):
            return recons_unrestricted

        def hook_restricted(activations, hook):
            return recons_restricted

        logits_som = model.run_with_hooks(full_tokens, fwd_hooks=[(hook_name, hook_unrestricted)])
        logits_hg = model.run_with_hooks(full_tokens, fwd_hooks=[(hook_name, hook_restricted)])

        base_score = continuation_avg_logprob_from_logits(base_logits, full_tokens, prompt_len)
        som_score = continuation_avg_logprob_from_logits(logits_som, full_tokens, prompt_len)
        hg_score = continuation_avg_logprob_from_logits(logits_hg, full_tokens, prompt_len)

    return ChoiceScores(base=base_score, som=som_score, hg=hg_score)


# ==========================================
# 5. EVALUATION LOGIC
# ==========================================
def evaluate_dataset(model_key, model, sae_adapter, hook_name, pool_mask, dataset_items, top_k, layer_tag):
    rows = []
    correct_base = 0
    correct_som = 0
    correct_hg = 0

    dataset_name = dataset_items[0]["dataset"] if len(dataset_items) > 0 else "dataset"
    for idx, ex in enumerate(tqdm(dataset_items, desc=f"{model_key} | {dataset_name}")):

    # for idx, ex in enumerate(tqdm(dataset_items, desc=f"{model_key} | {ex['dataset'] if dataset_items else 'dataset'}")):
        prompt = ex["prompt"]
        choices = ex["choices"]
        label = ex["label"]
        dataset_name = ex["dataset"]

        scores = []
        for choice in choices:
            full_text = prompt + choice
            prompt_text = prompt
            scores.append(
                score_choice_with_base_and_proxies(
                    model=model,
                    sae_adapter=sae_adapter,
                    hook_name=hook_name,
                    pool_mask=pool_mask,
                    full_text=full_text,
                    prompt_text=prompt_text,
                    top_k=top_k,
                )
            )

        pred_base = int(np.argmax([s.base for s in scores]))
        pred_som = int(np.argmax([s.som for s in scores]))
        pred_hg = int(np.argmax([s.hg for s in scores]))

        is_base = int(pred_base == label)
        is_som = int(pred_som == label)
        is_hg = int(pred_hg == label)

        correct_base += is_base
        correct_som += is_som
        correct_hg += is_hg

        rows.append({
            "Model": model_key,
            "Layer": layer_tag,
            "Dataset": dataset_name,
            "Index": idx,
            "Label": label,
            "Pred_M": pred_base,
            "Pred_SoM": pred_som,
            "Pred_hG": pred_hg,
            "Correct_M": is_base,
            "Correct_SoM": is_som,
            "Correct_hG": is_hg,
        })

    n = len(dataset_items)
    summary = {
        "Model": model_key,
        "Layer": layer_tag,
        "Dataset": dataset_items[0]["dataset"] if len(dataset_items) > 0 else "unknown",
        "N": n,
        "Acc_M": correct_base / n if n > 0 else 0.0,
        "Acc_SoM": correct_som / n if n > 0 else 0.0,
        "Acc_hG": correct_hg / n if n > 0 else 0.0,
        "Drop_SoM": (correct_base - correct_som) / n if n > 0 else 0.0,
        "Drop_hG": (correct_base - correct_hg) / n if n > 0 else 0.0,
    }
    return rows, summary


def run_for_model_and_layer(model_key, config, layer=None):
    print(f"\n🚀 STARTING DOWNSTREAM EXPERIMENT: {model_key} | layer={layer}")

    torch.cuda.empty_cache()
    gc.collect()

    try:
        model, sae_adapter, hook_name, dtype = load_model_and_sae(config, layer=layer)
    except Exception as e:
        print(f"Error loading {model_key} layer={layer}: {e}")
        return [], []

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

    print(f"Pool size P for {model_key} layer={layer}: {P}")

    all_rows = []
    all_summaries = []

    dataset_groups = load_all_datasets(CONFIG["MAX_EXAMPLES_PER_DATASET"])

    for dataset_items in dataset_groups:
        rows, summary = evaluate_dataset(
            model_key=model_key,
            model=model,
            sae_adapter=sae_adapter,
            hook_name=hook_name,
            pool_mask=pool_mask,
            dataset_items=dataset_items,
            top_k=top_k,
            layer_tag=layer if layer is not None else "fixed",
        )
        summary["P"] = P
        all_rows.extend(rows)
        all_summaries.append(summary)

        print(
            f"[{summary['Dataset']}] "
            f"Acc_M={summary['Acc_M']:.4f} | "
            f"Acc_SoM={summary['Acc_SoM']:.4f} | "
            f"Acc_hG={summary['Acc_hG']:.4f}"
        )

    return all_rows, all_summaries


# ==========================================
# 6. MAIN
# ==========================================
def main():
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    all_rows = []
    all_summaries = []

    for model_key, cfg in CONFIG["MODELS"].items():
        layers = cfg.get("layer_sweep", [None])

        for layer in layers:
            rows, summaries = run_for_model_and_layer(
                model_key=model_key,
                config=cfg,
                layer=layer,
            )
            all_rows.extend(rows)
            all_summaries.extend(summaries)

    df_rows = pd.DataFrame(all_rows)
    df_summary = pd.DataFrame(all_summaries)

    rows_path = os.path.join(CONFIG["OUTPUT_DIR"], "zero_shot_base_vs_proxy_results_final.csv")
    summary_path = os.path.join(CONFIG["OUTPUT_DIR"], "zero_shot_base_vs_proxy_summary_final.csv")

    df_rows.to_csv(rows_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    print(f"\nSaved detailed results to: {rows_path}")
    print(f"Saved summary results to: {summary_path}")


if __name__ == "__main__":
    main()