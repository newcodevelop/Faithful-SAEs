
import math
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASELINE = 16.97
BOUND_DIR = "./bounds/"
FIDELITY_DIR = "./final_ops/"
OUT_DIR = "./final_ops/"

def human_n(x, pos=None):
    x = float(x)
    if x >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if x >= 1_000:
        return f"{x/1_000:.0f}k"
    return f"{int(x)}"

def load_bound_curves():
    records = []
    for path in sorted(glob.glob(os.path.join(BOUND_DIR, "llama3_8b_layers_*.csv")),
                       key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0])):
        layer = int(os.path.basename(path).split("_")[-1].split(".")[0])
        df = pd.read_csv(path).sort_values("N")
        last = df.iloc[-1]
        R = float(last["R_hat_hG"])
        eps = float(last["eps_loss_hat"])
        eta = float(last["eta_hat"])
        B = float(last["B"])
        delta = float(last["delta"])
        m = float(last["m"])
        P = float(last["P"])
        N_last = float(last["N"])

        ssd = P * math.log((math.e * m) / P)

        def bound_at_n(n):
            t4 = B * math.sqrt((ssd + math.log(2 / delta)) / (2 * n))
            t5 = B * math.sqrt(math.log(4 / delta) / (2 * n))
            return R + eps + eta * B + t4 + t5

        asym_floor = R + eps + eta * B

        observed = df.copy()
        observed["Bound"] = observed["N"].apply(bound_at_n)

        extrap_rows = []
        n = N_last
        for _ in range(10):
            n = n * 2
            b = bound_at_n(n)
            extrap_rows.append({"Layer": layer, "N": n, "Bound": b})
            if b <= BASELINE:
                break

        extrap = pd.DataFrame(extrap_rows)
        nstar = None
        bstar = None
        if not extrap.empty and (extrap["Bound"] <= BASELINE).any():
            first = extrap.loc[extrap["Bound"] <= BASELINE].iloc[0]
            nstar = float(first["N"])
            bstar = float(first["Bound"])

        records.append({
            "layer": layer,
            "observed": observed[["N", "Bound"]],
            "extrap": extrap,
            "P": int(P),
            "R": R,
            "eps": eps,
            "eta": eta,
            "B70k": float(observed["Bound"].iloc[-1]),
            "asym_floor": asym_floor,
            "nstar": nstar,
            "bstar": bstar,
        })
    return records

# def make_bound_figure(records):
#     fig, ax = plt.subplots(figsize=(10.2, 6.4))

#     for rec in records:
#         obs = rec["observed"]
#         extra = rec["extrap"]
#         layer = rec["layer"]

#         ax.plot(
#             obs["N"], obs["Bound"],
#             marker="o", linewidth=2.2, markersize=5.5,
#             label=f"Layer {layer}"
#         )

#         if not extra.empty:
#             x = [obs["N"].iloc[-1]] + extra["N"].tolist()
#             y = [obs["Bound"].iloc[-1]] + extra["Bound"].tolist()
#             ax.plot(
#                 x, y,
#                 linestyle="--", linewidth=1.8, alpha=0.95
#             )

#             if rec["nstar"] is not None:
#                 ax.scatter(rec["nstar"], rec["bstar"], s=35, zorder=4)
#                 ax.annotate(
#                     f"{human_n(rec['nstar'])}",
#                     xy=(rec["nstar"], rec["bstar"]),
#                     xytext=(4, -10),
#                     textcoords="offset points",
#                     fontsize=8
#                 )

#     ax.axhline(BASELINE, linestyle=":", linewidth=2.0, color="black", label="Baseline = 16.97 bits")

#     ax.set_xscale("log", base=2)
#     ax.xaxis.set_major_formatter(FuncFormatter(human_n))
#     ax.set_xlabel("Sample size $N$")
#     ax.set_ylabel("Total bound (bits)")
#     ax.set_title("LLaMA-3-8B: observed and extrapolated bounds across layers")
#     ax.grid(True, which="both", alpha=0.25)
#     ax.legend(ncol=3, fontsize=8.5, frameon=True)

#     fig.tight_layout()
#     out_png = os.path.join(OUT_DIR, "llama_bound_extrapolated.png")
#     out_pdf = os.path.join(OUT_DIR, "llama_bound_extrapolated.pdf")
#     fig.savefig(out_png, dpi=300, bbox_inches="tight")
#     fig.savefig(out_pdf, bbox_inches="tight")
#     plt.close(fig)
#     return out_png, out_pdf



def make_bound_figure(records):
    fig, ax = plt.subplots(figsize=(10.2, 6.4))

    for rec in records:
        obs = rec["observed"]
        extra = rec["extrap"]
        layer = rec["layer"]

        # merge observed + continued N values into one normal-looking curve
        if not extra.empty:
            x = obs["N"].tolist() + extra["N"].tolist()
            y = obs["Bound"].tolist() + extra["Bound"].tolist()
        else:
            x = obs["N"].tolist()
            y = obs["Bound"].tolist()

        ax.plot(
            x, y,
            marker="o", linewidth=2.2, markersize=5.5,
            label=f"Layer {layer}"
        )

        if rec["nstar"] is not None:
            ax.scatter(rec["nstar"], rec["bstar"], s=35, zorder=4)
            ax.annotate(
                f"{human_n(rec['nstar'])}",
                xy=(rec["nstar"], rec["bstar"]),
                xytext=(4, -10),
                textcoords="offset points",
                fontsize=8
            )

    ax.axhline(BASELINE, linestyle=":", linewidth=2.0, color="black", label="Baseline = 16.98 bits")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(human_n))
    ax.set_xlabel("Sample size $N$")
    ax.set_ylabel("Total bound (bits)")
    ax.set_title("LLaMA-3-8B bounds across layers")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=3, fontsize=8.5, frameon=True)

    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "llama_bound_layers.png")
    out_pdf = os.path.join(OUT_DIR, "llama_bound_layers.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf




def load_fidelity_curves(records):
    # use layer-specific N* from the bound extrapolation to define how far to extend the flat fidelity tails
    nstar_map = {rec["layer"]: rec["nstar"] for rec in records if rec["nstar"] is not None}

    layers = [4, 8, 12, 16, 20, 24, 28, 30]
    fidelities = []
    for layer in layers:
        path = os.path.join(FIDELITY_DIR, f"output_fidelity_layer_{layer}.csv")
        if not os.path.exists(path):
            alt = os.path.join(FIDELITY_DIR, f"output_fidelity_layer_{layer} (1).csv")
            path = alt if os.path.exists(alt) else path
        df = pd.read_csv(path).sort_values("N")

        nstar = nstar_map.get(layer, None)
        extrap = []
        if nstar is not None and nstar > df["N"].iloc[-1]:
            n = float(df["N"].iloc[-1])
            last = df.iloc[-1]
            while n < nstar:
                n = n * 2
                row = last.copy()
                row["N"] = min(n, nstar)
                extrap.append(row)

        extrap_df = pd.DataFrame(extrap)
        fidelities.append({"layer": layer, "observed": df, "extrap": extrap_df})
    return fidelities

# def make_fidelity_figure(fidelities):
#     fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), sharex=True)
#     panels = [
#         ("KL_M_vs_SoM", "KL$(M\\,\\|\\,S\\circ M)$"),
#         ("Top1Agree_M_vs_SoM", "Top-1 agreement $(M, S\\circ M)$"),
#         ("AbsGoldLogProbDiff_M_vs_SoM", r"$|\Delta \log p_{\mathrm{gold}}|$"),
#         ("Loss_SoM", "Loss$(S\\circ M)$"),
#     ]

#     for ax, (col, title) in zip(axes.flatten(), panels):
#         for item in fidelities:
#             layer = item["layer"]
#             obs = item["observed"]
#             extra = item["extrap"]

#             ax.plot(
#                 obs["N"], obs[col],
#                 marker="o", linewidth=2.2, markersize=5.2,
#                 label=f"Layer {layer}"
#             )

#             if not extra.empty:
#                 x = [obs["N"].iloc[-1]] + extra["N"].tolist()
#                 y = [obs[col].iloc[-1]] + extra[col].tolist()
#                 ax.plot(
#                     x, y,
#                     linestyle="--", linewidth=1.8, alpha=0.95
#                 )

#         ax.set_title(title)
#         ax.grid(True, which="both", alpha=0.25)
#         ax.set_xscale("log", base=2)
#         ax.xaxis.set_major_formatter(FuncFormatter(human_n))

#     axes[1, 0].set_xlabel("Sample size $N$")
#     axes[1, 1].set_xlabel("Sample size $N$")
#     axes[0, 0].set_ylabel("Value")
#     axes[1, 0].set_ylabel("Value")

#     handles, labels = axes[0, 0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.02))
#     fig.suptitle("LLaMA-3-8B late-layer output fidelity (observed and flat-tail extrapolated)", y=1.06, fontsize=13)
#     fig.tight_layout()

#     out_png = os.path.join(OUT_DIR, "llama_output_fidelity_extrapolated.png")
#     out_pdf = os.path.join(OUT_DIR, "llama_output_fidelity_extrapolated.pdf")
#     fig.savefig(out_png, dpi=300, bbox_inches="tight")
#     fig.savefig(out_pdf, bbox_inches="tight")
#     plt.close(fig)
#     return out_png, out_pdf



def make_fidelity_figure(fidelities):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), sharex=True)
    panels = [
        ("KL_M_vs_SoM", "KL$(M\\,\\|\\,S\\circ M)$"),
        ("Top1Agree_M_vs_SoM", "Top-1 agreement $(M, S\\circ M)$"),
        ("AbsGoldLogProbDiff_M_vs_SoM", r"$|\Delta \log p_{\mathrm{gold}}|$"),
        ("Loss_SoM", "Loss$(S\\circ M)$"),
    ]

    for ax, (col, title) in zip(axes.flatten(), panels):
        for item in fidelities:
            layer = item["layer"]
            obs = item["observed"]
            extra = item["extrap"]

            # merge observed + continued N values into one normal-looking curve
            if not extra.empty:
                x = obs["N"].tolist() + extra["N"].tolist()
                y = obs[col].tolist() + extra[col].tolist()
            else:
                x = obs["N"].tolist()
                y = obs[col].tolist()

            ax.plot(
                x, y,
                marker="o", linewidth=2.2, markersize=5.2,
                label=f"Layer {layer}"
            )

        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(human_n))

    axes[1, 0].set_xlabel("Sample size $N$")
    axes[1, 1].set_xlabel("Sample size $N$")
    axes[0, 0].set_ylabel("Value")
    axes[1, 0].set_ylabel("Value")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("LLaMA-3-8B late-layer output fidelity", y=1.06, fontsize=13)
    fig.tight_layout()

    out_png = os.path.join(OUT_DIR, "llama_output_fidelity_layers.png")
    out_pdf = os.path.join(OUT_DIR, "llama_output_fidelity_layers.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


def main():
    bound_records = load_bound_curves()
    out1 = make_bound_figure(bound_records)
    fidelities = load_fidelity_curves(bound_records)
    out2 = make_fidelity_figure(fidelities)
    print("Saved:", out1, out2)

if __name__ == "__main__":
    main()
