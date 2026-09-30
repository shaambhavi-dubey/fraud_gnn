"""Step 7: subgroup plots and shift curves (mean +/- sd across seeds)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

COLORS = {"xgboost": "#7a7a7a", "gcn": "#1f77b4", "graphsage": "#d62728"}
METRICS = [("aucpr", "AUC-PR"), ("recall", "Recall"), ("fpr_licit", "FP rate among licit")]


def subgroup_plot():
    df = pd.read_csv(C.RAW_DIR / "per_group_metrics.csv")
    for fs in df.feature_set.unique():
        d = df[df.feature_set == fs]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (m, label) in zip(axes, METRICS):
            for j, model in enumerate(C.MODELS):
                g = d[d.model == model].groupby("group")[m]
                mean, sd = g.mean().reindex(list(C.GROUP_LABELS.values())), g.std().reindex(list(C.GROUP_LABELS.values()))
                ax.bar(np.arange(3) + j * 0.27, mean, 0.27, yerr=sd, label=model, color=COLORS[model], capsize=2)
            ax.set_xticks(np.arange(3) + 0.27)
            ax.set_xticklabels(list(C.GROUP_LABELS.values()))
            ax.set_xlabel("transaction connectivity (in-degree group)")
            ax.set_title(label)
        under = sorted(d[d.underpowered].group.unique())
        axes[0].legend()
        fig.suptitle(f"Subgroups ({fs}); underpowered groups (<{C.MIN_ILLICIT} illicit): {under or 'none'}")
        fig.tight_layout()
        fig.savefig(C.PLOTS_DIR / f"subgroups_{fs}.png", dpi=150)
        plt.close(fig)


def shift_plots():
    df = pd.read_csv(C.RAW_DIR / "shift_metrics.csv")
    for st, xlabel in (("activity", "share of in-degree-0 transactions"), ("prevalence", "illicit share (x original)")):
        d = df[df.shift_type == st]
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        for ax, (m, label) in zip(axes, METRICS + [("fnr", "False-negative rate")]):
            for model in C.MODELS:
                g = d[d.model == model].groupby("level_num")[m]
                mean, sd = g.mean(), g.std()
                ax.errorbar(mean.index, mean.values, yerr=sd.values, marker="o", label=model,
                            color=COLORS[model], capsize=2)
            ax.set_xlabel(xlabel)
            ax.set_title(label)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(C.PLOTS_DIR / f"shift_{st}.png", dpi=150)
        plt.close(fig)

    d = df[df.shift_type == "temporal"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    order = list(C.TEMPORAL_WINDOWS)
    for ax, (m, label) in zip(axes, METRICS + [("fnr", "False-negative rate")]):
        for j, model in enumerate(C.MODELS):
            g = d[d.model == model].groupby("level")[m]
            mean, sd = g.mean().reindex(order), g.std().reindex(order)
            ax.bar(np.arange(len(order)) + j * 0.27, mean, 0.27, yerr=sd, label=model, color=COLORS[model], capsize=2)
        ax.set_xticks(np.arange(len(order)) + 0.27)
        ax.set_xticklabels(order)
        ax.set_xlabel("test time steps")
        ax.set_title(label)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "shift_temporal.png", dpi=150)
    plt.close(fig)


def inductive_plot():
    path = C.RAW_DIR / "inductive_metrics.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (m, label) in zip(axes, METRICS):
        g = d.groupby("model")[m]
        mean, sd = g.mean().reindex(C.MODELS), g.std().reindex(C.MODELS)
        ax.bar(C.MODELS, mean, yerr=sd, color=[COLORS[k] for k in C.MODELS], capsize=3)
        ax.set_title(f"Inductive: {label}")
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "inductive.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    subgroup_plot()
    shift_plots()
    inductive_plot()
    print("plots saved to", C.PLOTS_DIR)
