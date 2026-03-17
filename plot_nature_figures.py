from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_latest_manifest(project_root: Path) -> Path:
    results_root = project_root / "training_results"
    bundles = sorted(
        [p for p in results_root.glob("nature_bundle_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for b in bundles:
        m = b / "nature_run_manifest.json"
        if m.exists():
            return m
    raise FileNotFoundError("No nature_run_manifest.json found.")


def moving_avg(x: pd.Series, window: int = 10) -> pd.Series:
    return x.rolling(window=window, min_periods=1).mean()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def beautify_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8, alpha=0.45)


def draw_sig(ax: plt.Axes, x1: float, x2: float, y: float, text: str) -> None:
    if text.strip() == "":
        return
    h = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#333333", linewidth=1.1)
    ax.text((x1 + x2) / 2.0, y + h * 1.12, text, ha="center", va="bottom", fontsize=12)


def panel_tag(ax: plt.Axes, tag: str) -> None:
    ax.text(
        -0.12,
        1.03,
        tag,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def load_metrics(manifest: Dict) -> Dict[str, pd.DataFrame]:
    out = {}
    for rn, info in manifest["runs"].items():
        out[rn] = pd.read_csv(info["metrics_csv"]).sort_values("iteration").reset_index(drop=True)
    return out


def summarize_last20(df: pd.DataFrame, col: str) -> tuple[float, float]:
    tail = df[df["iteration"] >= 180][col]
    if tail.empty:
        tail = df[col].tail(20)
    arr = tail.astype(float).to_numpy()
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def plot_figure2(bundle_dir: Path, run0: pd.DataFrame, run3: pd.DataFrame) -> None:
    set_style()
    c0 = "#1f4e79"
    c3 = "#b13a3a"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Panel A
    ax = axes[0]
    for df, c, lab in [(run0, c0, "RUN0 Perfect"), (run3, c3, "RUN3 No Curriculum")]:
        ax.plot(df["iteration"], df["task_completion_rate"], color=c, alpha=0.22, linewidth=1.0)
        ax.plot(df["iteration"], moving_avg(df["task_completion_rate"], 10), color=c, linewidth=2.6, label=lab)
    ax.axvline(50, color="#6e6e6e", linestyle="--", linewidth=1.3)
    ax.annotate(
        "Curriculum Switch",
        xy=(50, 0.85),
        xytext=(68, 0.92),
        fontsize=10.5,
        arrowprops=dict(arrowstyle="->", lw=1.0, color="#555555"),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Task Completion Rate")
    ax.set_ylim(0.0, 1.0)
    beautify_ax(ax)
    panel_tag(ax, "A")
    ax.legend(frameon=False, loc="lower right")

    # Panel B
    ax = axes[1]
    for df, c, lab in [(run0, c0, "RUN0 Perfect"), (run3, c3, "RUN3 No Curriculum")]:
        ax.plot(df["iteration"], df["episode_reward_mean"], color=c, alpha=0.22, linewidth=1.0)
        ax.plot(df["iteration"], moving_avg(df["episode_reward_mean"], 10), color=c, linewidth=2.6, label=lab)
    ax.axvline(50, color="#6e6e6e", linestyle="--", linewidth=1.3)
    yhi = float(max(run0["episode_reward_mean"].max(), run3["episode_reward_mean"].max()))
    ylo = float(min(run0["episode_reward_mean"].min(), run3["episode_reward_mean"].min()))
    ax.annotate(
        "Curriculum Switch",
        xy=(50, yhi - 0.12 * (yhi - ylo + 1e-6)),
        xytext=(68, yhi - 0.03 * (yhi - ylo + 1e-6)),
        fontsize=10.5,
        arrowprops=dict(arrowstyle="->", lw=1.0, color="#555555"),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Episode Reward")
    beautify_ax(ax)
    panel_tag(ax, "B")
    ax.legend(frameon=False, loc="best")

    fig.savefig(bundle_dir / "Figure_2_Learning_Dynamics.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(bundle_dir / "Figure_2_Learning_Dynamics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure3(bundle_dir: Path, run_metrics: Dict[str, pd.DataFrame], stats_df: pd.DataFrame) -> None:
    set_style()
    order = [
        "RUN0_Perfect",
        "RUN1_Baseline_MAPPO",
        "RUN2_Ablation_No_RTH",
        "RUN3_Ablation_No_Curr",
    ]
    labels = ["Perfect", "Baseline", "No RTH", "No Curr."]
    colors = ["#1f4e79", "#2f7e79", "#b07d2c", "#b13a3a"]

    task_mean, task_std = [], []
    rew_mean, rew_std = [], []
    for r in order:
        m, s = summarize_last20(run_metrics[r], "task_completion_rate")
        task_mean.append(m)
        task_std.append(s)
        m, s = summarize_last20(run_metrics[r], "episode_reward_mean")
        rew_mean.append(m)
        rew_std.append(s)

    sig_task = {row["run"]: str(row.get("sig_task", "")) for _, row in stats_df.iterrows()}
    sig_rew = {row["run"]: str(row.get("sig_reward", "")) for _, row in stats_df.iterrows()}

    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Panel A
    ax = axes[0]
    ax.bar(x, task_mean, yerr=task_std, capsize=4, color=colors, alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel("Task Completion Rate (Last 20 iters)")
    ax.set_ylim(0.0, 1.0)
    beautify_ax(ax)
    panel_tag(ax, "A")
    y_base = max(task_mean) + max(task_std) + 0.05
    for i, r in enumerate(order[1:], start=1):
        draw_sig(ax, 0, i, y_base + 0.05 * (i - 1), sig_task.get(r, ""))

    # Panel B
    ax = axes[1]
    ax.bar(x, rew_mean, yerr=rew_std, capsize=4, color=colors, alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel("Mean Episode Reward (Last 20 iters)")
    beautify_ax(ax)
    panel_tag(ax, "B")
    ymax = max(rew_mean) + max(rew_std) + 0.02
    for i, r in enumerate(order[1:], start=1):
        draw_sig(ax, 0, i, ymax + 0.02 * (i - 1), sig_rew.get(r, ""))

    fig.savefig(bundle_dir / "Figure_3_Ablation_Bars.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(bundle_dir / "Figure_3_Ablation_Bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure4(bundle_dir: Path, run_metrics: Dict[str, pd.DataFrame], l_scale_df: pd.DataFrame) -> None:
    set_style()
    # S-scale proxy: last20 means from each run metrics.
    s0_c, s0_r = summarize_last20(run_metrics["RUN0_Perfect"], "task_completion_rate")[0], summarize_last20(
        run_metrics["RUN0_Perfect"], "episode_reward_mean"
    )[0]
    s1_c, s1_r = summarize_last20(run_metrics["RUN1_Baseline_MAPPO"], "task_completion_rate")[0], summarize_last20(
        run_metrics["RUN1_Baseline_MAPPO"], "episode_reward_mean"
    )[0]

    l0 = l_scale_df[l_scale_df["run_label"] == "RUN0_Perfect"]
    l1 = l_scale_df[l_scale_df["run_label"] == "RUN1_Baseline_MAPPO"]
    if l0.empty or l1.empty:
        raise ValueError("L_scale_metrics.csv must contain RUN0_Perfect and RUN1_Baseline_MAPPO.")
    l0_c = float(l0["task_completion_rate"].iloc[0])
    l1_c = float(l1["task_completion_rate"].iloc[0])
    l0_r = float(l0["episode_reward_mean"].iloc[0])
    l1_r = float(l1["episode_reward_mean"].iloc[0])

    def degr(s: float, l: float) -> float:
        if abs(s) < 1e-8:
            return float("nan")
        return float((s - l) / abs(s))

    d0_c, d1_c = degr(s0_c, l0_c), degr(s1_c, l1_c)
    d0_r, d1_r = degr(s0_r, l0_r), degr(s1_r, l1_r)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    labels = ["RUN0", "RUN1"]
    x = np.arange(2)
    width = 0.34

    # Panel A: completion S vs L
    ax = axes[0]
    ax.bar(x - width / 2, [s0_c, s1_c], width=width, color="#1f4e79", label="S-Scale (5km)")
    ax.bar(x + width / 2, [l0_c, l1_c], width=width, color="#b13a3a", label="L-Scale (15km)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Task Completion Rate")
    ax.set_ylim(0.0, 1.0)
    beautify_ax(ax)
    panel_tag(ax, "A")
    ax.legend(frameon=False, loc="upper right")
    ax.text(x[0], 0.06, f"Degrade: {d0_c*100:.1f}%", ha="center", fontsize=10.5)
    ax.text(x[1], 0.06, f"Degrade: {d1_c*100:.1f}%", ha="center", fontsize=10.5)

    # Panel B: reward S vs L
    ax = axes[1]
    ax.bar(x - width / 2, [s0_r, s1_r], width=width, color="#1f4e79", label="S-Scale (5km)")
    ax.bar(x + width / 2, [l0_r, l1_r], width=width, color="#b13a3a", label="L-Scale (15km)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Episode Reward")
    beautify_ax(ax)
    panel_tag(ax, "B")
    low = min(s0_r, s1_r, l0_r, l1_r)
    ax.text(x[0], low * 0.9 if low < 0 else low + 0.01, f"Degrade: {d0_r*100:.1f}%", ha="center", fontsize=10.5)
    ax.text(x[1], low * 0.9 if low < 0 else low + 0.01, f"Degrade: {d1_r*100:.1f}%", ha="center", fontsize=10.5)

    fig.savefig(bundle_dir / "Figure_4_ZeroShot_Scalability.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(bundle_dir / "Figure_4_ZeroShot_Scalability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=r"E:\HetGAT\HetGAT-HRL")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--stats-csv", type=str, default="")
    parser.add_argument("--lscale-csv", type=str, default="")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    manifest_path = Path(args.manifest) if str(args.manifest).strip() else find_latest_manifest(project_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    bundle_dir = manifest_path.parent

    run_metrics = load_metrics(manifest)
    stats_csv = Path(args.stats_csv) if str(args.stats_csv).strip() else bundle_dir / "Table_2_Statistics.csv"
    if not stats_csv.exists():
        raise FileNotFoundError(f"Missing statistics table: {stats_csv}")
    stats_df = pd.read_csv(stats_csv)

    lscale_csv = Path(args.lscale_csv) if str(args.lscale_csv).strip() else Path(manifest["l_scale_metrics_csv"])
    if not lscale_csv.exists():
        raise FileNotFoundError(f"Missing L-scale metrics csv: {lscale_csv}")
    lscale_df = pd.read_csv(lscale_csv)

    plot_figure2(bundle_dir, run_metrics["RUN0_Perfect"], run_metrics["RUN3_Ablation_No_Curr"])
    plot_figure3(bundle_dir, run_metrics, stats_df)
    plot_figure4(bundle_dir, run_metrics, lscale_df)
    print(f"Saved figures into: {bundle_dir}")


if __name__ == "__main__":
    main()
