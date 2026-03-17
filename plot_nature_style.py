from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:
    sns = None


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def validate_columns(df: pd.DataFrame) -> None:
    required = {"iteration", "task_completion_rate", "episode_reward_mean"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            f"metrics.csv 缺少必要列: {missing}. 需要包含 {sorted(required)}"
        )


def set_nature_style() -> None:
    if sns is not None:
        sns.set_theme(style="white", context="paper")
    else:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.titlesize": 13,
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


def add_curriculum_marker(ax: plt.Axes, switch_iter: int, text_y: float) -> None:
    ax.axvline(
        x=switch_iter,
        color="#6e6e6e",
        linestyle="--",
        linewidth=1.4,
        alpha=0.95,
        zorder=1,
    )
    ax.annotate(
        "Curriculum Switch:\nAutopilot OFF",
        xy=(switch_iter, text_y),
        xytext=(switch_iter + 16, text_y),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=10.5,
        color="#333333",
        arrowprops=dict(arrowstyle="->", lw=1.0, color="#555555", shrinkA=2, shrinkB=2),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RL learning curves in Scientific Reports/Nature-like style."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="metrics.csv",
        help="输入 CSV 路径（默认: 当前目录 metrics.csv）",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="滑动平均窗口大小（默认: 10）",
    )
    parser.add_argument(
        "--switch-iter",
        type=int,
        default=50,
        help="课程切换迭代点（默认: 50）",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="Figure_2_Learning_Curves",
        help="输出文件名前缀（默认: Figure_2_Learning_Curves）",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path.resolve()}")

    df = pd.read_csv(input_path)
    validate_columns(df)
    df = df.sort_values("iteration").reset_index(drop=True)

    x = df["iteration"]
    y_task = df["task_completion_rate"].astype(float)
    y_reward = df["episode_reward_mean"].astype(float)

    y_task_smooth = moving_average(y_task, args.window)
    y_reward_smooth = moving_average(y_reward, args.window)

    set_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    if sns is not None:
        palette = sns.color_palette("deep", 6)
        c_task = palette[0]  # deep blue
        c_reward = palette[3]  # red/orange
    else:
        c_task = "#1f4e79"
        c_reward = "#b13a3a"

    # Panel A: Task Completion Rate
    ax = axes[0]
    ax.plot(x, y_task, color=c_task, alpha=0.25, linewidth=1.1, label="Raw")
    ax.plot(
        x, y_task_smooth, color=c_task, alpha=1.0, linewidth=2.7, label=f"Moving Average (w={args.window})"
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Task Completion Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    add_curriculum_marker(ax, args.switch_iter, text_y=0.88)
    ax.legend(loc="lower right", frameon=False)

    # Panel B: Mean Episode Reward
    ax = axes[1]
    ax.plot(x, y_reward, color=c_reward, alpha=0.25, linewidth=1.1, label="Raw")
    ax.plot(
        x,
        y_reward_smooth,
        color=c_reward,
        alpha=1.0,
        linewidth=2.7,
        label=f"Moving Average (w={args.window})",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    y_min = float(np.min(y_reward))
    y_max = float(np.max(y_reward))
    if np.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    margin = 0.10 * (y_max - y_min)
    ax.set_ylim(y_min - margin, y_max + margin)
    add_curriculum_marker(ax, args.switch_iter, text_y=y_max - 0.05 * (y_max - y_min))
    ax.legend(loc="best", frameon=False)

    # Common aesthetics
    for i, ax in enumerate(axes):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.grid(axis="y", color="#e8e8e8", linewidth=0.8, alpha=0.5)
        panel_tag = "A" if i == 0 else "B"
        ax.text(
            -0.13,
            1.05,
            panel_tag,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    out_prefix = Path(args.output_prefix)
    pdf_path = out_prefix.with_suffix(".pdf")
    png_path = out_prefix.with_suffix(".png")

    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {pdf_path.resolve()}")
    print(f"Saved: {png_path.resolve()}")


if __name__ == "__main__":
    main()
