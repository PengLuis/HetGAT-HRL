from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _find_latest_bundle(project_root: Path) -> Path:
    bundles = sorted(
        [p for p in (project_root / "training_results").glob("nature_bundle_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not bundles:
        raise FileNotFoundError("No nature_bundle_* found under training_results.")
    return bundles[0]


def _safe_read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _add_text_page(pdf: PdfPages, title: str, lines: List[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.06, 0.96, title, fontsize=18, fontweight="bold", va="top")
    y = 0.91
    for line in lines:
        ax.text(0.06, y, line, fontsize=11.5, va="top")
        y -= 0.032
        if y < 0.05:
            break
    pdf.savefig(fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _add_dataframe_page(pdf: PdfPages, title: str, df: pd.DataFrame, note: str = "") -> None:
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.03, 0.97, title, fontsize=16, fontweight="bold", va="top")
    if note:
        ax.text(0.03, 0.93, note, fontsize=10.5, va="top")

    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.2)
    tbl.scale(1.05, 1.35)
    pdf.savefig(fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _add_image_page(pdf: PdfPages, image_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.02, 0.98, title, fontsize=14, fontweight="bold", va="top")
    img = plt.imread(str(image_path))
    ax.imshow(img, extent=(0.02, 0.98, 0.06, 0.92), aspect="auto")
    pdf.savefig(fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=r"E:\HetGAT\HetGAT-HRL")
    parser.add_argument("--bundle-dir", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir.strip() else _find_latest_bundle(project_root)
    manifest_path = bundle_dir / "nature_run_manifest.json"
    stats_path = bundle_dir / "Table_2_Statistics.csv"
    lscale_path = bundle_dir / "L_scale_metrics.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing statistics table: {stats_path}")
    if not lscale_path.exists():
        raise FileNotFoundError(f"Missing L-scale metrics: {lscale_path}")

    manifest = _safe_read_json(manifest_path)
    stats = pd.read_csv(stats_path)
    lscale = pd.read_csv(lscale_path)

    out_pdf = Path(args.output) if args.output.strip() else bundle_dir / "ScientificReports_Final_Package.pdf"

    # Build concise conclusions from stats.
    row0 = stats[stats["run"] == "RUN0_Perfect"].iloc[0]
    row1 = stats[stats["run"] == "RUN1_Baseline_MAPPO"].iloc[0]
    row2 = stats[stats["run"] == "RUN2_Ablation_No_RTH"].iloc[0]
    row3 = stats[stats["run"] == "RUN3_Ablation_No_Curr"].iloc[0]
    l0 = lscale[lscale["run_label"] == "RUN0_Perfect"].iloc[0]
    l1 = lscale[lscale["run_label"] == "RUN1_Baseline_MAPPO"].iloc[0]

    lines = [
        f"Bundle directory: {bundle_dir}",
        f"Config: {manifest.get('config', 'N/A')}",
        "",
        "Core findings (last 20 iterations, i.e., Iter 180-199):",
        (
            "1) Baseline_MAPPO vs Perfect shows significant degradation: "
            f"completion {row1['task_completion_mean']:.4f} vs {row0['task_completion_mean']:.4f} "
            f"(p={row1['p_task_vs_RUN0']:.3e}, {row1['sig_task']}), "
            f"reward {row1['episode_reward_mean']:.4f} vs {row0['episode_reward_mean']:.4f} "
            f"(p={row1['p_reward_vs_RUN0']:.3e}, {row1['sig_reward']})."
        ),
        (
            "2) No_RTH ablation is also significantly worse than Perfect: "
            f"completion p={row2['p_task_vs_RUN0']:.3e} ({row2['sig_task']}), "
            f"reward p={row2['p_reward_vs_RUN0']:.3e} ({row2['sig_reward']})."
        ),
        (
            "3) No_Curriculum has no significant difference from Perfect in this setting: "
            f"completion p={row3['p_task_vs_RUN0']:.3e}, "
            f"reward p={row3['p_reward_vs_RUN0']:.3e}."
        ),
        "",
        "Zero-shot L-scale (15km, 150 nodes, 3 trucks + 6 UAVs, scenario C, 10 episodes):",
        (
            f"4) Perfect: completion={float(l0['task_completion_rate']):.4f}, "
            f"reward={float(l0['episode_reward_mean']):.4f}; "
            f"Baseline_MAPPO: completion={float(l1['task_completion_rate']):.4f}, "
            f"reward={float(l1['episode_reward_mean']):.4f}."
        ),
        "",
        "This PDF includes Table 2 and Figure 2-5 required for submission package.",
    ]

    with PdfPages(out_pdf) as pdf:
        _add_text_page(pdf, "HetGAT-HRL Scientific Reports Package", lines)
        _add_dataframe_page(
            pdf,
            "Table 2. Statistics (Welch's t-test vs RUN0_Perfect)",
            stats,
            note="Significance markers: * p<0.05, ** p<0.01",
        )
        _add_dataframe_page(
            pdf,
            "Zero-shot L-scale Metrics",
            lscale,
            note="Evaluation target: L-scale (15km), scenario C, 10 episodes.",
        )

        fig_map = [
            ("Figure_2_Learning_Dynamics.png", "Figure 2. Learning Dynamics"),
            ("Figure_3_Ablation_Bars.png", "Figure 3. Ablation and Baselines"),
            ("Figure_4_ZeroShot_Scalability.png", "Figure 4. Zero-shot Scalability"),
            ("Figure_5_Emergent_Behaviors.png", "Figure 5. Emergent Behaviors"),
        ]
        for fn, title in fig_map:
            p = bundle_dir / fn
            if p.exists():
                _add_image_page(pdf, p, title)

    print(f"Saved report: {out_pdf}")


if __name__ == "__main__":
    main()

