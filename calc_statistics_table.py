from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


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
    raise FileNotFoundError("No nature_run_manifest.json found under training_results/nature_bundle_*")


def welch_t_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan")
    m1, m2 = float(np.mean(x)), float(np.mean(y))
    v1, v2 = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    den = math.sqrt(v1 / n1 + v2 / n2)
    if den <= 0:
        return float("nan")
    t = (m1 - m2) / den
    dof_num = (v1 / n1 + v2 / n2) ** 2
    dof_den = ((v1 / n1) ** 2) / max(n1 - 1, 1) + ((v2 / n2) ** 2) / max(n2 - 1, 1)
    dof = dof_num / max(dof_den, 1e-12)
    try:
        from scipy.stats import t as tdist  # type: ignore

        p = 2.0 * (1.0 - float(tdist.cdf(abs(t), dof)))
    except Exception:
        # Fallback normal approximation.
        p = math.erfc(abs(t) / math.sqrt(2.0))
    return float(p)


def star(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def summarize_last20(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, float, float, float]:
    if "iteration" in df.columns:
        tail = df[df["iteration"] >= 180][col].astype(float).to_numpy()
        if tail.size == 0:
            tail = df[col].astype(float).tail(20).to_numpy()
    else:
        tail = df[col].astype(float).tail(20).to_numpy()
    m = float(np.mean(tail))
    s = float(np.std(tail, ddof=1)) if tail.size > 1 else 0.0
    ci95 = float(1.96 * s / math.sqrt(max(tail.size, 1)))
    return tail, m, s, ci95


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=r"E:\HetGAT\HetGAT-HRL")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--output-csv", type=str, default="")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    manifest_path = Path(args.manifest) if str(args.manifest).strip() else find_latest_manifest(project_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))

    runs: Dict[str, Dict[str, str]] = manifest["runs"]
    run_names = [
        "RUN0_Perfect",
        "RUN1_Baseline_MAPPO",
        "RUN2_Ablation_No_RTH",
        "RUN3_Ablation_No_Curr",
    ]
    data = {}
    for rn in run_names:
        csv_path = Path(runs[rn]["metrics_csv"])
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing metrics csv for {rn}: {csv_path}")
        data[rn] = pd.read_csv(csv_path)

    out_rows = []
    run0 = data["RUN0_Perfect"]
    x_task, _, _, _ = summarize_last20(run0, "task_completion_rate")
    x_rew, _, _, _ = summarize_last20(run0, "episode_reward_mean")

    for rn in run_names:
        d = data[rn]
        task_raw, task_mean, task_std, task_ci95 = summarize_last20(d, "task_completion_rate")
        rew_raw, rew_mean, rew_std, rew_ci95 = summarize_last20(d, "episode_reward_mean")

        if rn == "RUN0_Perfect":
            p_task = float("nan")
            p_rew = float("nan")
        else:
            p_task = welch_t_pvalue(task_raw, x_task)
            p_rew = welch_t_pvalue(rew_raw, x_rew)

        out_rows.append(
            {
                "run": rn,
                "n_last20": int(len(task_raw)),
                "task_completion_mean": task_mean,
                "task_completion_std": task_std,
                "task_completion_ci95": task_ci95,
                "episode_reward_mean": rew_mean,
                "episode_reward_std": rew_std,
                "episode_reward_ci95": rew_ci95,
                "p_task_vs_RUN0": p_task,
                "sig_task": star(p_task),
                "p_reward_vs_RUN0": p_rew,
                "sig_reward": star(p_rew),
            }
        )

    out_df = pd.DataFrame(out_rows)
    if str(args.output_csv).strip():
        out_csv = Path(args.output_csv)
    else:
        out_csv = manifest_path.parent / "Table_2_Statistics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
