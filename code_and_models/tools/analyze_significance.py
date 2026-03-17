from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from hetgat_hrl.eval.stats_welch import mean_ci95, welch_t_test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="ablation_results.csv path")
    p.add_argument("--metric", type=str, default="task_completion_rate")
    p.add_argument("--baseline", type=str, default="full_model")
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.csv)
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    by_variant: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(float(row[args.metric]))

    baseline = by_variant.get(args.baseline, [])
    out_rows = []
    for var, vals in by_variant.items():
        ci = mean_ci95(vals)
        if var == args.baseline:
            tt = {"t": 0.0, "dof": 0.0, "pvalue_two_sided": 1.0}
        else:
            tt = welch_t_test(vals, baseline)
        out_rows.append(
            {
                "variant": var,
                "n": len(vals),
                "mean": ci["mean"],
                "ci95_low": ci["ci95_low"],
                "ci95_high": ci["ci95_high"],
                "t_vs_baseline": tt["t"],
                "dof_vs_baseline": tt["dof"],
                "pvalue_vs_baseline": tt["pvalue_two_sided"],
            }
        )

    out = Path(args.out) if args.out else path.parent / f"significance_{args.metric}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "n",
                "mean",
                "ci95_low",
                "ci95_high",
                "t_vs_baseline",
                "dof_vs_baseline",
                "pvalue_vs_baseline",
            ],
        )
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    print("saved:", out)


if __name__ == "__main__":
    main()

