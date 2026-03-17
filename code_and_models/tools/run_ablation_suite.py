from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List

from hetgat_hrl.agents.actor_critic import AttentionGuidedLowLevelPolicy
from hetgat_hrl.core.mdp_spec import EnvConfig
from hetgat_hrl.envs.base_env import BaseHeteroDisasterEnv
from hetgat_hrl.hrl.planner import RiskTriggeredHRLPlanner
from hetgat_hrl.training.runner import EpisodeRunner


class NoHRLPlanner:
    def plan(self, env):
        return {aid: None for aid in env.state.agents}


def run_variant(cfg: EnvConfig, seed: int, episodes: int, planner_kind: str) -> Dict[str, float]:
    env = BaseHeteroDisasterEnv(cfg)
    low = AttentionGuidedLowLevelPolicy(obs_dim=20, hidden_dim=64, seed=seed)
    if planner_kind == "no_hrl":
        high = NoHRLPlanner()
    else:
        high = RiskTriggeredHRLPlanner(decision_interval=cfg.hrl_interval, seed=seed)
        if planner_kind == "no_risk_mask":
            high.risk_gat.beta = 0.0
    runner = EpisodeRunner(env=env, low_policy=low, high_planner=high)

    rs, cs, ivs = [], [], []
    for ep in range(episodes):
        m = runner.run_episode(seed=seed + ep)
        rs.append(float(m.episode_reward_mean))
        cs.append(float(m.task_completion_rate))
        ivs.append(float(m.invalid_action_mean))
    return {
        "episode_reward_mean": float(mean(rs)),
        "task_completion_rate": float(mean(cs)),
        "invalid_action_mean": float(mean(ivs)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=str, default="B")
    p.add_argument("--scale", type=str, default="M")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--episodes-per-seed", type=int, default=10)
    p.add_argument(
        "--results-root",
        type=str,
        default=r"E:\HetGAT\HetGAT-HRL\training_results",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = EnvConfig(scenario=args.scenario.upper(), phase=args.scale.upper())
    if args.scale.upper() == "S":
        base = replace(base, num_nodes=40, num_edges=64, num_trucks=1, num_uavs=1)
    elif args.scale.upper() == "M":
        base = replace(base, num_nodes=80, num_edges=140, num_trucks=2, num_uavs=3)
    else:
        base = replace(base, num_nodes=150, num_edges=260, num_trucks=3, num_uavs=6)

    variants = {
        "full_model": base,
        "no_risk_mask": base,
        "no_hrl": base,
        "no_weather": replace(base, stochastic_weather=False),
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.results_root) / f"ablation_{args.scale}_{args.scenario}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ablation_results.csv"

    rows: List[Dict[str, float]] = []
    for name, cfg in variants.items():
        for seed in range(int(args.seeds)):
            met = run_variant(cfg=replace(cfg, seed=seed), seed=seed, episodes=int(args.episodes_per_seed), planner_kind=name if name in ("no_risk_mask", "no_hrl") else "full")
            row = {"variant": name, "seed": seed, **met}
            rows.append(row)
            print(row)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "seed",
                "episode_reward_mean",
                "task_completion_rate",
                "invalid_action_mean",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("saved:", out_csv)


if __name__ == "__main__":
    main()

