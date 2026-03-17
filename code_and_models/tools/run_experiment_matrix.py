from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from hetgat_hrl.agents.actor_critic import AttentionGuidedLowLevelPolicy
from hetgat_hrl.core.mdp_spec import EnvConfig
from hetgat_hrl.envs.base_env import BaseHeteroDisasterEnv
from hetgat_hrl.hrl.planner import RiskTriggeredHRLPlanner
from hetgat_hrl.training.runner import EpisodeRunner


def scale_cfg(scale: str, base: EnvConfig) -> EnvConfig:
    s = scale.upper()
    if s == "S":
        return replace(base, phase="S", num_nodes=40, num_edges=64, num_trucks=1, num_uavs=1)
    if s == "M":
        return replace(base, phase="M", num_nodes=80, num_edges=140, num_trucks=2, num_uavs=3)
    if s == "L":
        return replace(base, phase="L", num_nodes=150, num_edges=260, num_trucks=3, num_uavs=6)
    raise ValueError(f"unknown scale={scale}")


def scenario_cfg(scenario: str, base: EnvConfig) -> EnvConfig:
    s = scenario.upper()
    if s == "A":
        return replace(base, scenario="A", stochastic_weather=False, comm_block_prob=0.0)
    if s == "B":
        return replace(base, scenario="B", stochastic_weather=True, comm_block_prob=0.08)
    if s == "C":
        return replace(base, scenario="C", stochastic_weather=True, comm_block_prob=0.18)
    raise ValueError(f"unknown scenario={scenario}")


def run_once(cfg: EnvConfig, episodes: int, seed: int) -> Dict[str, float]:
    env = BaseHeteroDisasterEnv(cfg)
    low = AttentionGuidedLowLevelPolicy(obs_dim=20, hidden_dim=64, seed=seed)
    high = RiskTriggeredHRLPlanner(decision_interval=cfg.hrl_interval, seed=seed)
    runner = EpisodeRunner(env=env, low_policy=low, high_planner=high)
    rs: List[float] = []
    cs: List[float] = []
    ivs: List[float] = []
    st: List[int] = []
    for ep in range(episodes):
        m = runner.run_episode(seed=seed + ep)
        rs.append(float(m.episode_reward_mean))
        cs.append(float(m.task_completion_rate))
        ivs.append(float(m.invalid_action_mean))
        st.append(int(m.steps))
    return {
        "episode_reward_mean": float(mean(rs)),
        "task_completion_rate": float(mean(cs)),
        "invalid_action_mean": float(mean(ivs)),
        "steps_mean": float(mean(st)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scales", type=str, default="S,M")
    p.add_argument("--scenarios", type=str, default="A,B,C")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--episodes-per-seed", type=int, default=10)
    p.add_argument(
        "--results-root",
        type=str,
        default=r"E:\HetGAT\HetGAT-HRL\training_results",
    )
    p.add_argument("--run-name", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scales = [x.strip() for x in args.scales.split(",") if x.strip()]
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    run_name = args.run_name or f"matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.results_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "matrix_results.csv"

    base = EnvConfig()
    rows: List[Dict[str, float]] = []
    for sc in scales:
        for sn in scenarios:
            for seed in range(int(args.seeds)):
                cfg = scenario_cfg(sn, scale_cfg(sc, replace(base, seed=seed)))
                met = run_once(cfg, episodes=int(args.episodes_per_seed), seed=seed)
                row = {
                    "scale": sc,
                    "scenario": sn,
                    "seed": int(seed),
                    "num_nodes": int(cfg.num_nodes),
                    "num_edges": int(cfg.num_edges),
                    "num_trucks": int(cfg.num_trucks),
                    "num_uavs": int(cfg.num_uavs),
                    **met,
                }
                rows.append(row)
                print(row)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "scale",
                "scenario",
                "seed",
                "num_nodes",
                "num_edges",
                "num_trucks",
                "num_uavs",
                "episode_reward_mean",
                "task_completion_rate",
                "invalid_action_mean",
                "steps_mean",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("saved:", out_csv)


if __name__ == "__main__":
    main()

