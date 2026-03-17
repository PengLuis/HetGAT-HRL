from __future__ import annotations

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


def build_base_cfg() -> EnvConfig:
    # User-provided historical setting mapped into current framework.
    return EnvConfig(
        phase="S",
        scenario="B",
        seed=0,
        num_nodes=100,
        n_nodes=100,
        num_edges=160,
        num_trucks=3,
        n_trucks=3,
        num_uavs=4,
        n_uavs=4,
        num_normal_tasks=5,
        n_normal_tasks=5,
        num_emergency_tasks=3,
        n_emergency_tasks=3,
        max_steps=150,
        dt=1.0,
        dt_seconds=1.0,
        hrl_interval=10,
        risk_spike_threshold=0.30,
        # Physics / battery
        uav_battery_init=0.70,
        uav_max_speed_mps=17.0,
        uav_idle_discharge_per_step=0.020,
        uav_flight_discharge_per_step=0.0,
        uav_flight_discharge_per_m=0.0001,
        uav_charge_rate_per_step=0.085,
        # Reward / penalties
        reward_step_penalty=-0.05,
        reward_invalid_action=-0.10,  # invalid_action_penalty + kind_invalid_penalty
        reward_idle_with_task=-0.08,
        reward_delivery_normal=5.0,
        reward_delivery_emergency=5.0,
        reward_pickup=1.0,
        penalty_timeout_normal=-0.5,
        penalty_timeout_emergency=-1.3,
        uav_crash_penalty=-5.0,
        # PBRS
        use_pbrs=True,
        pbrs_scale=5.0,
        # Weather/comms
        stochastic_weather=True,
        comm_block_prob=0.10,
        # Task trigger radius
        uav_delivery_radius_m=50.0,
    )


def variant_cfgs(base: EnvConfig) -> Dict[str, EnvConfig]:
    return {
        "v0_legacy_baseline": base,
        # Reduce crash pressure slightly; keep direction similar.
        "v1_energy_relaxed": replace(
            base,
            uav_idle_discharge_per_step=0.015,
            uav_flight_discharge_per_m=0.00008,
            comm_block_prob=0.08,
        ),
        # Keep energy pressure, but reduce invalid-action punishment to stabilize early behavior.
        "v2_invalid_soft": replace(
            base,
            reward_invalid_action=-0.07,
            uav_delivery_radius_m=60.0,
        ),
        # Throughput-focused variant: easier connectivity and execution bandwidth.
        "v3_throughput_boost": replace(
            base,
            num_edges=220,
            truck_speed_mps=11.0,
            uav_delivery_radius_m=90.0,
            comm_block_prob=0.05,
            max_steps=180,
        ),
        # Conservative completion-oriented variant.
        "v4_completion_guarded": replace(
            base,
            num_edges=200,
            truck_speed_mps=10.0,
            uav_delivery_radius_m=80.0,
            comm_block_prob=0.06,
            max_steps=170,
            reward_idle_with_task=-0.06,
        ),
    }


def evaluate_variant(name: str, cfg: EnvConfig, episodes: int = 6) -> Dict[str, float]:
    env = BaseHeteroDisasterEnv(cfg)
    low = AttentionGuidedLowLevelPolicy(obs_dim=20, hidden_dim=64, seed=cfg.seed)
    high = RiskTriggeredHRLPlanner(decision_interval=cfg.hrl_interval)
    runner = EpisodeRunner(env=env, low_policy=low, high_planner=high)

    rewards: List[float] = []
    completions: List[float] = []
    invalids: List[float] = []
    crashes: List[int] = []
    steps: List[int] = []
    attn_entropy: List[float] = []

    for ep in range(episodes):
        m = runner.run_episode(seed=cfg.seed + ep)
        rewards.append(m.episode_reward_mean)
        completions.append(m.task_completion_rate)
        invalids.append(m.invalid_action_mean)
        crashes.append(m.crashed_uav_count)
        steps.append(m.steps)
        attn_entropy.append(
            float(runner.last_episode_debug.get("attention_entropy_mean", 0.0))
        )

    return {
        "variant": name,
        "episode_reward_mean_avg": float(mean(rewards)),
        "task_completion_rate_avg": float(mean(completions)),
        "invalid_action_mean_avg": float(mean(invalids)),
        "crashed_uav_count_avg": float(mean(crashes)),
        "steps_avg": float(mean(steps)),
        "attention_entropy_mean_avg": float(mean(attn_entropy)),
    }


def score_row(row: Dict[str, float]) -> float:
    # Higher completion, lower invalid/crash are preferred.
    return (
        4.0 * row["task_completion_rate_avg"]
        - 1.5 * row["invalid_action_mean_avg"]
        - 0.2 * row["crashed_uav_count_avg"]
        + 0.5 * row["episode_reward_mean_avg"]
    )


def main() -> None:
    base = build_base_cfg()
    variants = variant_cfgs(base)

    rows: List[Dict[str, float]] = []
    for name, cfg in variants.items():
        row = evaluate_variant(name, cfg, episodes=6)
        row["score"] = score_row(row)
        rows.append(row)
        print(row)

    rows.sort(key=lambda r: r["score"], reverse=True)
    best = rows[0]
    print("\nBest variant:", best["variant"], "score=", best["score"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(r"E:\HetGAT\HetGAT-HRL\training_results") / f"calib_short_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "calibration_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "episode_reward_mean_avg",
                "task_completion_rate_avg",
                "invalid_action_mean_avg",
                "crashed_uav_count_avg",
                "steps_avg",
                "attention_entropy_mean_avg",
                "score",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    best_path = out_dir / "best_variant.txt"
    best_path.write_text(
        f"best_variant={best['variant']}\nscore={best['score']}\n",
        encoding="utf-8",
    )
    print("saved:", csv_path)
    print("saved:", best_path)


if __name__ == "__main__":
    main()
