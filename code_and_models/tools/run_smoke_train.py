from __future__ import annotations

import argparse
import csv
from pathlib import Path

from hetgat_hrl.agents.actor_critic import AttentionGuidedLowLevelPolicy
from hetgat_hrl.core.mdp_spec import EnvConfig
from hetgat_hrl.envs.base_env import BaseHeteroDisasterEnv
from hetgat_hrl.hrl.planner import RiskTriggeredHRLPlanner
from hetgat_hrl.training.runner import EpisodeRunner

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", type=str, default="S")
    p.add_argument("--scenario", type=str, default="B")
    p.add_argument("--num-trucks", type=int, default=2)
    p.add_argument("--num-uavs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--run-name", type=str, default="smoke_rebuild")
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument(
        "--results-root",
        type=str,
        default=r"E:\HetGAT\HetGAT-HRL\training_results",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EnvConfig(
        phase=args.phase,
        scenario=args.scenario,
        seed=args.seed,
        num_trucks=args.num_trucks,
        num_uavs=args.num_uavs,
        max_steps=args.max_steps,
    )
    env = BaseHeteroDisasterEnv(cfg)
    low = AttentionGuidedLowLevelPolicy(obs_dim=20, hidden_dim=64, seed=args.seed)
    high = RiskTriggeredHRLPlanner(decision_interval=cfg.hrl_interval)
    runner = EpisodeRunner(env=env, low_policy=low, high_planner=high)

    out_dir = Path(args.results_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    tb_dir = out_dir / "tb"
    writer = None
    if not args.no_tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(tb_dir))

    rows = []
    for ep in range(args.episodes):
        m = runner.run_episode(seed=args.seed + ep)
        row = {
            "episode": ep,
            "episode_reward_mean": m.episode_reward_mean,
            "task_completion_rate": m.task_completion_rate,
            "invalid_action_mean": m.invalid_action_mean,
            "invalid_action_count_mean": runner.last_episode_debug.get(
                "invalid_action_count_mean", 0.0
            ),
            "uav_follow_bind_count_step_mean": runner.last_episode_debug.get(
                "uav_follow_bind_count_step_mean", 0.0
            ),
            "uav_follow_bind_count_total_last": runner.last_episode_debug.get(
                "uav_follow_bind_count_total_last", 0.0
            ),
            "crashed_uav_count": m.crashed_uav_count,
            "steps": m.steps,
            "attention_entropy_mean": runner.last_episode_debug.get(
                "attention_entropy_mean", 0.0
            ),
        }
        rows.append(row)
        if writer is not None:
            writer.add_scalar("train/episode_reward_mean", float(row["episode_reward_mean"]), ep)
            writer.add_scalar("train/task_completion_rate", float(row["task_completion_rate"]), ep)
            writer.add_scalar("train/invalid_action_count_mean", float(row["invalid_action_count_mean"]), ep)
            writer.add_scalar("train/uav_follow_bind_count_step_mean", float(row["uav_follow_bind_count_step_mean"]), ep)
            writer.add_scalar("train/uav_follow_bind_count_total_last", float(row["uav_follow_bind_count_total_last"]), ep)
        print(row)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "episode_reward_mean",
                "task_completion_rate",
                "invalid_action_mean",
                "invalid_action_count_mean",
                "uav_follow_bind_count_step_mean",
                "uav_follow_bind_count_total_last",
                "crashed_uav_count",
                "steps",
                "attention_entropy_mean",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if writer is not None:
        writer.flush()
        writer.close()
    print(f"saved: {csv_path}")
    if writer is not None:
        print(f"saved tensorboard: {tb_dir}")
    elif not args.no_tensorboard:
        print("tensorboard disabled: SummaryWriter unavailable in current env")


if __name__ == "__main__":
    main()
