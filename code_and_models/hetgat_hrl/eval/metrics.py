from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeMetrics:
    episode_reward_mean: float
    task_completion_rate: float
    invalid_action_mean: float
    crashed_uav_count: int
    steps: int

