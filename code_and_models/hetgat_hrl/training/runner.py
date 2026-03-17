from __future__ import annotations

from typing import Dict, List

from hetgat_hrl.eval.metrics import EpisodeMetrics


class EpisodeRunner:
    def __init__(self, env, low_policy, high_planner):
        self.env = env
        self.low_policy = low_policy
        self.high_planner = high_planner
        self.last_episode_debug: Dict[str, float] = {}

    def run_episode(self, seed: int = 0) -> EpisodeMetrics:
        state = self.env.reset(seed=seed)
        total_reward = 0.0
        invalid_total = 0.0
        invalid_count_total = 0.0
        invalid_steps = 0
        crashed_uav: Dict[str, bool] = {}
        attention_entropy_total = 0.0
        attention_steps = 0
        follow_bind_step_total = 0.0
        follow_bind_steps = 0
        follow_bind_total_last = 0.0

        while not state.done:
            planner_goals = self.high_planner.plan(self.env)
            if hasattr(self.low_policy, "infer_attention_goals"):
                attn_goals, attn_summary = self.low_policy.infer_attention_goals(
                    self.env, fallback_goals=planner_goals
                )
                high_goals = attn_goals
                attention_entropy_total += float(
                    attn_summary.get("attention_entropy_mean", 0.0)
                )
                attention_steps += 1
            else:
                high_goals = planner_goals
            if hasattr(self.env, "set_recommended_goals"):
                self.env.set_recommended_goals(high_goals)
            actions = self.low_policy.act(self.env, high_goals=high_goals)
            out = self.env.step(actions)
            state = out.state
            total_reward += sum(out.rewards.values()) / max(len(out.rewards), 1)
            invalid_total += float(out.info.get("invalid_action_mean", 0.0))
            invalid_count_total += float(out.info.get("invalid_action_count", 0.0))
            invalid_steps += 1
            follow_bind_step_total += float(out.info.get("uav_follow_bind_count_step", 0.0))
            follow_bind_total_last = float(out.info.get("uav_follow_bind_count_total", follow_bind_total_last))
            follow_bind_steps += 1
            for aid, a in state.agents.items():
                if a.kind.name == "UAV" and a.crashed:
                    crashed_uav[aid] = True

        completion = out.info.get("task_completion_rate", 0.0)
        invalid_mean = invalid_total / max(invalid_steps, 1)
        invalid_count_mean = invalid_count_total / max(invalid_steps, 1)
        follow_bind_step_mean = follow_bind_step_total / max(follow_bind_steps, 1)
        attn_entropy_mean = attention_entropy_total / max(attention_steps, 1)
        self.last_episode_debug = {
            "attention_entropy_mean": float(attn_entropy_mean),
            "attention_steps": float(attention_steps),
            "invalid_action_count_mean": float(invalid_count_mean),
            "uav_follow_bind_count_step_mean": float(follow_bind_step_mean),
            "uav_follow_bind_count_total_last": float(follow_bind_total_last),
        }
        return EpisodeMetrics(
            episode_reward_mean=float(total_reward / max(state.step_index, 1)),
            task_completion_rate=float(completion),
            invalid_action_mean=float(invalid_mean),
            crashed_uav_count=len(crashed_uav),
            steps=int(state.step_index),
        )
