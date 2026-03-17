from hetgat_hrl.agents.actor_critic import AttentionGuidedLowLevelPolicy
from hetgat_hrl.core.mdp_spec import EnvConfig
from hetgat_hrl.envs.base_env import BaseHeteroDisasterEnv
from hetgat_hrl.hrl.planner import RiskTriggeredHRLPlanner
from hetgat_hrl.training.runner import EpisodeRunner


def test_runner_smoke() -> None:
    env = BaseHeteroDisasterEnv(
        EnvConfig(num_trucks=2, num_uavs=3, max_steps=20, num_nodes=20, num_edges=30)
    )
    low = AttentionGuidedLowLevelPolicy(obs_dim=20, hidden_dim=64, seed=0)
    high = RiskTriggeredHRLPlanner(decision_interval=5)
    runner = EpisodeRunner(env=env, low_policy=low, high_planner=high)
    m = runner.run_episode(seed=0)
    assert m.steps > 0
    assert 0.0 <= m.task_completion_rate <= 1.0
