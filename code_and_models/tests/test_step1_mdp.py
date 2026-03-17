from hetgat_hrl.core.mdp_spec import EnvConfig
from hetgat_hrl.envs.base_env import BaseHeteroDisasterEnv


def test_step1_interfaces() -> None:
    env = BaseHeteroDisasterEnv(EnvConfig(num_trucks=2, num_uavs=3, max_steps=10))
    state = env.reset(seed=42)
    assert len(state.agents) == 5
    assert len(state.tasks) >= 1
    obs = env.observe()
    legal = env.legal_actions()
    assert set(obs.keys()) == set(state.agents.keys())
    assert set(legal.keys()) == set(state.agents.keys())

