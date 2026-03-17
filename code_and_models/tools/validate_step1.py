from hetgat_hrl.core.mdp_spec import EnvConfig, TruckAction, UAVAction
from hetgat_hrl.envs.base_env import BaseHeteroDisasterEnv


def main() -> None:
    env = BaseHeteroDisasterEnv(
        EnvConfig(
            phase="S",
            scenario="B",
            num_nodes=40,
            num_edges=64,
            num_trucks=2,
            num_uavs=3,
            max_steps=5,
        )
    )
    state = env.reset(seed=0)
    print("reset_step:", state.step_index)
    print("agents:", sorted(state.agents.keys()))
    print("tasks:", sorted(state.tasks.keys()))
    print("hrl_trigger_at_reset:", env.should_trigger_hrl())
    print("truck_0_neighbors:", env.legal_actions()["truck_0"]["neighbors"])
    print("obs_dim:", len(env.observe()["truck_0"]))
    tmat = env.observe_task_matrix()["truck_0"]
    print("task_matrix_shape:", (len(tmat), len(tmat[0]) if tmat else 0))

    for t in range(3):
        truck_neighbors = env.legal_actions()["truck_0"]["neighbors"]
        act = {
            "truck_0": TruckAction(
                target_node=(truck_neighbors[0] if truck_neighbors else None),
                stay=(len(truck_neighbors) == 0),
            ),
            "uav_0": UAVAction(vx=8.0, vy=0.0),
        }
        step_out = env.step(action=act)
        print(
            f"step={t+1} terminated={step_out.terminated} "
            f"hrl_trigger={step_out.info['hrl_trigger']} "
            f"blocked={step_out.info['blocked_ratio']:.3f} "
            f"rain={step_out.info['rainfall_mean']:.2f} "
            f"invalid={step_out.info['invalid_action_count']} "
            f"comm_blocked={step_out.info['comm_blocked_count']} "
            f"uav0_battery={step_out.state.agents['uav_0'].battery:.3f} "
            f"follow_bind_total={step_out.info['uav_follow_bind_count_total']} "
            f"charge_gain_total={step_out.info['uav_charge_energy_gain_total']:.3f} "
            f"done_rate={step_out.info['task_completion_rate']:.3f} "
            f"reward_keys={len(step_out.rewards)}"
        )


if __name__ == "__main__":
    main()
