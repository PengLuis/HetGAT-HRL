from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from hetgat_hrl.agents.actor_critic import LearnableLowLevelPolicy
from hetgat_hrl.core.mdp_spec import AgentKind, EnvConfig
from hetgat_hrl.envs.base_env import BaseHeteroDisasterEnv
from hetgat_hrl.hrl.planner import RiskTriggeredHRLPlanner


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=r"E:\HetGAT\HetGAT-HRL\code_and_models\configs\calibrated_short_v1.yaml",
    )
    p.add_argument(
        "--checkpoints",
        type=str,
        default="",
        help="逗号分隔的 checkpoint 路径列表（推荐模式）",
    )
    p.add_argument(
        "--labels",
        type=str,
        default="",
        help="逗号分隔的标签，与 checkpoints 一一对应",
    )
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--scenario", type=str, default="C")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-root", type=str, default=r"E:\HetGAT\HetGAT-HRL\training_results")
    p.add_argument("--output-csv", type=str, default="")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--use-hetgat", type=_str2bool, default=True)
    p.add_argument("--enable-rth-mask", type=_str2bool, default=True)
    return p.parse_args()


def _flatten_cfg(cfg_yaml: Dict[str, Any], seed: int) -> EnvConfig:
    env = dict(cfg_yaml.get("env", {}))
    phy = dict(cfg_yaml.get("physics", {}))
    rew = dict(cfg_yaml.get("reward", {}))
    dst = dict(cfg_yaml.get("disturbance", {}))
    merged: Dict[str, Any] = {}
    merged.update(env)
    merged.update(phy)
    merged.update(rew)
    merged.update(dst)
    merged["seed"] = int(seed)
    return EnvConfig(**merged)


def _agent_xy(env: BaseHeteroDisasterEnv, aid: str) -> Tuple[float, float]:
    st = env.state.agents[aid]
    if st.pos_xy is not None:
        return float(st.pos_xy[0]), float(st.pos_xy[1])
    return env._node_xy(int(st.node or 0))


def _nearest_truck(env: BaseHeteroDisasterEnv, aid: str) -> Tuple[Optional[str], float]:
    ax, ay = _agent_xy(env, aid)
    best_id: Optional[str] = None
    best_d = float("inf")
    for tid, ts in env.state.agents.items():
        if ts.kind != AgentKind.TRUCK:
            continue
        tx, ty = _agent_xy(env, str(tid))
        d = float(((ax - tx) ** 2 + (ay - ty) ** 2) ** 0.5)
        if d < best_d:
            best_d = d
            best_id = str(tid)
    return best_id, best_d


def _required_rth_battery(env: BaseHeteroDisasterEnv, aid: str, dist_to_truck: float) -> float:
    st = env.state.agents[aid]
    base_discharge_per_m = float(max(getattr(env.cfg, "uav_flight_discharge_per_m", 1e-6), 1e-6))
    headwind_coeff = float(max(getattr(env.cfg, "uav_headwind_energy_coeff", 0.04), 0.0))
    rain_coeff = float(max(getattr(env.cfg, "uav_rain_energy_coeff", 0.02), 0.0))
    base_wind = float(max(getattr(env.cfg, "base_wind_mps", 0.0), 0.0))
    base_rain = float(max(getattr(env.cfg, "base_rainfall_mmh", 0.0), 0.0))
    cargo_unit_kg = float(max(getattr(env.cfg, "cargo_unit_kg", 40.0), 1e-6))
    m_load_kg = float(max(getattr(st, "cargo", 0.0), 0.0)) * cargo_unit_kg
    load_factor = 1.0 + 0.018 * m_load_kg
    weather_factor = 1.0 + headwind_coeff * base_wind + rain_coeff * base_rain
    safe_discharge_rate = base_discharge_per_m * weather_factor * load_factor
    return float(max(0.0, dist_to_truck) * safe_discharge_rate)


def select_goals_deterministic(
    planner: RiskTriggeredHRLPlanner, env: BaseHeteroDisasterEnv
) -> Dict[str, Optional[str]]:
    goals: Dict[str, Optional[str]] = {}
    used_tasks = set()
    use_hetgat = bool(getattr(env.cfg, "use_hetgat", True))
    enable_rth_mask = bool(getattr(env.cfg, "enable_rth_mask", True))

    ordered = sorted(
        env.state.agents.keys(),
        key=lambda aid: 0 if env.state.agents[aid].kind == AgentKind.UAV else 1,
    )
    for aid in ordered:
        st = env.state.agents[aid]
        if st.kind == AgentKind.UAV and bool(st.crashed):
            goals[aid] = None
            continue

        cands = planner._candidate_tasks(env, aid)
        tids: List[str] = []
        feats: List[List[float]] = []
        for t in cands:
            tid = str(t.task_id)
            if tid in used_tasks:
                continue
            dist_norm = float(env._agent_distance_to_task(aid, t) / 3000.0)
            emer = 1.0 if str(t.kind.value) == "emergency" else 0.0
            tids.append(tid)
            feats.append([dist_norm, emer])

        if st.kind == AgentKind.UAV:
            near_tid, near_dist = _nearest_truck(env, aid)
            if near_tid is not None and np.isfinite(near_dist):
                truck_feat = [float(near_dist / 3000.0), 0.0]
                tids = [near_tid] + tids
                feats = [truck_feat] + feats

                force_rth = bool(float(getattr(st, "cargo", 0.0)) <= 0.0)
                if not force_rth:
                    req_batt = _required_rth_battery(env, aid, float(near_dist))
                    if float(getattr(st, "battery", 0.0)) < float(req_batt * 1.2):
                        force_rth = True
                if enable_rth_mask and force_rth:
                    tids = [near_tid]
                    feats = [truck_feat]

        if not tids:
            goals[aid] = None
            continue

        if use_hetgat:
            x = torch.tensor(feats, dtype=torch.float32)
        else:
            arr = np.asarray(feats, dtype=np.float32)
            pooled = np.mean(arr, axis=0, keepdims=True)
            pooled = np.repeat(pooled, repeats=arr.shape[0], axis=0)
            x = torch.tensor(pooled, dtype=torch.float32)

        logits = planner.allocator(x).view(-1)
        a_idx = int(torch.argmax(logits).item())
        chosen_tid = tids[a_idx]
        goals[aid] = chosen_tid
        if chosen_tid in env.state.tasks:
            used_tasks.add(chosen_tid)
    return goals


def eval_checkpoint(
    checkpoint: Path,
    cfg_yaml: Dict[str, Any],
    label: str,
    scenario: str,
    episodes: int,
    seed: int,
    device: str,
    use_hetgat: bool,
    enable_rth_mask: bool,
) -> Dict[str, Any]:
    base = _flatten_cfg(cfg_yaml, seed=seed)
    cfg = EnvConfig(
        **{
            **base.__dict__,
            "phase": "L",
            "scenario": str(scenario).upper().strip(),
            "num_nodes": 150,
            "num_edges": 260,
            "num_trucks": 3,
            "num_uavs": 6,
            "map_size_m": 15000.0,
            "use_hetgat": bool(use_hetgat),
            "enable_rth_mask": bool(enable_rth_mask),
        }
    )
    env = BaseHeteroDisasterEnv(cfg)
    planner = RiskTriggeredHRLPlanner(decision_interval=cfg.hrl_interval, seed=seed)
    low = LearnableLowLevelPolicy(seed=seed, obs_dim=12, hidden_dim=128, device=device)

    payload = torch.load(checkpoint, map_location="cpu")
    high_state = payload.get("allocator_state_dict", payload.get("high_state_dict", None))
    low_state = payload.get("low_level_state_dict", None)
    if isinstance(high_state, dict):
        planner.allocator.load_state_dict(high_state, strict=True)
    if isinstance(low_state, dict):
        low.model.load_state_dict(low_state, strict=True)

    rs, cs, iv = [], [], []
    for ep in range(int(episodes)):
        state = env.reset(seed=seed * 1000 + ep)
        reward_sum = 0.0
        steps = 0
        invalid_sum = 0.0
        last_info: Dict[str, Any] = {}
        while not state.done:
            goals = select_goals_deterministic(planner, env)
            env.set_recommended_goals(goals)
            actions, _ = low.act(env, high_goals=goals, deterministic=True)
            out = env.step(actions)
            state = out.state
            last_info = out.info
            reward_sum += float(sum(out.rewards.values()) / max(len(out.rewards), 1))
            invalid_sum += float(out.info.get("invalid_action_count", 0.0))
            steps += 1
        rs.append(float(reward_sum / max(steps, 1)))
        cs.append(float(last_info.get("task_completion_rate", 0.0)))
        iv.append(float(invalid_sum / max(steps, 1)))

    return {
        "run_label": str(label),
        "checkpoint": str(checkpoint),
        "target_scale": "L",
        "target_scenario": str(scenario).upper().strip(),
        "episodes": int(episodes),
        "task_completion_rate": float(mean(cs)),
        "episode_reward_mean": float(mean(rs)),
        "invalid_action_mean": float(mean(iv)),
    }


def main() -> None:
    args = parse_args()
    cfg_yaml = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    checkpoints = [Path(s.strip()) for s in str(args.checkpoints).split(",") if s.strip()]
    labels = [s.strip() for s in str(args.labels).split(",") if s.strip()]
    if checkpoints and labels and len(labels) != len(checkpoints):
        raise ValueError("labels 数量必须与 checkpoints 数量一致")
    if checkpoints and not labels:
        labels = [f"RUN_{i}" for i in range(len(checkpoints))]

    rows: List[Dict[str, Any]] = []
    for i, ckpt in enumerate(checkpoints):
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        label = labels[i] if i < len(labels) else f"RUN_{i}"
        row = eval_checkpoint(
            checkpoint=ckpt,
            cfg_yaml=cfg_yaml,
            label=label,
            scenario=str(args.scenario).upper().strip(),
            episodes=int(args.episodes),
            seed=int(args.seed),
            device=str(args.device),
            use_hetgat=bool(args.use_hetgat),
            enable_rth_mask=bool(args.enable_rth_mask),
        )
        rows.append(row)
        print(row)

    if str(args.output_csv).strip():
        out_csv = Path(str(args.output_csv).strip())
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.results_root) / f"zero_shot_L_ckpt_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "L_scale_metrics.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run_label",
                "checkpoint",
                "target_scale",
                "target_scenario",
                "episodes",
                "task_completion_rate",
                "episode_reward_mean",
                "invalid_action_mean",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("saved:", out_csv)


if __name__ == "__main__":
    main()

