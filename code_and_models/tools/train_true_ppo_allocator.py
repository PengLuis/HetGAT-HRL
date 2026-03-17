from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from hetgat_hrl.agents.actor_critic import LearnableLowLevelPolicy
from hetgat_hrl.core.mdp_spec import AgentKind, EnvConfig, UAVAction
from hetgat_hrl.envs.base_env import BaseHeteroDisasterEnv
from hetgat_hrl.hrl.planner import RiskTriggeredHRLPlanner

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


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
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--episodes-per-iter", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--low-lr", type=float, default=3e-4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--low-update-epochs", type=int, default=4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--low-entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--low-hidden-dim", type=int, default=128)
    p.add_argument("--curriculum-warmup-iters", type=int, default=50)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--use-hetgat", type=_str2bool, default=True)
    p.add_argument("--enable-rth-mask", type=_str2bool, default=True)
    p.add_argument("--decision-interval", type=int, default=1)
    p.add_argument("--run-name", type=str, default="ppo_allocator_sb")
    p.add_argument(
        "--results-root",
        type=str,
        default=r"E:\HetGAT\HetGAT-HRL\training_results",
    )
    p.add_argument("--load-allocator", type=str, default="")
    p.add_argument("--save-allocator", type=str, default="")
    p.add_argument("--no-tensorboard", action="store_true")
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


def _ordered_agents(env: BaseHeteroDisasterEnv) -> List[str]:
    return sorted(
        env.state.agents.keys(),
        key=lambda aid: 0 if env.state.agents[aid].kind == AgentKind.UAV else 1,
    )


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


def sample_goals_with_logprob(
    planner: RiskTriggeredHRLPlanner,
    env: BaseHeteroDisasterEnv,
) -> Tuple[Dict[str, Optional[str]], List[Dict[str, Any]]]:
    goals: Dict[str, Optional[str]] = {}
    records: List[Dict[str, Any]] = []
    used_tasks = set()
    use_hetgat = bool(getattr(env.cfg, "use_hetgat", True))
    enable_rth_mask = bool(getattr(env.cfg, "enable_rth_mask", True))

    for aid in _ordered_agents(env):
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

        # Add nearest-truck virtual candidate for UAV so charging behavior is trainable.
        if st.kind == AgentKind.UAV:
            near_tid, near_dist = _nearest_truck(env, aid)
            if near_tid is not None and np.isfinite(near_dist):
                truck_feat = [float(near_dist / 3000.0), 0.0]
                tids = [near_tid] + tids
                feats = [truck_feat] + feats

                # RTH feasibility guard: low cargo or low battery forces return-to-truck candidate.
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

        # Baseline_MAPPO ablation: disable graph-aware hetero discrimination by
        # replacing per-candidate features with pooled mean features.
        if use_hetgat:
            x = torch.tensor(feats, dtype=torch.float32)
        else:
            arr = np.asarray(feats, dtype=np.float32)
            pooled = np.mean(arr, axis=0, keepdims=True)
            pooled = np.repeat(pooled, repeats=arr.shape[0], axis=0)
            x = torch.tensor(pooled, dtype=torch.float32)
        logits = planner.allocator(x).view(-1)
        dist = Categorical(logits=logits)
        a_idx = int(dist.sample().item())
        chosen_tid = tids[a_idx]
        old_logp = float(dist.log_prob(torch.tensor(a_idx)).item())

        goals[aid] = chosen_tid
        used_tasks.add(chosen_tid)
        records.append(
            {
                "aid": aid,
                "features": feats,
                "candidate_ids": list(tids),
                "action_idx": a_idx,
                "old_logp": old_logp,
                "return": 0.0,
                "adv": 0.0,
            }
        )
    return goals, records


def _ppo_update_high(
    planner: RiskTriggeredHRLPlanner,
    records: List[Dict[str, Any]],
    optimizer: Adam,
    clip_eps: float,
    entropy_coef: float,
    update_epochs: int,
    max_grad_norm: float,
) -> Dict[str, float]:
    if not records:
        return {"loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

    returns = torch.tensor([float(r["return"]) for r in records], dtype=torch.float32)
    adv = returns - returns.mean()
    adv = adv / (adv.std(unbiased=False) + 1e-8)
    for i, a in enumerate(adv.tolist()):
        records[i]["adv"] = float(a)

    loss_hist: List[float] = []
    entropy_hist: List[float] = []
    kl_hist: List[float] = []
    idx_all = list(range(len(records)))
    for _ in range(update_epochs):
        perm = torch.randperm(len(idx_all)).tolist()
        for j in perm:
            rec = records[idx_all[j]]
            x = torch.tensor(rec["features"], dtype=torch.float32)
            logits = planner.allocator(x).view(-1)
            dist = Categorical(logits=logits)
            act = torch.tensor(int(rec["action_idx"]), dtype=torch.long)
            new_logp = dist.log_prob(act)
            old_logp = torch.tensor(float(rec["old_logp"]), dtype=torch.float32)
            adv_t = torch.tensor(float(rec["adv"]), dtype=torch.float32)

            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
            entropy = dist.entropy()
            loss = -torch.min(surr1, surr2) - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(planner.allocator.parameters(), max_grad_norm)
            optimizer.step()

            loss_hist.append(float(loss.item()))
            entropy_hist.append(float(entropy.item()))
            kl_hist.append(float((old_logp - new_logp).item()))

    return {
        "loss": float(sum(loss_hist) / max(len(loss_hist), 1)),
        "entropy": float(sum(entropy_hist) / max(len(entropy_hist), 1)),
        "approx_kl": float(sum(kl_hist) / max(len(kl_hist), 1)),
    }


def _ppo_update_low(
    low: LearnableLowLevelPolicy,
    records: List[Dict[str, Any]],
    optimizer: Adam,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    update_epochs: int,
    max_grad_norm: float,
    device: str,
) -> Dict[str, float]:
    if not records:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "num_samples": 0.0,
        }

    dev = torch.device(device)
    obs = torch.tensor([r["obs"] for r in records], dtype=torch.float32, device=dev)
    raw_action = torch.tensor([r["raw_action"] for r in records], dtype=torch.float32, device=dev)
    old_logp = torch.tensor([r["old_logp"] for r in records], dtype=torch.float32, device=dev)
    old_value = torch.tensor([r["old_value"] for r in records], dtype=torch.float32, device=dev)
    returns = torch.tensor([r["return"] for r in records], dtype=torch.float32, device=dev)
    adv = returns - old_value
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    loss_hist: List[float] = []
    pl_hist: List[float] = []
    vl_hist: List[float] = []
    ent_hist: List[float] = []
    kl_hist: List[float] = []

    low.train()
    for _ in range(update_epochs):
        new_logp, entropy, value = low.evaluate_actions(obs, raw_action)
        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        value_pred_clipped = old_value + torch.clamp(value - old_value, -clip_eps, clip_eps)
        value_loss_unclipped = (value - returns).pow(2)
        value_loss_clipped = (value_pred_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

        entropy_mean = entropy.mean()
        loss = policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy_mean

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(list(low.parameters()), max_grad_norm)
        optimizer.step()

        approx_kl = (old_logp - new_logp).mean()
        loss_hist.append(float(loss.item()))
        pl_hist.append(float(policy_loss.item()))
        vl_hist.append(float(value_loss.item()))
        ent_hist.append(float(entropy_mean.item()))
        kl_hist.append(float(approx_kl.item()))

    return {
        "loss": float(sum(loss_hist) / max(len(loss_hist), 1)),
        "policy_loss": float(sum(pl_hist) / max(len(pl_hist), 1)),
        "value_loss": float(sum(vl_hist) / max(len(vl_hist), 1)),
        "entropy": float(sum(ent_hist) / max(len(ent_hist), 1)),
        "approx_kl": float(sum(kl_hist) / max(len(kl_hist), 1)),
        "num_samples": float(obs.shape[0]),
    }


def main() -> None:
    args = parse_args()
    cfg_yaml = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg_base = _flatten_cfg(cfg_yaml, seed=args.seed)
    cfg = EnvConfig(
        **{
            **cfg_base.__dict__,
            "use_hetgat": bool(args.use_hetgat),
            "enable_rth_mask": bool(args.enable_rth_mask),
        }
    )

    env = BaseHeteroDisasterEnv(cfg)
    planner = RiskTriggeredHRLPlanner(
        decision_interval=max(int(args.decision_interval), 1),
        seed=args.seed,
    )
    low = LearnableLowLevelPolicy(
        seed=args.seed,
        obs_dim=12,
        hidden_dim=int(args.low_hidden_dim),
        device=str(args.device),
    )
    joint_optimizer = Adam(
        [
            {"params": list(planner.allocator.parameters()), "lr": float(args.lr)},
            {"params": list(low.parameters()), "lr": float(args.low_lr)},
        ]
    )

    if str(args.load_allocator).strip():
        ckpt = Path(str(args.load_allocator).strip())
        payload = torch.load(ckpt, map_location="cpu")
        state = payload.get("allocator_state_dict", payload.get("high_state_dict", payload))
        planner.allocator.load_state_dict(state, strict=True)
        low_state = payload.get("low_level_state_dict", None)
        if isinstance(low_state, dict):
            low.model.load_state_dict(low_state, strict=True)
        print(f"loaded allocator checkpoint: {ckpt}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.results_root) / f"{args.run_name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.json").write_text(
        json.dumps(
            {
                "script": "train_true_ppo_allocator.py",
                "args": vars(args),
                "config": cfg_yaml,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    writer = None
    if not args.no_tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    csv_path = out_dir / "metrics.csv"
    rows: List[Dict[str, float]] = []
    global_seed = int(args.seed)
    for it in range(int(args.iterations)):
        # Soft curriculum: warm-start with environment takeover, then remove it.
        warmup_on = bool(it < int(args.curriculum_warmup_iters))
        cfg_curr = dict(env.cfg.__dict__)
        cfg_curr["uav_auto_approach_enabled"] = bool(warmup_on)
        cfg_curr["uav_monitor_snap_enabled"] = bool(warmup_on)
        cfg_curr["enable_auto_approach"] = bool(warmup_on)
        cfg_curr["enable_monitor_snap"] = bool(warmup_on)
        env.cfg = EnvConfig(**cfg_curr)
        if hasattr(env, "hazards") and hasattr(env.hazards, "cfg"):
            env.hazards.cfg = env.cfg

        iter_records: List[Dict[str, Any]] = []
        iter_low_records: List[Dict[str, Any]] = []
        iter_rewards: List[float] = []
        iter_completion: List[float] = []
        iter_invalid_count_mean: List[float] = []
        iter_follow_bind_step_mean: List[float] = []
        iter_alive_uav_pick_truck_ratio: List[float] = []
        iter_follow_bind_success_rate: List[float] = []
        iter_steps: List[int] = []
        iter_alive_goal_total = 0
        iter_alive_pick_truck_total = 0
        iter_bind_attempt_total = 0
        iter_bind_success_total = 0

        for ep in range(int(args.episodes_per_iter)):
            seed_ep = global_seed + it * int(args.episodes_per_iter) + ep
            state = env.reset(seed=seed_ep)
            ep_records: List[Dict[str, Any]] = []
            ep_low_records: List[Dict[str, Any]] = []
            low_agent_indices: Dict[str, List[int]] = {}
            timeline: List[Tuple[float, List[int]]] = []

            reward_sum = 0.0
            invalid_sum = 0.0
            follow_sum = 0.0
            alive_goal_with_target = 0
            alive_pick_truck = 0
            bind_attempt_count = 0
            bind_success_count = 0
            steps = 0
            last_info: Dict[str, Any] = {}

            while not state.done:
                goals, step_records = sample_goals_with_logprob(planner, env)
                for uid, st in env.state.agents.items():
                    if st.kind != AgentKind.UAV or bool(st.crashed):
                        continue
                    gid = goals.get(uid, None)
                    if gid is None:
                        continue
                    alive_goal_with_target += 1
                    tgt = env.state.agents.get(str(gid), None)
                    if tgt is not None and tgt.kind == AgentKind.TRUCK:
                        alive_pick_truck += 1
                env.set_recommended_goals(goals)
                actions, low_step_records = low.act(
                    env, high_goals=goals, deterministic=False
                )
                bind_attempt_targets: Dict[str, str] = {}
                for uid, st in env.state.agents.items():
                    if st.kind != AgentKind.UAV or bool(st.crashed):
                        continue
                    act = actions.get(uid, None)
                    if isinstance(act, UAVAction) and act.bind_truck_id is not None:
                        bind_attempt_count += 1
                        bind_attempt_targets[str(uid)] = str(act.bind_truck_id)
                out = env.step(actions)
                state = out.state
                last_info = out.info

                # Attach per-agent rewards for low-level PPO rollouts.
                for rec in low_step_records:
                    aid_rec = str(rec["aid"])
                    rec["reward"] = float(out.rewards.get(aid_rec, 0.0))
                    rec["done"] = 1.0 if bool(state.done) else 0.0
                    idx_low = len(ep_low_records)
                    ep_low_records.append(rec)
                    low_agent_indices.setdefault(aid_rec, []).append(idx_low)

                for uid, tid in bind_attempt_targets.items():
                    ust = out.state.agents.get(uid, None)
                    if ust is None or bool(ust.crashed):
                        continue
                    if ust.follow_target is not None and str(ust.follow_target) == str(tid):
                        bind_success_count += 1

                r_step = float(sum(out.rewards.values()) / max(len(out.rewards), 1))
                reward_sum += r_step
                invalid_sum += float(out.info.get("invalid_action_count", 0.0))
                follow_sum += float(out.info.get("uav_follow_bind_count_step", 0.0))
                steps += 1

                idxs: List[int] = []
                for rec in step_records:
                    idxs.append(len(ep_records))
                    ep_records.append(rec)
                timeline.append((r_step, idxs))

            g = 0.0
            returns_t = [0.0 for _ in range(len(timeline))]
            for t in range(len(timeline) - 1, -1, -1):
                g = float(timeline[t][0]) + float(args.gamma) * g
                returns_t[t] = g
            for t, (_, idxs) in enumerate(timeline):
                for idx in idxs:
                    ep_records[idx]["return"] = float(returns_t[t])

            # Per-UAV discounted returns for low-level policy.
            for aid_uav, idxs in low_agent_indices.items():
                g_low = 0.0
                for idx in reversed(idxs):
                    r = float(ep_low_records[idx]["reward"])
                    d = float(ep_low_records[idx]["done"])
                    g_low = r + float(args.gamma) * g_low * (1.0 - d)
                    ep_low_records[idx]["return"] = float(g_low)

            iter_records.extend(ep_records)
            iter_low_records.extend(ep_low_records)
            iter_rewards.append(float(reward_sum / max(steps, 1)))
            iter_completion.append(float(last_info.get("task_completion_rate", 0.0)))
            iter_invalid_count_mean.append(float(invalid_sum / max(steps, 1)))
            iter_follow_bind_step_mean.append(float(follow_sum / max(steps, 1)))
            iter_alive_uav_pick_truck_ratio.append(
                float(alive_pick_truck / max(alive_goal_with_target, 1))
            )
            iter_follow_bind_success_rate.append(
                float(bind_success_count / max(bind_attempt_count, 1))
            )
            iter_steps.append(int(steps))
            iter_alive_goal_total += int(alive_goal_with_target)
            iter_alive_pick_truck_total += int(alive_pick_truck)
            iter_bind_attempt_total += int(bind_attempt_count)
            iter_bind_success_total += int(bind_success_count)

        upd_high = _ppo_update_high(
            planner=planner,
            records=iter_records,
            optimizer=joint_optimizer,
            clip_eps=float(args.clip_eps),
            entropy_coef=float(args.entropy_coef),
            update_epochs=int(args.update_epochs),
            max_grad_norm=float(args.max_grad_norm),
        )
        upd_low = _ppo_update_low(
            low=low,
            records=iter_low_records,
            optimizer=joint_optimizer,
            clip_eps=float(args.clip_eps),
            entropy_coef=float(args.low_entropy_coef),
            value_coef=float(args.value_coef),
            update_epochs=int(args.low_update_epochs),
            max_grad_norm=float(args.max_grad_norm),
            device=str(args.device),
        )

        row = {
            "iteration": float(it),
            "episode_reward_mean": float(sum(iter_rewards) / max(len(iter_rewards), 1)),
            "task_completion_rate": float(sum(iter_completion) / max(len(iter_completion), 1)),
            "invalid_action_count_mean": float(
                sum(iter_invalid_count_mean) / max(len(iter_invalid_count_mean), 1)
            ),
            "uav_follow_bind_count_step_mean": float(
                sum(iter_follow_bind_step_mean) / max(len(iter_follow_bind_step_mean), 1)
            ),
            "alive_uav_pick_truck_ratio": float(
                sum(iter_alive_uav_pick_truck_ratio) / max(len(iter_alive_uav_pick_truck_ratio), 1)
            ),
            "alive_uav_pick_truck_ratio_weighted": float(
                iter_alive_pick_truck_total / max(iter_alive_goal_total, 1)
            ),
            "follow_bind_success_rate": float(
                sum(iter_follow_bind_success_rate) / max(len(iter_follow_bind_success_rate), 1)
            ),
            "follow_bind_success_rate_weighted": float(
                iter_bind_success_total / max(iter_bind_attempt_total, 1)
            ),
            "alive_uav_goal_with_target_total": float(iter_alive_goal_total),
            "alive_uav_pick_truck_total": float(iter_alive_pick_truck_total),
            "follow_bind_attempt_total": float(iter_bind_attempt_total),
            "follow_bind_success_total": float(iter_bind_success_total),
            "steps_mean": float(sum(iter_steps) / max(len(iter_steps), 1)),
            "ppo_high_loss": float(upd_high["loss"]),
            "ppo_high_entropy": float(upd_high["entropy"]),
            "ppo_high_approx_kl": float(upd_high["approx_kl"]),
            "ppo_low_loss": float(upd_low["loss"]),
            "ppo_low_policy_loss": float(upd_low["policy_loss"]),
            "ppo_low_value_loss": float(upd_low["value_loss"]),
            "ppo_low_entropy": float(upd_low["entropy"]),
            "ppo_low_approx_kl": float(upd_low["approx_kl"]),
            "low_level_samples": float(upd_low["num_samples"]),
            "num_decisions": float(len(iter_records)),
            "num_low_level_decisions": float(len(iter_low_records)),
            "curriculum_autopilot_on": float(1.0 if warmup_on else 0.0),
            "use_hetgat": float(1.0 if bool(env.cfg.use_hetgat) else 0.0),
            "enable_rth_mask": float(1.0 if bool(env.cfg.enable_rth_mask) else 0.0),
        }
        rows.append(row)
        print(row)

        if writer is not None:
            step = int(it)
            writer.add_scalar("train/episode_reward_mean", row["episode_reward_mean"], step)
            writer.add_scalar("train/task_completion_rate", row["task_completion_rate"], step)
            writer.add_scalar("train/invalid_action_count_mean", row["invalid_action_count_mean"], step)
            writer.add_scalar("train/uav_follow_bind_count_step_mean", row["uav_follow_bind_count_step_mean"], step)
            writer.add_scalar("train/alive_uav_pick_truck_ratio", row["alive_uav_pick_truck_ratio"], step)
            writer.add_scalar(
                "train/alive_uav_pick_truck_ratio_weighted",
                row["alive_uav_pick_truck_ratio_weighted"],
                step,
            )
            writer.add_scalar("train/follow_bind_success_rate", row["follow_bind_success_rate"], step)
            writer.add_scalar(
                "train/follow_bind_success_rate_weighted",
                row["follow_bind_success_rate_weighted"],
                step,
            )
            writer.add_scalar("train/ppo_high_loss", row["ppo_high_loss"], step)
            writer.add_scalar("train/ppo_high_entropy", row["ppo_high_entropy"], step)
            writer.add_scalar("train/ppo_high_approx_kl", row["ppo_high_approx_kl"], step)
            writer.add_scalar("train/ppo_low_loss", row["ppo_low_loss"], step)
            writer.add_scalar("train/ppo_low_policy_loss", row["ppo_low_policy_loss"], step)
            writer.add_scalar("train/ppo_low_value_loss", row["ppo_low_value_loss"], step)
            writer.add_scalar("train/ppo_low_entropy", row["ppo_low_entropy"], step)
            writer.add_scalar("train/ppo_low_approx_kl", row["ppo_low_approx_kl"], step)
            writer.add_scalar("train/num_low_level_decisions", row["num_low_level_decisions"], step)
            writer.add_scalar("train/curriculum_autopilot_on", row["curriculum_autopilot_on"], step)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "episode_reward_mean",
                "task_completion_rate",
                "invalid_action_count_mean",
                "uav_follow_bind_count_step_mean",
                "alive_uav_pick_truck_ratio",
                "alive_uav_pick_truck_ratio_weighted",
                "follow_bind_success_rate",
                "follow_bind_success_rate_weighted",
                "alive_uav_goal_with_target_total",
                "alive_uav_pick_truck_total",
                "follow_bind_attempt_total",
                "follow_bind_success_total",
                "steps_mean",
                "ppo_high_loss",
                "ppo_high_entropy",
                "ppo_high_approx_kl",
                "ppo_low_loss",
                "ppo_low_policy_loss",
                "ppo_low_value_loss",
                "ppo_low_entropy",
                "ppo_low_approx_kl",
                "low_level_samples",
                "num_decisions",
                "num_low_level_decisions",
                "curriculum_autopilot_on",
                "use_hetgat",
                "enable_rth_mask",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if writer is not None:
        writer.flush()
        writer.close()

    save_path = str(args.save_allocator).strip()
    if not save_path:
        save_path = str(out_dir / "allocator_final.pt")
    save_obj = {
        "allocator_state_dict": planner.allocator.state_dict(),
        "low_level_state_dict": low.model.state_dict(),
        "high_state_dict": planner.allocator.state_dict(),
        "args": vars(args),
    }
    torch.save(save_obj, save_path)
    print(f"saved allocator checkpoint: {save_path}")

    print(f"saved metrics: {csv_path}")
    if SummaryWriter is not None and not args.no_tensorboard:
        print(f"saved tensorboard: {out_dir / 'tb'}")
    print(f"run dir: {out_dir}")


if __name__ == "__main__":
    main()
