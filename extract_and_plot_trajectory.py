from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from hetgat_hrl.agents.actor_critic import LearnableLowLevelPolicy
from hetgat_hrl.core.mdp_spec import AgentKind, EnvConfig, TaskKind
from hetgat_hrl.envs.base_env import BaseHeteroDisasterEnv
from hetgat_hrl.hrl.planner import RiskTriggeredHRLPlanner


def find_latest_manifest(project_root: Path) -> Path:
    results_root = project_root / "training_results"
    bundles = sorted(
        [p for p in results_root.glob("nature_bundle_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for b in bundles:
        m = b / "nature_run_manifest.json"
        if m.exists():
            return m
    raise FileNotFoundError("No nature_run_manifest.json found.")


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


def plot_gradient(ax: plt.Axes, xs: List[float], ys: List[float], cmap_name: str, lw: float = 1.8) -> None:
    if len(xs) < 2:
        return
    cmap = plt.get_cmap(cmap_name)
    n = len(xs) - 1
    for i in range(n):
        c = cmap(i / max(n - 1, 1))
        ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]], color=c, linewidth=lw, alpha=0.9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=r"E:\HetGAT\HetGAT-HRL")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--config", type=str, default=r"E:\HetGAT\HetGAT-HRL\code_and_models\configs\calibrated_short_v1.yaml")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--scenario", type=str, default="B")
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    manifest_path = Path(args.manifest) if str(args.manifest).strip() else find_latest_manifest(project_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    bundle_dir = Path(args.output_dir) if str(args.output_dir).strip() else manifest_path.parent

    if str(args.checkpoint).strip():
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = Path(manifest["runs"]["RUN0_Perfect"]["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    cfg_yaml = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = _flatten_cfg(cfg_yaml, seed=int(args.seed))
    cfg = EnvConfig(
        **{
            **base.__dict__,
            "phase": "S",
            "scenario": str(args.scenario).upper().strip(),
            "use_hetgat": True,
            "enable_rth_mask": True,
            "uav_auto_approach_enabled": False,
            "uav_monitor_snap_enabled": False,
            "enable_auto_approach": False,
            "enable_monitor_snap": False,
        }
    )
    env = BaseHeteroDisasterEnv(cfg)
    planner = RiskTriggeredHRLPlanner(decision_interval=cfg.hrl_interval, seed=int(args.seed))
    low = LearnableLowLevelPolicy(seed=int(args.seed), obs_dim=12, hidden_dim=128, device=str(args.device))

    payload = torch.load(ckpt_path, map_location="cpu")
    high_state = payload.get("allocator_state_dict", payload.get("high_state_dict", None))
    low_state = payload.get("low_level_state_dict", None)
    if isinstance(high_state, dict):
        planner.allocator.load_state_dict(high_state, strict=True)
    if isinstance(low_state, dict):
        low.model.load_state_dict(low_state, strict=True)

    state = env.reset(seed=int(args.seed))
    traj: Dict[str, List[Tuple[float, float]]] = {aid: [] for aid in state.agents}
    blocked_union = set()

    # Pick a representative isolated emergency task (farthest from depot node 0).
    depot = env.topology.nodes[0]
    iso_task = None
    max_d = -1.0
    for t in state.tasks.values():
        if t.kind != TaskKind.EMERGENCY:
            continue
        n = env.topology.nodes[int(t.demand_node)]
        d = float(np.hypot(n.x - depot.x, n.y - depot.y))
        if d > max_d:
            max_d = d
            iso_task = t

    while not state.done:
        for aid in state.agents:
            traj[aid].append(_agent_xy(env, aid))
        blocked_union.update(env.topology.blocked_edges)

        goals = select_goals_deterministic(planner, env)
        env.set_recommended_goals(goals)
        actions, _ = low.act(env, high_goals=goals, deterministic=True)
        out = env.step(actions)
        state = out.state

    # final append
    for aid in state.agents:
        traj[aid].append(_agent_xy(env, aid))
    blocked_union.update(env.topology.blocked_edges)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 7.0), constrained_layout=True)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Topology edges
    plotted = set()
    for src, nbs in env.topology.adjacency.items():
        for dst in nbs:
            k = (min(int(src), int(dst)), max(int(src), int(dst)))
            if k in plotted:
                continue
            plotted.add(k)
            a = env.topology.nodes[int(src)]
            b = env.topology.nodes[int(dst)]
            ax.plot([a.x, b.x], [a.y, b.y], color="#d0d0d0", linewidth=0.8, alpha=0.65, zorder=1)

    # Blocked edges markers
    for src, dst in blocked_union:
        a = env.topology.nodes[int(src)]
        b = env.topology.nodes[int(dst)]
        mx, my = (a.x + b.x) * 0.5, (a.y + b.y) * 0.5
        ax.scatter([mx], [my], marker="x", s=48, color="#c43333", linewidths=1.6, zorder=3)

    # Truck trajectories
    for aid, s in state.agents.items():
        if s.kind != AgentKind.TRUCK:
            continue
        pts = traj.get(aid, [])
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color="#1f4e79", linewidth=2.6, alpha=0.95, zorder=4, label=f"{aid} path")
        ax.scatter([xs[0]], [ys[0]], color="#1f4e79", s=36, zorder=5)
        ax.scatter([xs[-1]], [ys[-1]], color="#1f4e79", s=36, zorder=5)

    # UAV trajectories with temporal gradients
    for aid, s in state.agents.items():
        if s.kind != AgentKind.UAV:
            continue
        pts = traj.get(aid, [])
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plot_gradient(ax, xs, ys, cmap_name="YlOrRd", lw=1.5)
        ax.scatter([xs[0]], [ys[0]], color="#f0ad00", s=18, zorder=6)

    # Annotations
    ax.scatter([depot.x], [depot.y], color="#0d6efd", s=90, marker="s", zorder=7)
    ax.annotate(
        "Mobile Depot (UGV)",
        xy=(depot.x, depot.y),
        xytext=(depot.x + 220, depot.y + 220),
        fontsize=11,
        arrowprops=dict(arrowstyle="->", lw=1.0, color="#0d6efd"),
        color="#0d6efd",
    )
    if iso_task is not None:
        n = env.topology.nodes[int(iso_task.demand_node)]
        ax.scatter([n.x], [n.y], color="#b13a3a", s=70, marker="^", zorder=7)
        ax.annotate(
            "Isolated Task",
            xy=(n.x, n.y),
            xytext=(n.x + 180, n.y + 180),
            fontsize=11,
            arrowprops=dict(arrowstyle="->", lw=1.0, color="#b13a3a"),
            color="#b13a3a",
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Figure 5. Emergent Behaviors (UGV-UAV Spatiotemporal Trajectories)", fontsize=13)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(axis="both", color="#ececec", linewidth=0.6, alpha=0.5)

    out_pdf = bundle_dir / "Figure_5_Emergent_Behaviors.pdf"
    out_png = bundle_dir / "Figure_5_Emergent_Behaviors.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
