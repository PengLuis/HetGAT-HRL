from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hetgat_hrl.agents.actor_critic import AttentionGuidedLowLevelPolicy
from hetgat_hrl.core.mdp_spec import AgentKind, EnvConfig, UAVAction
from hetgat_hrl.envs.base_env import BaseHeteroDisasterEnv
from hetgat_hrl.hrl.planner import RiskTriggeredHRLPlanner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, default="sb_2p3_40n_iter30")
    p.add_argument("--dt-seconds", type=float, default=10.0)
    p.add_argument(
        "--results-root",
        type=str,
        default=r"E:\HetGAT\HetGAT-HRL\training_results",
    )
    p.add_argument(
        "--panel-template",
        type=str,
        default=r"E:\HetGAT\HetGAT-HRL\code_and_models\tools\templates\hetgat_command_center_panel.html",
    )
    return p.parse_args()


def _round(v: Optional[float], nd: int = 3) -> Optional[float]:
    if v is None:
        return None
    return round(float(v), nd)


def snapshot(
    env: BaseHeteroDisasterEnv,
    info: Optional[Dict[str, Any]],
    rewards: Dict[str, float],
) -> Dict[str, Any]:
    agents: Dict[str, Any] = {}
    for aid, a in env.state.agents.items():
        agents[aid] = {
            "kind": str(a.kind.value),
            "node": None if a.node is None else int(a.node),
            "pos_xy": None
            if a.pos_xy is None
            else [_round(a.pos_xy[0], 2), _round(a.pos_xy[1], 2)],
            "battery": _round(a.battery, 4),
            "crashed": bool(a.crashed),
            "follow_target": a.follow_target,
            "transit": None
            if a.transit is None
            else [int(a.transit[0]), int(a.transit[1]), _round(a.transit[2], 4)],
        }

    tasks: Dict[str, Any] = {}
    for tid, t in env.state.tasks.items():
        tasks[tid] = {
            "kind": str(t.kind.value),
            "demand_node": int(t.demand_node),
            "deadline_step": int(t.deadline_step),
            "status": str(t.status.value),
            "assigned_to": t.assigned_to,
            "in_service_by": t.in_service_by,
            "service_remaining": int(t.service_remaining),
            "delivered_by": t.delivered_by,
            "delivered_step": t.delivered_step,
        }

    blocked = [[int(a), int(b)] for (a, b) in sorted(env.topology.blocked_edges)]
    node_hazards: Dict[str, Any] = {}
    for nid, node in env.topology.nodes.items():
        hz = env.hazards.node_weather(int(nid))
        wx, wy = env.hazards.wind_vector_at((float(node.x), float(node.y)))
        node_hazards[str(int(nid))] = {
            "rain": _round(hz.rain, 4),
            "wind": _round(hz.wind, 4),
            "quake": _round(hz.quake, 4),
            "wx": _round(wx, 4),
            "wy": _round(wy, 4),
        }
    hazard_field: Dict[str, Any] = {}
    edge_pstep: Dict[str, Any] = {}
    for (ea, eb), ep in getattr(env.hazards, "last_edge_pstep", {}).items():
        ka, kb = min(int(ea), int(eb)), max(int(ea), int(eb))
        edge_pstep[f"{ka}_{kb}"] = _round(float(ep), 6)

    gfield = getattr(env.hazards, "field", None)
    if gfield is not None:
        sc = getattr(gfield, "storm_center", None)
        qe = getattr(gfield, "quake_epicenter", None)
        if sc is not None and len(sc) >= 2:
            hazard_field["storm_center"] = [_round(float(sc[0]), 3), _round(float(sc[1]), 3)]
        hazard_field["rmw"] = _round(getattr(gfield, "rmw", None), 3)
        if qe is not None and len(qe) >= 2:
            hazard_field["quake_epicenter"] = [_round(float(qe[0]), 3), _round(float(qe[1]), 3)]
        hazard_field["quake_depth"] = _round(getattr(gfield, "quake_depth", None), 3)
    return {
        "step": int(env.state.step_index),
        "info": info or {},
        "rewards": {k: _round(v, 4) for k, v in rewards.items()},
        "agents": agents,
        "tasks": tasks,
        "blocked_edges": blocked,
        "node_hazards": node_hazards,
        "hazard_field": hazard_field,
        "edge_pstep": edge_pstep,
        "cfg": {
            "uav_delivery_radius_m": float(getattr(env.cfg, "uav_delivery_radius_m", 40.0)),
            "uav_monitor_radius_m": float(getattr(env.cfg, "uav_monitor_radius_m", 340.0)),
            "uav_max_sortie_m": float(getattr(env.cfg, "uav_max_sortie_m", 3000.0)),
        },
        "recommended_goals": deepcopy(getattr(env, "_recommended_goals", {})),
    }


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _task_status_to_panel(
    status: str,
    in_service_by: Optional[str] = None,
    service_remaining: int = 0,
) -> str:
    st = str(status)
    if st in {"delivered", "done"}:
        return "done"
    if st == "failed":
        return "failed"
    if st == "claimed":
        if in_service_by is not None and int(service_remaining) > 0:
            return "serving"
        return "claimed"
    return "pending"


def _resolve_agent_xy_for_panel(
    a: Dict[str, Any],
    node_xy: Dict[int, Tuple[float, float]],
    aid: Optional[str] = None,
    transit_total_cache: Optional[Dict[str, Tuple[int, int, float]]] = None,
    dt_seconds: float = 1.0,
) -> Tuple[float, float, Optional[int], Optional[int], Optional[float]]:
    node = a.get("node")
    transit = a.get("transit")

    if isinstance(transit, list) and len(transit) >= 3:
        n0, n1, raw = int(transit[0]), int(transit[1]), float(transit[2])
        p0 = node_xy.get(n0)
        p1 = node_xy.get(n1)
        if p0 is not None and p1 is not None:
            progress = 0.0
            if transit_total_cache is not None and aid is not None:
                key = str(aid)
                prev = transit_total_cache.get(key)
                if prev is None or int(prev[0]) != n0 or int(prev[1]) != n1:
                    total = max(float(raw), float(dt_seconds), 1e-6)
                    transit_total_cache[key] = (n0, n1, float(total))
                else:
                    total = max(float(prev[2]), float(raw), 1e-6)
                    transit_total_cache[key] = (n0, n1, float(total))
                progress = 1.0 - float(raw / max(total, 1e-6))
            else:
                # Fallback when no cache is provided.
                progress = 0.0
            progress = _clamp(float(progress), 0.0, 1.0)
            x = p0[0] + (p1[0] - p0[0]) * progress
            y = p0[1] + (p1[1] - p0[1]) * progress
            return x, y, n0, n1, progress
    else:
        if transit_total_cache is not None and aid is not None:
            transit_total_cache.pop(str(aid), None)

    pos = a.get("pos_xy")
    if isinstance(pos, list) and len(pos) >= 2:
        return float(pos[0]), float(pos[1]), (int(node) if node is not None else None), None, None

    if node is not None and int(node) in node_xy:
        x, y = node_xy[int(node)]
        return x, y, int(node), None, None

    return 0.0, 0.0, None, None, None


def adapt_trace_to_command_center_data(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = raw_payload.get("nodes", [])
    edge_items: List[Any] = raw_payload.get("edges", [])
    raw_frames: List[Dict[str, Any]] = raw_payload.get("frames", [])

    node_xy: Dict[int, Tuple[float, float]] = {
        int(n["id"]): (float(n["x"]), float(n["y"])) for n in nodes
    }

    base_edges = []
    for item in edge_items:
        if isinstance(item, dict):
            aa, bb = int(item.get("a", 0)), int(item.get("b", 0))
            pa = node_xy.get(aa, (0.0, 0.0))
            pb = node_xy.get(bb, (0.0, 0.0))
            length_m = float(item.get("length_m", (((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5)))
            base_edges.append({
                "id": str(item.get("id", f"e_{min(aa, bb)}_{max(aa, bb)}")),
                "a": aa,
                "b": bb,
                "length_m": length_m,
                "roughness_norm": float(item.get("roughness_norm", 0.0)),
                "building_density_norm": float(item.get("building_density_norm", 0.0)),
                "infra_bottleneck_norm": float(item.get("infra_bottleneck_norm", 0.0)),
                "base_vulnerability": float(item.get("base_vulnerability", 0.0)),
            })
        else:
            a, b = item
            aa, bb = int(a), int(b)
            pa = node_xy.get(aa, (0.0, 0.0))
            pb = node_xy.get(bb, (0.0, 0.0))
            length_m = float(((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5)
            base_edges.append({"id": f"e_{min(aa, bb)}_{max(aa, bb)}", "a": aa, "b": bb, "length_m": length_m,
                               "roughness_norm": 0.0, "building_density_norm": 0.0,
                               "infra_bottleneck_norm": 0.0, "base_vulnerability": 0.0})

    panel_frames: List[Dict[str, Any]] = []
    prev_agent_xy: Dict[str, Tuple[float, float]] = {}
    transit_total_cache: Dict[str, Tuple[int, int, float]] = {}
    prev_task_state: Dict[str, Dict[str, Any]] = {}
    prev_agent_state: Dict[str, Dict[str, Any]] = {}

    for rf in raw_frames:
        info = rf.get("info", {}) or {}
        dt = float(info.get("dt_seconds", 10.0) or 10.0)
        node_hazards = rf.get("node_hazards", {}) or {}
        hazard_field = rf.get("hazard_field", {}) or {}

        frame_nodes: List[Dict[str, Any]] = []
        for n in nodes:
            nid = str(int(n["id"]))
            hz = node_hazards.get(nid, {}) or {}
            frame_nodes.append({
                "id": int(n["id"]),
                "x": float(n["x"]),
                "y": float(n["y"]),
                "rain": float(hz.get("rain", 0.0)),
                "wind": float(hz.get("wind", 0.0)),
                "quake": float(hz.get("quake", 0.0)),
                "wx": float(hz.get("wx", 0.0)),
                "wy": float(hz.get("wy", 0.0)),
            })

        blocked_set = {
            (min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1])))
            for e in (rf.get("blocked_edges") or [])
            if isinstance(e, list) and len(e) >= 2
        }

        edge_pstep_map = rf.get("edge_pstep", {}) or {}
        edges = []
        for e in base_edges:
            key = (min(e["a"], e["b"]), max(e["a"], e["b"]))
            pkey = f"{key[0]}_{key[1]}"
            edges.append({
                "id": e["id"],
                "a": e["a"],
                "b": e["b"],
                "length_m": float(e.get("length_m", 0.0)),
                "blocked": key in blocked_set,
                "step_block_prob": float(edge_pstep_map.get(pkey, 0.0) or 0.0),
                "roughness_norm": float(e.get("roughness_norm", 0.0)),
                "building_density_norm": float(e.get("building_density_norm", 0.0)),
                "infra_bottleneck_norm": float(e.get("infra_bottleneck_norm", 0.0)),
                "base_vulnerability": float(e.get("base_vulnerability", 0.0)),
            })

        tasks_list: List[Dict[str, Any]] = []
        task_map = rf.get("tasks", {}) or {}
        rec_goals = rf.get("recommended_goals", {}) or {}
        for tid, t in task_map.items():
            dn = int(t.get("demand_node", 0))
            nx, ny = node_xy.get(dn, (0.0, 0.0))
            assigned = t.get("assigned_to")
            in_service_by = t.get("in_service_by")
            service_remaining = int(t.get("service_remaining", 0) or 0)
            delivered_by = t.get("delivered_by")
            recommended_for = None
            for aid, rec_tid in rec_goals.items():
                if rec_tid is not None and str(rec_tid) == str(tid):
                    recommended_for = str(aid)
                    break
            panel_status = _task_status_to_panel(
                str(t.get("status", "pending")),
                in_service_by=in_service_by,
                service_remaining=service_remaining,
            )
            tasks_list.append({
                "id": str(tid),
                "type": str(t.get("kind", "normal")),
                "deadline": int(t.get("deadline_step", 0)),
                "status": panel_status,
                "rawStatus": str(t.get("status", "pending")),
                "x": float(nx),
                "y": float(ny),
                "demandNode": int(dn),
                "assignedTo": assigned,
                "inServiceBy": in_service_by,
                "serviceRemaining": service_remaining,
                "deliveredBy": delivered_by,
                "recommendedFor": recommended_for,
                "_ph": (abs(hash(str(tid))) % 314) / 100.0,
            })
        task_by_id: Dict[str, Dict[str, Any]] = {str(t["id"]): t for t in tasks_list}
        active_task_by_agent: Dict[str, Dict[str, Any]] = {}
        for t in tasks_list:
            if str(t.get("status", "")) not in {"pending", "claimed", "serving"}:
                continue
            for k in ("inServiceBy", "assignedTo", "recommendedFor"):
                aa = t.get(k)
                if aa is not None and str(aa) not in active_task_by_agent:
                    active_task_by_agent[str(aa)] = t

        key_events: List[str] = []
        tasks_sorted = sorted(tasks_list, key=lambda x: str(x.get("id", "")))
        for tt in tasks_sorted:
            tid = str(tt.get("id"))
            cur_status = str(tt.get("status", "pending"))
            cur_remain = int(tt.get("serviceRemaining", 0) or 0)
            actor = tt.get("inServiceBy") or tt.get("assignedTo") or "unknown_agent"
            prev = prev_task_state.get(tid)

            if prev is None:
                if cur_status == "serving":
                    key_events.append(f"{actor} 开始交付任务 {tid}（剩余{cur_remain}回合）")
            else:
                prev_status = str(prev.get("status", "pending"))
                prev_remain = int(prev.get("serviceRemaining", 0) or 0)
                if prev_status != cur_status:
                    if cur_status == "serving":
                        key_events.append(f"{actor} 开始交付任务 {tid}（剩余{cur_remain}回合）")
                    elif cur_status == "done":
                        finisher = tt.get("deliveredBy") or actor
                        key_events.append(f"{finisher} 完成交付任务 {tid}")
                    elif cur_status == "failed":
                        key_events.append(f"任务 {tid} 超时失败")
                elif cur_status == "serving" and cur_remain != prev_remain:
                    key_events.append(f"任务 {tid} 交付进行中（剩余{cur_remain}回合）")

            prev_task_state[tid] = {
                "status": cur_status,
                "serviceRemaining": cur_remain,
            }

        if not key_events and int(info.get("uav_follow_bind_count_step", 0) or 0) > 0:
            key_events.append("无人机执行了搭载/充电绑定")

        agents_list: List[Dict[str, Any]] = []
        agent_map = rf.get("agents", {}) or {}
        n_agents = max(1, len(agent_map))
        uav_delivery_radius = float((rf.get("cfg", {}) or {}).get("uav_delivery_radius_m", info.get("uav_delivery_radius_m", 40.0)) or 40.0)
        for aid, a in agent_map.items():
            kind = str(a.get("kind", "uav"))
            x, y, cur_node, target_node, progress = _resolve_agent_xy_for_panel(
                a,
                node_xy,
                aid=str(aid),
                transit_total_cache=transit_total_cache,
                dt_seconds=dt,
            )
            prev = prev_agent_xy.get(str(aid))
            if prev is None:
                vx, vy = 0.0, 0.0
            else:
                vx = (x - prev[0]) / max(dt, 1e-6)
                vy = (y - prev[1]) / max(dt, 1e-6)
            prev_agent_xy[str(aid)] = (x, y)

            mode = "moving" if target_node is not None else "idle"
            if kind == "uav":
                if bool(a.get("crashed", False)):
                    mode = "crashed"
                elif a.get("follow_target") is not None:
                    mode = f"piggyback_{a.get('follow_target')}"
                else:
                    mode = "mission"

            aid_str = str(aid)
            next_node: Optional[int] = target_node
            action_text = "待命"
            target_task_id: Optional[str] = None
            target_x: Optional[float] = None
            target_y: Optional[float] = None
            if kind == "truck":
                if target_node is not None and (cur_node is None or int(target_node) != int(cur_node)):
                    action_text = f"前往节点 {int(target_node)}"
                elif progress is not None and float(progress) < 1.0:
                    action_text = f"沿路段行驶 {int(round(float(progress) * 100.0))}%"
                else:
                    action_text = "驻留"
            else:
                rec_tid = rec_goals.get(aid_str)
                if rec_tid is not None and str(rec_tid) in task_by_id:
                    tt = task_by_id[str(rec_tid)]
                    target_task_id = str(tt["id"])
                    target_x = float(tt["x"])
                    target_y = float(tt["y"])
                elif aid_str in active_task_by_agent:
                    tt = active_task_by_agent[aid_str]
                    target_task_id = str(tt["id"])
                    target_x = float(tt["x"])
                    target_y = float(tt["y"])

                if bool(a.get("crashed", False)):
                    action_text = "坠毁"
                elif a.get("follow_target") is not None:
                    action_text = f"搭载 {a.get('follow_target')}"
                elif target_task_id is not None:
                    action_text = f"飞向任务 {target_task_id}"
                else:
                    action_text = "巡航/待命"

            cur_battery = float(a.get("battery", 1.0) if a.get("battery") is not None else 1.0)
            was = prev_agent_state.get(aid_str)
            if kind == "uav":
                if was is not None and (not bool(was.get("crashed", False))) and bool(a.get("crashed", False)):
                    key_events.append(f"{aid_str} 坠毁（电量耗尽）")
                if was is not None and float(was.get("battery", 0.0)) >= 0.50 and cur_battery <= 0.01:
                    key_events.append(f"{aid_str} 电量突降至0（请检查能耗参数）")
            prev_agent_state[aid_str] = {"crashed": bool(a.get("crashed", False)), "battery": cur_battery}

            agents_list.append({
                "id": aid_str,
                "type": "truck" if kind == "truck" else "uav",
                "x": float(x),
                "y": float(y),
                "transit": a.get("transit"),
                "node": cur_node,
                "targetNode": target_node,
                "nextNode": next_node,
                "progress": progress,
                "vx": float(vx),
                "vy": float(vy),
                "battery": cur_battery,
                "inBlackout": False,
                "mode": mode,
                "actionText": action_text,
                "targetTaskId": target_task_id,
                "targetX": target_x,
                "targetY": target_y,
                "deliveryRadius": uav_delivery_radius if kind == "uav" else None,
            })

        normal_total = sum(1 for t in tasks_list if t["type"] == "normal")
        emer_total = sum(1 for t in tasks_list if t["type"] == "emergency")
        normal_done = sum(1 for t in tasks_list if t["type"] == "normal" and t["status"] == "done")
        emer_done = sum(1 for t in tasks_list if t["type"] == "emergency" and t["status"] == "done")

        rewards = rf.get("rewards", {}) or {}
        avg_reward = float(sum(float(v) for v in rewards.values()) / max(1, len(rewards)))

        epi_node = int(info.get("epicenter_node", -1))
        quake_epi = hazard_field.get("quake_epicenter")
        if (
            (not isinstance(quake_epi, list) or len(quake_epi) < 2)
            and epi_node in node_xy
        ):
            nx, ny = node_xy[epi_node]
            quake_epi = [float(nx), float(ny)]

        panel_frames.append({
            "step": int(rf.get("step", 0)),
            "nodes": frame_nodes,
            "edges": edges,
            "tasks": tasks_list,
            "agents": agents_list,
            "weather": {
                "rainfall": float(info.get("rainfall_mean", 0.0)),
                "wind": float(info.get("wind_mean", 0.0)),
                "commBlockRate": _clamp(float(info.get("comm_blocked_count", 0.0)) / float(n_agents), 0.0, 1.0),
                "riskSpike": bool(info.get("risk_spike", info.get("hrl_trigger", False))),
                "epicenterNode": epi_node,
                "stormCenter": hazard_field.get("storm_center"),
                "rmw": float(hazard_field.get("rmw", 0.0) or 0.0),
                "quakeEpicenter": quake_epi,
            },
            "metrics": {
                "normalDelivered": int(normal_done),
                "normalTotal": int(normal_total),
                "emergencyDelivered": int(emer_done),
                "emergencyTotal": int(emer_total),
                "dockingCount": int(info.get("uav_follow_bind_count_total", 0)),
                "avgReward": avg_reward,
                "invalidActionCount": int(info.get("invalid_action_count", 0)),
            },
            "keyEvents": key_events[:8],
        })

    return {"frames": panel_frames}


def build_viewer_html_with_template(
    panel_data: Dict[str, Any],
    panel_template: Path,
) -> str:
    template = panel_template.read_text(encoding="utf-8")
    payload = json.dumps(panel_data, ensure_ascii=False)
    inject = f"<script>window.__HETGAT_DATA__ = {payload};</script>\n<script>"
    if "<script>" not in template:
        raise RuntimeError(f"Invalid panel template (no <script>): {panel_template}")
    return template.replace("<script>", inject, 1)


def main() -> None:
    args = parse_args()
    panel_template = Path(args.panel_template)

    cfg = EnvConfig(
        phase="S",
        scenario="B",
        seed=args.seed,
        num_nodes=40,
        num_edges=64,
        num_trucks=2,
        num_uavs=3,
        max_steps=150,
        dt_seconds=float(args.dt_seconds),
        dt=float(args.dt_seconds),
        use_pbrs=True,
        pbrs_scale=5.0,
    )

    env = BaseHeteroDisasterEnv(cfg)
    low = AttentionGuidedLowLevelPolicy(obs_dim=20, hidden_dim=64, seed=args.seed)
    high = RiskTriggeredHRLPlanner(decision_interval=cfg.hrl_interval, seed=args.seed)

    out_dir = (
        Path(args.results_root)
        / f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_rows: List[Dict[str, Any]] = []
    best_trace: Optional[List[Dict[str, Any]]] = None
    best_nodes: Optional[List[Dict[str, Any]]] = None
    best_edges: Optional[List[Dict[str, Any]]] = None
    best_key = (-1.0, -1e18)
    best_seed = args.seed

    for ep in range(args.iterations):
        seed = args.seed + ep
        state = env.reset(seed=seed)
        uav_ids = [
            aid
            for aid, st in env.state.agents.items()
            if st.kind == AgentKind.UAV
        ]
        truck_slot_pick_count = 0
        uav_goal_decisions = 0
        alive_uav_goal_with_target = 0
        alive_uav_pick_truck_count = 0
        follow_bind_attempt_count = 0
        follow_bind_success_count = 0
        uav_low_cargo_goal_decisions = 0
        uav_low_cargo_pick_truck_count = 0
        truck_slot_pick_by_uav: Dict[str, int] = {aid: 0 for aid in uav_ids}
        low_cargo_goal_decisions_by_uav: Dict[str, int] = {aid: 0 for aid in uav_ids}
        low_cargo_pick_truck_by_uav: Dict[str, int] = {aid: 0 for aid in uav_ids}
        ep_nodes = [
            {"id": int(i), "x": float(n.x), "y": float(n.y)}
            for i, n in sorted(env.topology.nodes.items(), key=lambda kv: kv[0])
        ]
        ep_edges: List[Dict[str, Any]] = []
        for src, nbs in env.topology.adjacency.items():
            for dst in nbs:
                if src < dst:
                    eattr = env.topology.edge_attr(int(src), int(dst))
                    ep_edges.append({
                        "id": f"e_{int(src)}_{int(dst)}",
                        "a": int(src),
                        "b": int(dst),
                        "length_m": float(env.topology.edge_distance(int(src), int(dst))),
                        "roughness_norm": float(getattr(eattr, "roughness_norm", 0.0)),
                        "building_density_norm": float(getattr(eattr, "building_density_norm", 0.0)),
                        "infra_bottleneck_norm": float(getattr(eattr, "infra_bottleneck_norm", 0.0)),
                        "base_vulnerability": float(getattr(eattr, "base_vulnerability", 0.0)),
                    })
        frames = [
            snapshot(
                env,
                info={"episode_start": True},
                rewards={aid: 0.0 for aid in state.agents},
            )
        ]

        total_reward = 0.0
        last_info: Dict[str, Any] = {}
        task_terminal_step: Optional[int] = None
        while not state.done:
            planner_goals = high.plan(env)
            attn_goals, _ = low.infer_attention_goals(env, fallback_goals=planner_goals)
            for uid in uav_ids:
                st = env.state.agents.get(uid, None)
                if st is None or bool(st.crashed):
                    continue
                uav_goal_decisions += 1
                gid = attn_goals.get(uid, None)
                is_low_cargo = bool(st is not None and float(getattr(st, "cargo", 0.0)) <= 0.0)
                if is_low_cargo:
                    uav_low_cargo_goal_decisions += 1
                    low_cargo_goal_decisions_by_uav[uid] = int(
                        low_cargo_goal_decisions_by_uav.get(uid, 0) + 1
                    )
                if not isinstance(gid, str):
                    continue
                tgt = env.state.agents.get(str(gid), None)
                if tgt is not None and tgt.kind == AgentKind.TRUCK:
                    truck_slot_pick_count += 1
                    truck_slot_pick_by_uav[uid] = int(
                        truck_slot_pick_by_uav.get(uid, 0) + 1
                    )
                if isinstance(gid, str):
                    alive_uav_goal_with_target += 1
                    if tgt is not None and tgt.kind == AgentKind.TRUCK:
                        alive_uav_pick_truck_count += 1
                    if is_low_cargo:
                        if tgt is not None and tgt.kind == AgentKind.TRUCK:
                            uav_low_cargo_pick_truck_count += 1
                            low_cargo_pick_truck_by_uav[uid] = int(
                                low_cargo_pick_truck_by_uav.get(uid, 0) + 1
                            )
            env.set_recommended_goals(attn_goals)
            actions = low.act(env, high_goals=attn_goals)
            bind_attempt_targets: Dict[str, str] = {}
            for uid in uav_ids:
                st_before = env.state.agents.get(uid, None)
                if st_before is None or bool(st_before.crashed):
                    continue
                act = actions.get(uid, None)
                if isinstance(act, UAVAction) and act.bind_truck_id is not None:
                    follow_bind_attempt_count += 1
                    bind_attempt_targets[uid] = str(act.bind_truck_id)
            out = env.step(actions)
            state = out.state
            for uid, tid in bind_attempt_targets.items():
                st_after = out.state.agents.get(uid, None)
                if st_after is None or bool(st_after.crashed):
                    continue
                if st_after.follow_target is not None and str(st_after.follow_target) == str(tid):
                    follow_bind_success_count += 1
            last_info = out.info
            if task_terminal_step is None and bool(out.info.get("tasks_terminal", False)):
                task_terminal_step = int(state.step_index)
            total_reward += sum(out.rewards.values()) / max(len(out.rewards), 1)
            frames.append(snapshot(env, info=out.info, rewards=out.rewards))

        completion = float(last_info.get("task_completion_rate", 0.0))
        reward_mean = float(total_reward / max(state.step_index, 1))
        invalid_mean = float(last_info.get("invalid_action_mean", 0.0))
        row = {
            "iteration": ep,
            "seed": seed,
            "episode_reward_mean": reward_mean,
            "task_completion_rate": completion,
            "invalid_action_mean": invalid_mean,
            "task_terminal_step": int(task_terminal_step) if task_terminal_step is not None else -1,
            "task_terminal_time_s": float(task_terminal_step * cfg.dt_seconds) if task_terminal_step is not None else -1.0,
            "uav_truck_slot_pick_count": int(truck_slot_pick_count),
            "uav_goal_decisions": int(uav_goal_decisions),
            "uav_truck_slot_pick_ratio": float(
                truck_slot_pick_count / max(uav_goal_decisions, 1)
            ),
            "alive_uav_goal_with_target": int(alive_uav_goal_with_target),
            "alive_uav_pick_truck_count": int(alive_uav_pick_truck_count),
            "alive_uav_pick_truck_ratio": float(
                alive_uav_pick_truck_count / max(alive_uav_goal_with_target, 1)
            ),
            "follow_bind_attempt_count": int(follow_bind_attempt_count),
            "follow_bind_success_count": int(follow_bind_success_count),
            "follow_bind_success_rate": float(
                follow_bind_success_count / max(follow_bind_attempt_count, 1)
            ),
            "uav_low_cargo_goal_decisions": int(uav_low_cargo_goal_decisions),
            "uav_low_cargo_pick_truck_count": int(uav_low_cargo_pick_truck_count),
            "uav_low_cargo_pick_truck_ratio": float(
                uav_low_cargo_pick_truck_count / max(uav_low_cargo_goal_decisions, 1)
            ),
            "steps": int(state.step_index),
        }
        for uid in sorted(uav_ids):
            row[f"uav_truck_slot_pick_count_{uid}"] = int(
                truck_slot_pick_by_uav.get(uid, 0)
            )
            row[f"uav_low_cargo_goal_decisions_{uid}"] = int(
                low_cargo_goal_decisions_by_uav.get(uid, 0)
            )
            row[f"uav_low_cargo_pick_truck_count_{uid}"] = int(
                low_cargo_pick_truck_by_uav.get(uid, 0)
            )
        episode_rows.append(row)
        print(row)

        key = (completion, reward_mean)
        if key > best_key:
            best_key = key
            best_trace = frames
            best_nodes = ep_nodes
            best_edges = ep_edges
            best_seed = seed

    csv_path = out_dir / "iter_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        uav_pick_cols: List[str] = []
        if episode_rows:
            uav_pick_cols = [
                k
                for k in episode_rows[0].keys()
                if k.startswith("uav_truck_slot_pick_count_uav_")
                or k.startswith("uav_low_cargo_goal_decisions_uav_")
                or k.startswith("uav_low_cargo_pick_truck_count_uav_")
            ]
            uav_pick_cols.sort()
        w = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "seed",
                "episode_reward_mean",
                "task_completion_rate",
                "invalid_action_mean",
                "task_terminal_step",
                "task_terminal_time_s",
                "uav_truck_slot_pick_count",
                "uav_goal_decisions",
                "uav_truck_slot_pick_ratio",
                "alive_uav_goal_with_target",
                "alive_uav_pick_truck_count",
                "alive_uav_pick_truck_ratio",
                "follow_bind_attempt_count",
                "follow_bind_success_count",
                "follow_bind_success_rate",
                "uav_low_cargo_goal_decisions",
                "uav_low_cargo_pick_truck_count",
                "uav_low_cargo_pick_truck_ratio",
                *uav_pick_cols,
                "steps",
            ],
        )
        w.writeheader()
        for r in episode_rows:
            w.writerow(r)

    truck_slot_csv_path = out_dir / "uav_truck_slot_metrics.csv"
    with truck_slot_csv_path.open("w", newline="", encoding="utf-8") as f:
        uav_pick_cols: List[str] = []
        if episode_rows:
            uav_pick_cols = [
                k
                for k in episode_rows[0].keys()
                if k.startswith("uav_truck_slot_pick_count_uav_")
                or k.startswith("uav_low_cargo_goal_decisions_uav_")
                or k.startswith("uav_low_cargo_pick_truck_count_uav_")
            ]
            uav_pick_cols.sort()
        w = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "seed",
                "uav_truck_slot_pick_count",
                "uav_goal_decisions",
                "uav_truck_slot_pick_ratio",
                "alive_uav_goal_with_target",
                "alive_uav_pick_truck_count",
                "alive_uav_pick_truck_ratio",
                "follow_bind_attempt_count",
                "follow_bind_success_count",
                "follow_bind_success_rate",
                "uav_low_cargo_goal_decisions",
                "uav_low_cargo_pick_truck_count",
                "uav_low_cargo_pick_truck_ratio",
                *uav_pick_cols,
                "task_completion_rate",
                "episode_reward_mean",
                "task_terminal_step",
                "task_terminal_time_s",
            ],
        )
        w.writeheader()
        for r in episode_rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    if best_trace is None:
        raise RuntimeError("No trace generated")
    if best_nodes is None or best_edges is None:
        raise RuntimeError("Best episode topology snapshot missing")

    raw_payload = {
        "meta": {
            "iterations": int(args.iterations),
            "best_seed": int(best_seed),
            "best_key": {
                "task_completion_rate": best_key[0],
                "episode_reward_mean": best_key[1],
            },
        },
        "nodes": best_nodes,
        "edges": best_edges,
        "frames": best_trace,
    }

    panel_data = adapt_trace_to_command_center_data(raw_payload)
    html_content = build_viewer_html_with_template(panel_data, panel_template)

    html_path = out_dir / "best_episode_command_center.html"
    html_path.write_text(html_content, encoding="utf-8")

    # 兼容旧文件名
    legacy_html_path = out_dir / "best_episode_step_viewer.html"
    legacy_html_path.write_text(html_content, encoding="utf-8")

    meta_path = out_dir / "summary.json"
    meta_path.write_text(
        json.dumps(raw_payload["meta"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"saved metrics: {csv_path}")
    print(f"saved truck-slot metrics: {truck_slot_csv_path}")
    print(f"saved viewer: {html_path}")
    print(f"saved viewer(legacy): {legacy_html_path}")
    print(f"saved summary: {meta_path}")


if __name__ == "__main__":
    main()




