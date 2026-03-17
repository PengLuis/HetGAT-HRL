from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from hetgat_hrl.core.mdp_spec import (
    AgentKind,
    AgentRuntimeState,
    DeliveryTask,
    EnvConfig,
    HazardSnapshot,
    HeteroDisasterMDP,
    JointAction,
    JointState,
    StepResult,
    TaskKind,
    TaskStatus,
    TruckAction,
    UAVAction,
)
from hetgat_hrl.core.topology import GraphTopology
from hetgat_hrl.envs.hazards import DynamicHazardField
from hetgat_hrl.envs.task_manager import DynamicTaskManager


class BaseHeteroDisasterEnv(HeteroDisasterMDP):
    """
    Step-1 executable skeleton.
    This class only locks the MDP/SMDP interfaces and state transitions shape.
    """

    def __init__(self, cfg: Optional[EnvConfig] = None):
        self.cfg = cfg or EnvConfig()
        # Resolve legacy-vs-new naming into effective runtime values.
        self._num_nodes = int(self.cfg.num_nodes or self.cfg.n_nodes)
        self._num_trucks = int(self.cfg.num_trucks or self.cfg.n_trucks)
        self._num_uavs = int(self.cfg.num_uavs or self.cfg.n_uavs)
        self._num_normal_tasks = int(self.cfg.num_normal_tasks or self.cfg.n_normal_tasks)
        self._num_emergency_tasks = int(
            self.cfg.num_emergency_tasks or self.cfg.n_emergency_tasks
        )
        self._dt_seconds = float(self.cfg.dt_seconds)
        if self._dt_seconds <= 0.0:
            raise ValueError(f"dt_seconds must be > 0, got dt_seconds={self._dt_seconds}")
        self.topology = GraphTopology.build_from_config(self.cfg)
        self.hazards = DynamicHazardField(
            topo=self.topology,
            seed=self.cfg.seed + 1,
            stochastic_weather=self.cfg.stochastic_weather,
            cfg=self.cfg,
        )
        self.state = self._build_initial_state()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.task_manager = DynamicTaskManager(self.topology)
        self.comm_blocked: Dict[str, bool] = {aid: False for aid in self.state.agents}
        self.task_attention_slots: int = 6
        self.task_feat_dim: int = 5
        self.follow_bind_count_total: int = 0
        self.follow_steps_total: int = 0
        self.follow_charge_energy_total: float = 0.0
        self.low_battery_events_total: int = 0
        self.low_battery_return_success_total: int = 0
        self._uav_low_battery_flag: Dict[str, bool] = {}
        self._recommended_goals: Dict[str, Optional[str]] = {}
        self._effective_goals: Dict[str, Optional[str]] = {}
        self._pbrs_lock: Dict[str, Tuple[Optional[str], Optional[float]]] = {}
        self._pbrs_switch_total: int = 0
        # task_id -> uav_id, consumed on next step to realize "340m next-round snap"
        self._uav_emergency_snap_pending: Dict[str, str] = {}
        self._uav_discovered_blocked_edges: set = set()
        self._uav_discovered_blocked_total: int = 0

    def _build_initial_state(self) -> JointState:
        agents: Dict[str, AgentRuntimeState] = {}
        for i in range(self._num_trucks):
            agents[f"truck_{i}"] = AgentRuntimeState(
                agent_id=f"truck_{i}",
                kind=AgentKind.TRUCK,
                node=0,
                pos_xy=self._node_xy(0),
                battery=1.0,
                cargo=float(self.cfg.truck_cargo_capacity_units),
                replenish_timer=0,
            )
        for i in range(self._num_uavs):
            agents[f"uav_{i}"] = AgentRuntimeState(
                agent_id=f"uav_{i}",
                kind=AgentKind.UAV,
                node=0,
                pos_xy=self._node_xy(0),
                vel_xy=(0.0, 0.0),
                battery=float(self.cfg.uav_battery_init),
                cargo=float(self.cfg.uav_cargo_capacity_units),
                replenish_timer=0,
            )

        tasks: Dict[str, DeliveryTask] = {}
        # Simple deterministic placement for Step-2 baseline.
        for i in range(self._num_normal_tasks):
            demand = int((i * 3 + 5) % self._num_nodes)
            tasks[f"task_normal_{i}"] = DeliveryTask(
                task_id=f"task_normal_{i}",
                kind=TaskKind.NORMAL,
                demand_node=demand,
                deadline_step=min(self.cfg.max_steps - 1, 120 + i * 5),
                status=TaskStatus.PENDING,
                demand_left=float(self.cfg.task_demand_normal_units),
            )
        for i in range(self._num_emergency_tasks):
            demand = int((i * 7 + 9) % self._num_nodes)
            tasks[f"task_emergency_{i}"] = DeliveryTask(
                task_id=f"task_emergency_{i}",
                kind=TaskKind.EMERGENCY,
                demand_node=demand,
                deadline_step=min(self.cfg.max_steps - 1, 80 + i * 4),
                status=TaskStatus.PENDING,
                demand_left=float(self.cfg.task_demand_emergency_units),
            )

        return JointState(
            step_index=0,
            agents=agents,
            tasks=tasks,
            hazard=HazardSnapshot(),
            done=False,
        )

    def reset(self, seed: Optional[int] = None) -> JointState:
        if seed is not None:
            self.cfg = EnvConfig(**{**self.cfg.__dict__, "seed": int(seed)})
            self.topology = GraphTopology.build_from_config(self.cfg)
            self.hazards = DynamicHazardField(
                topo=self.topology,
                seed=self.cfg.seed + 1,
                stochastic_weather=self.cfg.stochastic_weather,
                cfg=self.cfg,
            )
            self.rng = np.random.default_rng(self.cfg.seed)
            self.task_manager = DynamicTaskManager(self.topology)
        self.state = self._build_initial_state()
        self.comm_blocked = {aid: False for aid in self.state.agents}
        self.follow_bind_count_total = 0
        self.follow_steps_total = 0
        self.follow_charge_energy_total = 0.0
        self.low_battery_events_total = 0
        self.low_battery_return_success_total = 0
        self._uav_low_battery_flag = {
            aid: False for aid, s in self.state.agents.items() if s.kind == AgentKind.UAV
        }
        self._recommended_goals = {}
        self._effective_goals = {}
        self._pbrs_lock = {aid: (None, None) for aid in self.state.agents}
        self._pbrs_switch_total = 0
        self._uav_emergency_snap_pending = {}
        self._uav_discovered_blocked_edges = set()
        self._uav_discovered_blocked_total = 0
        return self.state

    def set_recommended_goals(self, goals: Dict[str, Optional[str]]) -> None:
        self._recommended_goals = {
            str(k): (None if v is None else str(v)) for k, v in goals.items()
        }
        # Communication blackout freeze: blocked agents keep previous effective goal.
        for aid in self.state.agents:
            incoming = self._recommended_goals.get(str(aid), None)
            if bool(self.comm_blocked.get(aid, False)):
                if aid not in self._effective_goals:
                    self._effective_goals[aid] = incoming
                continue
            self._effective_goals[aid] = incoming

    def _has_active_assigned_task(self, aid: str) -> bool:
        for t in self.state.tasks.values():
            if (
                t.status in (TaskStatus.PENDING, TaskStatus.CLAIMED)
                and t.assigned_to is not None
                and str(t.assigned_to) == str(aid)
            ):
                return True
        return False

    def _assigned_task(self, aid: str) -> Optional[DeliveryTask]:
        agent_kind = self.state.agents[str(aid)].kind
        chosen: Optional[DeliveryTask] = None
        for t in self.state.tasks.values():
            if (
                t.status in (TaskStatus.PENDING, TaskStatus.CLAIMED)
                and t.assigned_to is not None
                and str(t.assigned_to) == str(aid)
            ):
                if agent_kind == AgentKind.UAV and t.kind != TaskKind.EMERGENCY:
                    continue
                if chosen is None or t.deadline_step < chosen.deadline_step:
                    chosen = t
        return chosen

    def _agent_task_distance_norm(self, aid: str) -> float:
        t = self._pbrs_target_task(aid)
        if t is None:
            return 1.0
        a = self.state.agents[aid]
        node = self.topology.nodes[int(t.demand_node)]
        if a.pos_xy is not None:
            d = float(np.hypot(a.pos_xy[0] - node.x, a.pos_xy[1] - node.y))
        else:
            cur = self.topology.nodes[int(a.node or 0)]
            d = float(np.hypot(cur.x - node.x, cur.y - node.y))
        return float(np.clip(d / 3000.0, 0.0, 1.0))

    def _agent_xy(self, aid: str) -> Tuple[float, float]:
        a = self.state.agents[aid]
        if a.pos_xy is not None:
            return float(a.pos_xy[0]), float(a.pos_xy[1])
        return self._node_xy(int(a.node or 0))

    def _agent_task_rel(self, aid: str, task: DeliveryTask) -> Tuple[float, float, float]:
        ax, ay = self._agent_xy(aid)
        tn = self.topology.nodes[int(task.demand_node)]
        dx = float(tn.x - ax)
        dy = float(tn.y - ay)
        dist = float(np.hypot(dx, dy))
        return (
            float(np.clip(dx / 3000.0, -1.0, 1.0)),
            float(np.clip(dy / 3000.0, -1.0, 1.0)),
            float(np.clip(dist / 3000.0, 0.0, 1.0)),
        )

    def _agent_distance_to_task(self, aid: str, task: DeliveryTask) -> float:
        a = self.state.agents[aid]
        node = self.topology.nodes[int(task.demand_node)]
        if a.kind == AgentKind.TRUCK and a.node is not None:
            g = self.topology.shortest_path_distance(
                int(a.node), int(task.demand_node), ignore_blocked=False
            )
            if np.isfinite(g):
                return float(g)
            cur = self.topology.nodes[int(a.node or 0)]
            return float(np.hypot(cur.x - node.x, cur.y - node.y))
        if a.pos_xy is not None:
            d = float(np.hypot(a.pos_xy[0] - node.x, a.pos_xy[1] - node.y))
        else:
            cur = self.topology.nodes[int(a.node or 0)]
            d = float(np.hypot(cur.x - node.x, cur.y - node.y))
        return d

    def _task_visible_to_agent(self, aid: str, task: DeliveryTask) -> bool:
        s = self.state.agents[str(aid)]
        if s.kind == AgentKind.UAV and task.kind != TaskKind.EMERGENCY:
            return False
        if task.status == TaskStatus.PENDING:
            return True
        if task.status == TaskStatus.CLAIMED and task.assigned_to is not None:
            return str(task.assigned_to) == str(aid)
        return False

    def _task_by_id_if_active(
        self, task_id: Optional[str], aid: Optional[str] = None
    ) -> Optional[DeliveryTask]:
        if task_id is None:
            return None
        t = self.state.tasks.get(str(task_id))
        if t is None:
            return None
        if aid is not None:
            a = self.state.agents.get(str(aid))
            if a is not None and a.kind == AgentKind.UAV and t.kind != TaskKind.EMERGENCY:
                return None
        if t.status == TaskStatus.PENDING:
            return t
        if aid is not None and t.status == TaskStatus.CLAIMED:
            if t.assigned_to is not None and str(t.assigned_to) == str(aid):
                return t
            return None
        return None

    def _pbrs_target_task(self, aid: str) -> Optional[DeliveryTask]:
        # HRL-authoritative target: only explicit recommendation.
        rec_tid = self._effective_goals.get(
            str(aid), self._recommended_goals.get(str(aid), None)
        )
        rec_task = self._task_by_id_if_active(rec_tid, aid=aid)
        if rec_task is not None:
            return rec_task
        return self._assigned_task(aid)

    def _service_rounds(self, aid: str, task: DeliveryTask) -> int:
        s = self.state.agents[aid]
        if s.kind == AgentKind.UAV:
            return int(max(1, int(self.cfg.unload_rounds_uav)))
        return int(max(1, int(self.cfg.unload_rounds_normal)))

    def _servicing_agents(self) -> set:
        out = set()
        for t in self.state.tasks.values():
            if (
                t.status == TaskStatus.CLAIMED
                and t.in_service_by is not None
                and int(t.service_remaining) > 0
            ):
                out.add(str(t.in_service_by))
        return out

    @staticmethod
    def _point_segment_distance(
        px: float, py: float, ax: float, ay: float, bx: float, by: float
    ) -> float:
        vx = float(bx - ax)
        vy = float(by - ay)
        wx = float(px - ax)
        wy = float(py - ay)
        vv = float(vx * vx + vy * vy)
        if vv <= 1e-12:
            return float(np.hypot(px - ax, py - ay))
        t = float(np.clip((wx * vx + wy * vy) / vv, 0.0, 1.0))
        cx = float(ax + t * vx)
        cy = float(ay + t * vy)
        return float(np.hypot(px - cx, py - cy))

    def _uav_visible_edge_ratio(self, aid: str, radius_m: float) -> float:
        s = self.state.agents[aid]
        if s.pos_xy is None:
            return 0.0
        px, py = float(s.pos_xy[0]), float(s.pos_xy[1])
        total = 0
        valid = 0
        for src, nbs in self.topology.adjacency.items():
            for dst in nbs:
                if int(src) >= int(dst):
                    continue
                a = self.topology.nodes[int(src)]
                b = self.topology.nodes[int(dst)]
                d = self._point_segment_distance(px, py, a.x, a.y, b.x, b.y)
                if d <= float(radius_m):
                    total += 1
                    if not self.topology.is_blocked(int(src), int(dst)):
                        valid += 1
        if total <= 0:
            return 0.0
        return float(valid / total)

    def _uav_visible_blocked_edges(self, aid: str, radius_m: float) -> List[Tuple[int, int]]:
        s = self.state.agents[aid]
        if s.pos_xy is None or s.crashed:
            return []
        if s.follow_target is not None:
            return []
        px, py = float(s.pos_xy[0]), float(s.pos_xy[1])
        visible: List[Tuple[int, int]] = []
        for (ea, eb) in self.topology.blocked_edges:
            a = self.topology.nodes[int(ea)]
            b = self.topology.nodes[int(eb)]
            d = self._point_segment_distance(px, py, a.x, a.y, b.x, b.y)
            if d <= float(radius_m):
                k = (min(int(ea), int(eb)), max(int(ea), int(eb)))
                visible.append(k)
        return visible

    def _uav_all_settled_for_termination(self) -> bool:
        for s in self.state.agents.values():
            if s.kind != AgentKind.UAV:
                continue
            if s.crashed:
                continue
            if s.follow_target is not None:
                continue
            return False
        return True

    def _start_service_for_arrived_agents(self) -> List[Tuple[str, DeliveryTask]]:
        started: List[Tuple[str, DeliveryTask]] = []
        busy = self._servicing_agents()

        # 0) Consume previous-step monitor-radius snap intents (next-round activation).
        snap_enabled = bool(getattr(self.cfg, "uav_monitor_snap_enabled", False)) or bool(
            getattr(self.cfg, "enable_monitor_snap", False)
        )
        if (not snap_enabled) and self._uav_emergency_snap_pending:
            # Safety: if snap rule is disabled, drop stale pending intents.
            self._uav_emergency_snap_pending = {}
        if snap_enabled and self._uav_emergency_snap_pending:
            pending_items = list(self._uav_emergency_snap_pending.items())
            self._uav_emergency_snap_pending = {}
            for tid, aid in pending_items:
                t = self.state.tasks.get(str(tid))
                s = self.state.agents.get(str(aid))
                if t is None or s is None:
                    continue
                if s.kind != AgentKind.UAV or s.crashed or str(aid) in busy:
                    continue
                if t.status != TaskStatus.PENDING or t.kind != TaskKind.EMERGENCY:
                    continue
                # Empty UAV cannot start unloading timer.
                if float(s.cargo) <= 0.0:
                    continue
                n = self.topology.nodes[int(t.demand_node)]
                # Snap UAV to demand node and start unloading service.
                s.pos_xy = (float(n.x), float(n.y))
                s.node = int(t.demand_node)
                s.vel_xy = (0.0, 0.0)
                t.status = TaskStatus.CLAIMED
                t.assigned_to = str(aid)
                t.in_service_by = str(aid)
                t.service_remaining = int(self._service_rounds(str(aid), t))
                started.append((str(aid), t))
                busy.add(str(aid))

        # 1) Regular arrival-based service trigger.
        for aid, s in self.state.agents.items():
            if aid in busy or s.crashed:
                continue
            if s.kind == AgentKind.TRUCK:
                if s.transit is not None or s.node is None:
                    continue
                if float(s.cargo) <= 0.0:
                    continue
                cands = [
                    t
                    for t in self.state.tasks.values()
                    if t.status == TaskStatus.PENDING and int(t.demand_node) == int(s.node)
                ]
                if not cands:
                    continue
                cands.sort(key=lambda t: (0 if t.kind == TaskKind.EMERGENCY else 1, t.deadline_step))
                t = cands[0]
                t.status = TaskStatus.CLAIMED
                t.assigned_to = str(aid)
                t.in_service_by = str(aid)
                t.service_remaining = int(self._service_rounds(aid, t))
                started.append((aid, t))
                busy.add(str(aid))
                continue

            # UAV logic
            if s.pos_xy is None:
                continue
            if float(s.cargo) <= 0.0:
                continue
            immediate_cands: List[DeliveryTask] = []
            monitor_cands: List[Tuple[float, DeliveryTask]] = []
            for t in self.state.tasks.values():
                if t.status != TaskStatus.PENDING or t.kind != TaskKind.EMERGENCY:
                    continue
                n = self.topology.nodes[int(t.demand_node)]
                d = float(np.hypot(float(s.pos_xy[0]) - n.x, float(s.pos_xy[1]) - n.y))
                if d <= float(self.cfg.uav_delivery_radius_m):
                    immediate_cands.append(t)
                elif snap_enabled and d <= float(
                    self.cfg.uav_monitor_radius_m
                ):
                    # Queue for next-round snap trigger (optional rule takeover).
                    monitor_cands.append((d, t))

            if immediate_cands:
                immediate_cands.sort(key=lambda t: t.deadline_step)
                t = immediate_cands[0]
                t.status = TaskStatus.CLAIMED
                t.assigned_to = str(aid)
                t.in_service_by = str(aid)
                t.service_remaining = int(self._service_rounds(aid, t))
                started.append((aid, t))
                busy.add(str(aid))
                continue

            if monitor_cands:
                monitor_cands.sort(key=lambda x: (x[0], x[1].deadline_step))
                _, t = monitor_cands[0]
                tid = str(t.task_id)
                # keep first reservation; avoid overwriting by later UAVs
                if tid not in self._uav_emergency_snap_pending and t.assigned_to is None:
                    self._uav_emergency_snap_pending[tid] = str(aid)

        return started

    def _advance_service_and_timeouts(
        self,
    ) -> Tuple[
        List[Tuple[str, DeliveryTask]],
        List[DeliveryTask],
        List[Tuple[str, DeliveryTask, float]],
    ]:
        delivered: List[Tuple[str, DeliveryTask]] = []
        timed_out: List[DeliveryTask] = []
        transfers: List[Tuple[str, DeliveryTask, float]] = []
        for task in self.state.tasks.values():
            if task.status in (TaskStatus.DELIVERED, TaskStatus.FAILED):
                continue
            if self.state.step_index > task.deadline_step:
                task.status = TaskStatus.FAILED
                task.in_service_by = None
                task.service_remaining = 0
                timed_out.append(task)
                continue
            if (
                task.status == TaskStatus.CLAIMED
                and task.in_service_by is not None
                and int(task.service_remaining) > 0
            ):
                task.service_remaining = int(task.service_remaining) - 1
                if int(task.service_remaining) <= 0:
                    aid = str(task.in_service_by)
                    agent = self.state.agents.get(aid)
                    # Backward compatibility: infer task demand if missing.
                    if float(task.demand_left) <= 0.0:
                        task.demand_left = (
                            float(self.cfg.task_demand_emergency_units)
                            if task.kind == TaskKind.EMERGENCY
                            else float(self.cfg.task_demand_normal_units)
                        )
                    agent_cargo = float(max(agent.cargo, 0.0)) if agent is not None else 0.0
                    transfer = float(min(agent_cargo, max(float(task.demand_left), 0.0)))
                    if agent is not None and transfer > 0.0:
                        agent.cargo = max(0.0, float(agent.cargo) - transfer)
                    task.demand_left = max(0.0, float(task.demand_left) - transfer)
                    transfers.append((aid, task, float(transfer)))

                    if float(task.demand_left) <= 1e-9:
                        task.status = TaskStatus.DELIVERED
                        task.delivered_by = aid
                        task.delivered_step = int(self.state.step_index)
                        task.in_service_by = None
                        task.service_remaining = 0
                        delivered.append((aid, task))
                    else:
                        # Partial delivery: release service lock for future agents.
                        task.status = TaskStatus.PENDING
                        task.assigned_to = None
                        task.in_service_by = None
                        task.service_remaining = 0
        return delivered, timed_out, transfers

    def _update_comm_blocked(self) -> None:
        # Hard-off switch for experiments without comm blackout.
        if (not bool(getattr(self.cfg, "enable_comm_blackout", False))) or float(self.cfg.comm_block_prob) <= 0.0:
            for aid in self.state.agents:
                self.comm_blocked[aid] = False
            return

        for aid, s in self.state.agents.items():
            node_id = int(s.node or 0)
            h = self.hazards.node_weather(node_id)
            r_tilde = float(np.clip(h.rain / max(self.cfg.base_rainfall_mmh, 1e-6), 0.0, 1.0))
            e_tilde = float(np.clip(h.quake, 0.0, 1.0))
            p_blocked = float(np.clip(self.state.hazard.blocked_ratio, 0.0, 1.0))
            risk_score = float(0.45 * p_blocked + 0.35 * r_tilde + 0.20 * e_tilde)
            p = float(
                np.clip(
                    self.cfg.comm_block_prob + 0.55 * risk_score,
                    0.0,
                    0.95,
                )
            )
            self.comm_blocked[aid] = bool(self.rng.uniform() < p)

    def _node_xy(self, node_id: int) -> Tuple[float, float]:
        node = self.topology.nodes[int(node_id)]
        return float(node.x), float(node.y)

    def _nearest_node(self, x: float, y: float) -> int:
        best_node = 0
        best_dist = 1e18
        for node_id, n in self.topology.nodes.items():
            d = (n.x - x) ** 2 + (n.y - y) ** 2
            if d < best_dist:
                best_dist = d
                best_node = int(node_id)
        return best_node

    def _has_open_emergency_tasks(self) -> bool:
        for t in self.state.tasks.values():
            if t.kind != TaskKind.EMERGENCY:
                continue
            if t.status in (TaskStatus.PENDING, TaskStatus.CLAIMED):
                return True
        return False

    def _advance_truck_transit(self, aid: str) -> None:
        s = self.state.agents[aid]
        if s.transit is None:
            return
        src, dst, remain = s.transit
        src = int(src)
        dst = int(dst)
        remain = max(0.0, float(remain) - float(self._dt_seconds))

        # Continuous truck position along edge to avoid teleport in rendering.
        dist = float(self.topology.edge_distance(src, dst))
        payload_kg = float(max(s.cargo, 0.0)) * float(self.cfg.cargo_unit_kg)
        speed = float(max(self._truck_speed_mps(src, dst, payload_kg=payload_kg), 1e-6))
        full_time = float(max(dist / speed, 1e-6))

        if remain <= 0.0:
            s.node = dst
            s.pos_xy = self._node_xy(dst)
            s.transit = None
        else:
            p0 = self._node_xy(src)
            p1 = self._node_xy(dst)
            progress = float(np.clip(1.0 - remain / full_time, 0.0, 1.0))
            x = (1.0 - progress) * p0[0] + progress * p1[0]
            y = (1.0 - progress) * p0[1] + progress * p1[1]
            s.node = src
            s.pos_xy = (float(x), float(y))
            s.transit = (src, dst, float(remain))

    def _start_truck_move(self, aid: str, target_node: int) -> bool:
        s = self.state.agents[aid]
        if s.transit is not None:
            return False
        if s.node is None:
            return False
        src = int(s.node)
        dst = int(target_node)
        if dst not in self.topology.neighbors(src):
            return False
        dist = self.topology.edge_distance(src, dst)
        payload_kg = float(max(s.cargo, 0.0)) * float(self.cfg.cargo_unit_kg)
        travel_time = dist / max(self._truck_speed_mps(src, dst, payload_kg=payload_kg), 1e-6)
        s.transit = (src, dst, float(travel_time))
        return True

    def _truck_speed_mps(self, src: int, dst: int, payload_kg: float = 0.0) -> float:
        e = self.topology.edge_attr(src, dst)
        slope_norm = float(
            np.clip(
                0.5
                * (self.topology.nodes[src].slope_norm + self.topology.nodes[dst].slope_norm),
                0.0,
                1.0,
            )
        )
        slope_deg = 35.0 * slope_norm
        rough = float(np.clip(e.roughness_norm, 0.0, 1.0))
        payload = float(max(payload_kg, 0.0))
        den = (1.0 + 0.015 * slope_deg) * (1.0 + 0.55 * rough) * (1.0 + 0.00035 * payload)
        return float(max(0.5, float(self.cfg.truck_speed_mps) / max(den, 1e-6)))

    def _apply_uav_action(
        self, aid: str, act: UAVAction
    ) -> Tuple[bool, float, bool, float, float]:
        """
        Returns:
            (invalid_action, moved_dist_m, new_bind, headwind_mps, rain_mmh)
        """
        s = self.state.agents[aid]
        if s.crashed:
            return False, 0.0, False, 0.0, 0.0
        if s.pos_xy is None:
            if s.node is not None:
                s.pos_xy = self._node_xy(int(s.node))
            else:
                s.pos_xy = (0.0, 0.0)
        force_hover = False
        vx = float(act.vx)
        vy = float(act.vy)

        # Follow-mode release.
        if s.follow_target is not None and bool(act.takeoff):
            s.follow_target = None
            s.replenish_timer = 0

        # Follow binding.
        if s.follow_target is None and act.bind_truck_id is not None:
            tid = str(act.bind_truck_id)
            ts = self.state.agents.get(tid)
            if ts is not None and ts.kind == AgentKind.TRUCK:
                followers = sum(
                    1
                    for uid, us in self.state.agents.items()
                    if uid != aid and us.kind == AgentKind.UAV and us.follow_target == tid
                )
                if followers >= int(max(self.cfg.uav_max_followers_per_truck, 0)):
                    # Queue full: reject bind, force hover, keep normal air-side energy consumption.
                    force_hover = True
                txy = ts.pos_xy if ts.pos_xy is not None else self._node_xy(int(ts.node or 0))
                uxy = s.pos_xy if s.pos_xy is not None else self._node_xy(int(s.node or 0))
                d = float(np.hypot(float(uxy[0]) - float(txy[0]), float(uxy[1]) - float(txy[1])))
                if (not force_hover) and d <= float(self.cfg.uav_bind_radius_m):
                    s.follow_target = tid
                    s.sortie_distance_m = 0.0
                    if (
                        float(s.cargo) < float(self.cfg.uav_cargo_capacity_units)
                        or float(s.battery) < 1.0
                    ):
                        s.replenish_timer = int(max(0, int(self.cfg.replenish_freeze_steps)))
                    return False, 0.0, True, 0.0, 0.0
                if force_hover:
                    # Force hovering in-place, not an invalid action.
                    vx = 0.0
                    vy = 0.0
                else:
                    return True, 0.0, False, 0.0, 0.0

        # If landed/following truck: physics overridden, no free movement.
        if s.follow_target is not None:
            return False, 0.0, False, 0.0, 0.0

        vx = float(vx)
        vy = float(vy)
        # Optional rule takeover: auto-approach when target enters monitor radius.
        tgt = self._pbrs_target_task(aid)
        auto_approach_enabled = bool(getattr(self.cfg, "uav_auto_approach_enabled", False)) or bool(
            getattr(self.cfg, "enable_auto_approach", False)
        )
        if (
            auto_approach_enabled
            and (not force_hover)
            and tgt is not None
            and s.pos_xy is not None
        ):
            tn = self.topology.nodes[int(tgt.demand_node)]
            dx = float(tn.x - s.pos_xy[0])
            dy = float(tn.y - s.pos_xy[1])
            d = float(np.hypot(dx, dy))
            if d <= float(self.cfg.uav_monitor_radius_m) and d > 1e-6:
                ux = dx / d
                uy = dy / d
                vx = float(ux * self.cfg.uav_max_speed_mps)
                vy = float(uy * self.cfg.uav_max_speed_mps)
        cmd_speed = float(np.hypot(vx, vy))
        headwind = 0.0
        rain = 0.0
        if cmd_speed > 1e-6 and s.pos_xy is not None:
            wx, wy = self.hazards.wind_vector_at(s.pos_xy)
            dir_x = vx / cmd_speed
            dir_y = vy / cmd_speed
            headwind = float(max(0.0, -(wx * dir_x + wy * dir_y)))
            weather = self.hazards.weather_at(s.pos_xy)
            rain = float(max(0.0, weather.rain))

        # 3.2 UAV speed attenuation formula with dynamic cargo coupling.
        # m_load_kg = cargo_units * cargo_unit_kg; cap fixed at 40 kg by spec.
        m_load_kg = float(max(s.cargo, 0.0)) * float(self.cfg.cargo_unit_kg)
        payload_ratio = float(m_load_kg / 40.0)
        f_load = float(max(0.25, 1.0 - 0.16 * payload_ratio))
        f_rain = float(max(0.35, 1.0 - 0.018 * rain))
        f_head = float(max(0.30, 1.0 - 0.03 * headwind))
        v_uav = float(self.cfg.uav_max_speed_mps) * f_load * f_rain * f_head

        if cmd_speed > v_uav and cmd_speed > 1e-6:
            scale = v_uav / max(cmd_speed, 1e-6)
            vx *= scale
            vy *= scale

        dt = float(self._dt_seconds)
        world_lim = float(max(getattr(self.cfg, "map_size_m", 3000.0), 1.0))
        nx = float(np.clip(s.pos_xy[0] + vx * dt, 0.0, world_lim))
        ny = float(np.clip(s.pos_xy[1] + vy * dt, 0.0, world_lim))
        dist = float(np.hypot(nx - s.pos_xy[0], ny - s.pos_xy[1]))
        s.vel_xy = (vx, vy)
        s.pos_xy = (nx, ny)
        s.node = self._nearest_node(nx, ny)
        s.sortie_distance_m += float(dist)
        s.lifetime_distance_m += float(dist)
        return False, dist, False, float(headwind), float(rain)

    def _sync_follow_and_charge(self, aid: str) -> float:
        s = self.state.agents[aid]
        if s.follow_target is None:
            return 0.0
        ts = self.state.agents.get(s.follow_target)
        if ts is None or ts.kind != AgentKind.TRUCK:
            s.follow_target = None
            s.replenish_timer = 0
            return 0.0
        if ts.transit is not None:
            # Keep following while truck is moving.
            src, dst, remain = ts.transit
            a = self.topology.nodes[int(src)]
            b = self.topology.nodes[int(dst)]
            full = self.topology.edge_distance(int(src), int(dst))
            truck_payload_kg = float(max(ts.cargo, 0.0)) * float(self.cfg.cargo_unit_kg)
            speed = float(
                max(
                    self._truck_speed_mps(int(src), int(dst), payload_kg=truck_payload_kg),
                    1e-6,
                )
            )
            progress = 1.0 - float(remain / max(full / speed, 1e-6))
            progress = float(np.clip(progress, 0.0, 1.0))
            x = (1.0 - progress) * a.x + progress * b.x
            y = (1.0 - progress) * a.y + progress * b.y
            s.pos_xy = (float(x), float(y))
            s.node = int(dst) if progress >= 0.5 else int(src)
        else:
            s.node = int(ts.node or 0)
            s.pos_xy = self._node_xy(int(s.node))
        s.vel_xy = (0.0, 0.0)

        need_replenish = bool(
            float(s.cargo) < float(self.cfg.uav_cargo_capacity_units)
            or float(s.battery) < 1.0
        )
        if int(s.replenish_timer) <= 0 and need_replenish:
            s.replenish_timer = int(max(0, int(self.cfg.replenish_freeze_steps)))

        if int(s.replenish_timer) > 0:
            s.replenish_timer = int(s.replenish_timer) - 1
            if int(s.replenish_timer) == 0:
                before = float(s.battery)
                s.battery = 1.0
                uav_cap = float(self.cfg.uav_cargo_capacity_units)
                if float(s.cargo) < uav_cap and float(ts.cargo) >= uav_cap:
                    ts.cargo = max(0.0, float(ts.cargo) - uav_cap)
                    s.cargo = uav_cap
                return float(max(0.0, s.battery - before))
            return 0.0

        return 0.0

    def _update_uav_energy_and_crash(
        self, aid: str, moved_dist_m: float, headwind_mps: float, rain_mmh: float
    ) -> bool:
        s = self.state.agents[aid]
        if s.kind != AgentKind.UAV or s.crashed:
            return False
        if s.follow_target is not None:
            # Charging already handled in follow sync.
            return False
        if moved_dist_m > 1e-6:
            m_load = float(max(s.cargo, 0.0)) * float(self.cfg.cargo_unit_kg)
            # 3.2 UAV energy model:
            # E_cost = d_fly * (1 + 0.04*W_head + 0.02*R) * (1 + 0.018*M_load)
            e_cost = float(moved_dist_m) * (
                1.0 + 0.04 * float(max(0.0, headwind_mps)) + 0.02 * float(max(0.0, rain_mmh))
            ) * (1.0 + 0.018 * m_load)
            s.battery -= float(self.cfg.uav_flight_discharge_per_m) * e_cost
        else:
            s.battery -= float(self.cfg.uav_idle_discharge_per_step)
        if s.battery <= 0.0:
            s.battery = 0.0
            s.crashed = True
            s.follow_target = None
            s.sortie_distance_m = 0.0
            return True
        return False

    def step(self, action: JointAction) -> StepResult:
        # Step-1: transition skeleton only (no full physics yet).
        if self.state.done:
            return StepResult(
                state=self.state,
                rewards={aid: 0.0 for aid in self.state.agents},
                terminated=True,
                truncated=False,
                info={"reason": "already_done"},
            )

        rewards = {
            aid: float(self.cfg.reward_step_penalty) for aid in self.state.agents
        }
        reward_step_total = float(self.cfg.reward_step_penalty) * float(
            len(self.state.agents)
        )
        reward_invalid_total = 0.0
        reward_idle_total = 0.0
        reward_delivery_total = 0.0
        reward_timeout_total = 0.0
        reward_pbrs_total = 0.0
        reward_crash_total = 0.0
        reward_discover_total = 0.0
        reward_docking_total = 0.0
        self.state.step_index += 1

        invalid_action_count = 0
        moved_dists: Dict[str, float] = {aid: 0.0 for aid in self.state.agents}
        moved_headwind: Dict[str, float] = {aid: 0.0 for aid in self.state.agents}
        moved_rain: Dict[str, float] = {aid: 0.0 for aid in self.state.agents}
        pre_step_battery: Dict[str, float] = {
            aid: float(s.battery)
            for aid, s in self.state.agents.items()
            if s.kind == AgentKind.UAV
        }
        step_follow_bind_count = 0
        step_follow_steps = 0
        step_follow_charge_energy = 0.0
        step_low_battery_events = 0
        step_low_battery_return_success = 0
        step_forced_takeoff_full = 0
        step_pbrs_switch_count = 0
        step_uav_discovered_blocked = 0

        reassigned = 0
        servicing_prev = self._servicing_agents()

        # 1) Advance existing truck transit.
        for aid, s in self.state.agents.items():
            if s.kind == AgentKind.TRUCK:
                self._advance_truck_transit(aid)
                # Depot refill: truck at node 0 gets full cargo each step.
                if s.transit is None and s.node is not None and int(s.node) == 0:
                    s.cargo = float(self.cfg.truck_cargo_capacity_units)

        # 2) Apply actions independently.
        for aid, s in self.state.agents.items():
            if aid in servicing_prev:
                # Agent is unloading cargo: ignore control inputs this step.
                continue
            a = action.get(aid, None)
            if s.kind == AgentKind.TRUCK:
                if a is None:
                    continue
                if not isinstance(a, TruckAction):
                    rewards[aid] += float(self.cfg.reward_invalid_action)
                    invalid_action_count += 1
                    reward_invalid_total += float(self.cfg.reward_invalid_action)
                    continue
                if bool(a.stay) or a.target_node is None:
                    continue
                ok = self._start_truck_move(aid, int(a.target_node))
                if not ok:
                    rewards[aid] += float(self.cfg.reward_invalid_action)
                    invalid_action_count += 1
                    reward_invalid_total += float(self.cfg.reward_invalid_action)
                else:
                    moved_dists[aid] = 1.0
            else:
                if a is None:
                    a = UAVAction()
                if not isinstance(a, UAVAction):
                    rewards[aid] += float(self.cfg.reward_invalid_action)
                    invalid_action_count += 1
                    reward_invalid_total += float(self.cfg.reward_invalid_action)
                    continue
                invalid, moved_dist, new_bind, hw, rain = self._apply_uav_action(aid, a)
                if invalid:
                    rewards[aid] += float(self.cfg.reward_invalid_action)
                    invalid_action_count += 1
                    reward_invalid_total += float(self.cfg.reward_invalid_action)
                if new_bind:
                    step_follow_bind_count += 1
                    batt_before = float(pre_step_battery.get(aid, 1.0))
                    if batt_before < float(self.cfg.docking_reward_battery_threshold):
                        dock_bonus = float(self.cfg.reward_docking_low_battery)
                        rewards[aid] += dock_bonus
                        reward_docking_total += dock_bonus
                moved_dists[aid] = moved_dist
                moved_headwind[aid] = float(hw)
                moved_rain[aid] = float(rain)

        # 2.5) Start unloading service for agents that reached task points.
        started_services = self._start_service_for_arrived_agents()
        servicing_now = self._servicing_agents()
        for aid in servicing_now:
            # No step penalty during unloading.
            rewards[aid] -= float(self.cfg.reward_step_penalty)
            reward_step_total -= float(self.cfg.reward_step_penalty)

        # 3) Follow sync + charging + crash.
        crashed_agents = []
        for aid, s in self.state.agents.items():
            if s.kind != AgentKind.UAV:
                continue
            if aid in servicing_now:
                # During unloading: no energy drain and no charging bookkeeping.
                s.vel_xy = (0.0, 0.0)
                continue
            # Hard rule: with full battery and unfinished emergency tasks, UAV cannot keep piggybacking.
            if (
                s.follow_target is not None
                and float(s.battery) >= float(self.cfg.uav_force_takeoff_battery_threshold)
                and float(s.cargo) >= float(self.cfg.uav_cargo_capacity_units)
                and int(s.replenish_timer) <= 0
                and self._has_open_emergency_tasks()
            ):
                s.follow_target = None
                s.replenish_timer = 0
                step_forced_takeoff_full += 1
            if float(s.battery) < 0.30 and not self._uav_low_battery_flag.get(aid, False):
                self._uav_low_battery_flag[aid] = True
                step_low_battery_events += 1

            if s.follow_target is not None:
                step_follow_steps += 1
            charged = self._sync_follow_and_charge(aid)
            step_follow_charge_energy += float(charged)

            if (
                self._uav_low_battery_flag.get(aid, False)
                and s.follow_target is not None
                and charged > 1e-9
            ):
                self._uav_low_battery_flag[aid] = False
                step_low_battery_return_success += 1

            crashed = self._update_uav_energy_and_crash(
                aid, moved_dists[aid], moved_headwind[aid], moved_rain[aid]
            )
            if crashed:
                rewards[aid] += float(self.cfg.uav_crash_penalty)
                reward_crash_total += float(self.cfg.uav_crash_penalty)
                crashed_agents.append(aid)
                self._uav_low_battery_flag[aid] = False

        # 4) Idle penalty only if agent has active assigned task.
        for aid, s in self.state.agents.items():
            if aid in servicing_now:
                continue
            if self._pbrs_target_task(aid) is None:
                continue
            stationary = moved_dists[aid] <= 1e-6
            if s.kind == AgentKind.TRUCK and s.transit is not None:
                stationary = False
            if s.kind == AgentKind.UAV and s.follow_target is not None:
                stationary = False
            if stationary:
                rewards[aid] += float(self.cfg.reward_idle_with_task)
                reward_idle_total += float(self.cfg.reward_idle_with_task)

        # 5) Delivery/timeout resolution.
        delivered, timed_out, transfers = self._advance_service_and_timeouts()
        delivered_normal = 0
        delivered_emergency = 0
        for aid, task, transfer in transfers:
            if transfer <= 0.0:
                continue
            if task.kind == TaskKind.NORMAL:
                denom = float(max(self.cfg.task_demand_normal_units, 1e-6))
                bonus = float(self.cfg.reward_delivery_normal) * float(transfer / denom)
            else:
                denom = float(max(self.cfg.task_demand_emergency_units, 1e-6))
                bonus = float(self.cfg.reward_delivery_emergency) * float(transfer / denom)
            rewards[aid] += float(bonus)
            reward_delivery_total += float(bonus)
        for _, task in delivered:
            if task.kind == TaskKind.NORMAL:
                delivered_normal += 1
            else:
                delivered_emergency += 1

        failed_normal = 0
        failed_emergency = 0
        for task in timed_out:
            if task.kind == TaskKind.NORMAL:
                penalty = float(self.cfg.penalty_timeout_normal)
                failed_normal += 1
            else:
                penalty = float(self.cfg.penalty_timeout_emergency)
                failed_emergency += 1
            if task.assigned_to is not None and str(task.assigned_to) in rewards:
                rewards[str(task.assigned_to)] += penalty
                reward_timeout_total += float(penalty)
            else:
                share = penalty / max(len(rewards), 1)
                for aid in rewards:
                    rewards[aid] += share
                reward_timeout_total += float(penalty)

        # 6) PBRS hook with strict target-consistency lock.
        if bool(self.cfg.use_pbrs):
            gamma = float(self.cfg.pbrs_gamma)
            norm_m = float(max(self.cfg.pbrs_distance_norm_m, 1e-6))
            for aid in rewards:
                current_task = self._pbrs_target_task(aid)
                if current_task is None:
                    self._pbrs_lock[aid] = (None, None)
                    continue
                cur_tid = str(current_task.task_id)
                cur_dist = self._agent_distance_to_task(aid, current_task)
                lock_tid, lock_dist = self._pbrs_lock.get(aid, (None, None))
                if lock_tid is None or lock_dist is None:
                    self._pbrs_lock[aid] = (cur_tid, float(cur_dist))
                    continue
                if str(lock_tid) != str(cur_tid):
                    # Hard reset on target switch: no shaping reward on switch step.
                    step_pbrs_switch_count += 1
                    self._pbrs_lock[aid] = (cur_tid, float(cur_dist))
                    continue
                phi_prev = -float(lock_dist) / norm_m
                phi_new = -float(cur_dist) / norm_m
                shaping = float(self.cfg.pbrs_scale) * (gamma * phi_new - phi_prev)
                rewards[aid] += shaping
                reward_pbrs_total += float(shaping)
                self._pbrs_lock[aid] = (cur_tid, float(cur_dist))

        rain_mean, wind_mean, blocked_ratio, epicenter = self.hazards.step()
        self.state.hazard.rainfall_mean = rain_mean
        self.state.hazard.wind_mean = wind_mean
        self.state.hazard.blocked_ratio = blocked_ratio
        self.state.hazard.epicenter_node = epicenter
        risk_spike = self.state.hazard.blocked_ratio >= self.cfg.risk_spike_threshold
        self.state.hazard.risk_spike = risk_spike
        self._update_comm_blocked()

        # 7) UAV scouting reward: first-time blocked-edge discovery within monitor radius.
        for aid, s in self.state.agents.items():
            if s.kind != AgentKind.UAV or s.crashed:
                continue
            visible_blocked = self._uav_visible_blocked_edges(
                aid, radius_m=float(self.cfg.uav_monitor_radius_m)
            )
            if not visible_blocked:
                continue
            new_count = 0
            for edge_key in visible_blocked:
                if edge_key in self._uav_discovered_blocked_edges:
                    continue
                self._uav_discovered_blocked_edges.add(edge_key)
                new_count += 1
            if new_count > 0:
                bonus = float(self.cfg.reward_uav_discover_blocked_edge) * float(new_count)
                rewards[aid] += bonus
                reward_discover_total += bonus
                step_uav_discovered_blocked += int(new_count)
                self._uav_discovered_blocked_total += int(new_count)

        tasks_terminal = all(
            t.status in (TaskStatus.DELIVERED, TaskStatus.FAILED)
            for t in self.state.tasks.values()
        )
        uav_settled = self._uav_all_settled_for_termination()
        if tasks_terminal and uav_settled:
            self.state.done = True
        if self.state.step_index >= self.cfg.max_steps:
            self.state.done = True

        self.follow_bind_count_total += int(step_follow_bind_count)
        self.follow_steps_total += int(step_follow_steps)
        self.follow_charge_energy_total += float(step_follow_charge_energy)
        self.low_battery_events_total += int(step_low_battery_events)
        self.low_battery_return_success_total += int(step_low_battery_return_success)
        self._pbrs_switch_total += int(step_pbrs_switch_count)
        uav_moves = [
            aid
            for aid, st in self.state.agents.items()
            if st.kind == AgentKind.UAV and moved_dists.get(aid, 0.0) > 1e-6
        ]
        mean_headwind = float(
            np.mean([moved_headwind[aid] for aid in uav_moves]) if uav_moves else 0.0
        )
        mean_rain = float(np.mean([moved_rain[aid] for aid in uav_moves]) if uav_moves else 0.0)
        uav_following_count = 0
        uav_follow_with_goal_count = 0
        uav_follow_near_goal_count = 0
        uav_follow_far_goal_count = 0
        for aid, s in self.state.agents.items():
            if s.kind != AgentKind.UAV or s.follow_target is None:
                continue
            uav_following_count += 1
            tgt = self._pbrs_target_task(aid)
            if tgt is None or s.pos_xy is None:
                continue
            uav_follow_with_goal_count += 1
            tn = self.topology.nodes[int(tgt.demand_node)]
            d = float(np.hypot(float(s.pos_xy[0]) - tn.x, float(s.pos_xy[1]) - tn.y))
            if d <= float(self.cfg.uav_monitor_radius_m):
                uav_follow_near_goal_count += 1
            else:
                uav_follow_far_goal_count += 1

        return StepResult(
            state=self.state,
            rewards=rewards,
            terminated=self.state.done,
            truncated=self.state.step_index >= self.cfg.max_steps,
            info={
                "hrl_trigger": self.should_trigger_hrl(),
                "accepted_actions": list(action.keys()),
                "invalid_action_count": int(invalid_action_count),
                "invalid_action_mean": float(
                    invalid_action_count / max(len(self.state.agents), 1)
                ),
                "crashed_agents": crashed_agents,
                "reassigned_count": int(reassigned),
                "service_started_count": int(len(started_services)),
                "servicing_agent_count": int(len(servicing_now)),
                "delivered_normal": int(delivered_normal),
                "delivered_emergency": int(delivered_emergency),
                "failed_normal": int(failed_normal),
                "failed_emergency": int(failed_emergency),
                "task_completion_rate": float(
                    sum(
                        1 for t in self.state.tasks.values() if t.status == TaskStatus.DELIVERED
                    )
                    / max(len(self.state.tasks), 1)
                ),
                "uav_follow_bind_count_step": int(step_follow_bind_count),
                "uav_follow_steps_step": int(step_follow_steps),
                "uav_charge_energy_gain_step": float(step_follow_charge_energy),
                "uav_low_battery_events_step": int(step_low_battery_events),
                "uav_low_battery_return_success_step": int(step_low_battery_return_success),
                "uav_follow_bind_count_total": int(self.follow_bind_count_total),
                "uav_follow_steps_total": int(self.follow_steps_total),
                "uav_charge_energy_gain_total": float(self.follow_charge_energy_total),
                "uav_low_battery_events_total": int(self.low_battery_events_total),
                "uav_low_battery_return_success_total": int(
                    self.low_battery_return_success_total
                ),
                "uav_forced_takeoff_full_step": int(step_forced_takeoff_full),
                "uav_low_battery_return_success_rate": float(
                    self.low_battery_return_success_total
                    / max(self.low_battery_events_total, 1)
                ),
                "pbrs_target_switch_count_step": int(step_pbrs_switch_count),
                "pbrs_target_switch_count_total": int(self._pbrs_switch_total),
                "comm_blocked_count": int(sum(1 for v in self.comm_blocked.values() if v)),
                "rainfall_mean": rain_mean,
                "wind_mean": wind_mean,
                "blocked_ratio": blocked_ratio,
                "epicenter_node": epicenter,
                "risk_spike": bool(risk_spike),
                "tasks_terminal": bool(tasks_terminal),
                "uav_settled_for_termination": bool(uav_settled),
                "dt_seconds": float(self._dt_seconds),
                "avg_degree": float(self.topology.average_degree()),
                "percolation_phase": str(self.hazards.last_percolation_phase),
                "percolation_lambda": float(self.hazards.last_lambda),
                "macro_block_prob_mean": float(self.hazards.last_pmacro_mean),
                "step_block_prob_mean": float(self.hazards.last_pstep_mean),
                "block_factor_bldg_mean": float(self.hazards.last_bldg_mean),
                "block_factor_infra_mean": float(self.hazards.last_infra_mean),
                "block_factor_length_norm_mean": float(self.hazards.last_length_norm_mean),
                "block_factor_length_ref_m": float(self.hazards.edge_len_ref_m),
                "reward_step_total": float(reward_step_total),
                "reward_invalid_total": float(reward_invalid_total),
                "reward_idle_total": float(reward_idle_total),
                "reward_delivery_total": float(reward_delivery_total),
                "reward_timeout_total": float(reward_timeout_total),
                "reward_pbrs_total": float(reward_pbrs_total),
                "reward_crash_total": float(reward_crash_total),
                "reward_discover_total": float(reward_discover_total),
                "reward_docking_total": float(reward_docking_total),
                "uav_headwind_mean_step": float(mean_headwind),
                "uav_rain_mean_step": float(mean_rain),
                "uav_discovered_blocked_step": int(step_uav_discovered_blocked),
                "uav_discovered_blocked_total": int(self._uav_discovered_blocked_total),
                "uav_following_count": int(uav_following_count),
                "uav_follow_with_goal_count": int(uav_follow_with_goal_count),
                "uav_follow_near_goal_count": int(uav_follow_near_goal_count),
                "uav_follow_far_goal_count": int(uav_follow_far_goal_count),
            },
        )

    def observe(self) -> Dict[str, List[float]]:
        # Step-1 observation shape baseline, will be replaced by full design.
        obs: Dict[str, List[float]] = {}
        pending_normal = sum(
            1
            for t in self.state.tasks.values()
            if t.status == TaskStatus.PENDING
            and t.kind == TaskKind.NORMAL
        )
        pending_emergency = sum(
            1
            for t in self.state.tasks.values()
            if t.status == TaskStatus.PENDING
            and t.kind == TaskKind.EMERGENCY
        )
        total_tasks = max(1, len(self.state.tasks))
        for aid, s in self.state.agents.items():
            node = int(s.node or 0)
            weather = self.hazards.node_weather(node)
            assigned = self._pbrs_target_task(aid)
            blocked = bool(self.comm_blocked.get(aid, False))
            goal_dx = 0.0
            goal_dy = 0.0
            if assigned is not None:
                goal_dx, goal_dy, _ = self._agent_task_rel(aid, assigned)
            global_blocked_ratio = 0.0 if blocked else float(self.state.hazard.blocked_ratio)
            global_pending_normal = 0.0 if blocked else float(pending_normal / total_tasks)
            global_pending_emergency = 0.0 if blocked else float(pending_emergency / total_tasks)
            if s.kind == AgentKind.TRUCK:
                total_nb = len(self.topology.adjacency.get(node, set()))
                valid_nb = len(self.topology.neighbors(node))
                nb_ratio = float(valid_nb / max(total_nb, 1))
            else:
                nb_ratio = self._uav_visible_edge_ratio(
                    aid, radius_m=float(self.cfg.uav_monitor_radius_m)
                )
            obs[aid] = [
                float(self.state.step_index / max(self.cfg.max_steps, 1)),
                float(s.battery),
                float(s.crashed),
                float(1.0 if s.kind == AgentKind.UAV else 0.0),
                float(1.0 if s.kind == AgentKind.TRUCK else 0.0),
                float(node / max(self._num_nodes - 1, 1)),
                float(weather.rain / 30.0),
                float(weather.wind / 20.0),
                float(weather.quake),
                float(goal_dx),
                float(goal_dy),
                global_blocked_ratio,
                global_pending_normal,
                global_pending_emergency,
                self._agent_task_distance_norm(aid),
                float(1.0 if (assigned is not None and assigned.kind == TaskKind.EMERGENCY) else 0.0),
                float(1.0 if s.follow_target is not None else 0.0),
                float(1.0 if blocked else 0.0),
                float(1.0 if self.state.hazard.risk_spike else 0.0),
                nb_ratio,
            ]
        return obs

    def observe_task_matrix(self) -> Dict[str, List[List[float]]]:
        """
        Returns fixed-size task matrix per agent:
        [task_attention_slots, task_feat_dim],
        features=[norm_dx, norm_dy, norm_dist, emergency_flag, is_recommended].
        """
        per_agent: Dict[str, List[List[float]]] = {}
        for aid in self.state.agents:
            s = self.state.agents[aid]
            blocked = bool(self.comm_blocked.get(aid, False))
            rec_tid = self._effective_goals.get(
                str(aid), self._recommended_goals.get(str(aid), None)
            )
            mat: List[List[float]] = []

            # Virtual task: nearest truck (for UAV energy-aware learning, no hard-coded battery rule).
            if s.kind == AgentKind.UAV:
                ax, ay = self._agent_xy(aid)
                nearest_tid: Optional[str] = None
                nearest_dist = float("inf")
                nearest_dx = 0.0
                nearest_dy = 0.0
                for tid, ts in self.state.agents.items():
                    if ts.kind != AgentKind.TRUCK:
                        continue
                    tx, ty = self._agent_xy(tid)
                    dx = float(tx - ax)
                    dy = float(ty - ay)
                    d = float(np.hypot(dx, dy))
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_tid = str(tid)
                        nearest_dx = dx
                        nearest_dy = dy
                if nearest_tid is not None:
                    mat.append(
                        [
                            float(np.clip(nearest_dx / 3000.0, -1.0, 1.0)),
                            float(np.clip(nearest_dy / 3000.0, -1.0, 1.0)),
                            float(np.clip(nearest_dist / 3000.0, 0.0, 1.0)),
                            0.0,  # emergency_flag
                            0.0,  # is_recommended
                        ]
                    )

            if not blocked:
                active_tasks = [
                    t for t in self.state.tasks.values() if self._task_visible_to_agent(aid, t)
                ]
                # Emergency first then earliest deadline.
                active_tasks.sort(
                    key=lambda t: (
                        0 if t.kind == TaskKind.EMERGENCY else 1,
                        t.deadline_step,
                    )
                )
                # Localized by ranking nearest tasks from active pool.
                ranked = sorted(
                    active_tasks,
                    key=lambda t: self._agent_distance_to_task(aid, t),
                )
                max_real_rows = max(0, int(self.task_attention_slots) - len(mat))
                for t in ranked[:max_real_rows]:
                    dx, dy, d = self._agent_task_rel(aid, t)
                    is_rec = 1.0 if (rec_tid is not None and str(t.task_id) == str(rec_tid)) else 0.0
                    mat.append(
                        [
                            float(dx),
                            float(dy),
                            float(d),
                            float(1.0 if t.kind == TaskKind.EMERGENCY else 0.0),
                            float(is_rec),
                        ]
                    )
            while len(mat) < self.task_attention_slots:
                mat.append([0.0, 0.0, 1.0, 0.0, 0.0])
            per_agent[aid] = mat
        return per_agent

    def observe_task_slots(self) -> Dict[str, List[Optional[str]]]:
        """
        Returns task ids aligned with observe_task_matrix rows for each agent.
        Row i in task matrix corresponds to slots[aid][i].
        """
        per_agent: Dict[str, List[Optional[str]]] = {}
        for aid in self.state.agents:
            s = self.state.agents[aid]
            blocked = bool(self.comm_blocked.get(aid, False))
            slots: List[Optional[str]] = []

            # Keep strict alignment with observe_task_matrix(): virtual nearest-truck slot first for UAV.
            if s.kind == AgentKind.UAV:
                ax, ay = self._agent_xy(aid)
                nearest_tid: Optional[str] = None
                nearest_dist = float("inf")
                for tid, ts in self.state.agents.items():
                    if ts.kind != AgentKind.TRUCK:
                        continue
                    tx, ty = self._agent_xy(tid)
                    d = float(np.hypot(tx - ax, ty - ay))
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_tid = str(tid)
                if nearest_tid is not None:
                    slots.append(nearest_tid)

            if not blocked:
                active_tasks = [
                    t for t in self.state.tasks.values() if self._task_visible_to_agent(aid, t)
                ]
                active_tasks.sort(
                    key=lambda t: (
                        0 if t.kind == TaskKind.EMERGENCY else 1,
                        t.deadline_step,
                    )
                )
                ranked = sorted(
                    active_tasks,
                    key=lambda t: self._agent_distance_to_task(aid, t),
                )
                max_real_slots = max(0, int(self.task_attention_slots) - len(slots))
                for t in ranked[:max_real_slots]:
                    slots.append(str(t.task_id))
            while len(slots) < self.task_attention_slots:
                slots.append(None)
            per_agent[aid] = slots
        return per_agent

    def legal_actions(self) -> Dict[str, object]:
        # Step-1 legality descriptor placeholder.
        legal: Dict[str, object] = {}
        for aid, s in self.state.agents.items():
            if s.kind == AgentKind.TRUCK:
                node = int(s.node or 0)
                legal[aid] = {
                    "type": "discrete_node",
                    "stay": True,
                    "neighbors": self.topology.neighbors(node),
                    "mask": [
                        1 if i in self.topology.neighbors(node) else 0
                        for i in range(self._num_nodes)
                    ],
                }
            else:
                legal[aid] = {
                    "type": "continuous_xy",
                    "bind_or_takeoff": True,
                    "vmax": self.cfg.uav_max_speed_mps,
                }
        return legal

    def should_trigger_hrl(self) -> bool:
        by_interval = self.state.step_index % max(self.cfg.hrl_interval, 1) == 0
        by_risk = bool(self.state.hazard.risk_spike)
        return bool(by_interval or by_risk)
