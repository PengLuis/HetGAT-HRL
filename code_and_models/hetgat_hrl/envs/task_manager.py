from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from hetgat_hrl.core.mdp_spec import (
    AgentKind,
    DeliveryTask,
    JointState,
    TaskKind,
    TaskStatus,
)
from hetgat_hrl.core.topology import GraphTopology


@dataclass
class TaskStats:
    delivered_normal: int = 0
    delivered_emergency: int = 0
    failed_normal: int = 0
    failed_emergency: int = 0
    reassigned: int = 0


class DynamicTaskManager:
    """
    Public pool + dynamic assignment + resolution rules.
    """

    def __init__(self, topology: GraphTopology):
        self.topology = topology

    def _agent_xy(self, state: JointState, aid: str) -> Tuple[float, float]:
        a = state.agents[aid]
        if a.pos_xy is not None:
            return float(a.pos_xy[0]), float(a.pos_xy[1])
        node = self.topology.nodes[int(a.node or 0)]
        return float(node.x), float(node.y)

    def _task_xy(self, task: DeliveryTask) -> Tuple[float, float]:
        n = self.topology.nodes[int(task.demand_node)]
        return float(n.x), float(n.y)

    def _distance(self, state: JointState, aid: str, task: DeliveryTask) -> float:
        ax, ay = self._agent_xy(state, aid)
        tx, ty = self._task_xy(task)
        return float(np.hypot(ax - tx, ay - ty))

    def _candidate_agents(
        self, state: JointState, task: DeliveryTask
    ) -> Iterable[str]:
        for aid, a in state.agents.items():
            if a.crashed:
                continue
            if task.kind == TaskKind.EMERGENCY:
                # Emergency prefers UAV first, but truck remains valid fallback.
                if a.kind in (AgentKind.UAV, AgentKind.TRUCK):
                    yield aid
            else:
                if a.kind == AgentKind.TRUCK:
                    yield aid

    def assign_tasks(self, state: JointState) -> int:
        """
        Deprecated in HRL-authoritative mode.
        Environment no longer performs any task allocation.
        """
        for task in state.tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.CLAIMED):
                task.assigned_to = None
                task.status = TaskStatus.PENDING
        return 0

    def resolve_deliveries(
        self,
        state: JointState,
        uav_delivery_radius_m: float,
    ) -> Tuple[List[Tuple[str, DeliveryTask]], List[DeliveryTask]]:
        delivered: List[Tuple[str, DeliveryTask]] = []
        timed_out: List[DeliveryTask] = []

        def _service_rounds(aid: str, task: DeliveryTask) -> int:
            a = state.agents[aid]
            if a.kind == AgentKind.UAV:
                return 1
            return 10

        def _arrived(aid: str, task: DeliveryTask) -> bool:
            a = state.agents[aid]
            if a.kind == AgentKind.TRUCK:
                return a.node is not None and int(a.node) == int(task.demand_node)
            if a.pos_xy is None:
                return False
            nx = self.topology.nodes[int(task.demand_node)]
            d = float(np.hypot(a.pos_xy[0] - nx.x, a.pos_xy[1] - nx.y))
            return d <= float(uav_delivery_radius_m)

        for task in state.tasks.values():
            if task.status in (TaskStatus.DELIVERED, TaskStatus.FAILED):
                continue
            if state.step_index > task.deadline_step:
                task.status = TaskStatus.FAILED
                task.in_service_by = None
                task.service_remaining = 0
                timed_out.append(task)
                continue

            # 1) Ongoing unloading countdown.
            if (
                task.status == TaskStatus.CLAIMED
                and task.in_service_by is not None
                and int(task.service_remaining) > 0
            ):
                task.service_remaining = int(task.service_remaining) - 1
                if int(task.service_remaining) <= 0:
                    aid = str(task.in_service_by)
                    a = state.agents.get(aid, None)
                    if float(task.demand_left) <= 0.0:
                        task.demand_left = 1.0 if task.kind == TaskKind.EMERGENCY else 8.0
                    cargo = float(max(a.cargo, 0.0)) if a is not None else 0.0
                    transfer = float(min(cargo, max(float(task.demand_left), 0.0)))
                    if a is not None and transfer > 0.0:
                        a.cargo = max(0.0, float(a.cargo) - transfer)
                    task.demand_left = max(0.0, float(task.demand_left) - transfer)

                    if float(task.demand_left) <= 1e-9:
                        task.status = TaskStatus.DELIVERED
                        task.delivered_by = aid
                        task.delivered_step = int(state.step_index)
                        task.in_service_by = None
                        task.service_remaining = 0
                        delivered.append((aid, task))
                    else:
                        # Partial fulfillment: reopen task for split delivery.
                        task.status = TaskStatus.PENDING
                        task.assigned_to = None
                        task.in_service_by = None
                        task.service_remaining = 0
                continue

            # 2) Start unloading when physically arrived and agent has cargo.
            if task.status != TaskStatus.PENDING:
                continue
            for aid, a in state.agents.items():
                if a.crashed:
                    continue
                if a.kind == AgentKind.UAV and task.kind != TaskKind.EMERGENCY:
                    continue
                if float(a.cargo) <= 0.0:
                    # Intercept: no cargo => cannot trigger unloading timer.
                    continue
                if not _arrived(aid, task):
                    continue
                task.status = TaskStatus.CLAIMED
                task.assigned_to = str(aid)
                task.in_service_by = str(aid)
                task.service_remaining = int(_service_rounds(aid, task))
                break

        return delivered, timed_out
