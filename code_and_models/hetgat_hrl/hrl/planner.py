from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from hetgat_hrl.agents.hetgat_risk import RiskMaskedHetGAT
from hetgat_hrl.core.mdp_spec import AgentKind, TaskKind, TaskStatus


@dataclass
class HRLPlannerState:
    step_last_refresh: int = 0
    goals: Dict[str, Optional[str]] = field(default_factory=dict)
    resolved_tasks_last: int = 0


class NeuralGoalAllocator(nn.Module):
    """
    Small neural scorer for high-level task assignment.
    Input features per (agent, task): [dist_norm, emergency_flag].
    Output: scalar preference score.
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Deterministic initializer to prefer near + emergency before training.
        with torch.no_grad():
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [N,2] -> score [N]
        return self.net(feat).squeeze(-1)


class RiskTriggeredHRLPlanner:
    """
    HRL high-level planner with unique authority on task recommendation.
    Refresh policy:
    - every decision_interval steps
    - on risk spike
    - on task completion/failure count change
    """

    def __init__(self, decision_interval: int = 5, seed: int = 0):
        self.decision_interval = max(int(decision_interval), 1)
        self.state = HRLPlannerState()
        torch.manual_seed(seed)
        self.allocator = NeuralGoalAllocator(hidden_dim=16)
        self.risk_gat = RiskMaskedHetGAT(in_dim=4, hidden_dim=16, beta=1.0)

    def _resolved_count(self, env) -> int:
        return int(
            sum(
                1
                for t in env.state.tasks.values()
                if t.status in (TaskStatus.DELIVERED, TaskStatus.FAILED)
            )
        )

    def _should_refresh(self, env) -> bool:
        by_interval = (
            (env.state.step_index - self.state.step_last_refresh) >= self.decision_interval
        )
        by_risk = bool(env.state.hazard.risk_spike)
        resolved_now = self._resolved_count(env)
        by_resolution = resolved_now != self.state.resolved_tasks_last
        by_arrival = False
        for aid, tid in self.state.goals.items():
            if tid is None:
                continue
            t = env.state.tasks.get(str(tid))
            if t is None or t.status in (TaskStatus.DELIVERED, TaskStatus.FAILED):
                by_arrival = True
                break
            a = env.state.agents[aid]
            if a.kind == AgentKind.TRUCK and a.node is not None:
                if int(a.node) == int(t.demand_node):
                    by_arrival = True
                    break
            if a.kind == AgentKind.UAV and a.pos_xy is not None:
                nx = env.topology.nodes[int(t.demand_node)]
                d = float(((a.pos_xy[0] - nx.x) ** 2 + (a.pos_xy[1] - nx.y) ** 2) ** 0.5)
                if d <= float(env.cfg.uav_delivery_radius_m):
                    by_arrival = True
                    break
        return bool(
            by_interval or by_risk or by_resolution or by_arrival or not self.state.goals
        )

    def _candidate_tasks(self, env, aid: str) -> List[object]:
        a = env.state.agents[aid]
        tasks = [
            t
            for t in env.state.tasks.values()
            if t.status == TaskStatus.PENDING
        ]
        if a.kind == AgentKind.UAV:
            tasks = [t for t in tasks if t.kind == TaskKind.EMERGENCY]
        return tasks

    def _score_candidates(self, env, aid: str, tasks: List[object]) -> List[Tuple[str, float]]:
        if not tasks:
            return []
        node_emb = self._risk_node_embedding(env)
        feats = []
        tids = []
        task_nodes = []
        for t in tasks:
            dist_norm = float(env._agent_distance_to_task(aid, t) / 3000.0)
            emer = 1.0 if t.kind == TaskKind.EMERGENCY else 0.0
            feats.append([dist_norm, emer])
            tids.append(str(t.task_id))
            task_nodes.append(int(t.demand_node))
        x = torch.tensor(feats, dtype=torch.float32)
        scores = self.allocator(x).detach().cpu().tolist()
        # Manual prior added to neural score for stability before training.
        stable_scores = []
        for i, sc in enumerate(scores):
            dist_norm, emer = feats[i]
            risk_prior = float(node_emb[task_nodes[i]].item()) if node_emb is not None else 0.0
            stable_scores.append(float(sc + 1.2 * emer - 1.8 * dist_norm + 0.25 * risk_prior))
        return list(zip(tids, stable_scores))

    def _risk_node_embedding(self, env) -> Optional[torch.Tensor]:
        n = len(env.topology.nodes)
        if n <= 0:
            return None
        x_rows = []
        for i in range(n):
            node = env.topology.nodes[i]
            hz = env.hazards.node_weather(i)
            x_rows.append(
                [
                    float(hz.rain / max(env.cfg.base_rainfall_mmh, 1e-6)),
                    float(hz.wind / max(env.cfg.base_wind_mps, 1e-6)),
                    float(hz.quake),
                    float(node.slope_norm),
                ]
            )
        x = torch.tensor(x_rows, dtype=torch.float32)
        edges = []
        risks = []
        for src, nbs in env.topology.adjacency.items():
            for dst in nbs:
                if src == dst:
                    continue
                edges.append((int(src), int(dst)))
                k = (min(int(src), int(dst)), max(int(src), int(dst)))
                risks.append(float(env.hazards.last_edge_pstep.get(k, 0.0)))
        if not edges:
            return None
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_risk = torch.tensor(risks, dtype=torch.float32)
        emb, _ = self.risk_gat(x, edge_index, edge_risk=edge_risk)
        # Higher value = lower risk preference (inverse hazard energy proxy).
        return emb.norm(dim=-1)

    def plan(self, env) -> Dict[str, Optional[str]]:
        if self._should_refresh(env):
            goals: Dict[str, Optional[str]] = {}
            used_tasks = set()
            # UAV first to secure emergency workload; then trucks.
            ordered_agents = sorted(
                env.state.agents.keys(),
                key=lambda aid: 0 if env.state.agents[aid].kind == AgentKind.UAV else 1,
            )
            for aid in ordered_agents:
                # Sticky lock: don't switch away from an active target if already engaged.
                prev_tid = self.state.goals.get(aid, None)
                if prev_tid is not None:
                    pt = env.state.tasks.get(str(prev_tid))
                    if pt is not None and pt.status in (TaskStatus.PENDING, TaskStatus.CLAIMED):
                        a = env.state.agents[aid]
                        keep = False
                        if pt.assigned_to is not None and str(pt.assigned_to) == str(aid):
                            keep = True
                        elif a.kind == AgentKind.UAV and pt.kind == TaskKind.EMERGENCY and a.pos_xy is not None:
                            nx = env.topology.nodes[int(pt.demand_node)]
                            d = float(((a.pos_xy[0] - nx.x) ** 2 + (a.pos_xy[1] - nx.y) ** 2) ** 0.5)
                            if d <= float(env.cfg.uav_monitor_radius_m):
                                keep = True
                        if keep and str(prev_tid) not in used_tasks:
                            goals[aid] = str(prev_tid)
                            used_tasks.add(str(prev_tid))
                            continue

                cands = self._candidate_tasks(env, aid)
                scored = self._score_candidates(env, aid, cands)
                if not scored:
                    goals[aid] = None
                    continue
                scored.sort(key=lambda it: it[1], reverse=True)
                picked = None
                for tid, _ in scored:
                    if tid not in used_tasks:
                        picked = tid
                        break
                goals[aid] = picked
                if picked is not None:
                    used_tasks.add(picked)

            self.state.goals = goals
            self.state.step_last_refresh = int(env.state.step_index)
            self.state.resolved_tasks_last = self._resolved_count(env)
        return dict(self.state.goals)
