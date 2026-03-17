from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal

from hetgat_hrl.agents.task_attention import TaskAttentionModule
from hetgat_hrl.core.mdp_spec import AgentKind, TaskKind, TruckAction, UAVAction


class SimpleActorCritic(nn.Module):
    """
    Lightweight actor-critic stub for rebuild stage.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: Tensor) -> Dict[str, Tensor]:
        h = self.encoder(obs)
        return {"logits": self.actor(h), "value": self.critic(h).squeeze(-1)}


class RuleBasedLowLevelPolicy:
    """
    Baseline executable policy that follows assigned tasks.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, high_goals: Optional[Dict[str, Optional[str]]] = None):
        actions: Dict[str, object] = {}
        legal_cache = env.legal_actions()
        full_battery_takeoff_thresh = 0.95

        def _agent_xy(aid: str) -> Tuple[float, float]:
            st = env.state.agents[aid]
            if st.pos_xy is not None:
                return float(st.pos_xy[0]), float(st.pos_xy[1])
            return env._node_xy(int(st.node or 0))

        def _full_speed_to(src: Tuple[float, float], dst: Tuple[float, float], vmax: float) -> Tuple[float, float]:
            dx = float(dst[0] - src[0])
            dy = float(dst[1] - src[1])
            norm = float(np.hypot(dx, dy))
            if norm <= 1e-6:
                return 0.0, 0.0
            return float(dx / norm * vmax), float(dy / norm * vmax)

        for aid, a in env.state.agents.items():
            goal_id: Optional[str] = None
            if high_goals is not None and aid in high_goals and high_goals[aid] is not None:
                goal_id = str(high_goals[aid])
            else:
                g = env._effective_goals.get(str(aid), env._recommended_goals.get(str(aid), None))
                if g is not None:
                    goal_id = str(g)

            if a.kind == AgentKind.TRUCK:
                legal = legal_cache[aid]["neighbors"]
                if not legal:
                    actions[aid] = TruckAction(stay=True)
                    continue
                t = env.state.tasks.get(goal_id) if goal_id is not None else None
                if t is None:
                    actions[aid] = TruckAction(stay=True)
                    continue
                best_nb = min(
                    legal,
                    key=lambda nb: env.topology.shortest_path_distance(
                        int(nb), int(t.demand_node), ignore_blocked=False
                    ),
                )
                actions[aid] = TruckAction(target_node=int(best_nb), stay=False)
                continue

            # UAV execution: strictly follow selected target semantics.
            if a.crashed:
                actions[aid] = UAVAction(vx=0.0, vy=0.0)
                continue

            target_agent = env.state.agents.get(goal_id) if goal_id is not None else None
            target_task = env.state.tasks.get(goal_id) if goal_id is not None else None
            cur_xy = _agent_xy(aid)
            vmax = float(env.cfg.uav_max_speed_mps)

            # Case 1: selected target is a truck (virtual task slot).
            if target_agent is not None and target_agent.kind == AgentKind.TRUCK:
                truck_id = str(goal_id)
                truck_xy = _agent_xy(truck_id)
                dist = float(np.hypot(cur_xy[0] - truck_xy[0], cur_xy[1] - truck_xy[1]))
                bind_radius = float(getattr(env.cfg, "uav_bind_radius_m", 50.0))

                # If currently attached to a different truck, detach first.
                if a.follow_target is not None and str(a.follow_target) != truck_id:
                    actions[aid] = UAVAction(takeoff=True)
                    continue
                if dist <= bind_radius:
                    actions[aid] = UAVAction(bind_truck_id=truck_id)
                else:
                    vx, vy = _full_speed_to(cur_xy, truck_xy, vmax=vmax)
                    actions[aid] = UAVAction(vx=vx, vy=vy)
                continue

            # Case 2: selected target is a real task (UAV can only execute emergency).
            if target_task is not None and target_task.kind == TaskKind.EMERGENCY:
                if a.follow_target is not None and float(a.battery) >= full_battery_takeoff_thresh:
                    actions[aid] = UAVAction(takeoff=True)
                    continue
                if a.follow_target is not None:
                    actions[aid] = UAVAction(vx=0.0, vy=0.0)
                    continue
                task_xy = env._node_xy(int(target_task.demand_node))
                vx, vy = _full_speed_to(cur_xy, task_xy, vmax=vmax)
                actions[aid] = UAVAction(vx=vx, vy=vy)
                continue

            # No valid selected goal: keep stable.
            actions[aid] = UAVAction(vx=0.0, vy=0.0)

        return actions


class AttentionGuidedLowLevelPolicy(RuleBasedLowLevelPolicy):
    """
    Rule-based execution with attention-driven goal selection in the forward chain.
    """

    def __init__(
        self,
        obs_dim: int = 20,
        hidden_dim: int = 64,
        seed: int = 0,
        use_hetgat: bool = True,
        enable_rth_mask: bool = True,
    ):
        super().__init__(seed=seed)
        torch.manual_seed(seed)
        self.use_hetgat = bool(use_hetgat)
        self.enable_rth_mask = bool(enable_rth_mask)
        self.attn = TaskAttentionModule(
            agent_dim=obs_dim,
            task_dim=5,
            hidden_dim=hidden_dim,
        )
        self.last_attention_summary: Dict[str, float] = {}

    def infer_attention_goals(
        self,
        env,
        fallback_goals: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[Dict[str, Optional[str]], Dict[str, float]]:
        obs = env.observe()
        task_mat = env.observe_task_matrix()
        task_slots = env.observe_task_slots() if hasattr(env, "observe_task_slots") else {}
        goals: Dict[str, Optional[str]] = {}
        entropies: List[float] = []
        rank_map: Dict[str, List[Tuple[float, Optional[str]]]] = {}
        forced_rth_mask_count = 0
        use_hetgat = bool(getattr(env.cfg, "use_hetgat", self.use_hetgat))
        enable_rth_mask = bool(
            getattr(env.cfg, "enable_rth_mask", self.enable_rth_mask)
        )

        def _agent_xy(aid: str) -> Tuple[float, float]:
            st = env.state.agents[aid]
            if st.pos_xy is not None:
                return float(st.pos_xy[0]), float(st.pos_xy[1])
            if hasattr(env, "_node_xy"):
                return env._node_xy(int(st.node or 0))
            return 0.0, 0.0

        def _nearest_truck_for(aid: str) -> Tuple[Optional[str], float]:
            ax, ay = _agent_xy(aid)
            best_id: Optional[str] = None
            best_d = float("inf")
            for tid, ts in env.state.agents.items():
                if ts.kind != AgentKind.TRUCK:
                    continue
                tx, ty = _agent_xy(str(tid))
                d = float(np.hypot(ax - tx, ay - ty))
                if d < best_d:
                    best_d = d
                    best_id = str(tid)
            return best_id, best_d

        def _required_rth_battery(agent_state, dist_to_truck: float) -> float:
            # Conservative per-meter energy estimate from configured physics coefficients.
            base_discharge_per_m = float(max(getattr(env.cfg, "uav_flight_discharge_per_m", 1e-6), 1e-6))
            headwind_coeff = float(max(getattr(env.cfg, "uav_headwind_energy_coeff", 0.04), 0.0))
            rain_coeff = float(max(getattr(env.cfg, "uav_rain_energy_coeff", 0.02), 0.0))
            base_wind = float(max(getattr(env.cfg, "base_wind_mps", 0.0), 0.0))
            base_rain = float(max(getattr(env.cfg, "base_rainfall_mmh", 0.0), 0.0))
            cargo_unit_kg = float(max(getattr(env.cfg, "cargo_unit_kg", 40.0), 1e-6))
            m_load_kg = float(max(getattr(agent_state, "cargo", 0.0), 0.0)) * cargo_unit_kg
            load_factor = 1.0 + 0.018 * m_load_kg
            weather_factor = 1.0 + headwind_coeff * base_wind + rain_coeff * base_rain
            safe_discharge_rate = base_discharge_per_m * weather_factor * load_factor
            return float(max(0.0, dist_to_truck) * safe_discharge_rate)

        for aid in env.state.agents:
            obs_vec = torch.tensor(obs[aid], dtype=torch.float32).view(1, 1, -1)
            task_tensor = torch.tensor(task_mat[aid], dtype=torch.float32).view(1, -1, 5)
            slot_ids = task_slots.get(aid, [None] * len(task_mat[aid]))
            mask_vals = [1.0 if sid is not None else 0.0 for sid in slot_ids]
            agent_state = env.state.agents[aid]
            if agent_state.kind == AgentKind.UAV and bool(agent_state.crashed):
                # Crashed UAV should not produce actionable goals.
                goals[aid] = None
                rank_map[aid] = []
                continue

            # Dynamic attention masking for alive UAV:
            # 1) cargo-empty interception
            # 2) return-to-truck feasibility interception based on energy budget
            if agent_state.kind == AgentKind.UAV and not bool(agent_state.crashed):
                nearest_truck_id, dist_to_truck = _nearest_truck_for(aid)
                force_rth = bool(float(agent_state.cargo) <= 0.0)
                if (not force_rth) and nearest_truck_id is not None and np.isfinite(dist_to_truck):
                    required_battery = _required_rth_battery(agent_state, float(dist_to_truck))
                    # Safety margin: 1.2x to avoid edge-of-failure behaviors.
                    if float(agent_state.battery) < float(required_battery * 1.2):
                        force_rth = True
                if enable_rth_mask and force_rth:
                    forced_rth_mask_count += 1
                    for i, sid in enumerate(slot_ids):
                        sid_str = "" if sid is None else str(sid)
                        if sid is None:
                            mask_vals[i] = 0.0
                            continue
                        if sid_str.startswith("truck"):
                            # Keep only nearest truck slot visible.
                            mask_vals[i] = 1.0 if (nearest_truck_id is not None and sid_str == nearest_truck_id) else 0.0
                        else:
                            mask_vals[i] = 0.0
                    # Safety fallback: if nearest truck slot not present in slots, keep any truck slot.
                    if float(sum(mask_vals)) <= 0.0:
                        for i, sid in enumerate(slot_ids):
                            sid_str = "" if sid is None else str(sid)
                            if sid is not None and sid_str.startswith("truck"):
                                mask_vals[i] = 1.0
            task_mask = torch.tensor(mask_vals, dtype=torch.float32).view(1, -1)

            if use_hetgat:
                _, weights = self.attn(obs_vec, task_tensor, task_mask=task_mask)
                w = weights[0, 0]  # [T]
            else:
                # Baseline_MAPPO-like fallback: no graph attention,
                # simple feature pooling/linear scoring on task slots.
                t = task_tensor[0]  # [T,5]
                dist_norm = t[:, 2]
                emer = t[:, 3]
                is_rec = t[:, 4]
                score = (-1.2 * dist_norm) + (0.45 * emer) + (0.15 * is_rec)
                score = score + (task_mask[0] - 1.0) * 1e9
                w = torch.softmax(score, dim=0)
            top_idx = int(torch.argmax(w).item())
            prob = torch.clamp(w, min=1e-9)
            entropy = float((-prob * torch.log(prob)).sum().item())
            entropies.append(entropy)
            ranked: List[Tuple[float, Optional[str]]] = []
            for i, sid in enumerate(slot_ids):
                if sid is None:
                    continue
                tid = str(sid)
                t = env.state.tasks.get(tid)
                if t is not None:
                    if (
                        env.state.agents[aid].kind == AgentKind.UAV
                        and t.kind != TaskKind.EMERGENCY
                    ):
                        continue
                    ranked.append((float(w[i].item()), tid))
                    continue
                # Virtual slot: truck agent id (e.g., "truck_0")
                ag = env.state.agents.get(tid)
                if ag is not None and ag.kind == AgentKind.TRUCK:
                    ranked.append((float(w[i].item()), tid))
            ranked.sort(key=lambda x: x[0], reverse=True)
            # Keep top_idx computation for diagnostics consistency.
            _ = top_idx
            rank_map[aid] = ranked

        used: set = set()
        ordered_agents = sorted(
            env.state.agents.keys(),
            key=lambda a: 0 if env.state.agents[a].kind == AgentKind.UAV else 1,
        )
        for aid in ordered_agents:
            st = env.state.agents[aid]
            if st.kind == AgentKind.UAV and bool(st.crashed):
                goals[aid] = None
                continue
            picked: Optional[str] = None
            for _, tid in rank_map.get(aid, []):
                if tid in env.state.tasks and tid in used:
                    continue
                picked = tid
                break
            if picked is None and fallback_goals is not None:
                fb = fallback_goals.get(aid, None)
                if fb is not None:
                    fb = str(fb)
                    t = env.state.tasks.get(fb)
                    if t is not None:
                        if env.state.agents[aid].kind == AgentKind.UAV and t.kind != TaskKind.EMERGENCY:
                            fb = None
                    if fb is not None and ((fb not in env.state.tasks) or (fb not in used)):
                        picked = fb
            goals[aid] = picked
            if picked is not None and picked in env.state.tasks:
                used.add(picked)

        summary = {
            "attention_entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
            "attention_agents": float(len(entropies)),
            "forced_rth_mask_count": float(forced_rth_mask_count),
        }
        self.last_attention_summary = summary
        return goals, summary


def build_task_feature_matrix(env) -> Dict[str, List[float]]:
    """
    Per-agent flattened task features [dx, dy, dist, emer_flag, is_rec, ...].
    """
    out: Dict[str, List[float]] = {}
    tasks = [
        t
        for t in env.state.tasks.values()
        if t.status.name in ("PENDING", "CLAIMED")
    ]
    for aid in env.state.agents:
        feats: List[float] = []
        rec_tid = env._effective_goals.get(str(aid), env._recommended_goals.get(str(aid), None))
        for t in tasks:
            dx, dy, d = env._agent_task_rel(aid, t)
            feats.extend(
                [
                    float(dx),
                    float(dy),
                    float(np.clip(d, 0.0, 1.0)),
                    1.0 if t.kind.name == "EMERGENCY" else 0.0,
                    1.0 if (rec_tid is not None and str(t.task_id) == str(rec_tid)) else 0.0,
                ]
            )
        out[aid] = feats
    return out


class _LowLevelGaussianActorCritic(nn.Module):
    """Continuous Gaussian actor-critic for UAV low-level control."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, 2)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((2,), -0.7))

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        h = self.encoder(obs)
        mu = self.mu_head(h)
        value = self.value_head(h).squeeze(-1)
        log_std = self.log_std.unsqueeze(0).expand_as(mu)
        return mu, log_std, value


class LearnableLowLevelPolicy:
    """
    Learnable UAV low-level policy.
    - UAV: Gaussian continuous control [vx, vy] with tanh squashing.
    - Truck: keep topological discrete heuristic execution.
    """

    def __init__(
        self,
        seed: int = 0,
        obs_dim: int = 12,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.model = _LowLevelGaussianActorCritic(obs_dim=self.obs_dim, hidden_dim=hidden_dim).to(
            self.device
        )
        self._truck_policy = RuleBasedLowLevelPolicy(seed=seed)
        self._eps = 1e-6

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def _agent_xy(self, env, aid: str) -> Tuple[float, float]:
        st = env.state.agents[aid]
        if st.pos_xy is not None:
            return float(st.pos_xy[0]), float(st.pos_xy[1])
        return env._node_xy(int(st.node or 0))

    def _resolve_goal_xy(
        self, env, aid: str, goal_id: Optional[str]
    ) -> Tuple[float, float, float, float, float]:
        """
        Returns:
            (goal_x, goal_y, goal_is_truck, goal_is_emergency, goal_valid)
        """
        if goal_id is None:
            ax, ay = self._agent_xy(env, aid)
            return float(ax), float(ay), 0.0, 0.0, 0.0
        tid = str(goal_id)
        ag = env.state.agents.get(tid)
        if ag is not None and ag.kind == AgentKind.TRUCK:
            tx, ty = self._agent_xy(env, tid)
            return float(tx), float(ty), 1.0, 0.0, 1.0
        t = env.state.tasks.get(tid)
        if t is not None:
            n = env.topology.nodes[int(t.demand_node)]
            emer = 1.0 if t.kind == TaskKind.EMERGENCY else 0.0
            return float(n.x), float(n.y), 0.0, float(emer), 1.0
        ax, ay = self._agent_xy(env, aid)
        return float(ax), float(ay), 0.0, 0.0, 0.0

    def _build_uav_obs(self, env, aid: str, goal_id: Optional[str]) -> np.ndarray:
        s = env.state.agents[aid]
        ax, ay = self._agent_xy(env, aid)
        gx, gy, goal_is_truck, goal_is_emergency, goal_valid = self._resolve_goal_xy(
            env, aid, goal_id
        )
        dx = float(gx - ax)
        dy = float(gy - ay)
        dist = float(np.hypot(dx, dy))
        vmax = float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0), 1e-6))
        wx, wy = env.hazards.wind_vector_at((float(ax), float(ay)))
        vx = float(s.vel_xy[0]) if s.vel_xy is not None else 0.0
        vy = float(s.vel_xy[1]) if s.vel_xy is not None else 0.0
        cargo_cap = float(max(getattr(env.cfg, "uav_cargo_capacity_units", 1.0), 1e-6))
        obs = np.array(
            [
                float(np.clip(dx / 3000.0, -1.0, 1.0)),
                float(np.clip(dy / 3000.0, -1.0, 1.0)),
                float(np.clip(dist / 3000.0, 0.0, 1.0)),
                float(np.clip(wx / 20.0, -1.0, 1.0)),
                float(np.clip(wy / 20.0, -1.0, 1.0)),
                float(np.clip(vx / vmax, -1.0, 1.0)),
                float(np.clip(vy / vmax, -1.0, 1.0)),
                float(np.clip(s.battery, 0.0, 1.0)),
                float(np.clip(float(s.cargo) / cargo_cap, 0.0, 1.0)),
                float(1.0 if s.follow_target is not None else 0.0),
                float(goal_is_truck),
                float(goal_is_emergency if goal_valid > 0.0 else 0.0),
            ],
            dtype=np.float32,
        )
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"UAV obs dim mismatch: got {obs.shape[0]}, expected {self.obs_dim}")
        return obs

    def _squashed_log_prob(self, mu: Tensor, log_std: Tensor, raw_action: Tensor) -> Tensor:
        std = torch.exp(log_std).clamp(min=1e-4, max=2.0)
        dist = Normal(mu, std)
        squashed = torch.tanh(raw_action)
        logp = dist.log_prob(raw_action) - torch.log(1.0 - squashed.pow(2) + self._eps)
        return logp.sum(dim=-1)

    def evaluate_actions(
        self, obs_batch: Tensor, raw_action_batch: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            (log_prob, entropy, value)
        """
        mu, log_std, value = self.model(obs_batch)
        std = torch.exp(log_std).clamp(min=1e-4, max=2.0)
        dist = Normal(mu, std)
        squashed = torch.tanh(raw_action_batch)
        logp = dist.log_prob(raw_action_batch) - torch.log(1.0 - squashed.pow(2) + self._eps)
        logp = logp.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return logp, entropy, value

    def act(
        self,
        env,
        high_goals: Optional[Dict[str, Optional[str]]] = None,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, object], List[Dict[str, float]]]:
        """
        Returns:
            actions, step_records_for_ppo
        """
        base_actions = self._truck_policy.act(env, high_goals=high_goals)
        actions: Dict[str, object] = {}
        records: List[Dict[str, float]] = []

        # Keep truck actions from topology-aware heuristic.
        for aid, s in env.state.agents.items():
            if s.kind == AgentKind.TRUCK:
                actions[aid] = base_actions.get(aid, TruckAction(stay=True))

        for aid, s in env.state.agents.items():
            if s.kind != AgentKind.UAV:
                continue
            if bool(s.crashed):
                actions[aid] = UAVAction(vx=0.0, vy=0.0)
                continue

            goal_id: Optional[str] = None
            if high_goals is not None and high_goals.get(aid, None) is not None:
                goal_id = str(high_goals[aid])
            else:
                g = env._effective_goals.get(str(aid), env._recommended_goals.get(str(aid), None))
                if g is not None:
                    goal_id = str(g)

            goal_agent = env.state.agents.get(str(goal_id)) if goal_id is not None else None
            goal_is_truck = bool(goal_agent is not None and goal_agent.kind == AgentKind.TRUCK)
            goal_task = env.state.tasks.get(str(goal_id)) if goal_id is not None else None

            # Discrete binding/takeoff interface is still handled at interface boundary.
            if s.follow_target is not None:
                if goal_is_truck and str(s.follow_target) == str(goal_id):
                    actions[aid] = UAVAction(vx=0.0, vy=0.0)
                    continue
                actions[aid] = UAVAction(takeoff=True)
                continue

            if goal_is_truck:
                cur_xy = self._agent_xy(env, aid)
                truck_xy = self._agent_xy(env, str(goal_id))
                dist = float(np.hypot(cur_xy[0] - truck_xy[0], cur_xy[1] - truck_xy[1]))
                bind_radius = float(getattr(env.cfg, "uav_bind_radius_m", 50.0))
                if dist <= bind_radius:
                    actions[aid] = UAVAction(bind_truck_id=str(goal_id))
                    continue

            # Continuous neural control.
            obs_np = self._build_uav_obs(env, aid, goal_id)
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device).view(1, -1)
            with torch.no_grad():
                mu, log_std, value = self.model(obs_t)
                std = torch.exp(log_std).clamp(min=1e-4, max=2.0)
                if deterministic:
                    raw = mu
                else:
                    raw = mu + std * torch.randn_like(std)
                squashed = torch.tanh(raw)
                logp = self._squashed_log_prob(mu, log_std, raw)

            vmax = float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0), 1e-6))
            cmd = squashed.squeeze(0).cpu().numpy()
            vx_cmd = float(np.clip(cmd[0], -1.0, 1.0) * vmax)
            vy_cmd = float(np.clip(cmd[1], -1.0, 1.0) * vmax)
            actions[aid] = UAVAction(vx=vx_cmd, vy=vy_cmd)
            records.append(
                {
                    "aid": str(aid),
                    "obs": obs_np.tolist(),
                    "raw_action": raw.squeeze(0).cpu().numpy().tolist(),
                    "old_logp": float(logp.item()),
                    "old_value": float(value.item()),
                    "reward": 0.0,
                    "done": 0.0,
                    "return": 0.0,
                }
            )

            # Optional safety: UAV should not be assigned to normal tasks.
            if goal_task is not None and goal_task.kind != TaskKind.EMERGENCY:
                actions[aid] = UAVAction(vx=0.0, vy=0.0)

        return actions, records
