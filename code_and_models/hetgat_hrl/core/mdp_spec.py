from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Tuple, Union


class AgentKind(str, Enum):
    TRUCK = "truck"
    UAV = "uav"


class TaskKind(str, Enum):
    NORMAL = "normal"
    EMERGENCY = "emergency"


class TaskStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    DELIVERED = "delivered"
    FAILED = "failed"


@dataclass(frozen=True)
class EnvConfig:
    # Legacy-compatible names (from previous framework)
    n_nodes: int = 40
    n_trucks: int = 2
    n_uavs: int = 3
    n_normal_tasks: int = 6
    n_emergency_tasks: int = 4
    dt: float = 20.0

    phase: str = "S"
    scenario: str = "B"  # A:平稳无灾害, B:风雨地震, C:风雨地震+通信屏蔽
    map_complexity: str = "M"
    seed: int = 0
    num_nodes: int = 40
    num_edges: int = 64
    avg_degree_min: float = 3.2
    avg_degree_max: float = 3.8
    map_source: str = "disaster_map"  # disaster_map | synthetic | osm_dem
    map_size_m: float = 5000.0
    min_node_spacing_m: float = 300.0
    redundant_edge_radius_m: float = 1000.0
    redundant_edge_prob: float = 0.8
    osm_graphml_path: str = ""
    dem_npy_path: str = ""
    num_trucks: int = 2
    num_uavs: int = 3
    num_normal_tasks: int = 6
    num_emergency_tasks: int = 4
    max_steps: int = 300
    dt_seconds: float = 20.0
    hrl_interval: int = 5
    risk_spike_threshold: float = 0.30
    # Truck physics
    truck_speed_mps: float = 8.0
    ignore_payload_constraints: bool = True
    truck_payload_kg: float = 0.0
    # Cargo unit conversion and capacity-demand scale:
    # 1.0 abstract cargo unit == cargo_unit_kg kilograms.
    cargo_unit_kg: float = 40.0
    truck_cargo_capacity_units: float = 40.0  # 1600 kg
    uav_cargo_capacity_units: float = 1.0     # 40 kg
    task_demand_normal_units: float = 8.0     # 320 kg
    task_demand_emergency_units: float = 1.0  # 40 kg
    replenish_freeze_steps: int = 2
    # UAV physics
    uav_max_speed_mps: float = 17.0
    uav_payload_kg: float = 0.0
    uav_payload_capacity_kg: float = 10.0
    uav_max_sortie_m: float = 3000.0
    uav_battery_init: float = 0.70
    uav_bind_radius_m: float = 170.0
    uav_max_followers_per_truck: int = 2
    uav_force_takeoff_battery_threshold: float = 0.98
    uav_idle_discharge_per_step: float = 0.02
    uav_flight_discharge_per_step: float = 0.04
    uav_flight_discharge_per_m: float = 0.0001
    uav_headwind_energy_coeff: float = 0.04
    uav_rain_energy_coeff: float = 0.02
    uav_energy_norm_wind_mps: float = 10.0
    uav_energy_norm_rain_mmh: float = 30.0
    uav_charge_rate_per_step: float = 0.085
    uav_crash_penalty: float = -5.0
    # UAV local sensing / auto-approach
    uav_monitor_radius_m: float = 340.0
    # Ablation toggles
    use_hetgat: bool = True
    enable_rth_mask: bool = True
    # Rule-based takeover switches (disable for pure RL exploration).
    uav_auto_approach_enabled: bool = False
    uav_monitor_snap_enabled: bool = False
    # Backward-compatible aliases used by some training loops.
    enable_auto_approach: bool = False
    enable_monitor_snap: bool = False
    # Service (unloading) rounds
    unload_rounds_normal: int = 10
    unload_rounds_emergency: int = 10
    unload_rounds_uav: int = 1
    # Reward
    reward_step_penalty: float = -0.01
    reward_invalid_action: float = -0.05
    reward_idle_with_task: float = -0.08
    reward_delivery_normal: float = 5.0
    reward_delivery_emergency: float = 10.0
    # Docking reward only for meaningful low-battery recovery.
    reward_docking_low_battery: float = 1.0
    docking_reward_battery_threshold: float = 0.5
    reward_uav_discover_blocked_edge: float = 0.5
    reward_pickup: float = 1.0
    reward_delivery_shared: float = 5.0
    penalty_timeout_normal: float = -0.5
    penalty_timeout_emergency: float = -1.3
    # Optional PBRS hook
    use_pbrs: bool = False
    pbrs_scale: float = 1.0
    pbrs_gamma: float = 0.99
    pbrs_distance_norm_m: float = 3000.0
    # Weather / comms
    stochastic_weather: bool = True
    weather_num_vortices: int = 3
    weather_num_rain_stations: int = 5
    base_rainfall_mmh: float = 12.0
    base_wind_mps: float = 6.0
    rain_idw_power: float = 2.0
    rain_idw_eps_m: float = 1.0
    weather_osc_amp: float = 0.18
    weather_osc_w1: float = 0.05
    weather_osc_w2: float = 0.03
    wind_vortex_strength_min: float = -1.0
    wind_vortex_strength_max: float = 1.0
    wind_vortex_radius_min_m: float = 300.0
    wind_vortex_radius_max_m: float = 900.0
    wind_vortex_scale_mps: float = 1.8
    # Percolation/logistic collapse model
    logistic_beta0: float = -3.4
    logistic_beta_slope: float = 1.4
    logistic_beta_rain: float = 1.2
    logistic_beta_quake: float = 1.8
    logistic_beta_sr: float = 3.0
    logistic_beta_re: float = 1.1
    logistic_beta_vbase: float = 1.0
    # Additional edge-structure effects (explicit terms).
    logistic_beta_bldg: float = 0.9
    logistic_beta_infra: float = 1.3
    logistic_beta_length: float = 0.8
    logistic_beta_lr: float = 0.9  # length-rain interaction
    lambda_aggressive: float = 0.045
    lambda_residual: float = 0.001
    percolation_lock_threshold: float = 0.35
    lambda_warmup_steps: int = 30
    lambda_warmup_min_factor: float = 0.35
    stochastic_block_max_prob: float = 0.85
    comm_block_prob: float = 0.10
    enable_comm_blackout: bool = False
    # Delivery radius for UAV
    uav_delivery_radius_m: float = 40.0

    def __post_init__(self) -> None:
        sc = str(self.scenario).upper().strip()
        if sc == "A":
            object.__setattr__(self, "stochastic_weather", False)
            object.__setattr__(self, "base_rainfall_mmh", 0.0)
            object.__setattr__(self, "base_wind_mps", 0.0)
            object.__setattr__(self, "lambda_aggressive", 0.0)
            object.__setattr__(self, "lambda_residual", 0.0)
            object.__setattr__(self, "comm_block_prob", 0.0)
            object.__setattr__(self, "enable_comm_blackout", False)
        elif sc == "B":
            object.__setattr__(self, "stochastic_weather", True)
            object.__setattr__(self, "comm_block_prob", 0.0)
            object.__setattr__(self, "enable_comm_blackout", False)
        elif sc == "C":
            object.__setattr__(self, "stochastic_weather", True)
            object.__setattr__(self, "enable_comm_blackout", True)
            if float(self.comm_block_prob) <= 0.0:
                object.__setattr__(self, "comm_block_prob", 0.10)

        # 复杂度预设（当前先固化 M）
        cx = str(self.map_complexity).upper().strip()
        if cx == "M":
            object.__setattr__(self, "map_size_m", 5000.0)
            object.__setattr__(self, "num_nodes", 40)
            object.__setattr__(self, "n_nodes", 40)
            object.__setattr__(self, "min_node_spacing_m", 300.0)
            object.__setattr__(self, "redundant_edge_radius_m", 1000.0)
            object.__setattr__(self, "redundant_edge_prob", 0.8)

        # Keep takeover switches synchronized with aliases.
        if bool(self.enable_auto_approach):
            object.__setattr__(self, "uav_auto_approach_enabled", True)
        if bool(self.enable_monitor_snap):
            object.__setattr__(self, "uav_monitor_snap_enabled", True)
        if bool(self.uav_auto_approach_enabled):
            object.__setattr__(self, "enable_auto_approach", True)
        if bool(self.uav_monitor_snap_enabled):
            object.__setattr__(self, "enable_monitor_snap", True)

        # Runtime-configurable delta t (physics and MDP share the same dt).
        dt_s = float(self.dt_seconds)
        if dt_s <= 0.0:
            dt_s = float(self.dt)
        if dt_s <= 0.0:
            raise ValueError(f"dt_seconds must be > 0, got dt_seconds={self.dt_seconds}, dt={self.dt}")
        object.__setattr__(self, "dt_seconds", dt_s)
        object.__setattr__(self, "dt", dt_s)


@dataclass
class AgentRuntimeState:
    agent_id: str
    kind: AgentKind
    node: Optional[int] = None
    pos_xy: Optional[Tuple[float, float]] = None
    vel_xy: Optional[Tuple[float, float]] = None
    battery: float = 1.0
    crashed: bool = False
    transit: Optional[Tuple[int, int, float]] = None
    follow_target: Optional[str] = None
    sortie_distance_m: float = 0.0
    lifetime_distance_m: float = 0.0
    cargo: float = 0.0
    replenish_timer: int = 0


@dataclass
class DeliveryTask:
    task_id: str
    kind: TaskKind
    demand_node: int
    deadline_step: int
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    in_service_by: Optional[str] = None
    service_remaining: int = 0
    demand_left: float = 0.0
    delivered_by: Optional[str] = None
    delivered_step: Optional[int] = None


@dataclass
class HazardSnapshot:
    rainfall_mean: float = 0.0
    wind_mean: float = 0.0
    blocked_ratio: float = 0.0
    risk_spike: bool = False
    epicenter_node: Optional[int] = None


@dataclass
class JointState:
    step_index: int
    agents: Dict[str, AgentRuntimeState] = field(default_factory=dict)
    tasks: Dict[str, DeliveryTask] = field(default_factory=dict)
    hazard: HazardSnapshot = field(default_factory=HazardSnapshot)
    done: bool = False


@dataclass(frozen=True)
class TruckAction:
    # For graph-constrained movement.
    target_node: Optional[int] = None
    stay: bool = False


@dataclass(frozen=True)
class UAVAction:
    # Continuous control + optional mode command.
    vx: float = 0.0
    vy: float = 0.0
    bind_truck_id: Optional[str] = None
    takeoff: bool = False


Action = Union[TruckAction, UAVAction]
JointAction = Mapping[str, Action]


@dataclass
class StepResult:
    state: JointState
    rewards: Dict[str, float]
    terminated: bool
    truncated: bool
    info: Dict[str, object]


class HeteroDisasterMDP(ABC):
    """
    Low-level MDP + high-level SMDP trigger interface.
    """

    cfg: EnvConfig

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> JointState:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: JointAction) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> Dict[str, List[float]]:
        """Per-agent observation vectors."""
        raise NotImplementedError

    @abstractmethod
    def observe_task_matrix(self) -> Dict[str, List[List[float]]]:
        """
        Per-agent task feature matrix with fixed slots:
        shape = [task_attention_slots, task_feat_dim]
        """
        raise NotImplementedError

    @abstractmethod
    def legal_actions(self) -> Dict[str, object]:
        """Per-agent legality descriptor or mask."""
        raise NotImplementedError

    @abstractmethod
    def should_trigger_hrl(self) -> bool:
        """SMDP decision trigger: interval or risk spike."""
        raise NotImplementedError
