from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from hetgat_hrl.core.mdp_spec import EnvConfig
from hetgat_hrl.core.topology import GraphTopology


@dataclass
class NodeHazard:
    rain: float
    wind: float
    quake: float


class GlobalHazardField:
    """
    Continuous geophysical field over 2D space.
    - Quake: single-source hypocentral attenuation (static residual field).
    - Wind: parametric vortex storm (moving center, Rankine-like profile).
    - Rain: eyewall Gaussian rainband coupled to storm center.
    """

    def __init__(
        self,
        map_bounds: Tuple[float, float, float, float],
        dt_seconds: float,
        seed: int,
        base_rainfall_mmh: float,
        base_wind_mps: float,
        enable_weather: bool = True,
    ):
        self.rng = np.random.default_rng(seed)
        self.dt = float(max(dt_seconds, 1e-6))
        self.x_min, self.x_max, self.y_min, self.y_max = map_bounds
        self.enable_weather = bool(enable_weather)

        sx = float(max(self.x_max - self.x_min, 1.0))
        sy = float(max(self.y_max - self.y_min, 1.0))
        diag = float(np.hypot(sx, sy))
        margin = float(0.20 * diag)

        # 1) Quake: single epicenter + hypocentral attenuation.
        self.quake_epicenter = (
            float(self.rng.uniform(self.x_min - margin, self.x_max + margin)),
            float(self.rng.uniform(self.y_min - margin, self.y_max + margin)),
        )
        self.quake_depth = 300.0
        self.quake_max_intensity = 1.0 if self.enable_weather else 0.0
        self.quake_gamma = 1.5

        # 2) Storm: moving vortex center.
        cx_lo, cx_hi = self.x_min + 0.15 * sx, self.x_max - 0.15 * sx
        cy_lo, cy_hi = self.y_min + 0.15 * sy, self.y_max - 0.15 * sy
        if cx_lo >= cx_hi:
            cx_lo, cx_hi = self.x_min, self.x_max
        if cy_lo >= cy_hi:
            cy_lo, cy_hi = self.y_min, self.y_max
        self.storm_center = np.array(
            [
                float(self.rng.uniform(cx_lo, cx_hi)),
                float(self.rng.uniform(cy_lo, cy_hi)),
            ],
            dtype=np.float64,
        )

        drift_speed = float(
            5.0 if base_wind_mps <= 0.0 else np.clip(0.6 * base_wind_mps, 2.0, 8.0)
        )
        drift_theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.storm_velocity = np.array(
            [drift_speed * np.cos(drift_theta), drift_speed * np.sin(drift_theta)],
            dtype=np.float64,
        )
        self.boundary_pad = float(0.25 * diag)

        self.rmw = float(np.clip(0.08 * diag, 250.0, 900.0))
        self.v_max = float(np.clip(max(base_wind_mps, 0.0) * 3.8, 0.0, 30.0))
        self.wind_decay_alpha = 0.55
        self.inflow_angle = float(np.deg2rad(20.0))

        # 3) Rainband coupled to storm.
        self.rain_max = float(np.clip(max(base_rainfall_mmh, 0.0) * 2.5, 0.0, 35.0))
        self.rain_sigma = float(np.clip(0.06 * diag, 180.0, 420.0))

    def step(self) -> None:
        if not self.enable_weather:
            return
        self.storm_center = self.storm_center + self.storm_velocity * self.dt
        # Smooth reflection at expanded boundary so storm keeps traversing map.
        low = np.array(
            [self.x_min - self.boundary_pad, self.y_min - self.boundary_pad],
            dtype=np.float64,
        )
        high = np.array(
            [self.x_max + self.boundary_pad, self.y_max + self.boundary_pad],
            dtype=np.float64,
        )
        for i in range(2):
            if self.storm_center[i] < low[i]:
                self.storm_center[i] = low[i] + (low[i] - self.storm_center[i])
                self.storm_velocity[i] *= -1.0
            elif self.storm_center[i] > high[i]:
                self.storm_center[i] = high[i] - (self.storm_center[i] - high[i])
                self.storm_velocity[i] *= -1.0

    def get_hazard_at(self, x: float, y: float) -> Tuple[float, float, float, float, float]:
        # 1) Quake attenuation from single hypocenter.
        dx_q = float(x - self.quake_epicenter[0])
        dy_q = float(y - self.quake_epicenter[1])
        r_epi = float(np.hypot(dx_q, dy_q))
        r_hypo = float(np.hypot(r_epi, self.quake_depth))
        if r_hypo <= 1e-9:
            quake = float(self.quake_max_intensity)
        else:
            quake = float(
                self.quake_max_intensity
                * (self.quake_depth / r_hypo) ** self.quake_gamma
            )
        quake = float(np.clip(quake, 0.0, 1.0))

        # 2) Parametric vortex wind.
        dx_s = float(x - self.storm_center[0])
        dy_s = float(y - self.storm_center[1])
        r = float(np.hypot(dx_s, dy_s))
        if (not self.enable_weather) or self.v_max <= 1e-9 or r <= 1e-6:
            vx, vy, v_mag = 0.0, 0.0, 0.0
        else:
            if r <= self.rmw:
                v_prof = float(self.v_max * (r / max(self.rmw, 1e-6)))
            else:
                v_prof = float(self.v_max * (self.rmw / max(r, 1e-6)) ** self.wind_decay_alpha)
            theta = float(np.arctan2(dy_s, dx_s))
            # CCW tangential + inward inflow angle.
            wind_dir = theta + 0.5 * np.pi + self.inflow_angle
            vx = float(v_prof * np.cos(wind_dir) + 0.5 * self.storm_velocity[0])
            vy = float(v_prof * np.sin(wind_dir) + 0.5 * self.storm_velocity[1])
            v_mag = float(np.hypot(vx, vy))

        # 3) Eyewall Gaussian rainband around RMW.
        if (not self.enable_weather) or self.rain_max <= 1e-9:
            rain = 0.0
        else:
            rain = float(
                self.rain_max
                * np.exp(-((r - self.rmw) ** 2) / (2.0 * max(self.rain_sigma**2, 1e-6)))
            )
        rain = float(max(rain, 0.0))
        return quake, v_mag, vx, vy, rain


class DynamicHazardField:
    """
    Wrapper that keeps existing env API stable while switching internals to
    a continuous global physical hazard field.
    """

    def __init__(
        self,
        topo: GraphTopology,
        seed: int,
        stochastic_weather: bool = True,
        cfg: Optional[EnvConfig] = None,
    ):
        self.topo = topo
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.stochastic_weather = bool(stochastic_weather)
        self.step_index = 0

        self.base_rainfall_mmh = float(cfg.base_rainfall_mmh if cfg is not None else 12.0)
        self.base_wind_mps = float(cfg.base_wind_mps if cfg is not None else 6.0)

        xs = np.array([n.x for n in self.topo.nodes.values()], dtype=np.float64)
        ys = np.array([n.y for n in self.topo.nodes.values()], dtype=np.float64)
        x_min = float(xs.min()) if xs.size else 0.0
        x_max = float(xs.max()) if xs.size else 3000.0
        y_min = float(ys.min()) if ys.size else 0.0
        y_max = float(ys.max()) if ys.size else 3000.0

        dt_s = float(cfg.dt_seconds if cfg is not None else 20.0)
        enable_weather = bool(self.stochastic_weather and (self.base_wind_mps > 0.0 or self.base_rainfall_mmh > 0.0))
        self.field = GlobalHazardField(
            map_bounds=(x_min, x_max, y_min, y_max),
            dt_seconds=dt_s,
            seed=seed + 1000,
            base_rainfall_mmh=self.base_rainfall_mmh,
            base_wind_mps=self.base_wind_mps,
            enable_weather=enable_weather,
        )

        self.node_hazard: Dict[int, NodeHazard] = {}
        self.epicenter_node = 0
        self.last_percolation_phase: str = "aggressive"
        self.last_lambda: float = float(cfg.lambda_aggressive) if cfg is not None else 0.012
        self.last_pmacro_mean: float = 0.0
        self.last_pstep_mean: float = 0.0
        self.last_edge_pstep: Dict[Tuple[int, int], float] = {}
        self.last_bldg_mean: float = 0.0
        self.last_infra_mean: float = 0.0
        self.last_length_norm_mean: float = 0.0
        edge_lens: List[float] = []
        for src, nbs in self.topo.adjacency.items():
            for dst in nbs:
                if src >= dst:
                    continue
                edge_lens.append(float(self.topo.edge_distance(int(src), int(dst))))
        if edge_lens:
            self.edge_len_ref_m = float(np.percentile(np.asarray(edge_lens), 90))
        else:
            self.edge_len_ref_m = 1.0
        self.edge_len_ref_m = float(max(self.edge_len_ref_m, 1e-6))
        self._init_field()

    def _init_field(self) -> None:
        self.epicenter_node = self._closest_node_id(
            float(self.field.quake_epicenter[0]),
            float(self.field.quake_epicenter[1]),
        )
        for node_id, node in self.topo.nodes.items():
            self.node_hazard[node_id] = self.weather_at((node.x, node.y))

    def _closest_node_id(self, x: float, y: float) -> int:
        best_id = 0
        best_d = float("inf")
        for nid, node in self.topo.nodes.items():
            d = float(np.hypot(float(node.x) - x, float(node.y) - y))
            if d < best_d:
                best_d = d
                best_id = int(nid)
        return best_id

    def wind_vector_at(
        self,
        point_xy: Tuple[float, float],
        base_wind_vector: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        _, _, vx, vy, _ = self.field.get_hazard_at(float(point_xy[0]), float(point_xy[1]))
        # Keep backward signature; optional base wind vector is treated as additive bias.
        if base_wind_vector is not None:
            vx += float(base_wind_vector[0])
            vy += float(base_wind_vector[1])
        return float(vx), float(vy)

    def rainfall_at(self, point_xy: Tuple[float, float]) -> float:
        _, _, _, _, rain = self.field.get_hazard_at(float(point_xy[0]), float(point_xy[1]))
        return float(rain)

    def weather_at(self, point_xy: Tuple[float, float]) -> NodeHazard:
        quake, wind, _, _, rain = self.field.get_hazard_at(float(point_xy[0]), float(point_xy[1]))
        return NodeHazard(
            rain=float(max(rain, 0.0)),
            wind=float(max(wind, 0.0)),
            quake=float(np.clip(quake, 0.0, 1.0)),
        )

    def step(self) -> Tuple[float, float, float, int]:
        self.step_index += 1
        self.field.step()
        self.epicenter_node = self._closest_node_id(
            float(self.field.quake_epicenter[0]),
            float(self.field.quake_epicenter[1]),
        )

        for node_id, node in self.topo.nodes.items():
            self.node_hazard[node_id] = self.weather_at((node.x, node.y))

        blocked_ratio_before = self.topo.blocked_ratio()
        threshold = float(self.cfg.percolation_lock_threshold) if self.cfg is not None else 0.30
        if blocked_ratio_before < threshold:
            lam_base = float(self.cfg.lambda_aggressive) if self.cfg is not None else 0.012
            phase = "aggressive"
        else:
            lam_base = float(self.cfg.lambda_residual) if self.cfg is not None else 0.0005
            phase = "residual"

        # Temporal warmup: slows early collapse growth while preserving later pressure.
        warm_steps = int(max(1, getattr(self.cfg, "lambda_warmup_steps", 1))) if self.cfg is not None else 1
        warm_min = float(np.clip(getattr(self.cfg, "lambda_warmup_min_factor", 1.0), 0.0, 1.0)) if self.cfg is not None else 1.0
        prog = float(np.clip(self.step_index / max(float(warm_steps), 1.0), 0.0, 1.0))
        lam = float(lam_base * (warm_min + (1.0 - warm_min) * prog))

        self.last_percolation_phase = phase
        self.last_lambda = float(lam)

        pmacro_vals: List[float] = []
        pstep_vals: List[float] = []
        bldg_vals: List[float] = []
        infra_vals: List[float] = []
        length_vals: List[float] = []
        self.last_edge_pstep = {}
        for src, nbs in self.topo.adjacency.items():
            for dst in nbs:
                if src >= dst:
                    continue
                hs, hd = self.node_hazard[int(src)], self.node_hazard[int(dst)]
                eattr = self.topo.edge_attr(src, dst)
                slope = float(
                    np.clip(
                        0.5
                        * (
                            self.topo.nodes[src].slope_norm + self.topo.nodes[dst].slope_norm
                        ),
                        0.0,
                        1.0,
                    )
                )
                rain = float(
                    np.clip(
                        0.5 * (hs.rain + hd.rain) / max(self.base_rainfall_mmh, 1e-6),
                        0.0,
                        2.5,
                    )
                )
                quake = float(np.clip(0.5 * (hs.quake + hd.quake), 0.0, 2.5))
                v_base = float(np.clip(eattr.base_vulnerability, 0.0, 1.0))
                bldg = float(np.clip(eattr.building_density_norm, 0.0, 1.0))
                infra = float(np.clip(eattr.infra_bottleneck_norm, 0.0, 1.0))
                edge_len = float(self.topo.edge_distance(int(src), int(dst)))
                len_norm = float(np.clip(edge_len / max(self.edge_len_ref_m, 1e-6), 0.0, 2.5))

                if self.cfg is None:
                    b0, bs, br, be, bsr, bre, bv = -4.0, 1.4, 1.2, 1.8, 3.0, 1.1, 1.0
                    bb, bi, bl, blr = 0.9, 1.3, 0.8, 0.9
                    pmax = 0.85
                else:
                    b0 = float(self.cfg.logistic_beta0)
                    bs = float(self.cfg.logistic_beta_slope)
                    br = float(self.cfg.logistic_beta_rain)
                    be = float(self.cfg.logistic_beta_quake)
                    bsr = float(self.cfg.logistic_beta_sr)
                    bre = float(self.cfg.logistic_beta_re)
                    bv = float(self.cfg.logistic_beta_vbase)
                    bb = float(self.cfg.logistic_beta_bldg)
                    bi = float(self.cfg.logistic_beta_infra)
                    bl = float(self.cfg.logistic_beta_length)
                    blr = float(self.cfg.logistic_beta_lr)
                    pmax = float(self.cfg.stochastic_block_max_prob)

                z = (
                    b0
                    + bs * slope
                    + br * rain
                    + be * quake
                    + bsr * (slope * rain)
                    + bre * (rain * quake)
                    + bb * bldg
                    + bi * infra
                    + bl * len_norm
                    + blr * (len_norm * rain)
                    + bv * v_base
                )
                p_macro = float(1.0 / (1.0 + np.exp(-z)))
                p_step = float(np.clip(p_macro * lam, 0.0, pmax))
                self.last_edge_pstep[(int(src), int(dst))] = p_step
                pmacro_vals.append(p_macro)
                pstep_vals.append(p_step)
                bldg_vals.append(bldg)
                infra_vals.append(infra)
                length_vals.append(len_norm)
                if self.rng.uniform() < p_step:
                    self.topo.set_blocked(src, dst, True)

        self.last_pmacro_mean = float(np.mean(pmacro_vals)) if pmacro_vals else 0.0
        self.last_pstep_mean = float(np.mean(pstep_vals)) if pstep_vals else 0.0
        self.last_bldg_mean = float(np.mean(bldg_vals)) if bldg_vals else 0.0
        self.last_infra_mean = float(np.mean(infra_vals)) if infra_vals else 0.0
        self.last_length_norm_mean = float(np.mean(length_vals)) if length_vals else 0.0

        rains = [h.rain for h in self.node_hazard.values()]
        winds = [h.wind for h in self.node_hazard.values()]
        blocked_ratio = self.topo.blocked_ratio()
        return (
            float(np.mean(rains)) if rains else 0.0,
            float(np.mean(winds)) if winds else 0.0,
            float(blocked_ratio),
            int(self.epicenter_node),
        )

    def node_weather(self, node_id: int) -> NodeHazard:
        return self.node_hazard[int(node_id)]
