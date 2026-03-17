from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _safe_base_wind_vector(sim) -> Tuple[float, float]:
    v = getattr(sim, "base_wind_vector_mps", None)
    if isinstance(v, (tuple, list, np.ndarray)) and len(v) >= 2:
        return float(v[0]), float(v[1])

    scalar = getattr(sim, "base_wind_mps", 0.0)
    return float(scalar), 0.0


def render_state(
    sim,
    ax: Optional[plt.Axes] = None,
    grid_n: int = 64,
    wind_subsample_step: int = 4,
) -> plt.Axes:
    """
    Render continuous weather overlays:
    - Rain: non-truncated base-noise-aware alpha field.
    - Wind: dense global quiver over downsampled Cartesian grid.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=120)

    nodes = list(sim.topology.nodes.values())
    if len(nodes) == 0:
        return ax

    xs = np.array([n.x for n in nodes], dtype=np.float64)
    ys = np.array([n.y for n in nodes], dtype=np.float64)
    pad_x = 0.05 * max(1.0, xs.max() - xs.min())
    pad_y = 0.05 * max(1.0, ys.max() - ys.min())
    x_min, x_max = float(xs.min() - pad_x), float(xs.max() + pad_x)
    y_min, y_max = float(ys.min() - pad_y), float(ys.max() + pad_y)

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, int(grid_n)),
        np.linspace(y_min, y_max, int(grid_n)),
    )

    base_rain = float(getattr(sim, "base_rainfall_mmh", 0.0))
    rain = np.zeros_like(xx, dtype=np.float64)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            rain[i, j] = float(sim.weather.rainfall_at((xx[i, j], yy[i, j]), base_rain))

    # --- Required fix: remove aggressive truncation ---
    # old: rain_lo = 0.45 * rain_max
    rain_lo = max(0.1, 0.2 * sim.base_rainfall_mmh)
    rain_hi = float(max(rain.max(), rain_lo + 1e-6))
    alpha = np.clip((rain - rain_lo) / (rain_hi - rain_lo), 0.0, 1.0)

    rgba = np.zeros((rain.shape[0], rain.shape[1], 4), dtype=np.float64)
    rgba[..., 0] = 0.20
    rgba[..., 1] = 0.45
    rgba[..., 2] = 0.95
    rgba[..., 3] = 0.60 * alpha
    ax.imshow(rgba, origin="lower", extent=[x_min, x_max, y_min, y_max], zorder=1)

    for src, nbs in sim.topology.adjacency.items():
        for dst in nbs:
            if src >= dst:
                continue
            n1 = sim.topology.nodes[src]
            n2 = sim.topology.nodes[dst]
            blocked = tuple(sorted((int(src), int(dst)))) in {
                tuple(sorted((int(a), int(b)))) for (a, b) in sim.topology.blocked_edges
            }
            if blocked:
                ax.plot(
                    [n1.x, n2.x],
                    [n1.y, n2.y],
                    color="#ef4444",
                    linewidth=1.4,
                    linestyle=(0, (4, 3)),
                    alpha=0.9,
                    zorder=2,
                )
            else:
                ax.plot(
                    [n1.x, n2.x],
                    [n1.y, n2.y],
                    color="#7dd3fc",
                    linewidth=0.8,
                    alpha=0.45,
                    zorder=2,
                )

    ax.scatter(xs, ys, s=12, c="#d1d5db", alpha=0.95, zorder=3)
    if 0 in sim.topology.nodes:
        n0 = sim.topology.nodes[0]
        ax.scatter([n0.x], [n0.y], s=80, c="#38bdf8", edgecolors="#ffffff", linewidths=0.8, zorder=4)

    # --- Required fix: dense global wind field from grid, not weather centers ---
    step = int(max(1, wind_subsample_step))
    xx_sub = xx[::step, ::step]
    yy_sub = yy[::step, ::step]
    U = np.zeros_like(xx_sub, dtype=np.float64)
    V = np.zeros_like(yy_sub, dtype=np.float64)
    base_wind_vec = _safe_base_wind_vector(sim)
    for i in range(xx_sub.shape[0]):
        for j in range(xx_sub.shape[1]):
            u, v = sim.weather.wind_at((xx_sub[i, j], yy_sub[i, j]), base_wind_vec)
            U[i, j] = float(u)
            V[i, j] = float(v)

    ax.quiver(
        xx_sub,
        yy_sub,
        U,
        V,
        color=(1.0, 1.0, 1.0, 0.55),
        angles="xy",
        scale_units="xy",
        scale=0.06,
        width=0.0023,
        pivot="mid",
        zorder=5,
    )

    # Removed: center circles / ax.add_patch(rc), because rainfall is continuous.
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#0b1220")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
