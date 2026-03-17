from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def welch_t_test(x: Iterable[float], y: Iterable[float]) -> Dict[str, float]:
    a = np.array(list(x), dtype=np.float64)
    b = np.array(list(y), dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return {"t": 0.0, "dof": 0.0, "pvalue_two_sided": 1.0}
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    na, nb = float(a.size), float(b.size)
    den = math.sqrt(max(va / na + vb / nb, 1e-12))
    t = (ma - mb) / den
    dof_num = (va / na + vb / nb) ** 2
    dof_den = (va * va) / (na * na * max(na - 1.0, 1.0)) + (vb * vb) / (
        nb * nb * max(nb - 1.0, 1.0)
    )
    dof = dof_num / max(dof_den, 1e-12)
    # Two-sided p-value (normal approximation; scipy not required).
    p = 2.0 * (1.0 - _normal_cdf(abs(float(t))))
    return {"t": float(t), "dof": float(dof), "pvalue_two_sided": float(np.clip(p, 0.0, 1.0))}


def mean_ci95(samples: Iterable[float]) -> Dict[str, float]:
    x = np.array(list(samples), dtype=np.float64)
    if x.size == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    m = float(np.mean(x))
    if x.size == 1:
        return {"mean": m, "ci95_low": m, "ci95_high": m}
    s = float(np.std(x, ddof=1))
    # z=1.96 approximation to keep dependency-free.
    half = 1.96 * s / math.sqrt(float(x.size))
    return {"mean": m, "ci95_low": float(m - half), "ci95_high": float(m + half)}

