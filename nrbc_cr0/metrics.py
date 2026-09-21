"""評価指標（論文 §4.2）。

評価範囲 Omega_eval = { n : 0.25 m <= r_n <= 0.45 m }（節点数 2,532）

  r_corr : |p| と |p_a| の Pearson 相関係数（振幅スケール非依存）
  eps_a  : ||p - s p_a|| / ||s p_a||（振幅・位相を含む相対誤差）
  W-bar  : dL = 20 log10(|p|/|p_a|) の球殻ごとの四分位幅の平均 [dB]

s は各手法解を解析解へ最小二乗の意味で射影した振幅の中央値であり、
節点単位荷重と音源強度の較正を一度だけ行うための全手法共通の実数である。
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm

from . import config as cfg


def radius(points, center):
    r = norm(np.asarray(points) - np.asarray(center), axis=1)
    r[r < 1e-12] = 1e-12
    return r


def analytic_pressure(r, freq=cfg.FREQ):
    """自由空間グリーン関数に基づく解析解 p_a(r) = i rho omega e^{ikr} / (4 pi r)。"""
    omega = 2.0 * np.pi * freq
    k = omega / cfg.C0
    return 1j * cfg.RHO * omega * np.exp(1j * k * r) / (4.0 * np.pi * r)


def eval_mask(r):
    return (r >= cfg.EVAL_R_MIN) & (r <= cfg.EVAL_R_MAX)


def common_scale(fields, p_a, mask, methods=cfg.SCALE_METHODS):
    """全手法共通の実数スケール s（各手法の最小二乗射影振幅の中央値）。"""
    alphas = []
    for name in methods:
        p = fields[name][mask]
        alphas.append(abs(np.vdot(p, p_a[mask]) / np.vdot(p, p)))
    return float(np.median(alphas))


def r_corr(p, p_a, mask):
    return float(np.corrcoef(np.abs(p[mask]), np.abs(p_a[mask]))[0, 1])


def eps_a(p, p_a_scaled, mask):
    return float(norm((p - p_a_scaled)[mask]) / norm(p_a_scaled[mask]))


def w_bar(p, p_a, r, detail=False):
    """球殻ごとの dL 四分位幅の平均 [dB]。"""
    dL = 20.0 * np.log10(np.maximum(np.abs(p), 1e-30)
                         / np.maximum(np.abs(p_a), 1e-30))
    edges = np.arange(cfg.EVAL_R_MIN, cfg.EVAL_R_MAX + cfg.W_BIN, cfg.W_BIN)
    inside = eval_mask(r)
    idx = np.digitize(r, edges) - 1
    widths, centers = [], []
    lo_pct, hi_pct = cfg.W_PERCENTILE
    for b in range(len(edges) - 1):
        sel = (idx == b) & inside
        if np.count_nonzero(sel) < 4:
            continue
        lo, hi = np.percentile(dL[sel], [lo_pct, hi_pct])
        widths.append(hi - lo)
        centers.append(0.5 * (edges[b] + edges[b + 1]))
    mean_w = float(np.mean(widths))
    if detail:
        return mean_w, np.asarray(centers), np.asarray(widths)
    return mean_w
