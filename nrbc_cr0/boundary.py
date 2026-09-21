"""直方体境界の面情報と、波数領域反射係数 C_r = 0 に対応する境界作用素。

提案手法の要点（論文 §2）:
  1. 境界 Gamma 上の節点を平面波成分に分解する変換行列
         [F_Gamma]_{jn} = (1/sqrt(w_n)) exp(-i k_{Gamma,j} . r_{Gamma,n})
     を作る。波数サンプル k_Gamma は伝搬円板 |k_Gamma| <= k 内に
     Fibonacci disk 配置で等方的に分布させる。
  2. 完全吸収条件 C_r = 0 のとき、波数領域アドミタンスは平面波値
         beta_0(k_Gamma) = k_perp / (Z0 * k)
     そのものとなる（B = B_0）。
  3. 空間領域へ戻した非局所作用素 B_hat = F_Gamma^H B F_Gamma を
     全体系 [K - k^2 M - i*rho*omega*C_hat] p = q に加える。
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import eigh, norm
from scipy.sparse import coo_matrix

from . import config as cfg

# 面名 -> (局所節点番号, 面内軸 u, 面内軸 v)
FACE_SPECS = (
    ("xmin", (0, 3, 7, 4), 1, 2),
    ("xmax", (1, 2, 6, 5), 1, 2),
    ("ymin", (0, 1, 5, 4), 0, 2),
    ("ymax", (3, 2, 6, 7), 0, 2),
    ("zmin", (0, 1, 2, 3), 0, 1),
    ("zmax", (4, 5, 6, 7), 0, 1),
)


def face_masks(points, bbox):
    """6 面それぞれに属する節点の真偽値マスク。"""
    xn, yn, zn, xx, yx, zx = bbox
    t = cfg.TOL_BC
    return {
        "xmin": np.abs(points[:, 0] - xn) < t,
        "xmax": np.abs(points[:, 0] - xx) < t,
        "ymin": np.abs(points[:, 1] - yn) < t,
        "ymax": np.abs(points[:, 1] - yx) < t,
        "zmin": np.abs(points[:, 2] - zn) < t,
        "zmax": np.abs(points[:, 2] - zx) < t,
    }


# --------------------------------------------------------------------
# 面要素（4 節点四角形）
# --------------------------------------------------------------------
def _quad_area(p4):
    """4 節点四角形の面積（2 つの三角形に分割）。"""
    return (0.5 * norm(np.cross(p4[1] - p4[0], p4[3] - p4[0]))
            + 0.5 * norm(np.cross(p4[2] - p4[1], p4[3] - p4[1])))


def _quad_mass(area):
    """4 節点四角形の要素質量行列。"""
    return (area / 36.0) * np.array([[4, 2, 1, 2],
                                     [2, 4, 2, 1],
                                     [1, 2, 4, 2],
                                     [2, 1, 2, 4]], dtype=np.complex128)


def surface_mass(points, hexes, mask, local_nodes, ids=None):
    """1 つの境界面上の質量行列（節点面積重み w_n(Gamma) を与える）。

    ids を省略すると mask の節点番号昇順を用いる。
    """
    if ids is None:
        ids = np.where(mask)[0].astype(int)
    m = len(ids)
    if m == 0:
        return ids, coo_matrix((0, 0), dtype=np.complex128).tocsr()
    idmap = {int(g): i for i, g in enumerate(ids)}
    rr, cc, dd = [], [], []
    ln = np.asarray(local_nodes, dtype=int)
    for hx in hexes:
        i4 = hx[ln]
        if not np.all(mask[i4]):
            continue
        Me = _quad_mass(_quad_area(points[hx][ln]))
        for a in range(4):
            ia = idmap[int(i4[a])]
            for b in range(4):
                rr.append(ia)
                cc.append(idmap[int(i4[b])])
                dd.append(Me[a, b])
    Mg = coo_matrix((np.asarray(dd, dtype=np.complex128),
                     (np.asarray(rr), np.asarray(cc))), shape=(m, m)).tocsr()
    return ids, Mg


# --------------------------------------------------------------------
# 波数サンプリングとホワイトニング
# --------------------------------------------------------------------
def fibonacci_disk(n, k_max):
    """半径 k_max の円板上に n 点を Fibonacci 配置する。"""
    i = np.arange(n, dtype=float)
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * i / golden ** 2
    radius = np.sqrt((i + 0.5) / n) * k_max
    return radius * np.cos(theta), radius * np.sin(theta)


def _whiten(F, w):
    """F^H diag(w) F ~= I となるように F を直交化する。"""
    H = F.conj().T @ (F * w[:, None])
    H = 0.5 * (H + H.conj().T)
    lam, U = eigh(H)
    keep = lam > max(cfg.PROP_EIG_EPS, cfg.PROP_EIG_REL * lam.max())
    if not np.any(keep):
        raise RuntimeError("ホワイトニングに失敗しました（有効な固有値がありません）。")
    Uk, Lk = U[:, keep], lam[keep]
    return F @ ((Uk / np.sqrt(Lk)) @ Uk.conj().T)


def _sqrt_psd(A):
    """半正定値 Hermite 行列の平方根。"""
    A = 0.5 * (A + A.conj().T)
    lam, U = eigh(A)
    return (U * np.sqrt(np.clip(lam, cfg.PROP_EIG_EPS, None))) @ U.conj().T


def plane_wave_basis(points, ids, u_axis, v_axis, k, n_k):
    """1 つの面に対する F_Gamma と波数領域アドミタンス重み w*beta_0 を返す。

    Returns
    -------
    F  : (n_k, N_b) ホワイトニング済みの平面波分解行列
    wy : (n_k,) 求積重み w_j と平面波アドミタンス beta_0_j の積
    """
    u, v = points[ids, u_axis], points[ids, v_axis]
    kx, ky = fibonacci_disk(n_k, k)
    k_perp = np.sqrt(np.maximum(k ** 2 - kx ** 2 - ky ** 2, 0.0))
    # 伝搬円板上の求積重み（立体角要素 dS = (k_perp/k) dk_x dk_y に対応）
    w = ((2.0 * np.pi * k ** 2 / n_k) * (k_perp / k)).astype(float)
    beta0 = (k_perp / (cfg.Z0 * k)).astype(np.complex128)   # C_r = 0 のときの B = B_0
    F = (1.0 / (2.0 * np.pi)) * np.exp(-1j * (np.outer(kx, u) + np.outer(ky, v)))
    if cfg.PROP_WHITEN:
        F = _whiten(F, w)
    return F, (w * beta0).astype(np.complex128)


def dense_face_operator(points, hexes, mask, local_nodes, u_axis, v_axis, k, n_k):
    """直接法用：1 面分の密な境界作用素 M^(1/2) B_hat M^(1/2) を構成する。"""
    ids = np.where(mask)[0].astype(int)
    if len(ids) == 0:
        return ids, np.zeros((0, 0), dtype=np.complex128)
    _, Mg = surface_mass(points, hexes, mask, local_nodes, ids)
    Mg12 = _sqrt_psd(Mg.toarray())
    F, wy = plane_wave_basis(points, ids, u_axis, v_axis, k, n_k)
    B = F.conj().T @ (wy[:, None] * F)
    B = 0.5 * (B + B.conj().T)
    C = Mg12 @ B @ Mg12
    return ids, 0.5 * (C + C.conj().T)
