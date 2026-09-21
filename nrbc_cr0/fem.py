"""8 節点六面体要素による 3 次元 Helmholtz 方程式の有限要素離散化。

未知数は速度ポテンシャル phi。音圧は p = i*rho*omega*phi で得る。
  標準領域 : A0 = K - k^2 M
  PML 領域 : 複素座標伸長 s_xi = 1 + i*sigma(xi)/k を要素中心で評価
"""

from __future__ import annotations

import numpy as np
import meshio
from scipy.sparse import coo_matrix

from . import config as cfg

# 2 点 Gauss 積分点（2x2x2）
_GP = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])

# 六面体の局所節点符号（gmsh の hexahedron 節点順）
_SX = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float)
_SY = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=float)
_SZ = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float)


# ====================================================================
# メッシュ
# ====================================================================
class Mesh:
    """構造化六面体メッシュ。

    Attributes
    ----------
    points : (N, 3) 節点座標
    hexes  : (E, 8) 要素-節点接続
    bbox   : (xmin, ymin, zmin, xmax, ymax, zmax)
    h      : 要素サイズ [m]
    """

    def __init__(self, points, hexes, h=cfg.H_MESH):
        self.points = np.asarray(points, dtype=float)[:, :3]
        self.hexes = np.asarray(hexes, dtype=int)
        self.h = float(h)
        p = self.points
        self.bbox = (p[:, 0].min(), p[:, 1].min(), p[:, 2].min(),
                     p[:, 0].max(), p[:, 1].max(), p[:, 2].max())

    @classmethod
    def read(cls, path=None, h=cfg.H_MESH):
        path = path or cfg.MESH_PATH
        m = meshio.read(path)
        return cls(m.points, m.cells_dict["hexahedron"], h)

    @property
    def n_nodes(self):
        return self.points.shape[0]

    @property
    def center(self):
        b = self.bbox
        return 0.5 * np.array([b[0] + b[3], b[1] + b[4], b[2] + b[5]])

    def source_node(self, center=None, allowed=None):
        """点音源を置く節点（指定点に最も近い節点）の番号を返す。"""
        center = self.center if center is None else np.asarray(center, float)
        d = np.linalg.norm(self.points - center, axis=1)
        if allowed is not None:
            d = np.where(allowed, d, d + 1e6)
        return int(np.argmin(d))


# ====================================================================
# 要素行列
# ====================================================================
def _shape(xi, eta, zeta):
    """形状関数 N と局所座標に関する微分 (dN/dxi, dN/deta, dN/dzeta)。"""
    return (0.125 * (1 + _SX * xi) * (1 + _SY * eta) * (1 + _SZ * zeta),
            0.125 * _SX * (1 + _SY * eta) * (1 + _SZ * zeta),
            0.125 * _SY * (1 + _SX * xi) * (1 + _SZ * zeta),
            0.125 * _SZ * (1 + _SX * xi) * (1 + _SY * eta))


def element_matrices(coords, stretch=None):
    """要素剛性・質量行列 (8x8)。

    Parameters
    ----------
    coords : (8, 3) 要素節点座標
    stretch : (sx, sy, sz) 複素伸長係数。None なら標準領域。
    """
    Ke = np.zeros((8, 8), dtype=np.complex128)
    Me = np.zeros((8, 8), dtype=np.complex128)
    if stretch is None:
        ax = ay = az = sv = 1.0
    else:
        sx, sy, sz = stretch
        ax, ay, az = (sy * sz) / sx, (sx * sz) / sy, (sx * sy) / sz
        sv = sx * sy * sz
    for xi in _GP:
        for eta in _GP:
            for zeta in _GP:
                N, dxi, deta, dzeta = _shape(xi, eta, zeta)
                J = np.column_stack(
                    [coords.T @ dxi, coords.T @ deta, coords.T @ dzeta])
                detJ = np.linalg.det(J)
                if abs(detJ) < cfg.EPS_DETJ:
                    continue
                dN = np.linalg.inv(J).T @ np.vstack([dxi, deta, dzeta])
                if stretch is None:
                    Ke += (dN.T @ dN) * detJ
                else:
                    Ke += (ax * np.outer(dN[0], dN[0])
                           + ay * np.outer(dN[1], dN[1])
                           + az * np.outer(dN[2], dN[2])) * detJ
                Me += (sv * np.outer(N, N)) * detJ
    return Ke, Me


def _sigma_1d(x, xmin, xmax, thickness):
    """PML の吸収プロファイル sigma(x)（2 乗則）。"""
    s = 0.0
    if x < xmin + thickness:
        s += cfg.SIGMA_MAX * min((xmin + thickness - x) / thickness, 1.0) ** 2
    if x > xmax - thickness:
        s += cfg.SIGMA_MAX * min((x - (xmax - thickness)) / thickness, 1.0) ** 2
    return s


def assemble(points, hexes, k, pml_bbox=None, pml_thickness=None):
    """全体剛性・質量行列を組み立てる。

    pml_bbox を与えると PML 領域として複素伸長を適用する。
    同形状・同伸長の要素行列はキャッシュして再利用する（一様メッシュで有効）。
    """
    points = np.asarray(points, float)
    n = points.shape[0]
    rows = np.empty(len(hexes) * 64, dtype=np.int64)
    cols = np.empty(len(hexes) * 64, dtype=np.int64)
    kdat = np.empty(len(hexes) * 64, dtype=np.complex128)
    mdat = np.empty(len(hexes) * 64, dtype=np.complex128)

    cache = {}
    pos = 0
    for hx in hexes:
        coords = points[hx]
        if pml_bbox is None:
            stretch = None
        else:
            xc, yc, zc = coords.mean(axis=0)
            stretch = (
                1 + 1j * _sigma_1d(xc, pml_bbox[0], pml_bbox[3], pml_thickness) / k,
                1 + 1j * _sigma_1d(yc, pml_bbox[1], pml_bbox[4], pml_thickness) / k,
                1 + 1j * _sigma_1d(zc, pml_bbox[2], pml_bbox[5], pml_thickness) / k,
            )
        shape_key = tuple(np.round(coords - coords[0], 12).ravel())
        key = (shape_key, stretch)
        if key not in cache:
            cache[key] = element_matrices(coords, stretch)
        Ke, Me = cache[key]

        idx = hx.astype(np.int64)
        rows[pos:pos + 64] = np.repeat(idx, 8)
        cols[pos:pos + 64] = np.tile(idx, 8)
        kdat[pos:pos + 64] = Ke.ravel()
        mdat[pos:pos + 64] = Me.ravel()
        pos += 64

    K = coo_matrix((kdat, (rows, cols)), shape=(n, n)).tocsr()
    M = coo_matrix((mdat, (rows, cols)), shape=(n, n)).tocsr()
    return K, M


# ====================================================================
# PML 用の外側メッシュ生成
# ====================================================================
def extend_mesh(points, n_layers, dh, tol=1e-10):
    """構造化メッシュの外側に n_layers 層を付加した拡張メッシュを作る。

    Returns
    -------
    pts_ext, hex_ext : 拡張メッシュ
    physical : (N_ext,) 元領域に属する節点の真偽値
    inv      : (N,) 元メッシュ節点 -> 拡張メッシュ節点の索引
    """
    points = np.asarray(points, float)

    def _unique(a):
        return np.unique(np.round(a / tol) * tol)

    xs, ys, zs = _unique(points[:, 0]), _unique(points[:, 1]), _unique(points[:, 2])
    if len(xs) * len(ys) * len(zs) != len(points):
        raise ValueError("構造化メッシュではありません。")

    def _pad(a):
        return np.concatenate([a[0] - dh * np.arange(n_layers, 0, -1),
                               a,
                               a[-1] + dh * np.arange(1, n_layers + 1)])

    xe, ye, ze = _pad(xs), _pad(ys), _pad(zs)
    nx, ny, nz = len(xe), len(ye), len(ze)
    X, Y, Z = np.meshgrid(xe, ye, ze, indexing="ij")
    # 節点番号は i + nx*(j + ny*k) の順（x が最内）
    pts_ext = np.column_stack([X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")])

    i0, j0, k0 = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1),
                             indexing="ij")
    i0, j0, k0 = i0.ravel(order="F"), j0.ravel(order="F"), k0.ravel(order="F")

    def nid(i, j, kk):
        return i + nx * (j + ny * kk)

    hex_ext = np.column_stack([
        nid(i0, j0, k0), nid(i0 + 1, j0, k0),
        nid(i0 + 1, j0 + 1, k0), nid(i0, j0 + 1, k0),
        nid(i0, j0, k0 + 1), nid(i0 + 1, j0, k0 + 1),
        nid(i0 + 1, j0 + 1, k0 + 1), nid(i0, j0 + 1, k0 + 1)]).astype(int)

    def _key(p):
        return (round(p[0] / tol), round(p[1] / tol), round(p[2] / tol))

    emap = {_key(pts_ext[i]): i for i in range(len(pts_ext))}
    inv = np.array([emap[_key(points[i])] for i in range(len(points))])

    physical = np.ones(len(pts_ext), dtype=bool)
    for ax in range(3):
        physical &= (pts_ext[:, ax] >= points[:, ax].min() - 1e-12)
        physical &= (pts_ext[:, ax] <= points[:, ax].max() + 1e-12)
    return pts_ext, hex_ext, physical, inv


def to_pressure(phi, omega):
    """速度ポテンシャル phi から音圧 p = i*rho*omega*phi を得る。"""
    return 1j * cfg.RHO * omega * phi
