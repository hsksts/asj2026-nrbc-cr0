"""各境界条件のソルバ。

いずれも領域中心の節点に単位節点荷重 b[src] = 1 を与えて速度ポテンシャルを解き、
音圧 p = i*rho*omega*phi を元メッシュの節点値として返す。

  solve_imp          : スカラアドミタンス境界（局所条件）
  solve_pml          : 外側に PML を n 層付加した参照解
  solve_prop         : 提案手法 C_r = 0（GMRES + LU 前処理）★主提案
  solve_prop_direct  : 提案手法の直接法版（メモリ比較の基準）
  solve_ie           : Astley-Leis 無限要素（論文本文では非採用・スケール推定用）
"""

from __future__ import annotations

import time

import numpy as np
from numpy.linalg import norm
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, gmres, splu, LinearOperator

from . import config as cfg
from . import boundary as bd
from .fem import assemble, extend_mesh, to_pressure


def _point_source(n, src):
    b = np.zeros(n, dtype=np.complex128)
    b[src] = 1.0
    return b


# ====================================================================
# IMP: スカラアドミタンス境界
# ====================================================================
def solve_imp(mesh, k, omega):
    """全方向に同一のアドミタンス Y = 1/Z0 を与える局所条件。"""
    t0 = time.perf_counter()
    pts, hexes = mesh.points, mesh.hexes
    n = mesh.n_nodes
    K, M = assemble(pts, hexes, k)

    masks = bd.face_masks(pts, mesh.bbox)
    rr, cc, dd = [], [], []
    for name, local_nodes, _, _ in bd.FACE_SPECS:
        mask = masks[name]
        ln = np.asarray(local_nodes, dtype=int)
        for hx in hexes:
            i4 = hx[ln]
            if not np.all(mask[i4]):
                continue
            Ce = bd._quad_mass(bd._quad_area(pts[hx][ln])) * cfg.Y_IMP
            rr.append(np.repeat(i4, 4))
            cc.append(np.tile(i4, 4))
            dd.append(Ce.ravel())
    C = coo_matrix((np.concatenate(dd),
                    (np.concatenate(rr), np.concatenate(cc))),
                   shape=(n, n)).tocsr()

    src = mesh.source_node()
    A = (K - k ** 2 * M - 1j * k * cfg.Z0 * C).tocsc()
    phi = spsolve(A, _point_source(n, src))
    return {"p": to_pressure(phi, omega), "src": src, "n_dof": n,
            "nnz": int(A.nnz), "time": time.perf_counter() - t0}


# ====================================================================
# PML
# ====================================================================
def solve_pml(mesh, k, omega, n_layers):
    """外側に PML を n_layers 層（層厚 h）付加した解。"""
    t0 = time.perf_counter()
    pts, hexes = mesh.points, mesh.hexes
    thickness = n_layers * mesh.h
    pe, he, physical, inv = extend_mesh(pts, n_layers, mesh.h)
    bbox_ext = (pe[:, 0].min(), pe[:, 1].min(), pe[:, 2].min(),
                pe[:, 0].max(), pe[:, 1].max(), pe[:, 2].max())
    K, M = assemble(pe, he, k, pml_bbox=bbox_ext, pml_thickness=thickness)

    d = norm(pe - mesh.center, axis=1)
    src_ext = int(np.argmin(np.where(physical, d, d + 1e6)))
    A = (K - k ** 2 * M).tocsc()
    phi = spsolve(A, _point_source(pe.shape[0], src_ext))
    p_ext = to_pressure(phi, omega)
    src = int(np.argmin(norm(pts - pe[src_ext], axis=1)))
    return {"p": p_ext[inv], "src": src, "n_dof": int(pe.shape[0]),
            "nnz": int(A.nnz), "layers": n_layers,
            "time": time.perf_counter() - t0}


# ====================================================================
# Prop: 提案手法（C_r = 0）
# ====================================================================
def solve_prop(mesh, k, omega, tol=cfg.GMRES_TOL, verbose=True):
    """提案手法（GMRES + LU 前処理）。

    非局所境界作用素 B_hat を陽に構成せず、GMRES の反復ごとに
        x -> F_Gamma -> diag(w*beta_0) -> F_Gamma^H -> M_Gamma
    の行列・ベクトル積として評価する。前処理には疎行列 A0 = K - k^2 M の
    LU 分解を用いる。
    """
    t0 = time.perf_counter()
    pts, hexes = mesh.points, mesh.hexes
    n = mesh.n_nodes
    K, M = assemble(pts, hexes, k)
    masks = bd.face_masks(pts, mesh.bbox)

    faces = []
    for name, local_nodes, u_axis, v_axis in bd.FACE_SPECS:
        mask = masks[name]
        ids, Mg = bd.surface_mass(pts, hexes, mask, local_nodes)
        if len(ids) == 0:
            continue
        n_k = int(cfg.PROP_N_K) if cfg.PROP_N_K is not None else len(ids)
        F, wy = bd.plane_wave_basis(pts, ids, u_axis, v_axis, k, n_k)
        faces.append((ids, F, wy, Mg))

    A0 = (K - k ** 2 * M).tocsc()
    lu = splu(A0)
    coeff = -1j * k * cfg.Z0

    def matvec(x):
        y = A0 @ x
        for ids, F, wy, Mg in faces:
            t = Mg @ (F.conj().T @ (wy * (F @ x[ids])))
            y[ids] += coeff * t
        return y

    src = mesh.source_node()
    n_iter = [0]
    phi, info = gmres(
        LinearOperator((n, n), matvec=matvec, dtype=np.complex128),
        _point_source(n, src),
        M=LinearOperator((n, n), matvec=lu.solve, dtype=np.complex128),
        restart=cfg.GMRES_RESTART, maxiter=cfg.GMRES_MAXITER, atol=tol,
        callback=lambda r: n_iter.__setitem__(0, n_iter[0] + 1),
        callback_type="legacy")
    dt = time.perf_counter() - t0
    if verbose:
        state = f"収束（{n_iter[0]} 反復）" if info == 0 else f"未収束 info={info}"
        print(f"  [Prop] {state}  M={faces[0][1].shape[0]}  {dt:.1f}s")
    return {"p": to_pressure(phi, omega), "src": src, "n_dof": n,
            "nnz": int(A0.nnz), "gmres_iters": n_iter[0], "gmres_info": int(info),
            "n_k": faces[0][1].shape[0], "time": dt}


def solve_prop_direct(mesh, k, omega, n_k=cfg.PROP_DIRECT_N_K):
    """提案手法の直接法版。密な境界ブロックを陽に保持するため非零要素数が増える。"""
    t0 = time.perf_counter()
    pts, hexes = mesh.points, mesh.hexes
    n = mesh.n_nodes
    K, M = assemble(pts, hexes, k)
    masks = bd.face_masks(pts, mesh.bbox)

    rr, cc, dd = [], [], []
    for name, local_nodes, u_axis, v_axis in bd.FACE_SPECS:
        ids, Cf = bd.dense_face_operator(
            pts, hexes, masks[name], local_nodes, u_axis, v_axis, k, n_k)
        if len(ids) == 0:
            continue
        rr.append(np.repeat(ids, len(ids)))
        cc.append(np.tile(ids, len(ids)))
        dd.append(Cf.ravel())
    Cb = coo_matrix((np.concatenate(dd),
                     (np.concatenate(rr), np.concatenate(cc))),
                    shape=(n, n)).tocsr()

    src = mesh.source_node()
    A = (K - k ** 2 * M).tocsc() - 1j * k * cfg.Z0 * Cb
    phi = spsolve(A, _point_source(n, src))
    return {"p": to_pressure(phi, omega), "src": src, "n_dof": n,
            "nnz": int(A.nnz), "n_k": n_k, "time": time.perf_counter() - t0}


# ====================================================================
# 無限要素（Astley-Leis, p 次）— 論文本文では非採用
# ====================================================================
_IE_CACHE = {}


def _ie_surface(p4):
    """無限要素の基底面における質量行列と接線方向剛性行列。"""
    Mf = np.zeros((4, 4), dtype=np.complex128)
    Kt = np.zeros((4, 4), dtype=np.complex128)
    x = np.asarray(p4, dtype=float)
    g = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    for r in g:
        for s in g:
            N = np.array([0.25 * (1 - r) * (1 - s), 0.25 * (1 + r) * (1 - s),
                          0.25 * (1 + r) * (1 + s), 0.25 * (1 - r) * (1 + s)])
            dr = np.array([-0.25 * (1 - s), 0.25 * (1 - s),
                           0.25 * (1 + s), -0.25 * (1 + s)])
            ds = np.array([-0.25 * (1 - r), -0.25 * (1 + r),
                           0.25 * (1 + r), 0.25 * (1 - r)])
            a1, a2 = x.T @ dr, x.T @ ds
            dA = float(norm(np.cross(a1, a2)))
            if dA < 1e-30:
                continue
            G = np.array([[a1 @ a1, a1 @ a2], [a1 @ a2, a2 @ a2]])
            if abs(np.linalg.det(G)) < 1e-30:
                continue
            Am = np.column_stack([a1, a2]) @ np.linalg.inv(G)
            grad = np.array([Am @ np.array([dr[i], ds[i]]) for i in range(4)])
            Mf += np.outer(N, N) * dA
            Kt += (grad @ grad.T) * dA
    return Mf, Kt


def _ie_radial(order, k):
    """動径方向の積分（外向き伝搬因子 exp(i k s) を含む）。"""
    from numpy.polynomial.legendre import leggauss
    gp, gw = leggauss(cfg.IE_NGAUSS)
    s = 0.5 * cfg.IE_SMAX * gp + 0.5 * cfg.IE_SMAX
    ws = 0.5 * cfg.IE_SMAX * gw
    xi = s / (s + cfg.IE_L)
    om = cfg.IE_L / (s + cfg.IE_L)
    dxi = cfg.IE_L / (s + cfg.IE_L) ** 2
    dom = -cfg.IE_L / (s + cfg.IE_L) ** 2
    phase = np.exp(1j * k * s)

    psi = np.zeros((order, len(s)), dtype=np.complex128)
    dpsi = np.zeros_like(psi)
    for m in range(order):
        xm = np.ones_like(xi) if m == 0 else xi ** m
        dxm = np.zeros_like(xi) if m == 0 else m * xi ** (m - 1) * dxi
        amp = xm * om
        damp = dxm * om + xm * dom
        psi[m] = amp * phase
        dpsi[m] = (damp + 1j * k * amp) * phase

    I0 = np.zeros((order, order), dtype=np.complex128)
    I1 = np.zeros_like(I0)
    for a in range(order):
        for b in range(order):
            I0[a, b] = ws @ (psi[a] * psi[b])
            I1[a, b] = ws @ (dpsi[a] * dpsi[b])
    return 0.5 * (I0 + I0.T), 0.5 * (I1 + I1.T)


def _ie_effective(p4, order, k):
    """内部自由度を静的凝縮した境界節点のみの等価行列。"""
    lengths = sorted([norm(p4[(i + 1) % 4] - p4[i]) for i in range(4)]
                     + [norm(p4[2] - p4[0]), norm(p4[3] - p4[1])])
    key = (order, round(k, 6), tuple(np.round(lengths, 8).tolist()))
    if key in _IE_CACHE:
        return _IE_CACHE[key].copy()

    Mf, Kt = _ie_surface(p4)
    I0, I1 = _ie_radial(order, k)
    A = np.kron(Mf, I1) + np.kron(Kt, I0) - k ** 2 * np.kron(Mf, I0)
    A = 0.5 * (A + A.T)
    bi = np.array([a * order for a in range(4)])
    ii = np.array([a * order + m for a in range(4) for m in range(1, order)])
    if ii.size == 0:
        Ae = A[np.ix_(bi, bi)]
    else:
        try:
            X = np.linalg.solve(A[np.ix_(ii, ii)], A[np.ix_(ii, bi)])
        except np.linalg.LinAlgError:
            X = np.linalg.solve(A[np.ix_(ii, ii)] + 1e-10j * np.eye(len(ii)),
                                A[np.ix_(ii, bi)])
        Ae = A[np.ix_(bi, bi)] - A[np.ix_(bi, ii)] @ X
    Ae = 0.5 * (Ae + Ae.T)
    _IE_CACHE[key] = Ae.copy()
    return Ae


def solve_ie(mesh, k, omega, order=cfg.IE_ORDER):
    """Astley-Leis 型無限要素（動径次数 p）。"""
    t0 = time.perf_counter()
    pts, hexes = mesh.points, mesh.hexes
    n = mesh.n_nodes
    K, M = assemble(pts, hexes, k)
    masks = bd.face_masks(pts, mesh.bbox)

    rr, cc, dd = [], [], []
    for name, local_nodes, _, _ in bd.FACE_SPECS:
        mask = masks[name]
        ln = np.asarray(local_nodes, dtype=int)
        for hx in hexes:
            i4 = hx[ln]
            if not np.all(mask[i4]):
                continue
            Ae = _ie_effective(pts[i4], order, k)
            rr.append(np.repeat(i4, 4))
            cc.append(np.tile(i4, 4))
            dd.append(Ae.ravel())
    Kie = coo_matrix((np.concatenate(dd),
                      (np.concatenate(rr), np.concatenate(cc))),
                     shape=(n, n)).tocsr()

    src = mesh.source_node()
    A = ((K - k ** 2 * M).tocsr() + Kie).tocsc()
    phi = spsolve(A, _point_source(n, src))
    return {"p": to_pressure(phi, omega), "src": src, "n_dof": n,
            "nnz": int(A.nnz), "order": order, "time": time.perf_counter() - t0}
