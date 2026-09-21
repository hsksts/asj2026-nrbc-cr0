#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""波数サンプル数 M に対する提案手法の収束性を調べる（論文 §4.1 の補足）。

論文本文は M = 441（各境界面の境界節点数）を用い、「評価指標は M が概ね 64 以上で
ほぼ一定となることを別途確認した」と記している。本スクリプトはその根拠を与える。

M ごとに提案手法を解き、次の 2 指標を評価範囲 0.25 m <= r <= 0.45 m で算出する。
  r_corr : |p| と解析解 |p_a| の Pearson 相関係数（論文 表 2 の主指標）
  sigma  : |p| * r の変動係数（1/r 減衰の一様さ）
全体行列の組立と前処理 LU(A0) は M に依存しないため、周波数ごとに 1 回だけ行う。

使用例:
    python scripts/run_m_convergence.py
    python scripts/run_m_convergence.py --m 8 16 32 64 128 441
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
from scipy.sparse.linalg import gmres, splu, LinearOperator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrbc_cr0 import config as cfg
from nrbc_cr0 import boundary as bd
from nrbc_cr0 import metrics
from nrbc_cr0.fem import Mesh, assemble, to_pressure

M_DEFAULT = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 441]
TOL_RCORR = 1.0e-3   # r_corr の絶対許容差
TOL_SIGMA = 2.0e-2   # sigma の相対許容差


def solve_at_m(mesh, A0, lu, rhs, k, omega, n_k):
    """波数サンプル数 M = n_k で提案手法を解く。"""
    pts, hexes = mesh.points, mesh.hexes
    masks = bd.face_masks(pts, mesh.bbox)
    faces = []
    for name, local_nodes, u_axis, v_axis in bd.FACE_SPECS:
        ids, Mg = bd.surface_mass(pts, hexes, masks[name], local_nodes)
        if len(ids) == 0:
            continue
        F, wy = bd.plane_wave_basis(pts, ids, u_axis, v_axis, k, n_k)
        faces.append((ids, F, wy, Mg))

    coeff = -1j * k * cfg.Z0

    def matvec(x):
        y = A0 @ x
        for ids, F, wy, Mg in faces:
            y[ids] += coeff * (Mg @ (F.conj().T @ (wy * (F @ x[ids]))))
        return y

    n = mesh.n_nodes
    n_iter = [0]
    phi, info = gmres(
        LinearOperator((n, n), matvec=matvec, dtype=np.complex128), rhs,
        M=LinearOperator((n, n), matvec=lu.solve, dtype=np.complex128),
        restart=cfg.GMRES_RESTART, maxiter=cfg.GMRES_MAXITER, atol=cfg.GMRES_TOL,
        callback=lambda r: n_iter.__setitem__(0, n_iter[0] + 1),
        callback_type="legacy")
    return to_pressure(phi, omega), n_iter[0], int(info)


def first_converged(m_values, values, reference, tol, relative=False):
    """M 昇順で、それ以降すべてが基準値の tol 以内に収まる最小の M。"""
    values = np.asarray(values)
    scale = abs(reference) if relative else 1.0
    ok = np.abs(values - reference) / scale <= tol
    for i in range(len(m_values)):
        if np.all(ok[i:]):
            return int(m_values[i])
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", type=float, nargs="+", default=[cfg.FREQ])
    ap.add_argument("--m", type=int, nargs="+", default=M_DEFAULT)
    ap.add_argument("--mesh", default=cfg.MESH_PATH)
    ap.add_argument("--out", default=os.path.join(cfg.RESULT_DIR,
                                                  "m_convergence.pkl"))
    args = ap.parse_args(argv)

    mesh = Mesh.read(args.mesh)
    center = mesh.center
    r = metrics.radius(mesh.points, center)
    mask = metrics.eval_mask(r)
    edge = mesh.bbox[3] - mesh.bbox[0]

    results = {}
    for freq in args.freq:
        omega = 2.0 * np.pi * freq
        k = omega / cfg.C0
        wavelength = cfg.C0 / freq
        bandwidth = (2.0 * edge / wavelength) ** 2   # 独立に分解できる平面波方向の目安
        p_a = metrics.analytic_pressure(r, freq)

        print(f"\n{'=' * 66}")
        print(f"  f = {freq:.0f} Hz  (k = {k:.2f} 1/m, λ = {wavelength:.3f} m, "
              f"(2L/λ)² = {bandwidth:.1f})")
        print(f"{'=' * 66}")

        t0 = time.perf_counter()
        K, M = assemble(mesh.points, mesh.hexes, k)
        A0 = (K - k ** 2 * M).tocsc()
        lu = splu(A0)
        rhs = np.zeros(mesh.n_nodes, dtype=np.complex128)
        rhs[mesh.source_node()] = 1.0
        print(f"  [組立 + LU] {time.perf_counter() - t0:.1f} s（周波数ごとに 1 回）")

        rec = {"M": [], "r_corr": [], "sigma": [], "iters": [], "time": [],
               "wavelength": wavelength, "bandwidth": bandwidth, "kh": k * mesh.h}
        for n_k in args.m:
            t1 = time.perf_counter()
            p, iters, info = solve_at_m(mesh, A0, lu, rhs, k, omega, n_k)
            dt = time.perf_counter() - t1
            rc = metrics.r_corr(p, p_a, mask)
            pr = np.abs(p[mask]) * r[mask]
            sigma = float(pr.std() / pr.mean())
            rec["M"].append(n_k)
            rec["r_corr"].append(rc)
            rec["sigma"].append(sigma)
            rec["iters"].append(iters)
            rec["time"].append(dt)
            flag = "" if info == 0 else f"  [gmres info={info}]"
            print(f"    M={n_k:4d} | r_corr={rc:.4f} | sigma={sigma:.4f} | "
                  f"反復={iters:3d} | {dt:5.1f}s | M/(2L/λ)²={n_k / bandwidth:4.1f}{flag}")
        results[freq] = rec

    print(f"\n{'=' * 66}")
    print("  収束判定（基準 = 最大 M の値）")
    print(f"{'=' * 66}")
    verdict = {}
    for freq, rec in results.items():
        m_rc = first_converged(rec["M"], rec["r_corr"], rec["r_corr"][-1], TOL_RCORR)
        m_sg = first_converged(rec["M"], rec["sigma"], rec["sigma"][-1],
                               TOL_SIGMA, relative=True)
        verdict[freq] = {"M_r_corr": m_rc, "M_sigma": m_sg}
        print(f"\n  f = {freq:.0f} Hz")
        print(f"    r_corr : 基準 {rec['r_corr'][-1]:.4f}（M={rec['M'][-1]}）, "
              f"|Δ| <= {TOL_RCORR} を満たす最小 M = {m_rc}")
        print(f"    sigma  : 基準 {rec['sigma'][-1]:.4f}（M={rec['M'][-1]}）, "
              f"相対変化 <= {TOL_SIGMA:.0%} を満たす最小 M = {m_sg}")
        candidates = [m for m in (m_rc, m_sg) if m is not None]
        if candidates:
            print(f"    → 両指標が収束する最小 M ≈ {max(candidates)} "
                  f"（(2L/λ)² ≈ {rec['bandwidth']:.0f} の約 "
                  f"{max(candidates) / rec['bandwidth']:.1f} 倍）")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({"results": results, "verdict": verdict,
                     "n_eval": int(mask.sum())}, fh)
    print(f"\n[保存] {args.out}")


if __name__ == "__main__":
    main()
