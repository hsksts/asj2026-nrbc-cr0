#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全手法を f = 1000 Hz で解き、音圧場と計算コストを results/state_1000Hz.pkl に保存する。

実行する手法:
  IMP          スカラアドミタンス境界
  PML1..PML4   PML（1〜4 層、4 層は収束参照）
  Prop         提案手法 C_r = 0（GMRES + LU 前処理）★主提案
  PropDirect   提案手法の直接法版（メモリ比較の基準。--skip-direct で省略可）
  IE4          Astley-Leis 無限要素 p=4（論文本文では非採用。共通スケール s の再現用）

使用例:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --skip-direct --skip-ie
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrbc_cr0 import config as cfg
from nrbc_cr0 import solvers
from nrbc_cr0.fem import Mesh


def _run(label, func, *args, **kwargs):
    """1 ケースを実行し、Python ヒープのピーク使用量を併せて記録する。

    注意: tracemalloc は Python が確保した配列のみを数えるため、
    疎行列 LU 分解のように C 側で確保される領域は含まれない
    （論文 表 2 のメモリ値もこの定義による）。
    """
    print(f"\n── {label} " + "─" * max(0, 46 - len(label)))
    tracemalloc.start()
    out = func(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out["peak_mb"] = peak / 1e6
    print(f"   DOF={out['n_dof']:,}  nnz={out['nnz']:,}  "
          f"{out['time']:.1f} s  peak {out['peak_mb']:.0f} MB")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", type=float, default=cfg.FREQ, help="解析周波数 [Hz]")
    ap.add_argument("--mesh", default=cfg.MESH_PATH, help="メッシュファイル")
    ap.add_argument("--out", default=None, help="出力 pickle のパス")
    ap.add_argument("--skip-direct", action="store_true", help="提案手法の直接法版を省く")
    ap.add_argument("--skip-ie", action="store_true", help="無限要素を省く")
    args = ap.parse_args(argv)

    mesh = Mesh.read(args.mesh)
    omega = 2.0 * np.pi * args.freq
    k = omega / cfg.C0
    print(f"[メッシュ] {os.path.basename(args.mesh)}  "
          f"節点 {mesh.n_nodes:,} / 要素 {len(mesh.hexes):,}")
    print(f"[条件] f = {args.freq:.0f} Hz,  k = {k:.3f} 1/m,  kh = {k * mesh.h:.3f}")

    results = {}
    results["IMP"] = _run("IMP", solvers.solve_imp, mesh, k, omega)
    for n_layers in cfg.PML_LAYERS_ALL:
        results[f"PML{n_layers}"] = _run(
            f"PML({n_layers} 層)", solvers.solve_pml, mesh, k, omega, n_layers)
    results["Prop"] = _run("Prop（提案手法・GMRES）", solvers.solve_prop, mesh, k, omega)
    if not args.skip_direct:
        results["PropDirect"] = _run("Prop（直接法）",
                                     solvers.solve_prop_direct, mesh, k, omega)
    if not args.skip_ie:
        results["IE4"] = _run("IE4（参考）", solvers.solve_ie, mesh, k, omega)

    state = {
        "freq": args.freq,
        "k": k,
        "omega": omega,
        "mesh": {"points": mesh.points, "hexes": mesh.hexes,
                 "bbox": mesh.bbox, "h": mesh.h},
        "physics": {"rho": cfg.RHO, "c0": cfg.C0},
        "results": results,
    }

    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    out = args.out or os.path.join(cfg.RESULT_DIR, f"state_{int(args.freq)}Hz.pkl")
    with open(out, "wb") as f:
        pickle.dump(state, f)
    print(f"\n[保存] {out}")

    if "PropDirect" in results:
        md = results["PropDirect"]["peak_mb"]
        mg = results["Prop"]["peak_mb"]
        print(f"[メモリ] 直接法 {md:.0f} MB → GMRES {mg:.0f} MB "
              f"（{(1 - mg / md) * 100:.0f} % 削減）")
    return state


if __name__ == "__main__":
    main()
