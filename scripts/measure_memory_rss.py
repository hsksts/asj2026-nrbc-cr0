#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""各手法の実メモリ使用量をプロセス最大常駐メモリ（maxRSS）で測る。

論文 表 2 のメモリ列は tracemalloc による Python ヒープのピーク値であり、
疎行列 LU 分解の因子（SuperLU が C 側で確保する支配項）を含まない。
本スクリプトは各ケースを別プロセスで実行し、resource.getrusage の ru_maxrss で
LU 因子を含めた実メモリを測る。インタプリタ起動直後の RSS を差し引いた
「正味」の値どうしを比較すること。

使用例:
    python scripts/measure_memory_rss.py
    python scripts/measure_memory_rss.py --case prop     # 単一ケース（内部利用）

注意: ru_maxrss の単位は Linux では KiB、macOS では byte。Windows では
resource モジュールが無いため、psutil の peak_wset などで代替する必要がある。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrbc_cr0 import config as cfg

CASES = ["baseline", "imp", "pml1", "pml2", "pml3", "pml4", "prop", "prop_direct"]


def _max_rss_mb():
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux: KiB, macOS: byte
    return rss / 1024.0 if sys.platform != "darwin" else rss / 1e6


def run_case(case):
    """子プロセス側：1 ケースを解いて maxRSS を報告する。"""
    from nrbc_cr0 import solvers
    from nrbc_cr0.fem import Mesh

    if case == "baseline":
        return {"case": case, "max_rss_mb": _max_rss_mb(), "n_dof": 0}

    mesh = Mesh.read()
    omega = 2.0 * np.pi * cfg.FREQ
    k = omega / cfg.C0
    if case == "imp":
        out = solvers.solve_imp(mesh, k, omega)
    elif case.startswith("pml"):
        out = solvers.solve_pml(mesh, k, omega, int(case[3:]))
    elif case == "prop":
        out = solvers.solve_prop(mesh, k, omega, verbose=False)
    elif case == "prop_direct":
        out = solvers.solve_prop_direct(mesh, k, omega)
    else:
        raise ValueError(f"未知のケース: {case}")
    return {"case": case, "max_rss_mb": _max_rss_mb(),
            "n_dof": out["n_dof"], "nnz": out["nnz"], "time": out["time"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default=None, help="単一ケースを実行して JSON を出力")
    args = ap.parse_args(argv)

    if args.case:
        print(json.dumps(run_case(args.case)))
        return

    print("各ケースを別プロセスで実行し maxRSS を測定します。\n")
    records = {}
    for case in CASES:
        proc = subprocess.run([sys.executable, os.path.abspath(__file__),
                               "--case", case],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  {case:<12} 失敗: {proc.stderr.strip().splitlines()[-1:]}")
            continue
        rec = json.loads(proc.stdout.strip().splitlines()[-1])
        records[case] = rec
        print(f"  {case:<12} maxRSS = {rec['max_rss_mb']:7.1f} MB"
              + (f"  (DOF {rec['n_dof']:,})" if rec["n_dof"] else "  (基準)"))

    base = records.get("baseline", {}).get("max_rss_mb", 0.0)
    print(f"\n基準（インタプリタ + import 直後）= {base:.1f} MB")
    print(f"\n{'ケース':<12}{'正味 [MB]':>12}{'DOF':>10}")
    for case in CASES[1:]:
        if case in records:
            rec = records[case]
            print(f"{case:<12}{rec['max_rss_mb'] - base:>12.1f}{rec['n_dof']:>10,}")
    print("\n※ 正味の値どうしを比較すること。論文 表 2 の値は Python ヒープの")
    print("   ピーク（LU 因子を含まない）であり、本測定とは定義が異なる。")


if __name__ == "__main__":
    main()
