#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""論文 表 2「解析解との一致度と計算コスト（f = 1000 Hz）」を再現する。

使用例:
    python scripts/compute_table.py
    python scripts/compute_table.py --latex results/table2.tex
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrbc_cr0 import config as cfg
from nrbc_cr0 import metrics

# 表に載せる行（論文の順序）とラベル
ROWS = [("IMP", "IMP"), ("PML1", "PML(1層)"), ("PML2", "PML(2層)"),
        ("PML3", "PML(3層)"), ("Prop", "Prop")]


def build_table(state):
    mesh = state["mesh"]
    points = np.asarray(mesh["points"])
    b = mesh["bbox"]
    center = 0.5 * np.array([b[0] + b[3], b[1] + b[4], b[2] + b[5]])

    r = metrics.radius(points, center)
    p_a = metrics.analytic_pressure(r, state["freq"])
    mask = metrics.eval_mask(r)

    fields = {name: np.asarray(res["p"]) for name, res in state["results"].items()}
    scale_methods = tuple(m for m in cfg.SCALE_METHODS if m in fields)
    s = metrics.common_scale(fields, p_a, mask, scale_methods)
    p_a_scaled = p_a * s

    rows = []
    for key, label in ROWS:
        if key not in fields:
            continue
        res = state["results"][key]
        rows.append({
            "key": key, "label": label,
            "r_corr": metrics.r_corr(fields[key], p_a, mask),
            "eps_a": metrics.eps_a(fields[key], p_a_scaled, mask),
            "w_bar": metrics.w_bar(fields[key], p_a, r),
            "n_dof": res["n_dof"],
            "mem_mb": res.get("peak_mb", float("nan")),
        })
    return rows, {"scale": s, "scale_methods": scale_methods,
                  "n_eval": int(mask.sum()), "r": r, "p_a": p_a,
                  "fields": fields}


def print_table(rows, info):
    print(f"[共通スケール s] {info['scale']:.4f} "
          f"（推定に用いた手法: {', '.join(info['scale_methods'])}）")
    print(f"[評価範囲] {cfg.EVAL_R_MIN:.2f} m <= r <= {cfg.EVAL_R_MAX:.2f} m, "
          f"節点数 {info['n_eval']:,}\n")
    head = f"{'名称':<10}{'r_corr':>8}{'eps_a':>8}{'W-bar[dB]':>11}{'DOF':>9}{'MB':>7}"
    print(head)
    print("-" * len(head.encode("utf-8").decode("utf-8")) )
    for row in rows:
        print(f"{row['label']:<10}{row['r_corr']:>8.3f}{row['eps_a']:>8.3f}"
              f"{row['w_bar']:>11.2f}{row['n_dof']:>9,}{row['mem_mb']:>7.0f}")


def write_latex(rows, path):
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{解析解との一致度と計算コスト（$f=1000$\\,Hz）。}",
        "\\label{tab:results}", "\\small",
        "\\setlength{\\tabcolsep}{2.6pt}",
        "\\begin{tabular}{l ccc cc}", "\\toprule",
        "名称 & $r_{\\rm corr}$ & $\\varepsilon_a$ & $\\overline{W}$\\,[dB]"
        " & DOF & メモリ\\,[MB] \\\\",
        "\\midrule",
    ]
    for row in rows:
        bold = row["key"] == "Prop"
        def f(x, fmt):
            t = format(x, fmt)
            return f"\\textbf{{{t}}}" if bold else t
        lines.append(f"{row['label']} & {f(row['r_corr'], '.3f')} & "
                     f"{f(row['eps_a'], '.3f')} & {row['w_bar']:.2f} & "
                     f"{row['n_dof']:,} & {row['mem_mb']:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n[保存] {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state",
                    default=os.path.join(cfg.RESULT_DIR, f"state_{int(cfg.FREQ)}Hz.pkl"))
    ap.add_argument("--latex", default=None, help="LaTeX 表の出力先")
    ap.add_argument("--shells", action="store_true", help="球殻ごとの四分位幅も表示")
    args = ap.parse_args(argv)

    with open(args.state, "rb") as fh:
        state = pickle.load(fh)
    rows, info = build_table(state)
    print_table(rows, info)

    if args.shells:
        print("\n--- 球殻ごとの四分位幅 [dB] ---")
        centers = None
        table = {}
        for row in rows:
            _, centers, widths = metrics.w_bar(
                info["fields"][row["key"]], info["p_a"], info["r"], detail=True)
            table[row["label"]] = widths
        print("r [m]     " + " ".join(f"{c:7.3f}" for c in centers))
        for label, widths in table.items():
            print(f"{label:<10}" + " ".join(f"{v:7.2f}" for v in widths))

    if args.latex:
        write_latex(rows, args.latex)


if __name__ == "__main__":
    main()
