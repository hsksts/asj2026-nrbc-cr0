#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""論文 図 3（距離減衰）と 図 4（解析解基準のレベル差 dL）を生成する。

図 3: 立体対角（角部）方向の節点について、球殻ごとの SPL 中央値を対数軸で示す。
      解析解 A/r を細破線で重ねる。PML は 1〜3 層を併記。
図 4: dL = 20 log10(|p|/|p_a|) の球殻ごとの中央値と四分位幅（25-75 %）。
      PML(1 層) は dL が図のスケールを超えるため除外する。

使用例:
    python scripts/make_fig_decay.py
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, NullLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrbc_cr0 import config as cfg
from nrbc_cr0 import metrics

P_REF = 20e-6          # SPL の基準音圧 [Pa]
BIN_DR = 0.05          # 球殻の幅 [m]
PERCENTILE = (25, 75)  # 四分位幅

# 手法ごとの体裁（色に頼らず線種＋マーカーで識別できるようにする）
STYLE = {
    "IMP":  dict(color="#1B9E77", marker="^", ls="--", label="IMP"),
    "PML1": dict(color="#6BAED6", marker="D", ls=":", label="PML(1)"),
    "PML2": dict(color="#2171B5", marker="X", ls="-.", label="PML(2)"),
    "PML3": dict(color="#08306B", marker="s", ls="-", label="PML(3)"),
    "Prop": dict(color="#C0392B", marker="o", ls="-", label="Prop"),
}
ORDER_SPL = ["IMP", "PML1", "PML2", "PML3", "Prop"]
ORDER_DB = ["IMP", "PML2", "PML3", "Prop"]


def _rc_setup():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.unicode_minus": False,
        "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.top": True, "ytick.right": True,
        "axes.linewidth": 0.7, "lines.linewidth": 1.1,
        "legend.fontsize": 7.3, "legend.framealpha": 1.0,
        "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def binned(r, q, edges):
    """球殻ごとの中央値と四分位点。節点数が 4 未満の球殻は捨てる。"""
    lo_pct, hi_pct = PERCENTILE
    idx = np.digitize(r, edges) - 1
    centers, median, lo, hi = [], [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if np.count_nonzero(sel) < 4:
            continue
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        median.append(np.median(q[sel]))
        lo.append(np.percentile(q[sel], lo_pct))
        hi.append(np.percentile(q[sel], hi_pct))
    return (np.asarray(centers), np.asarray(median),
            np.asarray(lo), np.asarray(hi))


def _legend(fig, handles, labels):
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.0), frameon=False,
               handlelength=1.5, columnspacing=0.9, handletextpad=0.4,
               fontsize=6.6)


def draw_spl(ax, ctx):
    """図 3: 立体対角方向の SPL 距離減衰。"""
    to_db = lambda p: 20 * np.log10(np.maximum(np.abs(p), 1e-30) / P_REF)
    rr = np.linspace(0.30, 0.90, 600)
    h_analytic, = ax.plot(rr, to_db(ctx["amp"] / rr), color="0.55",
                          lw=1.1, ls=(0, (3, 2)), zorder=1)

    handles, labels = [], []
    r_show = ctx["edges_diag"][1]
    for key in ORDER_SPL:
        st = STYLE[key]
        centers, median, _, _ = binned(
            ctx["r_diag"], to_db(ctx["fields"][key][ctx["mask_diag"]]),
            ctx["edges_diag"])
        sel = centers >= r_show
        h, = ax.plot(centers[sel], median[sel], marker=st["marker"], ls=st["ls"],
                     color=st["color"], ms=3.6, mfc=st["color"], mec="white",
                     mew=0.5, zorder=4)
        handles.append(h)
        labels.append(st["label"])

    ax.text(0.96, 0.96, "space diagonal (to corner)", transform=ax.transAxes,
            fontsize=7.2, color="0.35", va="top", ha="right")
    ax.set_xlim(0.30, 0.90)
    ax.set_ylim(145.5, 161.0)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(np.arange(0.3, 0.9, 0.1)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}"))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel(r"$r$ [m]")
    ax.set_ylabel("SPL [dB re 20 μPa]")
    handles.append(h_analytic)
    labels.append(r"Analytical ($\propto r^{-1}$)")
    return handles, labels


def draw_delta_l(ax, ctx):
    """図 4: 解析解基準のレベル差 dL（中央値＋四分位幅）。"""
    mask, r = ctx["mask_shell"], ctx["r_shell"]
    ref = np.abs(ctx["p_a"][mask])
    edges = ctx["edges_shell"]
    bin_width = edges[1] - edges[0]

    ax.axhline(0.0, color="0.0", lw=1.2, zorder=3)
    ax.text(edges[0] + 0.02 * (cfg.EVAL_R_MAX - edges[0]), 0.0,
            r"exact (ideal $1/r$)", fontsize=6.4, color="0.25",
            va="bottom", ha="left", zorder=6,
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7))

    handles, labels = [], []
    for i, key in enumerate(ORDER_DB):
        st = STYLE[key]
        dL = 20 * np.log10(np.abs(ctx["fields"][key][mask]) / np.maximum(ref, 1e-30))
        centers, median, lo, hi = binned(r, dL, edges)
        sel = centers <= cfg.EVAL_R_MAX
        dx = (i - (len(ORDER_DB) - 1) / 2) * bin_width * 0.17
        h = ax.errorbar(centers[sel] + dx, median[sel],
                        yerr=[median[sel] - lo[sel], hi[sel] - median[sel]],
                        marker=st["marker"], ls=st["ls"], color=st["color"],
                        ms=3.4, mfc=st["color"], mec="white", mew=0.5,
                        elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=4)
        handles.append(h)
        labels.append(st["label"])

    ax.text(0.035, 0.955, rf"$kh={ctx['kh']:.2f}$", transform=ax.transAxes,
            fontsize=7.2, color="0.35", ha="left", va="top")
    ax.set_xlim(edges[0], cfg.EVAL_R_MAX)
    ax.set_ylim(-0.15, 1.40)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel(r"$r$ [m]")
    ax.set_ylabel(r"$\Delta L = 20\log_{10}(|p|/|p_a|)$ [dB]")
    return handles, labels


def build_context(state):
    mesh = state["mesh"]
    points = np.asarray(mesh["points"])
    bbox, h = mesh["bbox"], mesh["h"]
    center = 0.5 * np.array([bbox[0] + bbox[3], bbox[1] + bbox[4], bbox[2] + bbox[5]])
    r_all = metrics.radius(points, center)
    p_a = metrics.analytic_pressure(r_all, state["freq"])
    fields = {k: np.asarray(v["p"]) for k, v in state["results"].items()}

    half = 0.5 * min(bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])
    r_lo = cfg.EVAL_R_MIN
    r_hi = 0.985 * half

    # 図 4 用: 全方向の球殻
    mask_shell = (r_all >= r_lo) & (r_all <= r_hi)

    # 図 3 用: 立体対角（角部）方向。方向余弦の 3 成分がいずれも大きい節点を選ぶ。
    direction = points - center
    with np.errstate(invalid="ignore"):
        cosines = np.abs(direction) / np.linalg.norm(
            direction, axis=1, keepdims=True).clip(1e-12)
    is_diagonal = np.sort(cosines, axis=1)[:, 0] > 0.45
    r_hi_diag = 0.98 * np.sqrt(3.0) * half      # 角部までは sqrt(3)*half 届く
    mask_diag = is_diagonal & (r_all >= r_lo) & (r_all <= r_hi_diag)

    mask_amp = (r_all >= r_lo) & (r_all <= cfg.EVAL_R_MAX)
    return {
        "fields": fields, "p_a": p_a,
        "mask_shell": mask_shell, "r_shell": r_all[mask_shell],
        "edges_shell": np.arange(r_lo, r_hi + BIN_DR, BIN_DR),
        "mask_diag": mask_diag, "r_diag": r_all[mask_diag],
        "edges_diag": np.arange(r_lo, r_hi_diag + BIN_DR, BIN_DR),
        "amp": float(np.mean(np.abs(p_a[mask_amp]) * r_all[mask_amp])),
        "kh": state["k"] * h,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state",
                    default=os.path.join(cfg.RESULT_DIR, f"state_{int(cfg.FREQ)}Hz.pkl"))
    ap.add_argument("--outdir", default=cfg.FIGURE_DIR)
    args = ap.parse_args(argv)

    with open(args.state, "rb") as fh:
        state = pickle.load(fh)
    ctx = build_context(state)
    os.makedirs(args.outdir, exist_ok=True)
    _rc_setup()

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.75))
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.20, right=0.90)
    _legend(fig, *draw_spl(ax, ctx))
    for ext in ("pdf", "png"):
        path = os.path.join(args.outdir, f"fig3_decay_spl.{ext}")
        fig.savefig(path, pad_inches=0.02)
        print(f"[保存] {path}")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.75))
    fig.subplots_adjust(top=0.83, bottom=0.15, left=0.16, right=0.97)
    handles, labels = draw_delta_l(ax, ctx)
    handles.append(ax.plot([], [], color="0.35", lw=1.5, ls=(0, (6, 3)))[0])
    labels.append(r"Analytical ($\propto r^{-1}$)")
    _legend(fig, handles, labels)
    for ext in ("pdf", "png"):
        path = os.path.join(args.outdir, f"fig4_delta_l.{ext}")
        fig.savefig(path, pad_inches=0.02)
        print(f"[保存] {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
