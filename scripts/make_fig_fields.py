#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""論文 図 2「音圧分布（XZ 断面、f = 1000 Hz）」を生成する。

上段 Re(p)、下段 |p| を IMP / PML(1) / PML(2) / PML(3) / Prop の順に並べる。
いずれも PML(3 層) の最大振幅で正規化する。

使用例:
    python scripts/make_fig_fields.py
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrbc_cr0 import config as cfg

METHODS = [("IMP", "IMP"), ("PML1", "PML(1)"), ("PML2", "PML(2)"),
           ("PML3", "PML(3)"), ("Prop", "Prop")]
VMAX_RE, VMAX_ABS = 0.02, 0.05


def _rc_setup():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.unicode_minus": False,
        "font.size": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "axes.linewidth": 0.5,
    })


def _slice_gridder(points, bbox, h):
    """y = 中央 の XZ 断面を構造格子に写す関数を返す。"""
    y_center = 0.5 * (bbox[1] + bbox[4])
    on_slice = np.where(np.abs(points[:, 1] - y_center) < 0.1 * h)[0]
    ix = np.round((points[on_slice, 0] - bbox[0]) / h).astype(int)
    iz = np.round((points[on_slice, 2] - bbox[2]) / h).astype(int)
    nx = int(round((bbox[3] - bbox[0]) / h)) + 1
    nz = int(round((bbox[5] - bbox[2]) / h)) + 1

    def to_grid(field):
        g = np.full((nz, nx), np.nan)
        g[iz, ix] = field[on_slice]
        return g

    return to_grid


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state",
                    default=os.path.join(cfg.RESULT_DIR, f"state_{int(cfg.FREQ)}Hz.pkl"))
    ap.add_argument("--outdir", default=cfg.FIGURE_DIR)
    args = ap.parse_args(argv)

    with open(args.state, "rb") as fh:
        state = pickle.load(fh)
    mesh = state["mesh"]
    points, bbox, h = np.asarray(mesh["points"]), mesh["bbox"], mesh["h"]
    results = state["results"]
    to_grid = _slice_gridder(points, bbox, h)
    scale = float(np.abs(results["PML3"]["p"]).max())

    _rc_setup()
    fig, axes = plt.subplots(2, len(METHODS), figsize=(1.32 * len(METHODS), 2.75))
    z_ticks = [bbox[2], 0.5 * (bbox[2] + bbox[5]), bbox[5]]
    x_ticks = [bbox[0], 0.5 * (bbox[0] + bbox[3]), bbox[3]]
    tick_labels = ["-0.5", "0", "0.5"]     # 領域中心を原点とした表示
    extent = [bbox[0], bbox[3], bbox[2], bbox[5]]
    im_re = im_abs = None

    for j, (key, label) in enumerate(METHODS):
        p = np.asarray(results[key]["p"])

        ax = axes[0, j]
        im_re = ax.imshow(to_grid(np.real(p)) / scale, origin="lower", extent=extent,
                          vmin=-VMAX_RE, vmax=VMAX_RE, cmap="RdBu_r",
                          interpolation="bilinear", aspect="equal")
        ax.set_title(label, fontsize=8.5,
                     fontweight="bold" if key == "Prop" else "normal", pad=2)
        ax.set_xticks([])
        ax.set_yticks(z_ticks)
        ax.set_yticklabels(tick_labels if j == 0 else [], fontsize=6.5)
        if j == 0:
            ax.set_ylabel(r"$z$ [m]", fontsize=7.5, labelpad=1)

        ax = axes[1, j]
        im_abs = ax.imshow(to_grid(np.abs(p)) / scale, origin="lower", extent=extent,
                           vmin=0.0, vmax=VMAX_ABS, cmap="magma",
                           interpolation="bilinear", aspect="equal")
        ax.set_xticks(x_ticks)
        ax.set_yticks(z_ticks)
        ax.set_xticklabels(tick_labels if j == 0 else [], fontsize=6.5)
        ax.set_yticklabels(tick_labels if j == 0 else [], fontsize=6.5)
        if j == 0:
            ax.set_xlabel(r"$x$ [m]", fontsize=7.5, labelpad=1)
            ax.set_ylabel(r"$z$ [m]", fontsize=7.5, labelpad=1)

        for ax in (axes[0, j], axes[1, j]):
            ax.tick_params(axis="both", length=2, width=0.4, pad=1)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)

    axes[0, 0].text(-0.62, 0.5, r"Re$(p)$", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=8.5)
    axes[1, 0].text(-0.62, 0.5, r"$|p|$", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=8.5)

    fig.subplots_adjust(right=0.9, hspace=0.06, wspace=0.08, top=0.93, bottom=0.04)
    for im, rect, label in (
            (im_re, [0.915, 0.55, 0.012, 0.37], r"Re$(p)/\max|p_{\rm PML}|$"),
            (im_abs, [0.915, 0.06, 0.012, 0.37], r"$|p|/\max|p_{\rm PML}|$")):
        cb = fig.colorbar(im, cax=fig.add_axes(rect))
        cb.set_label(label, fontsize=7)
        cb.ax.tick_params(labelsize=6, width=0.4)
        cb.outline.set_linewidth(0.4)

    os.makedirs(args.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(args.outdir, f"fig2_fields.{ext}")
        fig.savefig(path)
        print(f"[保存] {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
