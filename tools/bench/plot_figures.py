#!/usr/bin/env python3
"""Paper-style scatter figures: time and cost against document length."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def main():
    OUT = sys.argv[1] if len(sys.argv) > 1 else "results/bench-doclen"

    DATA = json.load(open(os.path.join(OUT, "figure_data.json")))
    ROWS = sorted(DATA["rows"], key=lambda r: r["pages"])

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


    def figure(key, ylabel, unit_fmt, color, fname, title):
        x = np.array([r["pages"] for r in ROWS], dtype=float)
        y = np.array([r[key] for r in ROWS], dtype=float)

        # proportional reference through the origin: least squares on y = k*x
        k = float((x @ y) / (x @ x))
        # coefficient of determination of that one-parameter fit
        ss_res = float(((y - k * x) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

        fig, ax = plt.subplots(figsize=(5.2, 3.9))
        xr = np.array([0, x.max() * 1.06])
        line, = ax.plot(xr, k * xr, ls="--", lw=1.1, color="0.55", zorder=1)
        dots = ax.scatter(x, y, s=42, color=color, edgecolor="white", linewidth=0.8,
                          zorder=3)
        # sample dot first: on the second row it reads as a stray data point
        handles = [dots, line]
        labels = ["document",
                  f"proportional fit  {unit_fmt(k * 1000)} / 1000 pages  ($R^2$={r2:.3f})"]

        ax.set_xlabel("Document length (pages)")
        ax.set_ylabel(ylabel)
        ax.text(0, 1.045, title, transform=ax.transAxes, fontsize=11,
                ha="left", va="bottom")
        ax.set_xlim(0, x.max() * 1.06)
        ax.set_ylim(0, max(y.max(), k * x.max()) * 1.12)
        ax.grid(True, color="0.9", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(handles, labels, loc="upper left", fontsize=8.5)

        fig.savefig(f"{OUT}/{fname}", facecolor="white")
        plt.close(fig)
        return k, r2


    NAME = "PageIndex Flash (with tree optimization)"

    kt, rt = figure("seconds", "End-to-end time (s)", lambda v: f"{v:,.0f} s",
                    "#2a78d6", "time_vs_pages.png",
                    f"{NAME}: time vs document length")
    kc, rc = figure("cost_cold", "Cost (USD)", lambda v: f"\\${v:,.2f}",
                    "#eb6834", "cost_vs_pages.png",
                    f"{NAME}: cost vs document length")

    print(f"time: {kt*1000:,.1f} s per 1000 pages   R^2={rt:.4f}")
    print(f"cost: ${kc*1000:,.3f} per 1000 pages   R^2={rc:.4f}")
    print(f"\nwrote {OUT}/time_vs_pages.png")
    print(f"wrote {OUT}/cost_vs_pages.png")


if __name__ == "__main__":
    main()
