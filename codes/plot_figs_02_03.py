# -*- coding: utf-8 -*-
"""
plot_figs_02_03

- Reads the TWO input CSVs directly from the configured input folder
- Saves ONLY the two requested figures to: <project_root>/results

Assumed project layout:
  <project_root>/
      codes/plot_figs_02_03.py   (this script)
      data/ESN_LSTM_comparison/
          wateryear_regional_summary.csv
          month_seasonality_regional_summary.csv
      results/   (will be created)

"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("qt5agg")  # or "Agg" for non-GUI backend
import matplotlib.pyplot as plt

#plt.style.use('seaborn-v0_8')

FONTSIZELABELS = 18
FONTSIZETICKS = 17
FONTSIZETITLE = 19
FONTSIZELEGEND = 16
plt.rc("axes", titlesize=FONTSIZETITLE)
plt.rc("axes", labelsize=FONTSIZELABELS)
plt.rc("xtick", labelsize=FONTSIZETICKS)
plt.rc("ytick", labelsize=FONTSIZETICKS)
plt.rc("legend", fontsize=FONTSIZELEGEND)

plt.rc("font", family="serif")
plt.rcParams["font.serif"] = ["Times New Roman"]

colors = [["#56ae6c",
"#8960b3",
"#b0923b",
"#ba495b"]]

#plt.rcParams["axes.linewidth"] = 2



# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # "one step back" from /codes
TABLES_DIR = PROJECT_ROOT / "data" / "ESN_LSTM_comparison"
OUTDIR = PROJECT_ROOT / "results"

# -----------------------------
# Plot config
# -----------------------------
MODELS_ORDER = ["ESN", "L-LSTM", "Best L-LSTM", "R-LSTM"]

def save_close(fig: plt.Figure, path: Path, dpi: int = 250) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    del fig
    gc.collect()

def ensure_order(cols: List[str]) -> List[str]:
    keep = [m for m in MODELS_ORDER if m in cols]
    rest = [c for c in cols if c not in keep]
    return keep + rest

def metric_tag(metric_col: str) -> str:
    s = str(metric_col)
    return s.split("_", 1)[1] if "_" in s else s

def plot_wy_median_nse_wl(wy_reg: pd.DataFrame, outdir: Path) -> None:
    var = "WL"
    metric = "median_NSE"
    d = wy_reg[wy_reg["var"].astype(str) == var].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    for m in ensure_order(d["model"].astype(str).unique().tolist()):
        sub = d[d["model"].astype(str) == m].sort_values("water_year")
        if sub.empty:
            continue
        ax.plot(
            sub["water_year"].to_numpy(dtype=float),
            sub[metric].to_numpy(dtype=float),
            marker="o",
            markersize=10,
            linewidth=1.6,
            label=m,
            color = colors[0][MODELS_ORDER.index(m)] if m in MODELS_ORDER else None,
        )

    ax.set_title(f"Water-year regional median {metric_tag(metric)} ({var})")
    ax.set_xlabel("Water year")
    ax.set_ylabel(f"Median {metric_tag(metric)}")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, frameon=True, fancybox=True)
    ax.set_ylim([0.68, 0.98])
    fig.tight_layout()
    fig.show()
    #save_close(fig, outdir / "fig 02.png")

def plot_month_seasonality_median_kge_wl(seas_reg: pd.DataFrame, outdir: Path) -> None:
    var = "WL"
    metric = "median_KGE"
    d = seas_reg[seas_reg["var"].astype(str) == var].copy()

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.subplots_adjust(
    top=0.911,
    bottom=0.156,
    left=0.084,
    right=0.985,
    hspace=0.2,
    wspace=0.2)
    for m in ensure_order(d["model"].astype(str).unique().tolist()):
        sub = d[d["model"].astype(str) == m].sort_values("month")
        if sub.empty:
            continue
        ax.plot(
            sub["month"].to_numpy(dtype=float),
            sub[metric].to_numpy(dtype=float),
            marker="o",
            markersize=10,
            color = colors[0][MODELS_ORDER.index(m)] if m in MODELS_ORDER else None,
            linewidth=1.6,
            label=m,
        )

    ax.set_title(f"Monthly seasonality – regional median {metric_tag(metric)} ({var})")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Median {metric_tag(metric)}")
    ax.set_xticks(np.arange(1, 13))
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, frameon=True, fancybox=True)
    ax.set_ylim([0.26, 0.92])

    fig.tight_layout()
    fig.show()
    #save_close(fig, outdir / "fig 03.png")

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    wy_reg = pd.read_csv(TABLES_DIR / "wateryear_regional_summary.csv")
    seas_reg = pd.read_csv(TABLES_DIR / "month_seasonality_regional_summary.csv")

    #plot_wy_median_nse_wl(wy_reg, OUTDIR)
    plot_month_seasonality_median_kge_wl(seas_reg, OUTDIR)

    #print(f"Saved 2 figures to: {OUTDIR.resolve()}")

if __name__ == "__main__":
    main()
