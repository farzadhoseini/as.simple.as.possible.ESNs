# -*- coding: utf-8 -*-
"""
plot_ecdf_failrate_thr05

- Reads URA_event_exam_summary_NSE_thresholds.csv
- Builds fail-rate per (basin, model, threshold)
- Plots ONLY: ECDF of basin fail-rate at NSE≥0.5 (events)
- Saves to: <project_root>/results/regional_ecdf_failrate_thr0.5.png

Assumed layout:
  <project_root>/
      codes/plot_figs_04.py   (this script)
      data/ESN_LSTM_comparison/URA_event_exam_summary_NSE_thresholds.csv
      results/  (created)

"""

from __future__ import annotations

import gc
import re
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


# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent            # one step back from /codes
CSV_PATH = PROJECT_ROOT / "data" / "ESN_LSTM_comparison" / "URA_event_exam_summary_NSE_thresholds.csv"
OUTDIR = PROJECT_ROOT / "results"

# -----------------------------
# Helpers
# -----------------------------
def save_close(fig: plt.Figure, path: Path, dpi: int = 250) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    del fig
    gc.collect()

def normalize_model_name(m: str) -> str:
    m = str(m)
    if m.startswith("Best"):
        return "Best L-LSTM"
    if m.strip() == "ESN":
        return "ESN"
    if m.strip() == "L-LSTM":
        return "L-LSTM"
    if m.strip() == "R-LSTM":
        return "R-LSTM"
    return m

def parse_thresholds(df: pd.DataFrame) -> List[float]:
    """Read thresholds from fail_NSE>=X columns in ascending order."""
    cols = [c for c in df.columns if c.startswith("fail_NSE>=")]
    thr = []
    for c in cols:
        m = re.search(r"fail_NSE>=(\d\.\d+)", c)
        if m:
            thr.append(float(m.group(1)))
    return sorted(list(set(thr)))

def build_fail_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return long df with one row per (basin, model, thr):
      total_events, pass_count, fail_count, pass_rate, fail_rate
    """
    df = df.copy()
    df["model2"] = df["model"].map(normalize_model_name)

    thrs = parse_thresholds(df)
    rows = []
    for thr in thrs:
        pass_col = f"pass_NSE>={thr:.1f}"
        fail_col = f"fail_NSE>={thr:.1f}"
        rate_col = f"pass_rate_NSE>={thr:.1f}"

        if pass_col not in df.columns or fail_col not in df.columns or rate_col not in df.columns:
            continue

        sub = df[["basin", "model2", "total_events", pass_col, fail_col, rate_col]].copy()
        sub.rename(
            columns={
                "model2": "model",
                pass_col: "pass_count",
                fail_col: "fail_count",
                rate_col: "pass_rate",
            },
            inplace=True,
        )
        sub["thr"] = thr
        sub["fail_rate"] = 1.0 - pd.to_numeric(sub["pass_rate"], errors="coerce")
        rows.append(sub)

    out = pd.concat(rows, axis=0, ignore_index=True)
    for c in ["total_events", "pass_count", "fail_count", "pass_rate", "fail_rate", "thr"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def ensure_model_order(models: List[str]) -> List[str]:
    pref = ["ESN", "L-LSTM", "Best L-LSTM", "R-LSTM"]
    keep = [m for m in pref if m in models]
    rest = [m for m in models if m not in keep]
    return keep + rest

def plot_ecdf_failrate_at_threshold(fail_long: pd.DataFrame, outdir: Path, thr: float) -> None:
    """ECDF of basin fail-rate at a given threshold (one curve per model)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    models = ensure_model_order(sorted(fail_long["model"].unique().tolist()))
    for m in models:
        sub = fail_long[(fail_long["model"] == m) & (np.isclose(fail_long["thr"], thr))]
        vals = sub["fail_rate"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        x = np.sort(vals)
        y = np.arange(1, x.size + 1) / x.size
        ax.plot(x, y, linewidth=1.6, label=m, color=colors[0][models.index(m)] if m in models else None)

    ax.set_title(f"ECDF of basin fail-rate at NSE≥{thr:.1f} (events)")
    ax.set_xlabel("Fail-rate (per basin)")
    ax.set_ylabel("ECDF")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=1, frameon=True, fancybox=True)
    fig.tight_layout()
    fig.show()

    #save_close(fig, outdir / "fig 04.png")

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df["model"] = df["model"].astype(str)

    fail_long = build_fail_long(df)

    plot_ecdf_failrate_at_threshold(fail_long, OUTDIR, thr=0.5)

    #print(f"Saved: {(OUTDIR / 'regional_ecdf_failrate_thr0.5.png').resolve()}")

if __name__ == "__main__":
    main()
