# -*- coding: utf-8 -*-
"""
Comparing LSTMs vs ESNs

Event-only hydrographs for URA basins from the unified "full_test_library_all_models.p".

Outputs (two versions per event):
  - Normalized:   results/hydrographs/Normalized/<basin>/*_norm.png
  - Not normalized: results/hydrographs/Not_Normalized/<basin>/*_raw.png

Plot layout (3 subplots):
  1) Streamflow (SF)  : normalized by peak(SFobs) within event window (or raw m3/s)
  2) Water level (WL) : normalized by peak(WLobs) within event window (or raw m)
  3) WL abs error     : abs(pred-obs)/peak(WLobs) (normalized) or abs(pred-obs) in cm (raw)

Models shown (3):
  - Local ESN
  - Local LSTM
  - Regional LSTM

Legend (figure-level, 4 rows with ncol=3):
  Row1: Obs
  Row2: Local ESN | Local LSTM | Regional LSTM
  Row3: Local ESN 95% | Local LSTM 95% | Regional LSTM 95%
  Row4: Err Local ESN | Err Local LSTM | Err Regional LSTM
"""

from __future__ import annotations

import gc
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

from matplotlib.patches import Patch
from matplotlib.dates import DateFormatter

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

# Project root (one level above /codes)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

FULL_LIB_PKL = DATA_DIR / "full_test_library_all_models.p"

OUT_ROOT = PROJECT_ROOT / "results" / "hydrographs"
OUT_NORM = OUT_ROOT / "Normalized"
OUT_RAW = OUT_ROOT / "Not_Normalized"

# If you want all basins, set TARGET_BASINS = None
TARGET_BASINS: List[str] | None = ["Lasarte"]

# Test period (used to slice the library)
TEST_START = "2015-10-01 00:00:00"
TEST_END = "2021-09-30 23:00:00"
TEST_START_TS = pd.to_datetime(TEST_START)
TEST_END_TS = pd.to_datetime(TEST_END)

# Event detection parameters
Q_HIGH_PCTL = 0.98
MAX_GAP_HOURS = 24
MIN_CORE_LEN_HOURS = 3
PRE_HOURS = 12
POST_HOURS = 12

# Save plot only if at least one of (Local ESN, Local LSTM) achieves NSE>=thr on SF
PLOT_NSE_GATE = 0.5

# Thresholds for event exam summary CSV
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

# Families in the unified library
FAMS = ["LocalESNs", "LocalLSTMs", "RegionalLSTMs"]
LABELS = {"LocalESNs": "Local ESN", "LocalLSTMs": "Local LSTM", "RegionalLSTMs": "Regional LSTM"}
COLORS = {"LocalESNs": colors[0][0], "LocalLSTMs": colors[0][2], "RegionalLSTMs": colors[0][3]}

# Output CSVs (kept for your workflow)
METRICS_CSV = OUT_ROOT / "URA_metrics_events_only_3models.csv"
EVENT_EXAM_CSV = OUT_ROOT / "URA_event_exam_summary_NSE_thresholds.csv"


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def load_full_lib() -> Dict[str, Any]:
    with open(FULL_LIB_PKL, "rb") as f:
        return pickle.load(f)


def save_close_fig(fig: plt.Figure, path: Path, dpi: int = 250) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    del fig
    gc.collect()


# -----------------------------------------------------------------------------
# Math / metrics
# -----------------------------------------------------------------------------

def _safe_peak(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    v = float(np.nanmax(x.to_numpy(dtype=float))) if x.size else float("nan")
    if not np.isfinite(v) or v == 0.0:
        return float("nan")
    return v


def nse_from_series(obs: pd.Series, sim: pd.Series) -> float:
    o = pd.to_numeric(obs, errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(sim, errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(o) & np.isfinite(s)
    if m.sum() < 2:
        return float("nan")
    o = o[m]; s = s[m]
    den = np.sum((o - np.mean(o)) ** 2)
    if den == 0:
        return float("nan")
    return 1.0 - (np.sum((o - s) ** 2) / den)


def _fmt_nse(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.3f}"


def build_nse_title_line(dfw: pd.DataFrame) -> str:
    """NSE summary line on SF for the three models."""
    obs = pd.to_numeric(dfw.get("SFobs", np.nan), errors="coerce")
    nse_esn = nse_from_series(obs, pd.to_numeric(dfw.get("SF_LocalESNs_median", np.nan), errors="coerce"))
    nse_ll  = nse_from_series(obs, pd.to_numeric(dfw.get("SF_LocalLSTMs_median", np.nan), errors="coerce"))
    nse_rl  = nse_from_series(obs, pd.to_numeric(dfw.get("SF_RegionalLSTMs_median", np.nan), errors="coerce"))
    return f"NSE: LocalESN={_fmt_nse(nse_esn)}  |  LocalLSTM={_fmt_nse(nse_ll)}  |  RegionalLSTM={_fmt_nse(nse_rl)}"


# -----------------------------------------------------------------------------
# Data assembly from full library
# -----------------------------------------------------------------------------

def quantile_band(members: Dict[str, pd.Series]) -> Tuple[pd.Series, pd.Series]:
    """
    Prefer explicit keys if present (q_low/q_high). Otherwise compute 2.5%/97.5% across seeds.
    """
    if "q_low" in members and "q_high" in members:
        return members["q_low"], members["q_high"]
    # Seed members: anything except "median"
    seed_series = [v for k, v in members.items() if k not in ("median",)]
    if not seed_series:
        nan = pd.Series(np.nan, index=members.get("median", pd.Series(dtype=float)).index)
        return nan, nan
    mat = np.vstack([pd.to_numeric(s, errors="coerce").to_numpy(dtype=float) for s in seed_series])
    lo = np.nanpercentile(mat, 2.5, axis=0)
    hi = np.nanpercentile(mat, 97.5, axis=0)
    idx = seed_series[0].index
    return pd.Series(lo, index=idx), pd.Series(hi, index=idx)


def build_basin_df(full: Dict[str, Any], basin: str) -> pd.DataFrame:
    """Obs + median + 95% band for SF and WL (if WL present). No 'best' member."""
    sf_obs = full["obs"]["streamflow"][basin].copy()
    sf_obs = sf_obs.loc[(sf_obs.index >= TEST_START_TS) & (sf_obs.index <= TEST_END_TS)]
    idx = sf_obs.index

    df = pd.DataFrame(index=idx)
    df["SFobs"] = pd.to_numeric(sf_obs, errors="coerce")

    wl_obs = full["obs"].get("water_level", {}).get(basin, pd.Series(np.nan, index=idx)).reindex(idx)
    df["WLobs"] = pd.to_numeric(wl_obs, errors="coerce")

    # Streamflow preds
    for fam in FAMS:
        members = full.get("streamflow_preds", {}).get(fam, {}).get(basin, {})
        if members:
            members = {k: pd.to_numeric(v.reindex(idx), errors="coerce") for k, v in members.items()}
            df[f"SF_{fam}_median"] = members.get("median", pd.Series(np.nan, index=idx))
            lo, hi = quantile_band(members)
            df[f"SF_{fam}_q_low"] = lo.reindex(idx)
            df[f"SF_{fam}_q_high"] = hi.reindex(idx)

    # WL preds (only if WL is in obs library)
    if basin in full.get("obs", {}).get("water_level", {}):
        for fam in FAMS:
            members = full.get("water_level_preds", {}).get(fam, {}).get(basin, {})
            if members:
                members = {k: pd.to_numeric(v.reindex(idx), errors="coerce") for k, v in members.items()}
                df[f"WL_{fam}_median"] = members.get("median", pd.Series(np.nan, index=idx))
                lo, hi = quantile_band(members)
                df[f"WL_{fam}_q_low"] = lo.reindex(idx)
                df[f"WL_{fam}_q_high"] = hi.reindex(idx)

    return df


# -----------------------------------------------------------------------------
# Event detection
# -----------------------------------------------------------------------------

def detect_highflow_events(
    times: pd.DatetimeIndex,
    hf_mask: pd.Series,
    max_gap_hours: int,
    min_core_len_hours: int,
    pre_hours: int,
    post_hours: int,
) -> List[Dict[str, Any]]:
    hf_times = times[hf_mask.values]
    if len(hf_times) == 0:
        return []

    core_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    start = hf_times[0]
    last = hf_times[0]
    for t in hf_times[1:]:
        if (t - last) / pd.Timedelta(hours=1) <= max_gap_hours:
            last = t
        else:
            core_intervals.append((start, last))
            start = t
            last = t
    core_intervals.append((start, last))

    events: List[Dict[str, Any]] = []
    eid = 1
    for c0, c1 in core_intervals:
        core_len = (c1 - c0) / pd.Timedelta(hours=1) + 1
        if core_len < min_core_len_hours:
            continue
        t0 = max(c0 - pd.Timedelta(hours=pre_hours), TEST_START_TS)
        t1 = min(c1 + pd.Timedelta(hours=post_hours), TEST_END_TS)
        events.append({"event_id": eid, "t_start": t0, "t_end": t1})
        eid += 1
    return events


# -----------------------------------------------------------------------------
# Plotting (3 panels, dual output)
# -----------------------------------------------------------------------------

def plot_event_window(
    dfw: pd.DataFrame,
    basin: str,
    event_id: int,
    out_file: Path,
    normalize: bool,
    mm_to_cms: float,
) -> None:
    """
    3-panel plot:
      1) SF (norm or raw)
      2) WL (norm or raw)
      3) WL abs error (norm or raw)
    """
    if dfw.empty:
        return

    # Require usable WL for this figure (your paper figures assume WL exists)
    wl_obs = pd.to_numeric(dfw.get("WLobs", np.nan), errors="coerce")
    if not np.isfinite(wl_obs.to_numpy(dtype=float)).any():
        return

    # Gate on SF NSE for Local ESN or Local LSTM (raw units for NSE)
    obs_sf_gate = pd.to_numeric(dfw.get("SFobs", np.nan), errors="coerce")
    sim_sf_esn  = pd.to_numeric(dfw.get("SF_LocalESNs_median", np.nan), errors="coerce")
    sim_sf_lstm = pd.to_numeric(dfw.get("SF_LocalLSTMs_median", np.nan), errors="coerce")
    nse_esn = nse_from_series(obs_sf_gate, sim_sf_esn)
    nse_ll  = nse_from_series(obs_sf_gate, sim_sf_lstm)
    if not ((np.isfinite(nse_esn) and nse_esn >= PLOT_NSE_GATE) or (np.isfinite(nse_ll) and nse_ll >= PLOT_NSE_GATE)):
        return

    # Peaks for normalization (computed on *raw* obs series in the window)
    sf_peak = _safe_peak(pd.to_numeric(dfw["SFobs"], errors="coerce") * mm_to_cms)
    wl_peak = _safe_peak(pd.to_numeric(dfw["WLobs"], errors="coerce"))

    if normalize:
        if not np.isfinite(sf_peak):
            sf_peak = 1.0
        if not np.isfinite(wl_peak):
            wl_peak = 1.0

    fig, (ax_sf, ax_wl, ax_err) = plt.subplots(
        3, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 3, 2]}
    )

    # ---- SF panel
    sf_obs = pd.to_numeric(dfw["SFobs"], errors="coerce") * mm_to_cms
    sf_plot = sf_obs / sf_peak if normalize else sf_obs
    ax_sf.plot(dfw.index, sf_plot, color="black", linewidth=1.6, label="Obs")

    for fam in FAMS:
        med = pd.to_numeric(dfw.get(f"SF_{fam}_median", np.nan), errors="coerce") * mm_to_cms
        lo  = pd.to_numeric(dfw.get(f"SF_{fam}_q_low", np.nan), errors="coerce") * mm_to_cms
        hi  = pd.to_numeric(dfw.get(f"SF_{fam}_q_high", np.nan), errors="coerce") * mm_to_cms

        if normalize:
            med = med / sf_peak
            lo  = lo / sf_peak
            hi  = hi / sf_peak

        c = COLORS[fam]
        ax_sf.plot(dfw.index, med, linewidth=2, color=c, label=LABELS[fam])
        if np.isfinite(lo.to_numpy(dtype=float)).any() and np.isfinite(hi.to_numpy(dtype=float)).any():
            ax_sf.fill_between(dfw.index, lo, hi, alpha=0.18, color=c)

    ax_sf.set_ylabel("SF / peak(SFobs)" if normalize else "Streamflow (m$^3$/s)")
    ax_sf.grid(True, color="0.9", linewidth=0.6)

    # ---- WL panel
    wl_plot = wl_obs / wl_peak if normalize else wl_obs
    ax_wl.plot(dfw.index, wl_plot, color="black", linewidth=2)  # no label (legend from ax_sf)

    for fam in FAMS:
        med = pd.to_numeric(dfw.get(f"WL_{fam}_median", np.nan), errors="coerce")
        lo  = pd.to_numeric(dfw.get(f"WL_{fam}_q_low", np.nan), errors="coerce")
        hi  = pd.to_numeric(dfw.get(f"WL_{fam}_q_high", np.nan), errors="coerce")

        if normalize:
            med = med / wl_peak
            lo  = lo / wl_peak
            hi  = hi / wl_peak

        c = COLORS[fam]
        ax_wl.plot(dfw.index, med, linewidth=2, color=c)
        if np.isfinite(lo.to_numpy(dtype=float)).any() and np.isfinite(hi.to_numpy(dtype=float)).any():
            ax_wl.fill_between(dfw.index, lo, hi, alpha=0.18, color=c)

    ax_wl.set_ylabel("WL / peak(WLobs)" if normalize else "Water level (m)")
    ax_wl.grid(True, color="0.9", linewidth=0.6)

    # ---- WL abs error panel
    for fam in FAMS:
        wl_med = pd.to_numeric(dfw.get(f"WL_{fam}_median", np.nan), errors="coerce")
        abs_err = (wl_med - wl_obs).abs()

        if normalize:
            err_plot = abs_err / wl_peak
            ax_err.plot(dfw.index, err_plot, linewidth=2.0, color=COLORS[fam], label=f"Err {LABELS[fam]}")
        else:
            err_cm = abs_err * 100.0  # m -> cm
            ax_err.plot(dfw.index, err_cm, linewidth=2.0, color=COLORS[fam], label=f"Err {LABELS[fam]}")

    #ax_err.axhline(0.0, color="0.7", linewidth=1.0)
    ax_err.set_ylabel("WL abs err / peak(WLobs)" if normalize else "WL abs error (cm)")
    ax_err.set_xlabel("Time")
    ax_err.grid(True, color="0.9", linewidth=0.6)
    ax_sf.set_xlim([dfw.index.min(), dfw.index.max()])
    ax_wl.set_xlim([dfw.index.min(), dfw.index.max()])
    ax_err.set_xlim([dfw.index.min(), dfw.index.max()])
    ax_sf.get_yaxis().set_label_coords(-0.05,0.5)
    ax_wl.get_yaxis().set_label_coords(-0.05,0.5)
    ax_err.get_yaxis().set_label_coords(-0.05,0.5)
    date_form = DateFormatter("%d/%m/%y\n%H:%M")
    ax_err.xaxis.set_major_formatter(date_form)

    # Title
    wy = dfw.index[0].year + (1 if dfw.index[0].month >= 10 else 0)  # water year
    title = f"{basin} - Event {event_id:03d} (WY {wy})  {build_nse_title_line(dfw)}"
    fig.suptitle(title, fontsize=FONTSIZETITLE)

    # ---- Figure legend (custom 4-row layout, ncol=3)
    # Build handle map from SF axis (Obs + model lines) and error axis (Err lines)
    handle_map: Dict[str, Any] = {}

    h1, l1 = ax_sf.get_legend_handles_labels()
    for hh, ll in zip(h1, l1):
        handle_map[ll] = hh

    h2, l2 = ax_err.get_legend_handles_labels()
    for hh, ll in zip(h2, l2):
        handle_map[ll] = hh

    # 95% band proxies
    for fam in FAMS:
        band_label = f"{LABELS[fam]} 95%"
        handle_map[band_label] = Patch(facecolor=COLORS[fam], alpha=0.18, edgecolor="none")

    blank = Patch(facecolor="none", edgecolor="none", label=" ")

    desired = [
        "Observed", "Local ESN", "Local LSTM", "Regional LSTM"
    ]

    ordered_h, ordered_l = [], []
    for lab in desired:
        if lab.strip() == "":
            ordered_h.append(blank)
            ordered_l.append(" ")
        else:
            if lab in handle_map:
                ordered_h.append(handle_map[lab])
                ordered_l.append(lab)

    fig.legend(
        ordered_h, ordered_l,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.93),
        ncol=1,
        frameon=True,
        fancybox=True
    )
    fig.tight_layout(rect=[0.03, 0.06, 0.97, 0.95])
    fig.show()

    #save_close_fig(fig, out_file)


# -----------------------------------------------------------------------------
# Metrics + event exam summary
# -----------------------------------------------------------------------------

def add_metrics_rows_for_window(
    rows: List[Dict[str, Any]],
    basin: str,
    event_id: int,
    dfw: pd.DataFrame,
) -> None:
    """Minimal metrics row (SF NSE for 3 models)."""
    obs = pd.to_numeric(dfw.get("SFobs", np.nan), errors="coerce")
    row = {
        "basin": basin,
        "event_id": int(event_id),
        "t0": str(dfw.index.min()),
        "t1": str(dfw.index.max()),
    }
    for fam in FAMS:
        sim = pd.to_numeric(dfw.get(f"SF_{fam}_median", np.nan), errors="coerce")
        row[f"NSE_SF_{LABELS[fam].replace(' ', '')}"] = nse_from_series(obs, sim)
    rows.append(row)


def flush_metrics(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)


# -----------------------------------------------------------------------------
# Main basin processing
# -----------------------------------------------------------------------------

def process_basin(
    basin: str,
    area_km2: float,
    full: Dict[str, Any],
    metrics_rows: List[Dict[str, Any]],
    exam_counts: Dict[Tuple[str, str], Dict[str, Any]],
) -> None:
    df = build_basin_df(full, basin)

    # Unit conversion for SF: mm/h * area(km2) -> m3/s (same as your older scripts)
    mm_to_cms = (area_km2 * 1e6 / 1000.0) / 3600.0  # (km2->m2, mm->m) / s

    # Detect events on SFobs (raw, before conversion)
    sf = pd.to_numeric(df["SFobs"], errors="coerce")
    qthr = np.nanquantile(sf.to_numpy(dtype=float), Q_HIGH_PCTL)
    hf_mask = sf >= qthr

    events = detect_highflow_events(
        times=df.index,
        hf_mask=hf_mask,
        max_gap_hours=MAX_GAP_HOURS,
        min_core_len_hours=MIN_CORE_LEN_HOURS,
        pre_hours=PRE_HOURS,
        post_hours=POST_HOURS,
    )

    # Output dirs
    out_norm = OUT_NORM / basin
    out_raw = OUT_RAW / basin
    out_norm.mkdir(parents=True, exist_ok=True)
    out_raw.mkdir(parents=True, exist_ok=True)

    for ev in events:
        dfw = df.loc[(df.index >= ev["t_start"]) & (df.index <= ev["t_end"])].copy()
        if dfw.empty:
            continue

        eid = int(ev["event_id"])

        if eid == 9 and basin == "Lasarte":

            # Save plots
            plot_event_window(
                dfw=dfw,
                basin=basin,
                event_id=eid,
                out_file=out_norm / f"{basin}_event_{eid:03d}_norm.png",
                normalize=True,
                mm_to_cms=mm_to_cms,
            )
            plot_event_window(
                dfw=dfw,
                basin=basin,
                event_id=eid,
                out_file=out_raw / f"{basin}_event_{eid:03d}_raw.png",
                normalize=False,
                mm_to_cms=mm_to_cms,
            )

        # Metrics row
        add_metrics_rows_for_window(metrics_rows, basin, eid, dfw)

        # Event exam counts (SF NSE thresholds) for 3 models
        obs = pd.to_numeric(dfw.get("SFobs", np.nan), errors="coerce")
        model_series = {
            "Local ESN": pd.to_numeric(dfw.get("SF_LocalESNs_median", np.nan), errors="coerce"),
            "Local LSTM": pd.to_numeric(dfw.get("SF_LocalLSTMs_median", np.nan), errors="coerce"),
            "Regional LSTM": pd.to_numeric(dfw.get("SF_RegionalLSTMs_median", np.nan), errors="coerce"),
        }
        for model, sim in model_series.items():
            key = (basin, model)
            if key not in exam_counts:
                exam_counts[key] = {
                    "total_events": 0,
                    "nse_values": [],
                    "pass": {thr: 0 for thr in THRESHOLDS},
                    "fail": {thr: 0 for thr in THRESHOLDS},
                }
            nse_val = nse_from_series(obs, sim)
            exam_counts[key]["total_events"] += 1
            exam_counts[key]["nse_values"].append(nse_val)
            for thr in THRESHOLDS:
                if np.isfinite(nse_val) and nse_val >= thr:
                    exam_counts[key]["pass"][thr] += 1
                else:
                    exam_counts[key]["fail"][thr] += 1


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    full = load_full_lib()

    # Basin areas
    basins_info_path = PROJECT_ROOT / "basins_info.csv"
    if basins_info_path.exists():
        basins_info = pd.read_csv(basins_info_path)
        basin_names = list(basins_info["basin"].astype(str))
        area_col = None
        for cand in ["area_km2", "AREA_KM2", "area", "Area"]:
            if cand in basins_info.columns:
                area_col = cand
                break
        if area_col is None:
            raise ValueError("No basin area column found in basins_info.csv (need area_km2 for m3/s conversion).")
        basin_to_area = dict(zip(basins_info["basin"].astype(str), basins_info[area_col].astype(float)))
    else:
        basin_names = sorted(list(full.get("obs", {}).get("streamflow", {}).keys()))
        basin_to_area = {b: 1.0 for b in basin_names}
        print("[WARN] basins_info.csv not found. Using area_km2=1.0 for all basins (SF scaling arbitrary).")

    if TARGET_BASINS is not None:
        basin_names = [b for b in basin_names if b in set(TARGET_BASINS)]

    # Reset CSVs
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if METRICS_CSV.exists():
        METRICS_CSV.unlink()

    metrics_rows: List[Dict[str, Any]] = []
    exam_counts: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for basin in basin_names:
        print(f"\n===== Basin: {basin} =====")
        if basin not in full.get("obs", {}).get("streamflow", {}):
            print("  [SKIP] basin not in full library obs.streamflow")
            continue
        if basin not in basin_to_area:
            print("  [SKIP] missing area in basins_info.csv")
            continue
        try:
            process_basin(basin, float(basin_to_area[basin]), full, metrics_rows, exam_counts)
        except Exception as ex:
            print(f"  [SKIP] Basin failed: {ex}")
            gc.collect()

    # Save metrics
    flush_metrics(metrics_rows, METRICS_CSV)
    print(f"\nSaved metrics to: {METRICS_CSV.resolve()}")

    # Save event exam summary
    rows_exam = []
    for (basin, model), rec in exam_counts.items():
        row = {"basin": basin, "model": model, "total_events": rec["total_events"]}
        nse_vals = np.array(rec.get("nse_values", []), dtype=float)
        row["mean_NSE"] = float(np.nanmean(nse_vals)) if nse_vals.size else np.nan
        for thr in THRESHOLDS:
            p = rec["pass"][thr]
            f = rec["fail"][thr]
            row[f"pass_NSE>={thr}"] = p
            row[f"fail_NSE>={thr}"] = f
            row[f"pass_rate_NSE>={thr}"] = (p / rec["total_events"]) if rec["total_events"] else np.nan
        rows_exam.append(row)

    df_exam = pd.DataFrame(rows_exam).sort_values(["basin", "model"])
    df_exam.to_csv(EVENT_EXAM_CSV, index=False)
    print(f"Saved event exam summary to: {EVENT_EXAM_CSV.resolve()}")


if __name__ == "__main__":
    main()
