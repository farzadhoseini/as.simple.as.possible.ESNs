# ComparingLSTMsvsESNs_URA_v01.py
# -----------------------------------------------------------------------------
# URA multi-basin comparison of LSTM vs ESN, following the Lasarte v05 logic:
# - Windows: full, water years, seasons, months, high-flow events
# - Hydrographs per basin & per window
# - Seed-level metrics (LSTM seeds, ESN seeds) + ensemble metrics (LSTM, ESN)
# - Boxplots of seeds (LSTM vs ESN) across windows and basins
#
# File structure expected (per basin B):
#   data/obs/B_obs_hr.txt                (date,qobs,lobs,prec,temp,PET)
#   data/LSTMsPreds/B_pred_qsim_hr.txt   (date,seed_*,ensemble_median)
#   data/LSTMsPreds/B_pred_lsim_hr.txt   (optional, same format for WL)
#   data/ESNsPreds/B/ESN_seedXX_preds.csv
#   data/ESNsPreds/B/ESN_ensemble_median_preds.csv
#
# Outputs:
#   codes/URA_compare_plots/URA_metrics_all_windows_LSTMseeds_vs_ESNseeds.csv
#   codes/URA_compare_plots/<BASIN>/hydrographs_*/...
#   codes/URA_compare_plots/Boxplot_*.png  (seed-level boxplots aggregated)
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import re

from esn_local_URA_v01 import ROOT_DIR, TEST_START, TEST_END  # uses same split
from metrics_rev import calculate_all_metrics, get_available_metrics

# Make sure we work with Timestamps, not strings
TEST_START_TS = pd.Timestamp(TEST_START)
TEST_END_TS   = pd.Timestamp(TEST_END)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DATA_DIR = ROOT_DIR / "data"
OBS_DIR = DATA_DIR / "obs"
LSTM_DIR = DATA_DIR / "LSTMsPreds"
ESN_DIR_ROOT = DATA_DIR / "ESNsPreds"

CODES_DIR = Path(__file__).resolve().parent
OUT_ROOT = CODES_DIR / "URA_compare_plots"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

METRICS_CSV = OUT_ROOT / "URA_metrics_all_windows_LSTMseeds_vs_ESNseeds.csv"

RESOLUTION = "1H"
TIME_COORD = "date"

# High-flow event definition (as Lasarte v05)
Q_HIGH_PCTL        = 0.98
MAX_GAP_HOURS      = 24
MIN_CORE_LEN_HOURS = 3
PRE_HOURS          = 12
POST_HOURS         = 12

# Metrics defined in metrics_rev
METRIC_NAMES = get_available_metrics()

# -----------------------------------------------------------------------------
# I/O helpers (taken from URA_v01, adapted to .txt/.csv preds)
# -----------------------------------------------------------------------------


def _safe_read_csv(path: Path, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    """Robust CSV/TXT reader for URA files.

    Some basin files can have inconsistent delimiters (comma/semicolon/tab/whitespace),
    malformed rows, or encoding quirks. We try a few sane fallbacks before giving up.
    """
    parse_dates = parse_dates or []

    # Ordered from most strict/fast to most forgiving.
    read_tries = [
        dict(sep=",", engine="c"),
        dict(sep=",", engine="python", on_bad_lines="skip"),
        dict(sep=";", engine="python", on_bad_lines="skip"),
        dict(sep="\t", engine="python", on_bad_lines="skip"),
        dict(sep=r"\s+", engine="python", on_bad_lines="skip"),
        dict(sep=None, engine="python", on_bad_lines="skip"),  # delimiter sniffing
    ]

    encodings = ["utf-8", "latin-1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        for kw in read_tries:
            try:
                df = pd.read_csv(
                    path,
                    encoding=enc,
                    parse_dates=parse_dates if parse_dates else None,
                    **kw,
                )
                return df
            except Exception as ex:
                last_err = ex
                continue
    raise last_err if last_err is not None else RuntimeError(f"Failed to read: {path}")


def _standardize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a DatetimeIndex named 'date' exists; drop rows with invalid timestamps."""
    if df.empty:
        return df

    cols_lower = {c.lower(): c for c in df.columns}
    if "date" in cols_lower:
        date_col = cols_lower["date"]
    elif "datetime" in cols_lower:
        date_col = cols_lower["datetime"]
        df = df.rename(columns={date_col: "date"})
        date_col = "date"
    else:
        # Fall back to first column
        first = df.columns[0]
        df = df.rename(columns={first: "date"})
        date_col = "date"

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.loc[df[date_col].notna()].copy()
    df = df.set_index(date_col).sort_index()
    df.index.name = "date"
    return df


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first matching column name (case-insensitive) from candidates."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _has_wl_data(df: pd.DataFrame) -> bool:
    """True if the basin has *usable* water-level observations.

    Some basins have no WL measurements, but their files may still contain a WL column
    filled with placeholders (e.g., all zeros). We consider WL 'available' only when:

    - WLobs exists
    - at least a few finite values exist
    - WLobs is not (near-)constant (std > eps)
    - WLobs is not all zeros

    This prevents plotting/metrics for WL when WL obs are effectively missing.
    """
    if "WLobs" not in df.columns:
        return False

    wl = pd.to_numeric(df["WLobs"], errors="coerce").to_numpy(dtype=float, copy=False)
    wl = wl[np.isfinite(wl)]

    if wl.size < 3:
        return False

    # treat "all zeros" as missing
    if np.nanmin(wl) == 0.0 and np.nanmax(wl) == 0.0:
        return False

    # treat constant (or nearly constant) as missing
    if np.nanstd(wl) < 1e-9:
        return False

    return True



def load_obs(basin: str) -> pd.DataFrame:
    """
    Load TEST observations for a basin from:
        data/obs/<BASIN>_obs_hr.txt

    Expected columns (case-insensitive, flexible):
        - date/datetime
        - qobs (required)  -> renamed to SFobs
        - lobs (optional) -> renamed to WLobs
    Other columns are kept if present.
    """
    path = OBS_DIR / f"{basin}_obs_hr.txt"
    if not path.exists():
        raise FileNotFoundError(f"Obs file not found for basin '{basin}': {path}")

    df = _safe_read_csv(path)
    df = _standardize_datetime_index(df)
    df = df.loc[(df.index >= TEST_START_TS) & (df.index <= TEST_END_TS)].copy()

    q_col = _pick_col(df, ["qobs", "sfobs", "q", "q_obs", "flow", "discharge"])
    if q_col is None:
        raise KeyError(f"Could not find a discharge column (qobs) in {path}. Columns: {list(df.columns)}")
    df = df.rename(columns={q_col: "SFobs"})

    wl_col = _pick_col(df, ["lobs", "wlobs", "wl_obs", "level", "stage", "water_level"])
    if wl_col is not None:
        df = df.rename(columns={wl_col: "WLobs"})
    else:
        df["WLobs"] = np.nan

    df["SFobs"] = pd.to_numeric(df["SFobs"], errors="coerce")
    df["WLobs"] = pd.to_numeric(df["WLobs"], errors="coerce")

    return df

def load_lstm_preds(basin: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load LSTM TEST predictions for a basin.

    Expects (streamflow required):
      <basin>_pred_qsim_hr.txt with a date/datetime column, seed_* members, and an ensemble median.
    Optional (water level):
      <basin>_pred_lsim_hr.txt in the same structure.
    """
    q_file = LSTM_DIR / f"{basin}_pred_qsim_hr.txt"
    if not q_file.exists():
        raise FileNotFoundError(f"LSTM Q file not found for '{basin}': {q_file}")

    df_q = _safe_read_csv(q_file)
    df_q = _standardize_datetime_index(df_q)
    df_q = df_q.loc[(df_q.index >= TEST_START_TS) & (df_q.index <= TEST_END_TS)].copy()
    df_q = df_q.reindex(idx)

    # Identify seed columns (flexible, but typically seed_*)
    seed_cols_q = [c for c in df_q.columns if str(c).lower().startswith("seed_") or re.match(r"^seed\d+", str(c).lower())]

    # Identify ensemble median column
    ens_q = _pick_col(df_q, ["ensemble_median", "median", "ensemble"])
    if ens_q is None:
        # Any non-seed column containing 'ensemble' or 'median' is accepted
        for c in df_q.columns:
            lc = str(c).lower()
            if c in seed_cols_q:
                continue
            if ("ensemble" in lc) or ("median" in lc):
                ens_q = c
                break
    if ens_q is None:
        raise KeyError(f"Could not find LSTM ensemble median column in {q_file}. Columns: {list(df_q.columns)}")

    df = pd.DataFrame(index=idx)
    # SF members
    for col in seed_cols_q:
        df[f"SF_LSTM_{col}"] = pd.to_numeric(df_q[col], errors="coerce")
    df["SF_LSTM_median"] = pd.to_numeric(df_q[ens_q], errors="coerce")

    # WL (optional)
    wl_file = LSTM_DIR / f"{basin}_pred_lsim_hr.txt"
    if wl_file.exists():
        df_wl = _safe_read_csv(wl_file)
        df_wl = _standardize_datetime_index(df_wl)
        df_wl = df_wl.loc[(df_wl.index >= TEST_START_TS) & (df_wl.index <= TEST_END_TS)].copy()
        df_wl = df_wl.reindex(idx)

        seed_cols_wl = [c for c in df_wl.columns if str(c).lower().startswith("seed_") or re.match(r"^seed\d+", str(c).lower())]
        ens_wl = _pick_col(df_wl, ["ensemble_median", "median", "ensemble"])
        if ens_wl is None:
            for c in df_wl.columns:
                lc = str(c).lower()
                if c in seed_cols_wl:
                    continue
                if ("ensemble" in lc) or ("median" in lc):
                    ens_wl = c
                    break

        for col in seed_cols_wl:
            df[f"WL_LSTM_{col}"] = pd.to_numeric(df_wl[col], errors="coerce")
        df["WL_LSTM_median"] = pd.to_numeric(df_wl[ens_wl], errors="coerce") if ens_wl is not None else np.nan
    else:
        df["WL_LSTM_median"] = np.nan


    # 95% reliability bands from per-seed members (used for plotting)
    sf_member_cols = [c for c in df.columns if c.startswith("SF_LSTM_") and c != "SF_LSTM_median"]
    if sf_member_cols:
        df["SF_LSTM_q_low"] = df[sf_member_cols].quantile(0.025, axis=1, numeric_only=True)
        df["SF_LSTM_q_high"] = df[sf_member_cols].quantile(0.975, axis=1, numeric_only=True)
    else:
        df["SF_LSTM_q_low"] = np.nan
        df["SF_LSTM_q_high"] = np.nan

    wl_member_cols = [c for c in df.columns if c.startswith("WL_LSTM_") and c != "WL_LSTM_median"]
    if wl_member_cols:
        df["WL_LSTM_q_low"] = df[wl_member_cols].quantile(0.025, axis=1, numeric_only=True)
        df["WL_LSTM_q_high"] = df[wl_member_cols].quantile(0.975, axis=1, numeric_only=True)
    else:
        df["WL_LSTM_q_low"] = np.nan
        df["WL_LSTM_q_high"] = np.nan
    return df

def load_esn_preds(basin: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load ESN TEST predictions for a basin from ESNsPreds/<basin>/.

    Per-seed files are expected as:
      ESN_seedXX_preds.csv
    with at least date + streamflow simulation column (e.g., SFsim). WLsim can be missing.

    The ensemble median file is optional:
      ESN_ensemble_median_preds.csv
    """
    basin_dir = ESN_DIR_ROOT / basin
    if not basin_dir.exists():
        raise FileNotFoundError(f"ESN preds folder not found for '{basin}': {basin_dir}")

    df = pd.DataFrame(index=idx)
    esn_sf_cols: List[str] = []
    esn_wl_cols: List[str] = []

    # Per-seed predictions
    for seed in range(1, 11):
        fpath = basin_dir / f"ESN_seed{seed:02d}_preds.csv"
        if not fpath.exists():
            continue

        tmp = _safe_read_csv(fpath)
        tmp = _standardize_datetime_index(tmp)
        tmp = tmp.loc[(tmp.index >= TEST_START_TS) & (tmp.index <= TEST_END_TS)].copy()
        tmp = tmp.reindex(idx)

        sf_sim_col = _pick_col(tmp, ["sfsim", "qsim", "sf_sim", "q_sim", "sim", "sf", "q"])
        if sf_sim_col is None:
            raise KeyError(f"Could not find streamflow simulation column in {fpath}. Columns: {list(tmp.columns)}")

        wl_sim_col = _pick_col(tmp, ["wlsim", "lsim", "wl_sim", "l_sim", "wl", "level", "stage"])

        label = f"ESN_seed{seed:02d}"
        sf_col = f"SF_{label}"
        wl_col = f"WL_{label}"

        df[sf_col] = pd.to_numeric(tmp[sf_sim_col], errors="coerce")
        esn_sf_cols.append(sf_col)

        if wl_sim_col is not None:
            df[wl_col] = pd.to_numeric(tmp[wl_sim_col], errors="coerce")
        else:
            df[wl_col] = np.nan
        esn_wl_cols.append(wl_col)

    if not esn_sf_cols:
        raise FileNotFoundError(f"No ESN seed prediction files found for basin '{basin}' in {basin_dir}")

    # Ensemble median file (optional)
    ens_file = basin_dir / "ESN_ensemble_median_preds.csv"
    if ens_file.exists():
        ens = _safe_read_csv(ens_file)
        ens = _standardize_datetime_index(ens)
        ens = ens.loc[(ens.index >= TEST_START_TS) & (ens.index <= TEST_END_TS)].copy()
        ens = ens.reindex(idx)

        # Flexible column picking
        sf_ens = _pick_col(ens, ["sfsim_ensemble_median", "sf_ensemble_median", "qsim_ensemble_median", "ensemble_median"])
        if sf_ens is None:
            for c in ens.columns:
                lc = str(c).lower()
                if ("ensemble" in lc) and ("median" in lc) and ("sf" in lc or "q" in lc):
                    sf_ens = c
                    break
        if sf_ens is not None:
            df["SF_ESN_median"] = pd.to_numeric(ens[sf_ens], errors="coerce")

        wl_ens = _pick_col(ens, ["wlsim_ensemble_median", "wl_ensemble_median", "lsim_ensemble_median"])
        if wl_ens is None:
            for c in ens.columns:
                lc = str(c).lower()
                if ("ensemble" in lc) and ("median" in lc) and ("wl" in lc or "l" in lc or "level" in lc):
                    wl_ens = c
                    break
        if wl_ens is not None:
            df["WL_ESN_median"] = pd.to_numeric(ens[wl_ens], errors="coerce")

    # Ensemble stats from members (mean + 95% band)
    df["SF_ESN"] = df[esn_sf_cols].mean(axis=1, skipna=True)
    df["SF_ESN_q_low"] = df[esn_sf_cols].quantile(0.025, axis=1, numeric_only=True)
    df["SF_ESN_q_high"] = df[esn_sf_cols].quantile(0.975, axis=1, numeric_only=True)

    # WL stats (may end up all-NaN; downstream we skip WL panels if truly absent)
    df["WL_ESN"] = df[esn_wl_cols].mean(axis=1, skipna=True)
    df["WL_ESN_q_low"] = df[esn_wl_cols].quantile(0.025, axis=1, numeric_only=True)
    df["WL_ESN_q_high"] = df[esn_wl_cols].quantile(0.975, axis=1, numeric_only=True)

    return df

# -----------------------------------------------------------------------------
# Utilities (ported & slightly generalized from Lasarte v05)
# -----------------------------------------------------------------------------

def water_year_label(dt: pd.Timestamp) -> int:
    """Water year: 1 Oct y → 30 Sep (y+1)."""
    return dt.year if dt.month >= 10 else dt.year - 1


def season_label_from_month(m: int) -> str:
    """Hydrological seasons: SON (Oct–Dec), DJF (Jan–Mar), MAM (Apr–Jun), JJA (Jul–Sep)."""
    if m in (10, 11, 12):
        return "SON"
    elif m in (1, 2, 3):
        return "DJF"
    elif m in (4, 5, 6):
        return "MAM"
    elif m in (7, 8, 9):
        return "JJA"
    return "UNK"


def detect_highflow_events(times: pd.DatetimeIndex,
                           hf_mask: pd.Series,
                           max_gap_hours: int,
                           min_core_len_hours: int,
                           pre_hours: int,
                           post_hours: int):
    """
    Detect high-flow events from boolean mask (SFobs >= threshold).
    Returns list of dicts with event_id, t_core_start, t_core_end, t_start, t_end.
    """
    hf_times = times[hf_mask.values]
    events = []
    if len(hf_times) == 0:
        return events

    current_start = hf_times[0]
    last_t = hf_times[0]
    core_intervals = []

    for t in hf_times[1:]:
        gap_hours = (t - last_t) / pd.Timedelta(hours=1)
        if gap_hours <= max_gap_hours:
            last_t = t
        else:
            core_intervals.append((current_start, last_t))
            current_start = t
            last_t = t
    core_intervals.append((current_start, last_t))

    event_id = 0
    for t_core_start, t_core_end in core_intervals:
        core_len_hours = (t_core_end - t_core_start) / pd.Timedelta(hours=1) + 1
        if core_len_hours < min_core_len_hours:
            continue

        t_start = t_core_start - pd.Timedelta(hours=pre_hours)
        t_end   = t_core_end   + pd.Timedelta(hours=post_hours)

        t_start = max(t_start, TEST_START_TS)
        t_end   = min(t_end,   TEST_END_TS)

        events.append({
            "event_id": event_id,
            "t_core_start": t_core_start,
            "t_core_end": t_core_end,
            "t_start": t_start,
            "t_end": t_end,
        })
        event_id += 1

    return events


def _metrics_from_da(obs_da: xr.DataArray,
                     sim_da: xr.DataArray,
                     resolution: str = "1H") -> dict:
    """
    Wrapper around calculate_all_metrics with safe NaN handling.
    Uses datetime_coord='date' to match DataArray dim/coord.
    """
    if np.all(np.isnan(obs_da.values)) or np.all(np.isnan(sim_da.values)):
        return {name: np.nan for name in METRIC_NAMES}
    try:
        res = calculate_all_metrics(
            obs_da,
            sim_da,
            resolution=resolution,
            datetime_coord="date",
        )
    except Exception:
        res = {name: np.nan for name in METRIC_NAMES}
    return res


def compute_window_metrics_for_model(
    df_window: pd.DataFrame,
    basin: str,
    var: str,
    sim_col: str,
    model_label: str,
    model_family: str,
    window_type: str,
    window_label: str,
    wy: Optional[int] = None,
    season: Optional[str] = None,
    month: Optional[int] = None,
) -> Optional[dict]:
    """
    Calculate all metrics (from metrics_rev) for a given window, variable and model.
    model_family is one of: 'LSTM_seed', 'ESN_seed', 'LSTM', 'ESN'.
    """
    obs_col = f"{var}obs"
    if obs_col not in df_window or sim_col not in df_window:
        return None

    obs = df_window[obs_col].astype(float)
    sim = df_window[sim_col].astype(float)

    if obs.dropna().empty or sim.dropna().empty:
        metrics_dict = {name: np.nan for name in METRIC_NAMES}
    else:
        obs_da = xr.DataArray(
            obs.to_numpy(),
            coords={"date": obs.index},
            dims=["date"],
        )
        sim_da = xr.DataArray(
            sim.to_numpy(),
            coords={"date": sim.index},
            dims=["date"],
        )
        metrics_dict = _metrics_from_da(obs_da, sim_da, resolution="1H")

    row = {
        "basin": basin,
        "window_type": window_type,
        "window_label": window_label,
        "var": var,                # "SF" or "WL"
        "model": model_label,      # e.g. "LSTM_seed_64410", "ESN_seed01", "LSTM", "ESN"
        "model_family": model_family,
        "wy": wy,
        "season": season,
        "month": month,
        "t_start": df_window.index.min(),
        "t_end": df_window.index.max(),
    }
    for mname in METRIC_NAMES:
        row[mname] = metrics_dict.get(mname, np.nan)

    return row


def _extract_core_metrics_for_annot(rows_for_annot: dict) -> tuple[str, str]:
    """
    Build two lines of text (for Q and WL) using metrics of:
      - LSTM ensemble median (model_family='LSTM')
      - ESN mean ensemble (model_family='ESN')
    rows_for_annot is like:
      {"LSTM": {"SF": row, "WL": row}, "ESN": {...}}
    """
    def fmt(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "nan"
        return f"{v:.3f}"

    def get(row, name):
        if row is None:
            return np.nan
        return row.get(name, np.nan)

    # --- Streamflow
    row_q_lstm = rows_for_annot.get("LSTM", {}).get("SF")
    row_q_esn  = rows_for_annot.get("ESN", {}).get("SF")

    nse_q_lstm = get(row_q_lstm, "NSE")
    nse_q_esn  = get(row_q_esn,  "NSE")
    kge_q_lstm = get(row_q_lstm, "KGE")
    kge_q_esn  = get(row_q_esn,  "KGE")
    fhv_q_lstm = get(row_q_lstm, "FHV")
    fhv_q_esn  = get(row_q_esn,  "FHV")

    line_q = (
        f"NSE(Q): LSTM={fmt(nse_q_lstm)}, ESN={fmt(nse_q_esn)}; "
        f"KGE(Q): LSTM={fmt(kge_q_lstm)}, ESN={fmt(kge_q_esn)}; "
        f"FHV(Q): LSTM={fmt(fhv_q_lstm)}, ESN={fmt(fhv_q_esn)}"
    )

    # --- Water level
    row_w_lstm = rows_for_annot.get("LSTM", {}).get("WL")
    row_w_esn  = rows_for_annot.get("ESN", {}).get("WL")

    nse_w_lstm = get(row_w_lstm, "NSE")
    nse_w_esn  = get(row_w_esn,  "NSE")
    kge_w_lstm = get(row_w_lstm, "KGE")
    kge_w_esn  = get(row_w_esn,  "KGE")
    fhv_w_lstm = get(row_w_lstm, "FHV")
    fhv_w_esn  = get(row_w_esn,  "FHV")

    line_w = (
        f"NSE(WL): LSTM={fmt(nse_w_lstm)}, ESN={fmt(nse_w_esn)}; "
        f"KGE(WL): LSTM={fmt(kge_w_lstm)}, ESN={fmt(kge_w_esn)}; "
        f"FHV(WL): LSTM={fmt(fhv_w_lstm)}, ESN={fmt(fhv_w_esn)}"
    )

    return line_q, line_w


def plot_hydrograph(
    df_window: pd.DataFrame,
    title: str,
    out_file: Path,
    rows_for_annot: Optional[dict],
    mm_to_cms_factor: float,
):
    """
    Hydrograph panels:

    - Always: Streamflow (Obs + LSTM median + ESN mean) with 95% bands for BOTH models (from seeds).
    - If WL exists (has at least one finite value): Water level hydrograph + WL error (cm).
    - If WL is missing: Streamflow error (m3/s) instead of WL panels.

    Colors for the error panel are kept identical to the corresponding hydrograph lines.
    """
    if df_window.empty:
        return

    plot_wl = _has_wl_data(df_window)

    if plot_wl:
        fig, (ax_sf, ax_wl, ax_err) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    else:
        fig, (ax_sf, ax_err) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # --- Streamflow (convert mm/h -> m3/s)
    sf_obs = pd.to_numeric(df_window.get("SFobs", np.nan), errors="coerce") * mm_to_cms_factor
    sf_lstm = pd.to_numeric(df_window.get("SF_LSTM_median", np.nan), errors="coerce") * mm_to_cms_factor
    sf_esn = pd.to_numeric(df_window.get("SF_ESN", np.nan), errors="coerce") * mm_to_cms_factor

    sf_lstm_low = pd.to_numeric(df_window.get("SF_LSTM_q_low", np.nan), errors="coerce") * mm_to_cms_factor
    sf_lstm_high = pd.to_numeric(df_window.get("SF_LSTM_q_high", np.nan), errors="coerce") * mm_to_cms_factor
    sf_esn_low = pd.to_numeric(df_window.get("SF_ESN_q_low", np.nan), errors="coerce") * mm_to_cms_factor
    sf_esn_high = pd.to_numeric(df_window.get("SF_ESN_q_high", np.nan), errors="coerce") * mm_to_cms_factor

    (l_obs,) = ax_sf.plot(df_window.index, sf_obs, label="Obs", linewidth=1.5)
    (l_lstm,) = ax_sf.plot(df_window.index, sf_lstm, label="LSTM median", linestyle="--", linewidth=1.3)
    (l_esn,) = ax_sf.plot(df_window.index, sf_esn, label="ESN mean", linestyle="-.", linewidth=1.3)

    # LSTM 95% band
    if not np.all(np.isnan(sf_lstm_low)) and not np.all(np.isnan(sf_lstm_high)):
        ax_sf.fill_between(
            df_window.index,
            sf_lstm_low,
            sf_lstm_high,
            alpha=0.22,
            color=l_lstm.get_color(),
            label="LSTM 95% band",
        )

    # ESN 95% band
    if not np.all(np.isnan(sf_esn_low)) and not np.all(np.isnan(sf_esn_high)):
        ax_sf.fill_between(
            df_window.index,
            sf_esn_low,
            sf_esn_high,
            alpha=0.22,
            color=l_esn.get_color(),
            label="ESN 95% band",
        )

    ax_sf.set_ylabel("Streamflow (m$^3$/s)")
    ax_sf.grid(True, color="0.9", linewidth=0.5)

    # Collect legend handles across axes
    handles_all: List[Any] = []
    labels_all: List[str] = []
    h, l = ax_sf.get_legend_handles_labels()
    handles_all += h
    labels_all += l

    if plot_wl:
        # --- Water level panel
        wl_obs = pd.to_numeric(df_window.get("WLobs", np.nan), errors="coerce")
        wl_lstm = pd.to_numeric(df_window.get("WL_LSTM_median", np.nan), errors="coerce")
        wl_esn = pd.to_numeric(df_window.get("WL_ESN", np.nan), errors="coerce")

        wl_lstm_low = pd.to_numeric(df_window.get("WL_LSTM_q_low", np.nan), errors="coerce")
        wl_lstm_high = pd.to_numeric(df_window.get("WL_LSTM_q_high", np.nan), errors="coerce")
        wl_esn_low = pd.to_numeric(df_window.get("WL_ESN_q_low", np.nan), errors="coerce")
        wl_esn_high = pd.to_numeric(df_window.get("WL_ESN_q_high", np.nan), errors="coerce")

        (w_obs,) = ax_wl.plot(df_window.index, wl_obs, label="Obs (WL)", linewidth=1.5)
        (w_lstm,) = ax_wl.plot(df_window.index, wl_lstm, label="LSTM median (WL)", linestyle="--", linewidth=1.3)
        (w_esn,) = ax_wl.plot(df_window.index, wl_esn, label="ESN mean (WL)", linestyle="-.", linewidth=1.3)

        if not np.all(np.isnan(wl_lstm_low)) and not np.all(np.isnan(wl_lstm_high)):
            ax_wl.fill_between(
                df_window.index,
                wl_lstm_low,
                wl_lstm_high,
                alpha=0.22,
                color=w_lstm.get_color(),
                label="LSTM 95% band (WL)",
            )
        if not np.all(np.isnan(wl_esn_low)) and not np.all(np.isnan(wl_esn_high)):
            ax_wl.fill_between(
                df_window.index,
                wl_esn_low,
                wl_esn_high,
                alpha=0.22,
                color=w_esn.get_color(),
                label="ESN 95% band (WL)",
            )

        ax_wl.set_ylabel("Water level (m)")
        ax_wl.grid(True, color="0.9", linewidth=0.5)

        h, l = ax_wl.get_legend_handles_labels()
        handles_all += h
        labels_all += l

        # --- Water level error (cm) with SAME colors as WL hydrograph
        err_lstm_cm = (wl_lstm - wl_obs) * 100.0
        err_esn_cm = (wl_esn - wl_obs) * 100.0

        ax_err.axhline(0.0, color="0.7", linewidth=1.0)
        ax_err.plot(df_window.index, err_lstm_cm, linestyle="--", linewidth=1.0, color=w_lstm.get_color(), label="LSTM error (WL, cm)")
        ax_err.plot(df_window.index, err_esn_cm, linestyle="-.", linewidth=1.0, color=w_esn.get_color(), label="ESN error (WL, cm)")
        ax_err.set_ylabel("WL error (cm)")
        ax_err.set_xlabel("Time")
        ax_err.grid(True, color="0.9", linewidth=0.5)

        h, l = ax_err.get_legend_handles_labels()
        handles_all += h
        labels_all += l
    else:
        # --- Streamflow error (m3/s) with SAME colors as SF hydrograph
        err_lstm = sf_lstm - sf_obs
        err_esn = sf_esn - sf_obs

        ax_err.axhline(0.0, color="0.7", linewidth=1.0)
        ax_err.plot(df_window.index, err_lstm, linestyle="--", linewidth=1.0, color=l_lstm.get_color(), label="LSTM error (SF)")
        ax_err.plot(df_window.index, err_esn, linestyle="-.", linewidth=1.0, color=l_esn.get_color(), label="ESN error (SF)")
        ax_err.set_ylabel("SF error (m$^3$/s)")
        ax_err.set_xlabel("Time")
        ax_err.grid(True, color="0.9", linewidth=0.5)

        h, l = ax_err.get_legend_handles_labels()
        handles_all += h
        labels_all += l

    # Single shared legend at bottom (outside plots)
    if handles_all:
        fig.legend(
            handles_all,
            labels_all,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=max(1, min(len(set(labels_all)), 4)),
            frameon=False,
        )

    fig.autofmt_xdate()

    # Title + metrics annotation
    full_title = title
    if rows_for_annot is not None:
        line_q, line_w = _extract_core_metrics_for_annot(rows_for_annot)
        if plot_wl:
            full_title = f"{title}\n{line_q}\n{line_w}"
        else:
            full_title = f"{title}\n{line_q}"

    fig.suptitle(full_title, y=0.98, fontsize=11)

    if plot_wl:
        fig.tight_layout(rect=[0.03, 0.08, 0.97, 0.93])
    else:
        fig.tight_layout(rect=[0.03, 0.08, 0.97, 0.92])

    fig.savefig(out_file, dpi=300)
    plt.close(fig)

def process_basin(
    basin: str,
    area_km2: Optional[float],
    metrics_rows: List[dict],
):
    """
    Process one basin:
      - build df (obs + LSTM seeds + ESN seeds + ensembles)
      - detect high flow events
      - loop over windows (events, WY, seasons, months, full)
      - append metrics rows
      - write hydrographs in basin-specific folders under OUT_ROOT
    """
    print(f"===== Basin: {basin} =====")

    # Per-basin output dirs
    basin_root = OUT_ROOT / basin
    hydro_events_dir = basin_root / "hydrographs_events"
    hydro_wy_dir     = basin_root / "hydrographs_WY"
    hydro_season_dir = basin_root / "hydrographs_seasons"
    hydro_month_dir  = basin_root / "hydrographs_months"
    hydro_full_dir   = basin_root / "hydrographs_full"
    for d in [basin_root, hydro_events_dir, hydro_wy_dir, hydro_season_dir,
              hydro_month_dir, hydro_full_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load data for basin
    # -----------------------------
    df_obs = load_obs(basin)
    idx = df_obs.index

    df_lstm = load_lstm_preds(basin, idx)
    df_esn  = load_esn_preds(basin, idx)

    df = df_obs.copy()
    for col in df_lstm.columns:
        df[col] = df_lstm[col]
    for col in df_esn.columns:
        df[col] = df_esn[col]

    # Identify member columns
    lstm_sf_member_cols = [c for c in df.columns if c.startswith("SF_LSTM_seed_")]
    lstm_wl_member_cols = [c for c in df.columns if c.startswith("WL_LSTM_seed_")]
    esn_sf_member_cols  = [c for c in df.columns if c.startswith("SF_ESN_seed")]
    esn_wl_member_cols  = [c for c in df.columns if c.startswith("WL_ESN_seed")]

    LSTM_MEMBER_COLS = {"SF": lstm_sf_member_cols, "WL": lstm_wl_member_cols}
    ESN_MEMBER_COLS  = {"SF": esn_sf_member_cols,  "WL": esn_wl_member_cols}

    # Ensemble means (if not already present)
    if "SF_ESN" not in df.columns and esn_sf_member_cols:
        df["SF_ESN"] = df[esn_sf_member_cols].mean(axis=1, skipna=True)
    if "WL_ESN" not in df.columns and esn_wl_member_cols:
        df["WL_ESN"] = df[esn_wl_member_cols].mean(axis=1, skipna=True)

    # Discharge conversion (mm/h -> m3/s) – area_km2 is REQUIRED
    if area_km2 is None or not np.isfinite(area_km2) or area_km2 <= 0:
        raise ValueError(
            f"Invalid or missing area_km2 for basin '{basin}'. "
            "Check basins_info.csv."
        )

    # 1 mm over 1 km^2 in 1 hour = 1000 m^3 / 3600 s
    mm_to_cms_factor = area_km2 * 1000.0 / 3600.0

    # ==========================
    # Helper: process a window
    # ==========================
    def process_window(
        df_win: pd.DataFrame,
        window_type: str,
        window_label: str,
        title: str,
        out_file: Path,
        wy: Optional[int] = None,
        season: Optional[str] = None,
        month: Optional[int] = None,
    ):
        nonlocal metrics_rows

        if df_win.empty:
            return

        rows_for_annot: dict = {"LSTM": {}, "ESN": {}}

        vars_to_do = ["SF"]
        if _has_wl_data(df_win):
            vars_to_do.append("WL")

        for var in vars_to_do:
            # --- LSTM seeds
            for col in LSTM_MEMBER_COLS[var]:
                seed_id = col.replace("SF_LSTM_seed_", "").replace("WL_LSTM_seed_", "")
                model_label = f"LSTM_seed_{seed_id}"
                row = compute_window_metrics_for_model(
                    df_win, basin, var, col,
                    model_label=model_label,
                    model_family="LSTM_seed",
                    window_type=window_type,
                    window_label=window_label,
                    wy=wy,
                    season=season,
                    month=month,
                )
                if row is not None:
                    metrics_rows.append(row)

            # --- ESN seeds
            for col in ESN_MEMBER_COLS[var]:
                # e.g. SF_ESN_seed01 -> ESN_seed01
                if col.startswith(f"{var}_"):
                    model_label = col[len(f"{var}_"):]
                else:
                    model_label = col
                row = compute_window_metrics_for_model(
                    df_win, basin, var, col,
                    model_label=model_label,
                    model_family="ESN_seed",
                    window_type=window_type,
                    window_label=window_label,
                    wy=wy,
                    season=season,
                    month=month,
                )
                if row is not None:
                    metrics_rows.append(row)

            # --- LSTM ensemble median
            sim_col_lstm = f"{var}_LSTM_median"
            if sim_col_lstm in df_win.columns:
                row_lstm = compute_window_metrics_for_model(
                    df_win, basin, var, sim_col_lstm,
                    model_label="LSTM",
                    model_family="LSTM",
                    window_type=window_type,
                    window_label=window_label,
                    wy=wy,
                    season=season,
                    month=month,
                )
                if row_lstm is not None:
                    metrics_rows.append(row_lstm)
                    rows_for_annot["LSTM"][var] = row_lstm

            # --- ESN mean ensemble
            sim_col_esn = f"{var}_ESN"
            if sim_col_esn in df_win.columns:
                row_esn = compute_window_metrics_for_model(
                    df_win, basin, var, sim_col_esn,
                    model_label="ESN",
                    model_family="ESN",
                    window_type=window_type,
                    window_label=window_label,
                    wy=wy,
                    season=season,
                    month=month,
                )
                if row_esn is not None:
                    metrics_rows.append(row_esn)
                    rows_for_annot["ESN"][var] = row_esn

        # Hydrograph
        plot_hydrograph(df_win, title, out_file, rows_for_annot, mm_to_cms_factor)

    # ==================================================
    # High-flow events
    # ==================================================
    q_thr = df["SFobs"].quantile(Q_HIGH_PCTL)
    print(f"  Q{int(Q_HIGH_PCTL*100)} threshold for SFobs = {q_thr:.4f}")
    hf_mask = df["SFobs"] >= q_thr
    events = detect_highflow_events(
        df.index,
        hf_mask,
        MAX_GAP_HOURS,
        MIN_CORE_LEN_HOURS,
        PRE_HOURS,
        POST_HOURS,
    )
    print(f"  Detected {len(events)} high-flow events.")

    for ev in events:
        t0, t1 = ev["t_start"], ev["t_end"]
        df_win = df.loc[t0:t1].copy()
        wy_peak = water_year_label(ev["t_core_start"])
        win_label = f"event_{ev['event_id']:03d}_WY{wy_peak}"
        title = f"{basin} – High-flow event {ev['event_id']} (WY {wy_peak})"
        out_file = hydro_events_dir / f"{basin}_Hydrograph_event_{ev['event_id']:03d}.png"
        process_window(
            df_win,
            window_type="event",
            window_label=win_label,
            title=title,
            out_file=out_file,
            wy=wy_peak,
        )

    # ==================================================
    # Water years, seasons, months, full period
    # ==================================================
    wys   = sorted({water_year_label(ts) for ts in df.index})
    years = sorted({ts.year for ts in df.index})

    # --- Water years
    for wy in wys:
        wy_start = pd.Timestamp(f"{wy}-10-01 00:00:00")
        wy_end   = pd.Timestamp(f"{wy+1}-09-30 23:00:00")
        df_win = df.loc[(df.index >= wy_start) & (df.index <= wy_end)].copy()
        if df_win.empty:
            continue
        win_label = f"WY_{wy}"
        title = f"{basin} – Hydrograph WY {wy}"
        out_file = hydro_wy_dir / f"{basin}_Hydrograph_WY_{wy}.png"
        process_window(
            df_win,
            window_type="wateryear",
            window_label=win_label,
            title=title,
            out_file=out_file,
            wy=wy,
        )

    # --- Seasons
    season_months = {
        "SON": [10, 11, 12],
        "DJF": [1, 2, 3],
        "MAM": [4, 5, 6],
        "JJA": [7, 8, 9],
    }
    for wy in wys:
        wy_start = pd.Timestamp(f"{wy}-10-01 00:00:00")
        wy_end   = pd.Timestamp(f"{wy+1}-09-30 23:00:00")
        df_wy = df.loc[(df.index >= wy_start) & (df.index <= wy_end)].copy()
        if df_wy.empty:
            continue
        for season, months in season_months.items():
            df_win = df_wy.loc[df_wy.index.month.isin(months)].copy()
            if df_win.empty:
                continue
            win_label = f"{season}_WY{wy}"
            title = f"{basin} – {season} season (WY {wy})"
            out_file = hydro_season_dir / f"{basin}_Hydrograph_{season}_WY{wy}.png"
            process_window(
                df_win,
                window_type="season",
                window_label=win_label,
                title=title,
                out_file=out_file,
                wy=wy,
                season=season,
            )

    # --- Months (calendar)
    for year in years:
        df_year = df.loc[df.index.year == year].copy()
        if df_year.empty:
            continue
        for month in range(1, 12 + 1):
            df_win = df_year.loc[df_year.index.month == month].copy()
            if df_win.empty:
                continue
            wy_for_month = water_year_label(df_win.index[0])
            season = season_label_from_month(month)
            win_label = f"{year}_{month:02d}"
            title = f"{basin} – Hydrograph {year}-{month:02d}"
            out_file = hydro_month_dir / f"{basin}_Hydrograph_{year}_{month:02d}.png"
            process_window(
                df_win,
                window_type="month",
                window_label=win_label,
                title=title,
                out_file=out_file,
                wy=wy_for_month,
                season=season,
                month=month,
            )

    # --- Full period
    df_full = df.copy()
    title_full = (
        f"{basin} – Hydrograph full period "
        f"{df_full.index.min().date()} – {df_full.index.max().date()}"
    )
    out_full = hydro_full_dir / f"{basin}_Hydrograph_full_period.png"
    process_window(
        df_full,
        window_type="full",
        window_label="full_period",
        title=title_full,
        out_file=out_full,
        wy=None,
    )


# -----------------------------------------------------------------------------
# Boxplots over all basins (seed-level distributions LSTM vs ESN)
# -----------------------------------------------------------------------------

def make_seed_boxplots(df_metrics: pd.DataFrame):
    """
    Make boxplots of seed distributions for:
      - NSE, KGE, FHV
      - window_type in {event, wateryear, season, month, full}
      - var in {SF, WL}
    Aggregated over all basins.
    """
    def boxplot_by_window_seedfamilies(
        dfm: pd.DataFrame,
        window_type: str,
        var: str,
        metric: str,
        out_name: str,
    ):
        sub = dfm[
            (dfm["window_type"] == window_type)
            & (dfm["var"] == var)
            & (dfm["model_family"].isin(["LSTM_seed", "ESN_seed"]))
        ].copy()

        if metric not in sub.columns:
            print(f"  [Boxplot] Metric {metric} not in dataframe columns")
            return

        sub = sub[np.isfinite(sub[metric])]
        if sub.empty:
            print(f"  [Boxplot] No seed data for {window_type}, {var}, {metric}")
            return

        families = ["LSTM_seed", "ESN_seed"]
        labels = ["LSTM seeds", "ESN seeds"]
        data = [sub[sub["model_family"] == fam][metric].values for fam in families]

        if any(len(d) == 0 for d in data):
            print(f"  [Boxplot] Missing data for some family: {window_type}, {var}, {metric}")
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = ["lightgray", "lightblue"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set(facecolor=c)

        ax.set_ylabel(metric)
        ax.set_title(f"{metric} – {var}, {window_type} windows (seed distributions)")

        if metric in ("NSE", "KGE"):
            all_vals = np.concatenate([d for d in data if len(d) > 0])
            if len(all_vals) > 0:
                thr = np.nanpercentile(all_vals, 10)
                y_min = max(0.5, thr)
                ax.set_ylim(y_min, 1.0)

        fig.tight_layout()
        fig.savefig(OUT_ROOT / out_name, dpi=300)
        plt.close(fig)

    for metric in ["NSE", "KGE", "FHV"]:
        for window_type in ["event", "wateryear", "season", "month", "full"]:
            for var in ["SF", "WL"]:
                out_name = f"Boxplot_{metric}_{var}_{window_type}_seeds_allBasins.png"
                boxplot_by_window_seedfamilies(df_metrics, window_type, var, metric, out_name)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    # Read basins_info to get basin names and areas if available
    basins_info = pd.read_csv(ROOT_DIR / "basins_info.csv")
    basin_names = list(basins_info["basin"].astype(str))

    # Try to find an area column (REQUIRED for m3/s conversion)
    area_col_candidates = ["area_km2", "AREA_KM2", "area", "Area"]
    area_col = None
    for cand in area_col_candidates:
        if cand in basins_info.columns:
            area_col = cand
            break

    print("Basins:", basin_names)
    print("Area column used:", area_col)

    if area_col is None:
        raise ValueError(
            "No basin area column found in basins_info.csv. "
            "Please add a column named 'area_km2' (or 'AREA_KM2', 'area', 'Area') "
            "with basin areas in km^2 so streamflow can be plotted in m^3/s."
        )

    metrics_rows: List[dict] = []

    for basin in basin_names:
        if area_col is not None:
            area_val = basins_info.loc[basins_info["basin"] == basin, area_col]
            area_km2 = float(area_val.iloc[0]) if not area_val.empty else None
        else:
            area_km2 = None

        try:
            process_basin(basin, area_km2, metrics_rows)
        except Exception as ex:
            print(f"[SKIP] Basin {basin} failed with error: {ex}")

    # Save metrics
    if metrics_rows:
        dfm = pd.DataFrame(metrics_rows)
        dfm.to_csv(METRICS_CSV, index=False)
        print(f"\nSaved URA metrics (all basins/windows) to: {METRICS_CSV}")

        # Seed-level boxplots over all basins
        make_seed_boxplots(dfm)
    else:
        print("No metrics computed (all basins skipped).")


if __name__ == "__main__":
    main()
