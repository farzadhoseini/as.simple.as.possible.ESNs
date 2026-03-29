# ComparingLSTMsvsESNs_URA_v01.py
# -----------------------------------------------------------------------------
# Compare LSTM vs ESN performance across URA basins on TEST period.
#
# Expects structure under ROOT_DIR:
#   data/obs/<BASIN>_obs_hr.txt               (date,qobs,lobs,prec,temp,PET)
#   data/LSTMsPreds/<BASIN>_pred_qsim_hr.csv  (date, seed_*, ensemble_median)
#   data/LSTMsPreds/<BASIN>_pred_lsim_hr.csv  (optional, same idea for WL)
#   data/ESNsPreds/<BASIN>/ESN_seedXX_preds.csv
#   data/ESNsPreds/<BASIN>/ESN_ensemble_median_preds.csv
#
# Outputs:
#   codes/URA_LSTM_vs_ESN_metrics_TEST.csv
#   codes/URA_compare_plots/<BASIN>_hydrograph_TEST.png
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from esn_local_URA_v01 import ROOT_DIR, TEST_START, TEST_END
from metrics_rev import calculate_all_metrics

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DATA_DIR = ROOT_DIR / "data"
OBS_DIR = DATA_DIR / "obs"
LSTM_DIR = DATA_DIR / "LSTMsPreds"
ESN_DIR_ROOT = DATA_DIR / "ESNsPreds"

CODES_DIR = Path(__file__).resolve().parent
OUT_DIR = CODES_DIR / "URA_compare_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = CODES_DIR / "URA_LSTM_vs_ESN_metrics_TEST.csv"

RESOLUTION = "1H"
TIME_COORD = "date"


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def load_obs(basin: str) -> pd.DataFrame:
    """
    Load TEST obs for a basin from:
        data/obs/<BASIN>_obs_hr.txt

    Expected columns:
        date, qobs, lobs, prec, temp, PET
    """
    path = OBS_DIR / f"{basin}_obs_hr.txt"  # <- NOTE: .txt here
    if not path.exists():
        raise FileNotFoundError(f"Obs file not found for basin '{basin}': {path}")

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[(df.index >= TEST_START) & (df.index <= TEST_END)]

    if "qobs" not in df.columns:
        raise KeyError(f"Column 'qobs' not found in {path} (needed for SFobs).")
    df.rename(columns={"qobs": "SFobs"}, inplace=True)

    if "lobs" in df.columns:
        df.rename(columns={"lobs": "WLobs"}, inplace=True)
    else:
        df["WLobs"] = np.nan

    df["SFobs"] = df["SFobs"].astype(float)
    df["WLobs"] = df["WLobs"].astype(float)

    return df

def load_lstm_preds(basin: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load LSTM TEST preds for a basin.

    Expects:
      <basin>_pred_qsim_hr.csv with 'date', 'seed_*', 'ensemble_median'
      <basin>_pred_lsim_hr.csv with same structure (for WL). WL file can be missing.
    """
    q_file = LSTM_DIR / f"{basin}_pred_qsim_hr.txt"
    if not q_file.exists():
        raise FileNotFoundError(f"LSTM Q file not found for '{basin}': {q_file}")

    df_q = pd.read_csv(q_file, parse_dates=["date"]).set_index("date").sort_index()
    df_q = df_q.loc[(df_q.index >= TEST_START) & (df_q.index <= TEST_END)]
    df_q = df_q.reindex(idx)

    sf_seed_cols_raw = [c for c in df_q.columns if c.startswith("seed_")]
    if "ensemble_median" in df_q.columns:
        df_q.rename(columns={"ensemble_median": "SF_LSTM_median"}, inplace=True)
    else:
        raise KeyError(f"Column 'ensemble_median' not found in {q_file}")

    # WL file (optional)
    wl_file = LSTM_DIR / f"{basin}_pred_lsim_hr.txt"
    if wl_file.exists():
        df_wl = pd.read_csv(wl_file, parse_dates=["date"]).set_index("date").sort_index()
        df_wl = df_wl.loc[(df_wl.index >= TEST_START) & (df_wl.index <= TEST_END)]
        df_wl = df_wl.reindex(idx)
        wl_seed_cols_raw = [c for c in df_wl.columns if c.startswith("seed_")]
        if "ensemble_median" in df_wl.columns:
            df_wl.rename(columns={"ensemble_median": "WL_LSTM_median"}, inplace=True)
        else:
            raise KeyError(f"Column 'ensemble_median' not found in {wl_file}")
    else:
        df_wl = pd.DataFrame(index=idx)
        wl_seed_cols_raw = []

    df = pd.DataFrame(index=idx)

    # SF members
    for col in sf_seed_cols_raw:
        new_col = f"SF_LSTM_{col}"   # e.g. SF_LSTM_seed_12345
        df[new_col] = df_q[col]

    # WL members
    for col in wl_seed_cols_raw:
        new_col = f"WL_LSTM_{col}"
        df[new_col] = df_wl[col]

    df["SF_LSTM_median"] = df_q["SF_LSTM_median"]
    df["WL_LSTM_median"] = (
        df_wl["WL_LSTM_median"] if "WL_LSTM_median" in df_wl.columns else np.nan
    )

    return df


def load_esn_preds(basin: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load ESN TEST preds for a basin from ESNsPreds/<basin> folder.

    Expects ESN_seeds_URA_v01.py outputs:
      ESN_seedXX_preds.csv: date, SFobs, WLobs, SFsim, WLsim
      ESN_ensemble_median_preds.csv: date, SFobs, WLobs,
                                     SFsim_ensemble_median, WLsim_ensemble_median
    """
    basin_dir = ESN_DIR_ROOT / basin
    if not basin_dir.exists():
        raise FileNotFoundError(f"ESN preds folder not found for '{basin}': {basin_dir}")

    df = pd.DataFrame(index=idx)

    esn_sf_cols = []
    esn_wl_cols = []

    # Per-seed predictions
    for seed in range(1, 11):
        fpath = basin_dir / f"ESN_seed{seed:02d}_preds.csv"
        if not fpath.exists():
            continue

        tmp = pd.read_csv(fpath, parse_dates=["date"]).set_index("date").sort_index()
        tmp = tmp.loc[(tmp.index >= TEST_START) & (tmp.index <= TEST_END)]
        tmp = tmp.reindex(idx)

        label = f"ESN_seed{seed:02d}"
        sf_col = f"SF_{label}"
        wl_col = f"WL_{label}"

        df[sf_col] = tmp["SFsim"].values
        esn_sf_cols.append(sf_col)

        if "WLsim" in tmp.columns:
            df[wl_col] = tmp["WLsim"].values
        else:
            df[wl_col] = np.nan
        esn_wl_cols.append(wl_col)

    # Ensemble median file
    ens_file = basin_dir / "ESN_ensemble_median_preds.csv"
    if ens_file.exists():
        ens = pd.read_csv(ens_file, parse_dates=["date"]).set_index("date").sort_index()
        ens = ens.loc[(ens.index >= TEST_START) & (ens.index <= TEST_END)]
        ens = ens.reindex(idx)
        if "SFsim_ensemble_median" in ens.columns:
            df["SF_ESN_median"] = ens["SFsim_ensemble_median"].values
        if "WLsim_ensemble_median" in ens.columns:
            df["WL_ESN_median"] = ens["WLsim_ensemble_median"].values

    # Ensemble stats from members (mean + 95% band)
    if esn_sf_cols:
        df["SF_ESN"] = df[esn_sf_cols].mean(axis=1, skipna=True)
        df["SF_ESN_q_low"] = df[esn_sf_cols].quantile(0.025, axis=1, numeric_only=True)
        df["SF_ESN_q_high"] = df[esn_sf_cols].quantile(0.975, axis=1, numeric_only=True)
    if esn_wl_cols:
        df["WL_ESN"] = df[esn_wl_cols].mean(axis=1, skipna=True)

    return df


# -----------------------------------------------------------------------------
# Metrics & plotting
# -----------------------------------------------------------------------------

def _metrics_for_pair(
    dates: pd.DatetimeIndex,
    obs: np.ndarray,
    sim: np.ndarray,
) -> Dict[str, float]:
    da_obs = xr.DataArray(obs, coords={TIME_COORD: dates}, dims=[TIME_COORD])
    da_sim = xr.DataArray(sim, coords={TIME_COORD: dates}, dims=[TIME_COORD])

    m = calculate_all_metrics(
        da_obs,
        da_sim,
        resolution=RESOLUTION,
        datetime_coord=TIME_COORD,
    )
    # convert to plain floats
    return {k: float(v) for k, v in m.items()}


def plot_hydrograph_test(basin: str, df: pd.DataFrame) -> None:
    """
    Simple TEST hydrograph plot with LSTM vs ESN for SF.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df.index, df["SFobs"], label="Obs", linewidth=1.2)
    ax.plot(df.index, df["SF_LSTM_median"], label="LSTM med", linewidth=1.0)

    if "SF_ESN_median" in df.columns:
        ax.plot(df.index, df["SF_ESN_median"], label="ESN med", linewidth=1.0)
    elif "SF_ESN" in df.columns:
        ax.plot(df.index, df["SF_ESN"], label="ESN mean", linewidth=1.0)

    if "SF_ESN_q_low" in df.columns and "SF_ESN_q_high" in df.columns:
        ax.fill_between(
            df.index,
            df["SF_ESN_q_low"],
            df["SF_ESN_q_high"],
            alpha=0.2,
            label="ESN 95% band",
        )

    ax.set_title(f"{basin} – TEST streamflow")
    ax.set_ylabel("SF (mm/h)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    out = OUT_DIR / f"{basin}_hydrograph_TEST.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    basins_info = pd.read_csv(ROOT_DIR / "basins_info.csv")
    basin_names = list(basins_info["basin"].astype(str))

    all_rows: List[Dict[str, Any]] = []

    for i, basin in enumerate(basin_names, start=1):
        print(f"===== Basin {i}/{len(basin_names)}: {basin} =====")

        # 1) obs
        try:
            df_obs = load_obs(basin)
        except Exception as ex:
            print(f"  [SKIP] Obs load failed for {basin}: {ex}")
            continue

        idx = df_obs.index

        # 2) LSTM preds
        try:
            df_lstm = load_lstm_preds(basin, idx)
        except Exception as ex:
            print(f"  [SKIP] LSTM preds load failed for {basin}: {ex}")
            continue

        # 3) ESN preds
        try:
            df_esn = load_esn_preds(basin, idx)
        except Exception as ex:
            print(f"  [SKIP] ESN preds load failed for {basin}: {ex}")
            continue

        # Merge
        df = df_obs.copy()
        df["SFobs"] = df["SFobs"].astype(float)
        df["WLobs"] = df["WLobs"].astype(float)

        for col in df_lstm.columns:
            df[col] = df_lstm[col]
        for col in df_esn.columns:
            df[col] = df_esn[col]

        # Metrics SF – LSTM vs ESN
        sf_obs = df["SFobs"].values
        sf_lstm = df["SF_LSTM_median"].values
        if "SF_ESN_median" in df.columns:
            sf_esn = df["SF_ESN_median"].values
        else:
            sf_esn = df["SF_ESN"].values

        m_lstm = _metrics_for_pair(df.index, sf_obs, sf_lstm)
        m_esn = _metrics_for_pair(df.index, sf_obs, sf_esn)

        row_lstm = {"basin": basin, "model": "LSTM", **m_lstm}
        row_esn = {"basin": basin, "model": "ESN", **m_esn}
        all_rows.append(row_lstm)
        all_rows.append(row_esn)

        # Plot hydrograph
        plot_hydrograph_test(basin, df)

    if all_rows:
        dfm = pd.DataFrame(all_rows)
        dfm.to_csv(METRICS_CSV, index=False)
        print(f"Saved URA LSTM vs ESN TEST metrics to {METRICS_CSV}")
    else:
        print("No metrics computed (all basins skipped).")


if __name__ == "__main__":
    main()
