# ESN_seeds_URA_v01.py
# -----------------------------------------------------------------------------
# After HPO, retrain best local ESN per basin on TRAIN+VALID (original) with
# multiple random seeds and evaluate on TEST.
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import xarray as xr

from esn_local_URA_v01 import (
    ROOT_DIR,
    BasinDataset,
    build_basin_dataset,
    build_and_train_for_params,
)
from metrics_rev import calculate_all_metrics, calculate_metrics

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

SEEDS = list(range(1, 11))
RESOLUTION = "1H"
TIME_COORD = "date"


def _compute_all_metrics(dates: pd.Series,
                         obs: np.ndarray,
                         sim: np.ndarray) -> Dict[str, float]:
    da_obs = xr.DataArray(
        obs,
        coords={TIME_COORD: pd.to_datetime(dates.values)},
        dims=[TIME_COORD],
    )
    da_sim = xr.DataArray(
        sim,
        coords={TIME_COORD: pd.to_datetime(dates.values)},
        dims=[TIME_COORD],
    )

    metrics = calculate_all_metrics(
        da_obs,
        da_sim,
        resolution=RESOLUTION,
        datetime_coord=TIME_COORD,
    )
    extra = calculate_metrics(
        da_obs,
        da_sim,
        metrics=["Missed-Peaks"],
        resolution=RESOLUTION,
        datetime_coord=TIME_COORD,
    )
    metrics.update(extra)
    return metrics


def run_for_basin(
    basin_name: str,
    best_params_row: pd.Series,
    esn_preds_root: Path,
) -> List[Dict[str, Any]]:
    from esn_local_URA_v01 import predict_series

    LOGGER.info("=== ESN seeds experiment for basin '%s' ===", basin_name)
    t0_basin = time.time()

    # IMPORTANT: no URAValidationHelper here → original TRAIN/VALID/TEST
    basin_ds = build_basin_dataset(
        basin_name,
        normalization="minmax",
    )

    hp_keys = [
        "reservoir_size",
        "washout",
        "spectral_radius",
        "leaking_rate",
        "ridge_alpha",
        "input_scaling",
        "rc_connectivity",
        "ema_alpha",
    ]
    params = {k: float(best_params_row[k]) for k in hp_keys}
    params["reservoir_size"] = int(params["reservoir_size"])
    params["washout"] = int(params["washout"])
    params["add_bias"] = True
    params["clamp_nonneg"] = True

    basin_dir = esn_preds_root / basin_name
    basin_dir.mkdir(parents=True, exist_ok=True)

    dates_test = pd.to_datetime(basin_ds.test["date"].values)
    q_obs = basin_ds.test["streamflowmean"].values.astype(float)
    wl_obs = (
        basin_ds.test["levelmean"].values.astype(float)
        if "levelmean" in basin_ds.test.columns
        else np.full_like(q_obs, np.nan, dtype=float)
    )

    all_sf = []
    all_wl = []
    metrics_rows: List[Dict[str, Any]] = []

    for seed in SEEDS:
        LOGGER.info("  Basin %s – seed %d", basin_name, seed)
        t_seed0 = time.time()

        reservoir, ro_sf, ro_wl = build_and_train_for_params(
            basin_ds,
            params,
            use_train_valid=True,
            seed=seed,
        )

        Xtest_scaled = basin_ds.X_test * float(params["input_scaling"])
        ytest_sf, ytest_wl = predict_series(
            reservoir,
            Xtest_scaled,
            ro_sf,
            ro_wl,
            ssf=basin_ds.ssf,
            swl=basin_ds.swl,
            clamp_nonneg=params["clamp_nonneg"],
            ema_alpha=float(params["ema_alpha"]),
        )

        all_sf.append(ytest_sf)
        all_wl.append(ytest_wl)

        df_seed = pd.DataFrame(
            {
                "date": dates_test,
                "SFobs": q_obs,
                "WLobs": wl_obs,
                "SFsim": ytest_sf,
                "WLsim": ytest_wl,
            }
        )
        out_seed = basin_dir / f"ESN_seed{seed:02d}_preds.csv"
        df_seed.to_csv(out_seed, index=False)

        m = _compute_all_metrics(dates_test, q_obs, ytest_sf)
        m_row = {
            "basin": basin_name,
            "seed": seed,
            "tag": "seed",
            **m,
            "runtime_sec": time.time() - t_seed0,
        }
        metrics_rows.append(m_row)

    all_sf_arr = np.vstack(all_sf).T
    ensemble_sf = np.median(all_sf_arr, axis=1)

    all_wl_arr = np.vstack(all_wl).T
    ensemble_wl = np.median(all_wl_arr, axis=1)

    df_ens = pd.DataFrame(
        {
            "date": dates_test,
            "SFobs": q_obs,
            "WLobs": wl_obs,
            "SFsim_ensemble_median": ensemble_sf,
            "WLsim_ensemble_median": ensemble_wl,
        }
    )
    out_ens = basin_dir / "ESN_ensemble_median_preds.csv"
    df_ens.to_csv(out_ens, index=False)

    m_ens = _compute_all_metrics(dates_test, q_obs, ensemble_sf)
    m_ens_row = {
        "basin": basin_name,
        "seed": np.nan,
        "tag": "ensemble_median",
        **m_ens,
        "runtime_sec": time.time() - t0_basin,
    }
    metrics_rows.append(m_ens_row)

    LOGGER.info(
        "Finished basin '%s' – ensemble NSE=%.3f, KGE=%.3f, total time %.1f s",
        basin_name,
        m_ens.get("NSE", np.nan),
        m_ens.get("KGE", np.nan),
        m_ens_row["runtime_sec"],
    )

    return metrics_rows


def main():
    codes_dir = Path(__file__).resolve().parent
    root_dir = ROOT_DIR

    hpo_best_file = codes_dir / "HPO_ESNs_URA.csv"
    if not hpo_best_file.exists():
        raise FileNotFoundError(
            f"HPO_ESNs_URA.csv not found at {hpo_best_file}. "
            "Run esn_local_URA_HPO_v01.py first."
        )

    df_best = pd.read_csv(hpo_best_file)
    basin_names = list(df_best["basin"].astype(str))

    data_dir = root_dir / "data"
    esn_preds_root = data_dir / "ESNsPreds"
    esn_preds_root.mkdir(parents=True, exist_ok=True)

    all_metrics_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []

    t0_all = time.time()
    for i, basin in enumerate(basin_names, start=1):
        LOGGER.info("----- Basin %d/%d: %s -----", i, len(basin_names), basin)
        best_row = df_best.loc[df_best["basin"] == basin].iloc[0]
        t_basin0 = time.time()
        rows = run_for_basin(basin, best_row, esn_preds_root)
        all_metrics_rows.extend(rows)
        runtime_rows.append(
            {
                "basin": basin,
                "total_runtime_sec": time.time() - t_basin0,
            }
        )

    if all_metrics_rows:
        df_metrics = pd.DataFrame(all_metrics_rows)
        out_metrics = codes_dir / "ESN_URA_seeds_metrics.csv"
        df_metrics.to_csv(out_metrics, index=False)
        LOGGER.info("Saved ESN seeds metrics to %s", out_metrics)

    if runtime_rows:
        df_runtime = pd.DataFrame(runtime_rows)
        out_runtime = codes_dir / "ESN_URA_seeds_runtime.csv"
        df_runtime.to_csv(out_runtime, index=False)
        LOGGER.info("Saved ESN seeds runtime log to %s", out_runtime)

    LOGGER.info("Total ESN seeds experiment time: %.1f s", time.time() - t0_all)


if __name__ == "__main__":
    main()
