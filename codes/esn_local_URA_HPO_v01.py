# esn_local_URA_HPO_v01.py
# -----------------------------------------------------------------------------
# Basin-wise HPO for local ESNs on URA data.
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from esn_local_URA_v01 import (
    ROOT_DIR,
    BASINS_INFO_FILE,
    BasinDataset,
    build_basin_dataset,
    build_and_train_for_params,
)

from metrics_rev import calculate_metrics

try:
    from skopt import Optimizer
    from skopt.space import Real, Integer
    HAS_SKOPT = True
except Exception:
    HAS_SKOPT = False

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@dataclass
class HPSpace:
    N_min: int = 700
    N_max: int = 1800
    washout_min: int = 60
    washout_max: int = 250
    sr_min: float = 0.3
    sr_max: float = 1.0
    lr_min: float = 1e-3
    lr_max: float = 0.1
    ridge_min: float = 1e-8
    ridge_max: float = 1e-3
    in_scale_min: float = 0.7
    in_scale_max: float = 1.5
    rc_conn_min: float = 0.05
    rc_conn_max: float = 0.4
    ema_min: float = 0.1
    ema_max: float = 0.45

    def to_skopt_space(self) -> List[Any]:
        if not HAS_SKOPT:
            raise RuntimeError("skopt is not available.")
        return [
            Integer(self.N_min, self.N_max, name="reservoir_size"),
            Integer(self.washout_min, self.washout_max, name="washout"),
            Real(self.sr_min, self.sr_max, prior="log-uniform", name="spectral_radius"),
            Real(self.lr_min, self.lr_max, prior="log-uniform", name="leaking_rate"),
            Real(self.ridge_min, self.ridge_max, prior="log-uniform", name="ridge_alpha"),
            Real(self.in_scale_min, self.in_scale_max, prior="uniform", name="input_scaling"),
            Real(self.rc_conn_min, self.rc_conn_max, prior="uniform", name="rc_connectivity"),
            Real(self.ema_min, self.ema_max, prior="uniform", name="ema_alpha"),
        ]

    def sample_random(self, rng: np.random.Generator) -> Dict[str, Any]:
        def loguniform(low, high):
            return float(np.exp(rng.uniform(np.log(low), np.log(high))))

        return {
            "reservoir_size": int(rng.integers(self.N_min, self.N_max + 1)),
            "washout": int(rng.integers(self.washout_min, self.washout_max + 1)),
            "spectral_radius": loguniform(self.sr_min, self.sr_max),
            "leaking_rate": loguniform(self.lr_min, self.lr_max),
            "ridge_alpha": loguniform(self.ridge_min, self.ridge_max),
            "input_scaling": float(rng.uniform(self.in_scale_min, self.in_scale_max)),
            "rc_connectivity": float(rng.uniform(self.rc_conn_min, self.rc_conn_max)),
            "ema_alpha": float(rng.uniform(self.ema_min, self.ema_max)),
            "add_bias": True,
            "clamp_nonneg": True,
        }


HP_SPACE = HPSpace()


def _metrics_for_valid(basin_ds: BasinDataset,
                       params: Dict[str, Any]) -> Tuple[float, float, float]:
    from esn_local_URA_v01 import predict_series

    if basin_ds.valid.empty or len(basin_ds.X_valid) == 0:
        return np.nan, np.nan, -1e6

    reservoir, ro_sf, ro_wl = build_and_train_for_params(
        basin_ds, params, use_train_valid=False, seed=None
    )

    Xv_scaled = basin_ds.X_valid * float(params["input_scaling"])
    yv_sf, _ = predict_series(
        reservoir,
        Xv_scaled,
        ro_sf,
        ro_wl,
        ssf=basin_ds.ssf,
        swl=basin_ds.swl,
        clamp_nonneg=bool(params.get("clamp_nonneg", True)),
        ema_alpha=float(params["ema_alpha"]),
    )

    obs = basin_ds.valid["streamflowmean"].values.astype(float)
    sim = yv_sf

    da_obs = xr.DataArray(
        obs,
        coords={"date": pd.to_datetime(basin_ds.valid["date"].values)},
        dims=["date"],
    )
    da_sim = xr.DataArray(
        sim,
        coords={"date": pd.to_datetime(basin_ds.valid["date"].values)},
        dims=["date"],
    )

    try:
        metrics = calculate_metrics(
            da_obs,
            da_sim,
            metrics=["NSE", "KGE"],
            resolution="1H",
            datetime_coord="date",
        )
        nse_val = float(metrics.get("NSE", np.nan))
        kge_val = float(metrics.get("KGE", np.nan))
    except Exception:
        nse_val, kge_val = np.nan, np.nan

    if not np.isfinite(nse_val) or not np.isfinite(kge_val):
        score = -1e6
    else:
        score = 0.5 * (nse_val + kge_val)

    return nse_val, kge_val, score


@dataclass
class HPOConfig:
    max_calls: int = 500
    min_calls: int = 100
    patience: int = 50
    random_state: int = 123


def run_hpo_for_basin(
    basin_name: str,
    hpo_cfg: HPOConfig,
    out_dir: Path,
) -> Optional[pd.Series]:
    basin_log_path = out_dir / f"HPO_{basin_name}.csv"
    records: List[Dict[str, Any]] = []

    LOGGER.info("=== HPO for basin '%s' ===", basin_name)
    t0_basin = time.time()

    try:
        basin_ds = build_basin_dataset(
            basin_name,
            normalization="minmax",
        )
    except Exception as ex:
        LOGGER.exception("Failed to build dataset for basin '%s': %s", basin_name, ex)
        return None

    if basin_ds.valid.empty:
        LOGGER.warning(
            "Basin '%s' has empty VALID segment even after URA filling – skipping.",
            basin_name,
        )
        return None

    rng = np.random.default_rng(hpo_cfg.random_state + hash(basin_name) % 10000)

    if HAS_SKOPT:
        space = HP_SPACE.to_skopt_space()
        optimizer = Optimizer(
            dimensions=space,
            random_state=hpo_cfg.random_state + hash(basin_name) % 10000,
            base_estimator="GP",
            acq_func="EI",
        )

        best_score = -1e9
        last_improvement = 0
        n_calls = hpo_cfg.max_calls

        keys = [
            "reservoir_size",
            "washout",
            "spectral_radius",
            "leaking_rate",
            "ridge_alpha",
            "input_scaling",
            "rc_connectivity",
            "ema_alpha",
        ]

        for i in range(n_calls):
            x = optimizer.ask()
            params = {k: v for k, v in zip(keys, x)}
            params["add_bias"] = True
            params["clamp_nonneg"] = True

            t_trial0 = time.time()
            try:
                nse_val, kge_val, score = _metrics_for_valid(basin_ds, params)
                loss = -score
            except Exception as ex:
                LOGGER.exception("Error in objective for basin '%s': %s", basin_name, ex)
                nse_val, kge_val, score, loss = np.nan, np.nan, -1e6, 1e6

            t_trial1 = time.time()
            optimizer.tell(x, loss)

            rec = {
                "basin": basin_name,
                "trial": i,
                **params,
                "nse_val": nse_val,
                "kge_val": kge_val,
                "score": score,
                "loss": loss,
                "elapsed_sec": t_trial1 - t_trial0,
            }
            records.append(rec)

            if score > best_score:
                best_score = score
                last_improvement = i

            if (i + 1) % 5 == 0:
                LOGGER.info(
                    "[%s] Trial %d/%d, score=%.4f (best=%.4f so far)",
                    basin_name,
                    i + 1,
                    n_calls,
                    score,
                    best_score,
                )

            if i + 1 >= hpo_cfg.min_calls and i - last_improvement >= hpo_cfg.patience:
                LOGGER.info(
                    "[%s] Early stopping at trial %d (no improvement for %d trials).",
                    basin_name,
                    i + 1,
                    hpo_cfg.patience,
                )
                break

    else:
        best_score = -1e9
        last_improvement = 0
        n_calls = hpo_cfg.max_calls

        for i in range(n_calls):
            params = HP_SPACE.sample_random(rng)
            t_trial0 = time.time()
            try:
                nse_val, kge_val, score = _metrics_for_valid(basin_ds, params)
                loss = -score
            except Exception as ex:
                LOGGER.exception("Error in objective for basin '%s': %s", basin_name, ex)
                nse_val, kge_val, score, loss = np.nan, np.nan, -1e6, 1e6
            t_trial1 = time.time()

            rec = {
                "basin": basin_name,
                "trial": i,
                **params,
                "nse_val": nse_val,
                "kge_val": kge_val,
                "score": score,
                "loss": loss,
                "elapsed_sec": t_trial1 - t_trial0,
            }
            records.append(rec)

            if score > best_score:
                best_score = score
                last_improvement = i

            if (i + 1) % 5 == 0:
                LOGGER.info(
                    "[%s] Trial %d/%d, score=%.4f (best=%.4f so far)",
                    basin_name,
                    i + 1,
                    n_calls,
                    score,
                    best_score,
                )

            if i + 1 >= hpo_cfg.min_calls and i - last_improvement >= hpo_cfg.patience:
                LOGGER.info(
                    "[%s] Early stopping at trial %d (no improvement for %d trials).",
                    basin_name,
                    i + 1,
                    hpo_cfg.patience,
                )
                break

    if not records:
        return None

    df_log = pd.DataFrame(records)
    df_log.to_csv(basin_log_path, index=False)
    LOGGER.info("Saved HPO log for basin '%s' to %s", basin_name, basin_log_path)

    idx_best = df_log["score"].idxmax()
    best_row = df_log.loc[idx_best].copy()
    best_row["best_score"] = best_row["score"]
    best_row["hpo_elapsed_sec"] = time.time() - t0_basin

    LOGGER.info(
        "[%s] HPO best score=%.4f (NSE=%.4f, KGE=%.4f), total HPO time %.1f s",
        basin_name,
        best_row["score"],
        best_row["nse_val"],
        best_row["kge_val"],
        best_row["hpo_elapsed_sec"],
    )

    return best_row


def main():
    codes_dir = Path(__file__).resolve().parent
    hpo_logs_dir = codes_dir / "HPO_logs"
    hpo_logs_dir.mkdir(parents=True, exist_ok=True)

    basins_info = pd.read_csv(BASINS_INFO_FILE)
    basin_names = list(basins_info["basin"].astype(str))

    LOGGER.info("Starting URA ESN HPO for %d basins.", len(basin_names))

    hpo_cfg = HPOConfig()

    best_rows: List[pd.Series] = []

    t0_all = time.time()
    for i, basin in enumerate(basin_names, start=1):
        LOGGER.info("----- Basin %d/%d: %s -----", i, len(basin_names), basin)
        best = run_hpo_for_basin(
            basin_name=basin,
            hpo_cfg=hpo_cfg,
            out_dir=hpo_logs_dir,
        )
        if best is not None:
            best_rows.append(best)

    if not best_rows:
        LOGGER.warning("No successful HPO results; HPO_ESNs_URA.csv not created.")
        return

    df_best = pd.DataFrame(best_rows)
    out_best = codes_dir / "HPO_ESNs_URA.csv"
    df_best.to_csv(out_best, index=False)
    LOGGER.info(
        "Saved best ESN hyperparameters for %d basins to %s",
        len(df_best),
        out_best,
    )

    t_all = time.time() - t0_all
    LOGGER.info("Total HPO wall-clock time: %.1f s", t_all)


if __name__ == "__main__":
    main()
