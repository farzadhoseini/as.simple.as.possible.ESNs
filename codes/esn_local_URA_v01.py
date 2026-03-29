# esn_local_URA_v01.py
# -----------------------------------------------------------------------------
# Core utilities for local ESN models on URA basins
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from reservoirpy.nodes import Reservoir, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler

LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

BASINS_INFO_FILE = ROOT_DIR / "basins_info.csv"
URA_HOURLY_DIR = ROOT_DIR / "URA_data" / "hourly"

# Shared date windows (dd/mm/yyyy)
# VALID: 01/10/2000–30/09/2005
# TRAIN: 01/10/2005–30/09/2015
# TEST : 01/10/2015–30/09/2021
VALID_START = "2000-10-01"
VALID_END   = "2005-09-30"

TRAIN_START = "2005-10-01"
TRAIN_END   = "2015-09-30"

TEST_START  = "2015-10-01"
TEST_END    = "2021-09-30"

# Minimum length for considering a basin as "having VAL data" (~1 year of hourly data)
MIN_VALID_LEN_HOURS = 24 * 365

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def ema1(x: np.ndarray, alpha: float) -> np.ndarray:
    """Simple first-order exponential moving average filter."""
    if alpha <= 0.0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


@dataclass
class BasinMeta:
    code: str
    name: str
    area_km2: float
    lon: float
    lat: float

    # Optional, per-basin split information coming from basins_info.csv
    val_start: Optional[pd.Timestamp] = None
    val_end: Optional[pd.Timestamp] = None
    train_start: Optional[pd.Timestamp] = None
    train_end: Optional[pd.Timestamp] = None
    test_start: Optional[pd.Timestamp] = None
    test_end: Optional[pd.Timestamp] = None

    # Whether this basin should use water level as a second target
    has_level: bool = False
@dataclass
class BasinDataset:
    """Container with pre-split and scaled data for one basin."""

    meta: BasinMeta

    df: pd.DataFrame
    train: pd.DataFrame
    valid: pd.DataFrame       # may be URA-filled; used only for HPO
    test: pd.DataFrame
    train_valid: pd.DataFrame # ALWAYS original TRAIN + original VALID

    sx: Any
    ssf: Any
    swl: Optional[Any]

    X_train: np.ndarray
    X_valid: np.ndarray
    X_train_valid: np.ndarray
    X_test: np.ndarray

    ysf_train_s: np.ndarray
    ysf_train_valid_s: np.ndarray
    ywl_train_s: Optional[np.ndarray]
    ywl_train_valid_s: Optional[np.ndarray]

    has_wl: bool


# -----------------------------------------------------------------------------
# Data loading and splitting
# -----------------------------------------------------------------------------

def load_basins_info(path: Path = BASINS_INFO_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "basin" not in df.columns:
        raise RuntimeError("basins_info.csv must contain a 'basin' column.")
    return df


def get_basin_meta(
    basin_name: str,
    basins_info: Optional[pd.DataFrame] = None,
) -> BasinMeta:
    """Return BasinMeta for a given basin, including split info.

    The split fields (val_start, train_start, etc.) and level flag are
    taken from basins_info.csv when available. If any field is missing
    or cannot be parsed, it is left as ``None`` and the fallback global
    windows are used in ``split_train_valid_test``.
    """
    if basins_info is None:
        basins_info = load_basins_info()

    row = basins_info.loc[basins_info["basin"] == basin_name]
    if row.empty:
        raise KeyError(f"Basin '{basin_name}' not found in basins_info.csv")
    row = row.iloc[0]

    def _parse_ts(key: str):
        if key not in row or pd.isna(row[key]):
            return None
        val = row[key]
        try:
            return pd.to_datetime(val)
        except Exception:  # pragma: no cover - defensive
            LOGGER.warning(
                "Could not parse %s='%s' for basin '%s'; using None.",
                key,
                val,
                basin_name,
            )
            return None

    val_start = _parse_ts("val_start")
    val_end = _parse_ts("val_end")
    train_start = _parse_ts("train_start")
    train_end = _parse_ts("train_end")
    test_start = _parse_ts("test_start")
    test_end = _parse_ts("test_end")

    level_flag = str(row.get("level_mean", "")).strip().lower()
    has_level = level_flag in {"yes", "y", "true", "1"}

    return BasinMeta(
        code=str(row.get("code", basin_name)),
        name=str(row["basin"]),
        area_km2=float(row.get("area(km2)", row.get("area_km2", 1.0))),
        lon=float(row.get("lon", 0.0)),
        lat=float(row.get("lat", 0.0)),
        val_start=val_start,
        val_end=val_end,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        has_level=has_level,
    )
def load_ura_basin_df(basin_name: str,
                      hourly_dir: Path = URA_HOURLY_DIR) -> pd.DataFrame:
    """Load hourly URA time series for a basin."""
    for ext in (".txt", ".csv"):
        path = hourly_dir / f"{basin_name}{ext}"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
            break
    else:
        raise FileNotFoundError(
            f"No hourly file found for basin '{basin_name}' in {hourly_dir} (txt/csv)."
        )

    required = [
        "date",
        "streamflowmean",
        "precipitation",
        "temperature",
        "potential_evapotranspiration",
    ]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing in {path}")
    return df


def split_train_valid_test(
    df: pd.DataFrame,
    meta: Optional[BasinMeta] = None,
    min_valid_hours: int = MIN_VALID_LEN_HOURS,
    valid_fraction: float = 0.30,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a basin's dataframe into TRAIN, VALID, TEST.

    Priority:
    1. If ``meta`` has fully specified per-basin windows
       (val_start/val_end, train_start/train_end, test_start/test_end),
       those are used.
    2. Otherwise, fall back to the original global/dynamic logic based on
       VALID_START/TRAIN_START/TEST_START and the 30/70 VAL/TRAIN split.

    The VALID segment must have at least ``min_valid_hours`` points;
    otherwise, the function falls back to the dynamic 30/70 split.
    """
    if "date" not in df.columns:
        raise KeyError("Expected a 'date' column in dataframe.")
    df = df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1) Per-basin windows from basins_info.csv (if fully available)
    # ------------------------------------------------------------------
    if meta is not None and all(
        getattr(meta, field) is not None
        for field in (
            "val_start",
            "val_end",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
        )
    ):
        vs, ve = meta.val_start, meta.val_end
        trs, tre = meta.train_start, meta.train_end
        ts, te = meta.test_start, meta.test_end

        valid_mask = (df["date"] >= vs) & (df["date"] <= ve)
        train_mask = (df["date"] >= trs) & (df["date"] <= tre)
        test_mask = (df["date"] >= ts) & (df["date"] <= te)

        valid = df.loc[valid_mask].reset_index(drop=True)
        train = df.loc[train_mask].reset_index(drop=True)
        test = df.loc[test_mask].reset_index(drop=True)

        if len(valid) >= min_valid_hours and not train.empty and not test.empty:
            LOGGER.info(
                "Using per-basin windows from basins_info for basin '%s': "
                "n_train=%d, n_valid=%d, n_test=%d",
                getattr(meta, "name", "<unknown>"),
                len(train),
                len(valid),
                len(test),
            )
            return train, valid, test

        LOGGER.warning(
            "Per-basin windows from basins_info for basin '%s' produced "
            "n_train=%d, n_valid=%d, n_test=%d (min_valid_hours=%d). "
            "Falling back to global/dynamic split.",
            getattr(meta, "name", "<unknown>"),
            len(train),
            len(valid),
            len(test),
            min_valid_hours,
        )

    # ------------------------------------------------------------------
    # 2) Fallback: reproduce original global/dynamic logic
    # ------------------------------------------------------------------
    valid_start = pd.to_datetime(VALID_START)
    valid_end = pd.to_datetime(VALID_END)
    train_start = pd.to_datetime(TRAIN_START)
    train_end = pd.to_datetime(TRAIN_END)
    test_start = pd.to_datetime(TEST_START)
    test_end = pd.to_datetime(TEST_END)

    # Official VALID window
    valid_official = df[
        (df["date"] >= valid_start) & (df["date"] <= valid_end)
    ].reset_index(drop=True)
    n_valid_official = len(valid_official)

    if n_valid_official >= min_valid_hours:
        # Use original hand-designed windows
        train_mask = (df["date"] >= train_start) & (df["date"] <= train_end)
        train = df.loc[train_mask].reset_index(drop=True)
        valid = valid_official

        if train.empty:
            raise RuntimeError(
                "TRAIN segment is empty even though VALID has enough data. "
                "Check TRAIN_START/TRAIN_END vs basin coverage."
            )

        LOGGER.info(
            "Using fixed VAL window [%s, %s] for this basin (%d points in VAL).",
            VALID_START,
            VALID_END,
            n_valid_official,
        )
    else:
        # No VAL or < min_valid_hours in VALID window: build 30/70 split
        LOGGER.warning(
            "Basin has < %d hourly steps in VAL window (%d found). "
            "Using 30%% earliest / 70%% latest split over VAL+TRAIN union.",
            min_valid_hours,
            n_valid_official,
        )

        # VAL+TRAIN union: from VALID_START up to TRAIN_END
        mask_union = (df["date"] >= valid_start) & (df["date"] <= train_end)
        tv = df.loc[mask_union].reset_index(drop=True)
        n = len(tv)

        if n < 10:
            raise RuntimeError(
                "Not enough data in [VALID_START, TRAIN_END] union for this "
                "basin to perform 30/70 split."
            )

        split_idx = int(np.floor(valid_fraction * n))
        valid = tv.iloc[:split_idx].reset_index(drop=True)
        train = tv.iloc[split_idx:].reset_index(drop=True)

        if train.empty:
            raise RuntimeError(
                "TRAIN segment empty after 30/70 split; check data coverage."
            )

        LOGGER.info(
            "Dynamic split for basin: n_total=%d, n_valid=%d (%.1f%%), "
            "n_train=%d (%.1f%%) over [VALID_START, TRAIN_END].",
            n,
            len(valid),
            100.0 * len(valid) / n,
            len(train),
            100.0 * len(train) / n,
        )

    # TEST always taken from [TEST_START, TEST_END]
    test_mask = (df["date"] >= test_start) & (df["date"] <= test_end)
    test = df.loc[test_mask].reset_index(drop=True)

    return train, valid, test
def build_scalers(train: pd.DataFrame,
                  normalization: str = "minmax"):
    X_cols = [
        "precipitation",
        "temperature",
        "potential_evapotranspiration",
    ]

    if normalization == "minmax":
        sx = MinMaxScaler()
        ssf = MinMaxScaler()
        swl = MinMaxScaler()
    else:
        sx = StandardScaler()
        ssf = StandardScaler()
        swl = StandardScaler()

    train_X = train[X_cols].values
    mask_X = np.all(np.isfinite(train_X), axis=1)
    if not mask_X.any():
        raise RuntimeError("No finite forcings in TRAIN to build scalers.")
    sx.fit(train_X[mask_X])

    mask_q = np.isfinite(train["streamflowmean"].values)
    if not mask_q.any():
        raise RuntimeError("No finite streamflow in TRAIN to build scaler.")
    ssf.fit(train.loc[mask_q, ["streamflowmean"]].values)

    if "levelmean" in train.columns:
        mask_wl = np.isfinite(train["levelmean"].values)
        if mask_wl.any():
            swl.fit(train.loc[mask_wl, ["levelmean"]].values)
        else:
            swl = None
    else:
        swl = None

    return sx, ssf, swl


def make_wl_scaled(df_seg: pd.DataFrame, swl):
    if swl is None or "levelmean" not in df_seg.columns:
        return None
    arr = df_seg["levelmean"].values.astype(float)
    y = np.full(len(arr), np.nan, dtype=float)
    m = np.isfinite(arr)
    if m.any():
        y[m] = swl.transform(arr[m].reshape(-1, 1)).ravel()
    return y


def build_basin_dataset(
    basin_name: str,
    normalization: str = "minmax",
    hourly_dir: Path = URA_HOURLY_DIR,
    basins_info: Optional[pd.DataFrame] = None,
) -> BasinDataset:
    """Build a BasinDataset for one basin.

    - TRAIN and TEST are always the original basin series.
    - train_valid (and X_train_valid) are ALWAYS built from original
      TRAIN + original VALID (no URA filling).
    """
    meta = get_basin_meta(basin_name, basins_info=basins_info)
    df = load_ura_basin_df(meta.name, hourly_dir=hourly_dir)

    train, valid, test = split_train_valid_test(df, meta=meta)

    # Keep original TRAIN+VALID union for final training of seeds
    train_valid = pd.concat([train, valid], axis=0)

    # Build scalers on TRAIN only
    sx, ssf, swl = build_scalers(train, normalization=normalization)

    X_cols = [
        "precipitation",
        "temperature",
        "potential_evapotranspiration",
    ]

    def transform_X(df_seg: pd.DataFrame) -> np.ndarray:
        X = df_seg[X_cols].values.astype(float)
        return sx.transform(X)

    X_train = transform_X(train)
    X_valid = transform_X(valid) if not valid.empty else np.empty((0, len(X_cols)))
    X_train_valid = transform_X(train_valid)
    X_test = transform_X(test)


    has_wl_meta = getattr(meta, "has_level", False)

    ysf_train_s = ssf.transform(train[["streamflowmean"]].values.astype(float)).ravel()
    ysf_train_valid_s = ssf.transform(
        train_valid[["streamflowmean"]].values.astype(float)
    ).ravel()

    ywl_train_s = make_wl_scaled(train, swl)
    ywl_train_valid_s = make_wl_scaled(train_valid, swl)

    # If basins_info says this basin has no water level target,
    # force WL targets to None so that only SF is modeled.
    if not has_wl_meta:
        ywl_train_s = None
        ywl_train_valid_s = None

    has_wl = has_wl_meta and (ywl_train_s is not None)
    return BasinDataset(
        meta=meta,
        df=df,
        train=train,
        valid=valid,  # possibly URA-filled; used only for HPO
        test=test,
        train_valid=train_valid,  # always original TRAIN+VALID
        sx=sx,
        ssf=ssf,
        swl=swl,
        X_train=X_train,
        X_valid=X_valid,
        X_train_valid=X_train_valid,
        X_test=X_test,
        ysf_train_s=ysf_train_s,
        ysf_train_valid_s=ysf_train_valid_s,
        ywl_train_s=ywl_train_s,
        ywl_train_valid_s=ywl_train_valid_s,
        has_wl=has_wl,
    )


# -----------------------------------------------------------------------------
# ESN core
# -----------------------------------------------------------------------------

def create_reservoir(params: Dict[str, Any],
                     seed: Optional[int] = None) -> Reservoir:
    return Reservoir(
        units=int(params["reservoir_size"]),
        sr=float(params["spectral_radius"]),
        lr=float(params["leaking_rate"]),
        rc_connectivity=float(params["rc_connectivity"]),
        seed=seed,
    )


def _states_with_washout(
    reservoir: Reservoir,
    Xs: np.ndarray,
    washout: int,
) -> np.ndarray:
    Z = reservoir.run(Xs)
    if washout > 0 and len(Z) > washout:
        return Z[washout:]
    return Z


def _augment_bias(Z: np.ndarray, add_bias: bool) -> np.ndarray:
    if not add_bias:
        return Z
    return np.hstack([Z, np.ones((len(Z), 1), dtype=Z.dtype)])


def fit_readouts(
    reservoir: Reservoir,
    X_train_s: np.ndarray,
    ysf_train_s: np.ndarray,
    ywl_train_s: Optional[np.ndarray],
    *,
    washout: int,
    add_bias: bool,
    ridge_alpha: float,
):
    """Fit Ridge readouts (SF and optional WL) on TRAIN or TRAIN+VALID."""
    Z = _states_with_washout(reservoir, X_train_s, washout)
    Zb = _augment_bias(Z, add_bias)

    # Streamflow readout with NaN masking
    ysf_tail = ysf_train_s[-len(Z):]
    mask_sf = np.isfinite(ysf_tail)
    if not mask_sf.any():
        raise RuntimeError("No finite streamflow targets to fit SF readout.")
    ro_sf = Ridge(ridge=ridge_alpha)
    ro_sf = ro_sf.fit(Zb[mask_sf], ysf_tail[mask_sf].reshape(-1, 1))

    # Water level readout (optional)
    ro_wl = None
    if ywl_train_s is not None:
        ywl_tail = ywl_train_s[-len(Z):]
        mask_wl = np.isfinite(ywl_tail)
        if mask_wl.any():
            ro_wl = Ridge(ridge=ridge_alpha).fit(
                Zb[mask_wl], ywl_tail[mask_wl].reshape(-1, 1)
            )

    return ro_sf, ro_wl


def predict_series(
    reservoir: Reservoir,
    Xs: np.ndarray,
    ro_sf: Ridge,
    ro_wl: Optional[Ridge],
    *,
    ssf,
    swl,
    clamp_nonneg: bool,
    ema_alpha: float,
):
    Z = reservoir.run(Xs)
    Zb = _augment_bias(Z, True)

    psf = np.asarray(ro_sf.run(Zb)).ravel()
    ysf = ssf.inverse_transform(psf.reshape(-1, 1)).ravel()
    if clamp_nonneg:
        ysf = np.maximum(ysf, 0.0)
    if ema_alpha > 0.0:
        ysf = ema1(ysf, ema_alpha)

    if ro_wl is not None and swl is not None:
        pwl = np.asarray(ro_wl.run(Zb)).ravel()
        ywl = swl.inverse_transform(pwl.reshape(-1, 1)).ravel()
    else:
        ywl = np.full_like(ysf, np.nan, dtype=float)

    return ysf, ywl


def build_and_train_for_params(
    basin_ds: BasinDataset,
    params: Dict[str, Any],
    use_train_valid: bool = False,
    seed: Optional[int] = None,
):
    reservoir = create_reservoir(params, seed=seed)

    if use_train_valid:
        Xs = basin_ds.X_train_valid * float(params["input_scaling"])
        ysf = basin_ds.ysf_train_valid_s
        ywl = basin_ds.ywl_train_valid_s
    else:
        Xs = basin_ds.X_train * float(params["input_scaling"])
        ysf = basin_ds.ysf_train_s
        ywl = basin_ds.ywl_train_s

    ro_sf, ro_wl = fit_readouts(
        reservoir,
        Xs,
        ysf,
        ywl,
        washout=int(params["washout"]),
        add_bias=bool(params.get("add_bias", True)),
        ridge_alpha=float(params["ridge_alpha"]),
    )

    return reservoir, ro_sf, ro_wl
