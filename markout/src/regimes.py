from __future__ import annotations

import numpy as np
import polars as pl

SPLITS = ["session", "volatility", "volume"]
LABEL_ORDER = {
    "session": ["open", "midday", "close"],
    "volatility": ["low", "mid", "high"],
    "volume": ["low", "mid", "high"],
}


def _terciles(col: str, alias: str) -> pl.Expr:
    """Rank-based terciles: exactly equal counts (up to remainder), ties
    broken by row order, no dependence on quantile interpolation."""
    bucket = (pl.col(col).rank(method="ordinal") - 1) * 3 // pl.len()
    return (
        pl.when(bucket == 0)
        .then(pl.lit("low"))
        .when(bucket == 1)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("high"))
        .alias(alias)
    )


def assign_regimes(features: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Session by local time of day; volatility and volume by full-sample
    terciles of the 60 s realized vol and trade intensity."""
    local = pl.col("ts_event").dt.convert_time_zone("America/New_York").dt.strftime("%H:%M:%S")
    session = (
        pl.when(local < config["regime_open_end"])
        .then(pl.lit("open"))
        .when(local < config["regime_close_start"])
        .then(pl.lit("midday"))
        .otherwise(pl.lit("close"))
        .alias("session")
    )
    return features.with_columns(
        session,
        _terciles("realized_vol_60", "volatility"),
        _terciles("intensity_60", "volume"),
    )


def weighted_mean_clustered(
    y: np.ndarray, w: np.ndarray, clusters: np.ndarray
) -> tuple[float, float]:
    """Size-weighted mean with a cluster-robust standard error: the
    sandwich for a weighted intercept-only fit, G/(G-1) corrected."""
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    total = w.sum()
    mean = float((w * y).sum() / total)
    _, inverse = np.unique(clusters, return_inverse=True)
    g = int(inverse.max()) + 1
    scores = np.bincount(inverse, weights=w * (y - mean), minlength=g)
    var = (scores**2).sum() / total**2 * (g / (g - 1)) if g > 1 else float("nan")
    return mean, float(np.sqrt(var))
