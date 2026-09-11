from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.features import NS_PER_SECOND

POLICIES = ["static", "vpin_gated", "composite", "random"]


def split_days(day_labels: list[str], train_share: float) -> tuple[list[str], list[str]]:
    days = sorted(day_labels)
    n_train = math.ceil(len(days) * train_share)
    return days[:n_train], days[n_train:]


@dataclass
class Thresholds:
    vpin_cut: float
    composite_cut: float
    sit_out_rate: float


def fit_thresholds(
    vpin_train: np.ndarray, predicted_train: np.ndarray, sit_out_rate: float
) -> Thresholds:
    """VPIN policy stands down above the train (1 - rate) quantile of VPIN;
    composite stands down below the train `rate` quantile of predicted
    markout. Both therefore sit out the same share of train trades."""
    return Thresholds(
        vpin_cut=float(np.quantile(vpin_train, 1 - sit_out_rate)),
        composite_cut=float(np.quantile(predicted_train, sit_out_rate)),
        sit_out_rate=sit_out_rate,
    )


def participation_masks(
    vpin: np.ndarray, predicted: np.ndarray, thresholds: Thresholds, seed: int
) -> dict[str, np.ndarray]:
    n = len(vpin)
    rng = np.random.default_rng(seed)
    return {
        "static": np.ones(n, dtype=bool),
        "vpin_gated": np.asarray(vpin) <= thresholds.vpin_cut,
        "composite": np.asarray(predicted) >= thresholds.composite_cut,
        "random": rng.random(n) >= thresholds.sit_out_rate,
    }


def fill_shares(size: np.ndarray, mask: np.ndarray, max_fill_shares: int) -> np.ndarray:
    return np.where(mask, np.minimum(np.asarray(size, dtype=np.float64), max_fill_shares), 0.0)


def inventory_path(ts_event: pl.Series, signed_fill: np.ndarray, hold_seconds: float) -> np.ndarray:
    """Signed shares still open at each trade time under hold-H-then-close:
    the sum of signed fills in (t - H, t]."""
    df = pl.DataFrame({"ts_event": ts_event, "q": signed_fill})
    period = f"{int(round(hold_seconds * NS_PER_SECOND))}ns"
    return (
        df.rolling(index_column="ts_event", period=period)
        .agg(pl.col("q").sum().alias("inv"))["inv"]
        .to_numpy()
    )


def max_drawdown(cum_pnl: np.ndarray) -> float:
    if len(cum_pnl) == 0:
        return 0.0
    path = np.concatenate([[0.0], np.asarray(cum_pnl, dtype=np.float64)])
    peak = np.maximum.accumulate(path)
    return float((peak - path).max())


def paired_daily_stats(policy_daily: np.ndarray, baseline_daily: np.ndarray) -> tuple[float, float]:
    d = np.asarray(policy_daily, dtype=np.float64) - np.asarray(baseline_daily, dtype=np.float64)
    n = len(d)
    mean = float(d.mean()) if n else float("nan")
    if n < 2:
        return mean, float("nan")
    se = d.std(ddof=1) / math.sqrt(n)
    return mean, (mean / se if se > 0 else float("nan"))
