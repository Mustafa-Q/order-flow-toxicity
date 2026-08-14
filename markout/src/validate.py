from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    blocking: bool = True


@dataclass
class ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_blocking_passed(self) -> bool:
        return all(c.passed for c in self.checks if c.blocking)


def run_validation_report(
    markouts: pl.DataFrame, se_table: pl.DataFrame, config: dict
) -> ValidationReport:
    checks = []

    # 1. h=0 ~= +half spread
    mean_h0 = markouts["markout_0s_dollars"].mean()
    mean_half_spread = (markouts["quoted_spread"] / 2).mean()
    diff = abs(mean_h0 - mean_half_spread)
    checks.append(
        CheckResult(
            "h0_equals_half_spread",
            passed=diff < 0.002 and mean_h0 > 0,
            detail=f"mean X(0)=${mean_h0:.5f} vs mean half-spread=${mean_half_spread:.5f} (diff=${diff:.5f})",
        )
    )

    # 2. aggressor buy share ~= 50%
    buy_share = (markouts["aggressor_side"] == 1).mean()
    checks.append(
        CheckResult(
            "aggressor_buy_share",
            passed=0.45 <= buy_share <= 0.55,
            detail=f"buy share={buy_share:.1%} (expect 45-55%)",
        )
    )

    # 3. mean quoted spread ~= $0.01
    mean_spread = markouts["quoted_spread"].mean()
    checks.append(
        CheckResult(
            "mean_spread",
            passed=mean_spread <= 0.02,
            detail=f"mean quoted spread=${mean_spread:.4f} (expect ~$0.01)",
        )
    )

    # 4. trade count per day plausible and stable
    counts = markouts.group_by("date").agg(pl.len().alias("n")).sort("date")
    n_list = counts["n"].to_list()
    ratio = (max(n_list) / min(n_list)) if n_list and min(n_list) > 0 else float("inf")
    checks.append(
        CheckResult(
            "trade_count_stability",
            passed=ratio <= 5,
            detail=f"daily trade counts={n_list} (max/min ratio={ratio:.1f})",
        )
    )

    # 5. no NaNs in markout columns
    markout_cols = [c for c in markouts.columns if c.startswith("markout_")]
    null_counts = {c: markouts[c].null_count() for c in markout_cols}
    total_nulls = sum(null_counts.values())
    checks.append(
        CheckResult(
            "no_nans",
            passed=total_nulls == 0,
            detail=f"null counts by column={null_counts}" if total_nulls else "no nulls found",
        )
    )

    # 6. monotone-ish decay (advisory only)
    horizons = sorted(config["horizons_seconds"])
    sw_means = []
    for h in horizons:
        row = se_table.filter(pl.col("horizon") == h)
        if len(row) and "mean_dollars_sw" in row.columns:
            sw_means.append(row["mean_dollars_sw"][0])
    non_increasing = sum(
        1 for a, b in zip(sw_means, sw_means[1:]) if b <= a
    )
    total_transitions = max(len(sw_means) - 1, 1)
    monotone_ratio = non_increasing / total_transitions
    checks.append(
        CheckResult(
            "monotone_decay",
            passed=monotone_ratio >= 0.7,
            detail=f"{non_increasing}/{total_transitions} transitions were non-increasing",
            blocking=False,
        )
    )

    return ValidationReport(checks=checks)


def print_report(report: ValidationReport) -> None:
    print("\n=== Validation report ===")
    for check in report.checks:
        status = "PASS" if check.passed else ("WARN" if not check.blocking else "FAIL")
        print(f"[{status}] {check.name}: {check.detail}")
    print(f"Overall: {'PASS' if report.all_blocking_passed else 'FAIL'}\n")
