# Design: Horse-Race Regression of Passive-Fill Markouts on Order-Flow Features

Phase 2 of the order-flow-toxicity project. Regresses short-horizon
passive-fill markouts on the full Phase 1 feature set and isolates VPIN's
marginal contribution. The deliverable is one clear finding either way.

## Objective

Produce, from the per-trade features files, a coefficient table with
day-clustered standard errors at three horizons, a VPIN marginal-value
table, a leave-one-out ranking of every feature, a decile sort of realized
markout by predicted markout, and a coefficient chart. Write the finding
into both READMEs.

## Scope guardrails

- No quoting simulator, no P&L, no policies. Phase 3.
- No fast-VPIN variant yet. If VPIN adds nothing, rerun `features.py` with
  `vpin_window_buckets: 10` as a follow-up; that is a config change, not
  new code.
- No out-of-sample split. With 1.9M rows and 15 parameters, overfitting is
  negligible; the README says the decile sort is in-sample.
- No new heavy dependencies. OLS with cluster-robust SEs is implemented in
  numpy and tested against known answers.

## Inputs

`data/processed/{SYM}_{day}_features.parquet` for every day, in date order.
Required columns: `date` is derived from the filename; `aggressor_side`,
`size`, every feature column from `features.feature_columns`, and the
targets `markout_{h}s_bps` for h in `regression_horizons_seconds`.

## Config additions

```yaml
regression_horizons_seconds: [5, 1, 60]   # first is the headline
winsor_quantile: 0.001                     # clip regressors at q and 1-q
n_deciles: 10
```

## Sample construction (`build_design`)

1. Concatenate all days with a `date` column.
2. Directional features are multiplied by `aggressor_side`:
   `signed_imbalance_*`, `ofi_*`, `momentum_*`, `depth_imbalance`,
   `run_length`. Positive then means "flow in the direction of the incoming
   trade", the adverse direction for the maker. Non-directional features
   enter as they are: `intensity_*`, `realized_vol_*`, `spread_bps`, `vpin`.
   Aligned columns keep their names; the alignment is documented, not
   renamed, so the leave-one-out table reads the same as the features table.
3. Drop rows with a null in any regressor or any target. Report the count
   and the share. This removes the VPIN warm-up prefix and empty windows.
4. Winsorize every regressor at `winsor_quantile` and `1 - winsor_quantile`
   over the pooled sample.
5. Z-score every regressor over the pooled sample. Coefficients read as
   "bps of markout per one standard deviation of the feature".
6. Targets are `markout_{h}s_bps`, untouched.

Returns a `Design` dataclass: `X` (float64 numpy, N × K), `feature_names`,
`targets` (dict horizon → float64 numpy N), `clusters` (int numpy N, day
index), `n_dropped`, `n_total`.

## Estimator (`ols_cluster`)

`ols_cluster(X, y, clusters, add_intercept=True) -> OlsResult` with fields
`coef`, `se`, `t`, `r2`, `n`, `k`, `n_clusters`, `names` (intercept named
`const`, placed first).

- β = (XᵀX)⁻¹Xᵀy via `numpy.linalg.lstsq`.
- Residuals e = y − Xβ.
- Meat M = Σ_g (X_gᵀe_g)(X_gᵀe_g)ᵀ over clusters g.
- V = (XᵀX)⁻¹ M (XᵀX)⁻¹ · G/(G−1) · (N−1)/(N−K), the Stata-style
  small-sample correction. G is the number of clusters, K includes the
  intercept.
- se = sqrt(diag V), t = β / se, R² = 1 − Σe² / Σ(y − ȳ)².

## Model set, per horizon

| Name | Regressors |
|---|---|
| `full` | all features |
| `no_vpin` | all features except `vpin` |
| `vpin_only` | `vpin` |
| `drop_<feature>` | all features except `<feature>`, for every feature |

## Outputs

All in `output/`, all committed.

1. `{SYM}_horse_race.csv`: one row per feature; columns `coef_{h}`,
   `se_{h}`, `t_{h}` for each horizon; then rows `const`, `r2`, `n_obs`,
   `n_days` (the last three carry the value in every `coef_{h}` column and
   null elsewhere). Feature rows ordered by |t| at the headline horizon,
   descending.
2. `{SYM}_vpin_marginal.csv`: one row per horizon; columns `horizon`,
   `r2_full`, `r2_no_vpin`, `delta_r2`, `vpin_coef`, `vpin_t`,
   `r2_vpin_only`.
3. `{SYM}_leave_one_out.csv`: one row per feature; columns `feature`,
   `delta_r2_{h}` per horizon (full R² minus the R² without that feature),
   sorted by the headline horizon descending.
4. `{SYM}_decile_sort.csv`: one row per decile of the `full` model's fitted
   value at the headline horizon; columns `decile` (1 = most negative
   predicted markout, i.e. most toxic-looking), `n_trades`,
   `predicted_mean_bps`, `realized_mean_bps` at the headline horizon, and
   `realized_mean_bps_{h}` for the other horizons using the same decile
   assignment. Plus a final row `spread` with decile 10 minus decile 1.
5. `{SYM}_horse_race.png`: one panel, features on the y-axis ordered as in
   the CSV, three series (one per horizon) of coefficient ± 2 SE as
   horizontal point-and-bar marks, vertical zero line. Made with matplotlib
   following the dataviz skill's guidance.

## Sanity checks (printed like the other stages)

Blocking:

1. `n_clusters >= 10`.
2. Every SE finite and positive.
3. `delta_r2` for every leave-one-out model is ≥ 0 (a nested model cannot
   fit better).

Advisory (reported, never blocks):

4. The decile sort's realized means are non-decreasing across at least 70%
   of adjacent deciles. In-sample, so weak by construction; it flags a
   broken fit, not a weak signal.

A draft of this spec had a blocking check that the aligned
`signed_imbalance_5` coefficient be negative, on the grounds that signed
flow is the canonical adverse-selection signal. On the real data that
coefficient is indistinguishable from zero, alone or in the full model.
That is a finding about SPY on a single-venue book, not a pipeline defect,
so the check was removed: gates test properties of a working fit, never
economic hypotheses.

## README content

Both READMEs get a "Phase 2 result" section: the finding in one paragraph,
the headline numbers (VPIN coefficient and t, ΔR² from VPIN vs. the best
feature's ΔR², decile-10-minus-decile-1 spread in bps), the chart, and the
caveats: R² of a few percent is normal at tick level, 18 clusters make SEs
optimistic, VPIN's identification is largely cross-day, decile sort is
in-sample, single-venue book.

## Module structure (`src/regress.py`)

- `DIRECTIONAL_PREFIXES`, `DIRECTIONAL_COLUMNS` constants.
- `align_direction(df) -> pl.DataFrame`
- `winsorize(X: np.ndarray, q: float) -> np.ndarray` (column-wise)
- `standardize(X: np.ndarray) -> np.ndarray` (column-wise z-score)
- `Design` dataclass; `build_design(features: pl.DataFrame, config) -> Design`
- `OlsResult` dataclass; `ols_cluster(X, y, clusters, names) -> OlsResult`
- `run_model_set(design, horizon) -> dict[str, OlsResult]`
- `horse_race_table(results_by_h, headline) -> pl.DataFrame`
- `vpin_marginal_table(results_by_h) -> pl.DataFrame`
- `leave_one_out_table(results_by_h, headline) -> pl.DataFrame`
- `decile_sort_table(design, results_by_h, headline, n_deciles) -> pl.DataFrame`
- `run_regression_checks(results_by_h, headline, loo) -> ValidationReport`
- `plot_coefficients(table, horizons, output_path)`
- `main()`

## Testing (`tests/test_regress.py`)

- `ols_cluster` on a 6-row, 2-cluster fixture with hand-computed β and
  clustered SE (worked out in the test's comment).
- With one observation per cluster, `ols_cluster` SEs equal HC1 SEs
  computed directly in the test.
- `align_direction` flips exactly the directional columns and leaves the
  rest untouched.
- `winsorize` clips exactly at the requested quantiles; `standardize`
  yields mean 0, std 1 per column.
- Synthetic recovery: y = 2·x₁ − 3·x₂ + noise over 5 clusters; `full`
  recovers coefficients within tolerance, |t| > 5 on x₁ and x₂,
  leave-one-out ΔR² is large for x₁, x₂ and near zero for the rest.
- `decile_sort_table`: with fitted values known, decile means match.
- Checks: a positive `signed_imbalance_5` coefficient fails check 3.
