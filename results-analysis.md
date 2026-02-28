# Results Analysis (from `notebooks/models/Results.ipynb`)

## 1) Executive Summary
The final calibrated soft-voting ensemble (`FiveModel_CalSoft_t0.25`) performs as a **useful but imperfect risk-screening model**.

At threshold `0.25` on the held-out test set (`n=235`):
- ROC AUC: `0.821`
- Accuracy: `0.830`
- Recall (At-Risk class): `0.680`
- Precision (At-Risk class): `0.586`
- F1 (At-Risk class): `0.630`
- MCC: `0.522`
- Balanced Accuracy: `0.775`
- Confusion matrix: TN `161`, FP `24`, FN `16`, TP `34`

Interpretation:
- The model separates classes reasonably well (AUC > 0.8).
- The major practical limitation is **missed at-risk convoys** (16 FNs), not random threshold noise.
- Performance is **not stable across years/time regimes**, indicating temporal drift and threshold instability.

## 2) Model Performance and Tradeoffs
### Baseline threshold (`0.25`)
`0.25` is the strongest operating point in this notebook for balanced utility:
- It produced the **best MCC** in the sweep (`0.522`).
- It kept false positive rate relatively low (`0.130`) while retaining moderate recall (`0.680`).

### Threshold sweep findings (`0.15` to `0.35`)
- Best accuracy occurred near `0.32` (`0.843`) but with lower recall (more missed positives).
- Lowering threshold increases recall but rapidly raises FP burden:
  - At `0.21`: recall `0.700`, FPR `0.227`, MCC `0.412`
  - At `0.19`: recall `0.760`, FPR `0.286`, MCC `0.398`
- Selection outputs confirm:
  - `min_fn_bounded_fp` -> `0.25`
  - `max_mcc_with_recall_constraint` -> `0.21`
  - `max_bal_acc_with_recall_constraint` -> `0.19`

Operational takeaway:
- If analyst capacity is limited, `0.25` is the best compromise.
- If mission cost of missed attacks is dominant, consider `0.21` or `0.19` with explicit acceptance of many more false alarms.

## 3) False Negative (FN) Diagnostics
FN threshold diagnostics:
- FN count: `16`
- Median FN probability: `0.168`
- Only `6.25%` of FNs are near the threshold window
- Heuristic: `structural miss`

Interpretation:
- Most FNs are not “just below cutoff”; they are scored far from positive.
- This is a **representation/coverage issue** (model blind spots), not just a threshold tuning issue.

## 4) Confusion-Group Pattern Analysis
Comparing group statistics (FN/FP/TP/TN):
- FN and FP groups tend to involve **larger convoys** and **more escorts** than TP.
- FN group has higher average **U-boat activity** than TP (mean ~`51.8` vs `33.0`).
- `Total Tons of Convoy` is higher in FN/FP than TP, suggesting size-related ambiguity.

Important caveat:
- Formal corrected hypothesis tests did **not** find statistically significant group differences after multiplicity correction:
  - Global Kruskal significant features after FDR: `0`
  - Pairwise MWU significant comparisons after adjustment: `0`

Interpretation:
- There are directional signals, but current sample size/noise means weak statistical confidence.

## 5) Feature Importance Triangulation
Three views were combined: permutation, aggregated base-model importance, and SHAP.

Top consensus features by average rank:
1. `Previous Month Avg Sink %`
2. `Number of Stragglers`
3. `Total Tons of Convoy`
4. `Avg Number of U-Boats in Atlantic`
5. `Time At Sea (Days)`

Agreement between methods:
- Base rank vs SHAP rank: Spearman `0.882` (strong, significant)
- Permutation vs base/shap: moderate only

Interpretation:
- Model-internal attribution methods are consistent with each other.
- Permutation adds a somewhat different perspective, indicating interaction/nonlinearity effects.

FN-specific SHAP delta (features more important inside FNs):
- Largest positive jump: `Month` (`+0.147`)
- Then: `Time At Sea (Days)` (`+0.024`), `Escort Ratio` (`+0.013`)

Interpretation:
- Misses are disproportionately linked to **seasonality/time context**, which global importance alone under-emphasizes.

## 6) Segment and Temporal Robustness
Year-level metrics show strong heterogeneity:
- 1942: recall `0.875`, MCC `0.667` (strong)
- 1943: recall `0.600`, MCC `0.476` (drop)
- 1944: recall `0.000`, MCC `0.000` (failure under tiny positive count)

FN distribution by year:
- Highest FN share in 1941 (`31.25%` of all FNs), then 1940/1944 (`18.75%` each)

Shift tests:
- Many features show highly significant distribution shifts across eras/segments.
- Early vs Late medians moved strongly in escorts, ships, sighting range, tons, and predicted probability.

Threshold stability:
- Segment-optimal thresholds vary materially (`~0.15` to `0.32`).
- Every evaluated year segment showed material deviation from global threshold.
- Some segments could not satisfy recall constraints (notably Late/1944 contexts).

Interpretation:
- A single global threshold is not robust over time.
- The model is exposed to **temporal regime change**.

## 7) Leakage and Data Quality Audit
Audit outcomes are mostly clean:
- Leakage flags: none
- Split integrity: no duplicate/ID overlap leakage detected
- Train/test alignment checks: all pass
- Missingness rates: effectively zero across confusion groups

Residual risk signals:
- `Number of Stragglers`: outlier concentration flagged in FP group (medium risk)
- `Total Tons of Convoy`: outlier concentration flagged in FN group (medium risk)

Interpretation:
- Pipeline quality is good; remaining error issues are not explained by obvious leakage or missingness.
- Tail behavior and outliers likely contribute to unstable edge-case predictions.

## 8) Potential Next Steps
1. Add time-aware validation (rolling-origin or year-block CV) and report per-era confidence intervals for recall/FN rate.
2. Introduce threshold policy by segment (e.g., early/late war or year buckets) instead of one global cutoff.
3. Address structural FNs using targeted reweighting/cost-sensitive learning or FN-focused hard-example retraining.
4. Add calibration-by-segment checks (ECE/Brier per era) to detect when global calibration degrades.
5. Investigate robust transforms/winsorization for `Total Tons of Convoy` and `Number of Stragglers` to reduce outlier-driven errors.
6. Expand feature space with lagged/interaction terms for seasonality and operational tempo (Month x U-boat pressure, escort density interactions).
7. Define deployment guardrails: drift triggers, retraining cadence, and “human review required” rules for uncertain probability bands.
