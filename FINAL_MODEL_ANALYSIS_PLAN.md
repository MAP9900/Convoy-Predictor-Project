# Final Model Analysis Plan

This checklist is for analyzing the final ensemble model (`FiveModel_CalSoft_t0.25`) and turning findings into clear next actions.

## 1) Reproducibility and Setup
- [ ] Restart kernel and run all cells in order.
- [ ] Confirm one canonical split is used everywhere (`train_size=0.8`, `random_state=1945`, `stratify=y`).
- [ ] Confirm feature names match model inputs exactly (no leaked/dropped-column mismatch).
- [ ] Save key artifacts from this run: confusion matrix, ROC, permutation importance, aggregated importance, SHAP plot.

## 2) Core Performance Snapshot
- [ ] Record: `Acc`, `Recall_1`, `Precision_1`, `F1_1`, `MCC`, `Bal_Acc`, `ROC_AUC`.
- [ ] Record confusion matrix counts: `TN`, `FP`, `FN`, `TP`.
- [ ] Track these metrics in a small history table by run date + model name.

## 3) Error-Group Deep Dive (FN / FP / TP / TN)
- [ ] Build and keep `all_rows_scored` with full original rows and flags:
- `Is_Test`, `Actual`, `Predicted`, `Pred_Prob`, `Is_False_Negative`, `Is_False_Positive`, `Is_True_Positive`, `Is_True_Negative`.
- [ ] Review each subset dataframe:
- `false_negatives`, `false_positives`, `true_positives`, `true_negatives`.
- [ ] Use `compare_confusion_group_describes(...)` for side-by-side `describe()` comparisons.

## 4) Statistical Testing (Recommended Next)
- [ ] For numeric features across FN/FP/TP/TN:
- Run Kruskal-Wallis test per feature.
- [ ] For significant features:
- Run pairwise Mann-Whitney U (or Dunn’s test) between relevant groups.
- [ ] Apply multiple-testing correction:
- Holm or FDR (Benjamini-Hochberg).
- [ ] Add effect sizes (not just p-values):
- Cliff’s delta or rank-biserial correlation.
- [ ] Build a ranked summary table:
- `feature`, `p_raw`, `p_adj`, `effect_size`, `direction`.

## 5) Calibration and Threshold Quality
- [ ] Plot calibration curve and compute Brier score.
- [ ] Inspect FN predicted probabilities:
- Are many FNs near threshold (e.g., 0.20–0.35)?
- [ ] Sweep threshold around current setting:
- e.g., `0.15` to `0.35` by `0.01`.
- [ ] Choose operating threshold using business objective:
- Min FN, bounded FP increase, or max MCC/Balanced Accuracy.

## 6) Feature Importance Triangulation
- [ ] Keep all three importance views:
- Permutation importance (ensemble-level),
- Aggregated base-model importance,
- SHAP (tree-compatible base models).
- [ ] Look for stable features appearing in top ranks across all 3 methods.
- [ ] Flag unstable features (high disagreement) for caution in interpretation.

## 7) Segment and Temporal Robustness
- [ ] Compare error rates by time segment:
- `Year`, `Month` bins, early-war vs late-war.
- [ ] If route labels are available, compare by convoy route segment.
- [ ] Check if FN concentration is segment-specific (distribution shift risk).

## 8) Leakage and Data Quality Checks
- [ ] Reconfirm no leakage columns are present in training inputs.
- [ ] Check missingness/outliers by group (especially FNs).
- [ ] Validate that metadata joins/index alignment are correct after split.
- [ ] Confirm no duplicated rows across train/test split.

## 9) Model Improvement Loop
- [ ] If FN-heavy in specific region:
- Lower threshold or try class weighting for recall-sensitive base models.
- [ ] Re-run evaluation and compare to baseline:
- delta in FN, FP, MCC, Bal_Acc, Recall_1.
- [ ] Keep a concise experiment log:
- config, threshold, metrics, key findings, decision.

## 10) Deliverables to Save
- [ ] Final metrics table (baseline vs tuned threshold).
- [ ] Final confusion matrix and ROC plot.
- [ ] FN/FP/TP/TN analysis tables.
- [ ] Statistical test summary with corrected p-values + effect sizes.
- [ ] Final written conclusion:
- What changed, why, and what threshold/model you selected.

---

## Suggested Immediate Next 3 Actions
1. Run Kruskal-Wallis + corrected pairwise tests on numeric features across FN/FP/TP/TN.
2. Run a focused threshold sweep (`0.15–0.35`) and choose threshold based on FN-vs-FP tradeoff.
3. Build a one-page summary table with metrics, top stable features, and top FN patterns.
