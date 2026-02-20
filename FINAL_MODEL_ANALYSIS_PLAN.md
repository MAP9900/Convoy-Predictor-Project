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

## 4) Statistical Testing DONE
- [ ] For numeric features across FN/FP/TP/TN:
- Run Kruskal-Wallis test per feature.
    - Kruskal-Wallis across 4 groups (all samples)
    - FDR correction (Benjamini-Hochberg)
    - Compute group medians (FN, FP, TP, TN)
    - FN vs TP (Mann-Whitney U, all samples)
        - Cliff’s delta
    - FP vs TN (Mann-Whitney U, all samples)
        - Cliff’s delta
    - FN vs FP (Mann-Whitney U, all samples)
        - Cliff’s delta
- [ ] For significant features (after global screen):
- Run targeted pairwise Mann-Whitney U (or Dunn’s test if keeping full 4-group framework).
- [ ] Apply multiple-testing correction:
- Holm (conservative) or FDR (Benjamini-Hochberg).
- [ ] Add effect sizes (prioritize magnitude over p-value):
- Cliff’s delta (preferred) or rank-biserial correlation.
- [ ] Define interpretation thresholds:
- Small / Medium / Large effect size (report criteria used).
- [ ] Build a ranked summary table:
- `feature`, `comparison`, `median_group1`, `median_group2`, `p_raw`, `p_adj`, `effect_size`, `direction`.

## 5) Calibration and Threshold Quality DONE
- [ ] Plot calibration curve and compute Brier score.
    - Compare to base rate.
    - (If available) compare pre- vs post-calibration.
- [ ] Inspect FN predicted probabilities:
    - Compute median FN probability.
    - % of FNs in 0.20–0.25 (just below threshold).
    - % of FNs in 0.25–0.35 (just above threshold).
    - Determine: threshold artifact vs structural miss.
- [ ] Sweep threshold around current setting:
    - e.g., `0.15` to `0.35` by `0.01`.
    - Track: Recall₁, Precision₁, FPR, MCC, Balanced Accuracy.
- [ ] Choose operating threshold using business objective:
    - Min FN with bounded FP increase, or
    - Max MCC / Balanced Accuracy subject to recall constraint.
- [ ] Check threshold stability:
    - Evaluate optimal threshold across CV folds.

## 6) Feature Importance Triangulation DONE
- [ ] Keep all three importance views:
    - Permutation importance (ensemble-level, evaluated on validation set).
    - Aggregated base-model importance (mean + variance across models).
    - SHAP (tree-compatible base models only).
- [ ] Rank features within each method and normalize scales before comparison.
- [ ] Identify stable features:
    - Appear in top-k across ≥2 methods.
- [ ] Quantify agreement:
    - Rank correlation (Spearman) between importance methods.
- [ ] Flag unstable features (high disagreement or high variance) for caution.
- [ ] Distinguish global vs local importance:
    - Global ranking vs FN-specific SHAP inspection.

## 7) Segment and Temporal Robustness DONE
- [ ] Compare performance metrics by time segment:
    - `Year`, `Month` bins, early-war vs late-war.
    - Track Recall₁, FPR, MCC per segment.
- [ ] If route labels are available:
    - Compare metrics by convoy route segment.
- [ ] Analyze FN distribution by segment:
    - % of total FNs per segment.
    - FN rate within segment.
- [ ] Test for distribution shift:
    - Compare feature distributions across time segments.
- [ ] Evaluate threshold stability by segment:
    - Does optimal threshold vary materially?

## 8) Leakage and Data Quality Checks
- [ ] Reconfirm no leakage columns are present in training inputs:
    - Remove post-outcome or future-derived variables.
- [ ] Validate train/test split integrity:
    - No duplicated rows across splits.
    - No shared convoy identifiers across splits (if applicable).
- [ ] Check metadata joins and index alignment after split.
- [ ] Analyze missingness by confusion group (especially FNs).
- [ ] Inspect outliers by group:
    - Extreme feature values driving predictions?
- [ ] Confirm preprocessing consistency:
    - Same scaling/encoding fitted only on training data.

## 9) Model Improvement Loop 
- [ ] If FN-heavy in specific region:
    - Lower threshold (controlled sweep).
    - Try class weighting or recall-sensitive tuning in base models.
- [ ] Re-run full evaluation:
    - Compare delta in FN, FP, MCC, Balanced Accuracy, Recall₁, Brier.
- [ ] Verify improvement consistency:
    - Cross-validation stability.
    - Segment-level robustness.
- [ ] Maintain concise experiment log:
    - model config, calibration method, threshold,
    - key metrics,
    - FN/FP breakdown,
    - decision and rationale.
- [ ] Stop criteria:
    - Improvement < predefined practical threshold or trade-off unacceptable.

## 10) Deliverables to Save
- [ ] Final metrics table (baseline vs tuned threshold).
- [ ] Final confusion matrix and ROC plot.
- [ ] FN/FP/TP/TN analysis tables.
- [ ] Statistical test summary with corrected p-values + effect sizes.
- [ ] Final written conclusion:
- What changed, why, and what threshold/model you selected.

---



USE PRIOR KNOWLEDGE FROM Model_Loading.md. IMPLEMENT EVERYTHING INTO THE EXISTING WORKFLOW WE JUST ESTABLISHED (DEDICATED .py FILES UNDER src/results, ADD TO FILES WHERE IT MAKES SENSE AND MAKE NEW ONES WHEN NEEDED, PROPER IMPORTS, ETC). MAKE SURE TO ADDITIONALLY, ADD THE CODE BLOCKS TO RUN THE ADDITIONS IN run_results.md AND UPDATE SCRIPTS.md. KEEP CODE SIMPLE AND NO NEED FOR ANY TEST FUNCTIONS. ALSO ADD CODE COMMENTS WHERE APPLICABLE (#WHAT CODE DOES, ETC)


Now can you create results.md with a detailed description of all the additions and what the aim to accomplish. Also include any other relevant information please. 