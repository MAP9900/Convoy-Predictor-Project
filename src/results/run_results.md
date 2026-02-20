# Run Results (Notebook Cells)

```python
# Cell 1: Core imports
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.models.model_artifacts import get_artifact_dir
from src.results.model_loading_core import make_shared_split, set_seed
from src.results.ensemble_models import load_final_ensemble_models, run_five_model_calsoft_t025
from src.results.visualization_functions import (
    plot_aggregated_base_model_importance,
    plot_confusion_matrix,
    plot_permutation_importance,
    plot_roc_curve,
    plot_shap_importance,
)
from src.results.confusion_groups import (
    compare_confusion_group_describes,
    get_false_negatives,
    get_false_positives,
    get_true_negatives,
    get_true_positives,
)
from src.results.statistical_testing import (
    build_summary_table,
    run_conditional_targeted_tests,
    run_global_kruskal,
    run_pairwise_tests,
)
from src.results.calibration_threshold_eval import (
    calibration_report,
    fn_probability_diagnostics,
    select_threshold,
    threshold_stability_cv,
    threshold_sweep,
)
from src.results.feature_importance_triangulation import (
    build_triangulation_report,
    compute_base_model_importance,
    compute_permutation_importance,
    compute_rank_agreement,
    compute_shap_importance,
    fn_specific_shap_analysis,
    identify_stable_unstable_features,
    normalize_importances,
)
from src.results.segment_temporal_robustness import (
    add_time_segments,
    build_segment_temporal_report,
    compute_fn_distribution,
    compute_segment_metrics,
    distribution_shift_tests,
    select_threshold_from_sweep,
    threshold_stability_by_segment,
    threshold_sweep_metrics,
)
from src.results.leakage_data_quality_checks import (
    audit_alignment,
    audit_preprocessing,
    build_risk_summary,
    check_split_integrity,
    confusion_groups,
    flag_leakage_columns,
    missingness_by_confusion_group,
    outlier_bounds_from_train,
    outliers_by_confusion_group,
)
```

```python
# Cell 2: Recreate model-ready dataset for X, y, feature_names
set_seed(1945)

raw_df = pd.read_csv('/Users/matthewplambeck/Desktop/Convoy Predictor/data/processed/Complete_Convoy_Data.csv')
raw_df = raw_df.drop(columns=['Unnamed: 0']).reset_index(drop=True)

model_df = raw_df.drop(columns=[
    'Convoy Number',
    'Number of Ships Sunk',
    'Depart_Date',
    'Arrival/Dispersal Date',
    'Number of Escorts Sunk',
    'Number of Stragglers Sunk',
    'Total Tons of Ships Sunk',
    'Escort Sink Percentage',
    'Straggler Sink Percentage',
])
model_df['Risk'] = (model_df['Overall Sink Percentage'] > 0).astype(int)

X = model_df.drop(columns=['Overall Sink Percentage', 'Risk'])
y = model_df['Risk'].values
feature_names = X.columns.tolist()
```

```python
# Cell 3: Shared train/test split
X_train, X_test, y_train, y_test = make_shared_split(X, y, train_size=0.8, random_state=1945)
```

```python
# Cell 4: Load saved base models used by final ensemble
ARTIFACT_DIR = get_artifact_dir('algorithm_test_3')

loaded = load_final_ensemble_models(
    artifact_dir=ARTIFACT_DIR,
    feature_names=feature_names,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)

dt = loaded['dt']
rf = loaded['rf']
et = loaded['et']
ada = loaded['ada']
qda = loaded['qda']
```

```python
# Cell 5: Base estimators for five-model ensemble
dt_ensemble = dt.best_model
rf_ensemble = rf.best_model
et_ensemble = et.best_model
ada_ensemble = ada.best_model
qda_ensemble = qda.best_model
```

```python
# Cell 6: Run calibrated soft-voting ensemble (threshold=0.25)
res_calsoft_025, voter_calsoft_025, cm_calsoft_025, rep_calsoft_025 = run_five_model_calsoft_t025(
    X_train,
    X_test,
    y_train,
    y_test,
    qda_ensemble,
    ada_ensemble,
    dt_ensemble,
    rf_ensemble,
    et_ensemble,
)

res_calsoft_025
```

```python
# Cell 7: ROC curve
model_name = 'FiveModel_CalSoft_t0.25'
y_proba = voter_calsoft_025.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
ROC_AUC = roc_auc_score(y_test, y_proba)
plot_roc_curve(fpr, tpr, ROC_AUC, model_name=model_name)
```

```python
# Cell 8: Confusion matrix plot
y_pred = (y_proba >= 0.25).astype(int)
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, model_name=model_name, class_labels=[0, 1])
```

```python
# Cell 9: Permutation importance
perm_df = plot_permutation_importance(
    voter_calsoft_025=voter_calsoft_025,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_names,
    model_name=model_name,
)
perm_df
```

```python
# Cell 10: Aggregated base-model importance
agg_df = plot_aggregated_base_model_importance(
    voter_calsoft_025=voter_calsoft_025,
    feature_names=feature_names,
    model_name=model_name,
)
agg_df
```

```python
# Cell 11: SHAP importance (tree base models only)
shap_df = plot_shap_importance(
    voter_calsoft_025=voter_calsoft_025,
    X_test=X_test,
    feature_names=feature_names,
    model_name=model_name,
)
shap_df
```

```python
# Cell 12: Confusion-group extracts
all_rows_fn, false_negatives = get_false_negatives(voter_calsoft_025, threshold=0.25)
all_rows_fp, false_positives = get_false_positives(voter_calsoft_025, threshold=0.25)
all_rows_tp, true_positives = get_true_positives(voter_calsoft_025, threshold=0.25)
all_rows_tn, true_negatives = get_true_negatives(voter_calsoft_025, threshold=0.25)

false_negatives
```

```python
# Cell 13: Inspect FP / TP / TN tables
false_positives
```

```python
true_positives
```

```python
true_negatives
```

```python
# Cell 14: Compare describe() metrics across FN / FP / TP / TN
numeric_summary, numeric_compare = compare_confusion_group_describes(
    false_negatives,
    false_positives,
    true_positives,
    true_negatives,
    include='numeric',
)

all_summary, all_compare = compare_confusion_group_describes(
    false_negatives,
    false_positives,
    true_positives,
    true_negatives,
    include='all',
)

numeric_compare
```

```python
# Cell 15: Save numeric comparison to Excel
numeric_compare.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Confusion_Matrix_Stats_Compare.xlsx')
```

```python
# Cell 16: Build FN/FP/TP/TN group dictionary for statistical testing
groups = {
    'FN': false_negatives,
    'FP': false_positives,
    'TP': true_positives,
    'TN': true_negatives,
}
```

```python
# Cell 17: Global 4-group Kruskal screen (FDR correction)
global_kruskal_df = run_global_kruskal(
    groups=groups,
    p_adjust_method='fdr_bh',
)
global_kruskal_df.head(20)
```

```python
# Cell 18: Pairwise Mann-Whitney tests on all numeric features
pairwise_all_df = run_pairwise_tests(
    groups=groups,
    comparisons=(('FN', 'TP'), ('FP', 'TN'), ('FN', 'FP')),
    p_adjust_method='fdr_bh',  # switch to 'holm' if preferred
    kruskal_results=global_kruskal_df,
)
pairwise_all_df.head(20)
```

```python
# Cell 19: Conditional targeted post-hoc tests for globally significant features
targeted_pairwise_df = run_conditional_targeted_tests(
    groups=groups,
    kruskal_results=global_kruskal_df,
    alpha=0.05,
    p_adjust_method='holm',  # parameterizable: 'holm' or 'fdr_bh'
)
targeted_pairwise_df.head(20)
```

```python
# Cell 20: Ranked summary table (sorted by global adjusted p then |effect size|)
stats_summary_df = build_summary_table(
    global_kruskal_results=global_kruskal_df,
    pairwise_results=pairwise_all_df,
)
stats_summary_df.head(30)
```

```python
# Cell 21: Optional exports for the new statistical testing outputs
global_kruskal_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Kruskal_Global_Screen.xlsx', index=False)
pairwise_all_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/MWU_Pairwise_All.xlsx', index=False)
targeted_pairwise_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/MWU_Targeted_PostHoc.xlsx', index=False)
stats_summary_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Statistical_Testing_Summary.xlsx', index=False)
```

```python
# Cell 22: Calibration report (post only, plus baseline)
cal_report = calibration_report(
    y_true=y_test,
    y_proba=y_proba,
    threshold=0.25,
    y_proba_pre=None,  # Add pre-calibration probabilities if available
    plot=True,
)
cal_report
```

```python
# Cell 23: FN probability diagnostics around threshold
fn_diag = fn_probability_diagnostics(
    y_true=y_test,
    y_proba=y_proba,
    threshold=0.25,
    band_below=(0.20, 0.25),
    band_above=(0.25, 0.35),
    near_window=0.05,
    threshold_artifact_cutoff=0.30,
)
fn_diag
```

```python
# Cell 24: Threshold sweep from 0.15 to 0.35
sweep_df = threshold_sweep(
    y_true=y_test,
    y_proba=y_proba,
    threshold_min=0.15,
    threshold_max=0.35,
    threshold_step=0.01,
)
sweep_df.head(10)
```

```python
# Cell 25: Select threshold by each business objective
sel_min_fn = select_threshold(
    sweep_df=sweep_df,
    current_threshold=0.25,
    objective='min_fn_bounded_fp',
    fp_increase_bound=0.20,
)

sel_max_mcc = select_threshold(
    sweep_df=sweep_df,
    current_threshold=0.25,
    objective='max_mcc_with_recall_constraint',
    recall_min=0.70,
)

sel_max_bal = select_threshold(
    sweep_df=sweep_df,
    current_threshold=0.25,
    objective='max_bal_acc_with_recall_constraint',
    recall_min=0.70,
)

sel_min_fn, sel_max_mcc, sel_max_bal
```

```python
# Cell 26: Optional CV stability analysis
# Expected shape for cv_folds:
# cv_folds = [
#     (y_true_fold, y_proba_fold),
#     {'y_true': y_true_fold, 'y_proba': y_proba_fold},
# ]
#
# Run only if cv_folds is already prepared in your notebook.
if 'cv_folds' in globals():
    stability_report = threshold_stability_cv(
        cv_folds=cv_folds,
        objective='min_fn_bounded_fp',
        current_threshold=0.25,
        fp_increase_bound=0.20,
        recall_min=0.70,
        threshold_min=0.15,
        threshold_max=0.35,
        threshold_step=0.01,
        plot=True,
        plot_kind='hist',
    )
    stability_report
else:
    print("cv_folds not found. Define cv_folds first to run stability analysis.")
```

```python
# Cell 27: Optional exports for calibration and threshold diagnostics
sweep_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Threshold_Sweep.xlsx', index=False)
pd.DataFrame([fn_diag]).to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/FN_Threshold_Diagnostics.xlsx', index=False)
pd.DataFrame([{
    'chosen_threshold_min_fn_bounded_fp': sel_min_fn['chosen_threshold'],
    'chosen_threshold_max_mcc_recall_constraint': sel_max_mcc['chosen_threshold'],
    'chosen_threshold_max_balacc_recall_constraint': sel_max_bal['chosen_threshold'],
}]).to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Threshold_Selections.xlsx', index=False)
```

```python
# Cell 28: Setup inputs for feature importance triangulation
X_val = X_test.copy()
y_val = y_test.copy()
ensemble_model = voter_calsoft_025
base_models = {
    'qda': qda_ensemble,
    'ada': ada_ensemble,
    'dt': dt_ensemble,
    'rf': rf_ensemble,
    'et': et_ensemble,
}
feature_names_val = X_val.columns.tolist()

# FN mask on validation set for FN-specific SHAP.
fn_mask = (y_val == 1) & (y_proba < 0.25)
```

```python
# Cell 29: Compute each importance view separately
perm_importance_df = compute_permutation_importance(
    ensemble_model=ensemble_model,
    X_val=X_val,
    y_val=y_val,
    feature_names=feature_names_val,
    n_repeats_perm=10,
    scoring='balanced_accuracy',
    random_state=1945,
)

base_report = compute_base_model_importance(
    base_models=base_models,
    feature_names=feature_names_val,
)
base_importance_df = base_report['importance_df']
base_model_details_df = base_report['model_details']

shap_report = compute_shap_importance(
    base_models=base_models,
    X_val=X_val,
    feature_names=feature_names_val,
)
shap_importance_df = shap_report['importance_df']

perm_importance_df.head(10), base_importance_df.head(10), shap_importance_df.head(10)
```

```python
# Cell 30: Merge + normalize + rank agreement + stability + FN-specific SHAP
triangulation_compare_df = perm_importance_df.merge(base_importance_df, on='feature', how='outer')
triangulation_compare_df = triangulation_compare_df.merge(shap_importance_df, on='feature', how='outer')
triangulation_compare_df = normalize_importances(triangulation_compare_df)

agreement_report = compute_rank_agreement(triangulation_compare_df)
agreement_table_df = agreement_report['agreement_table']
agreement_matrix_df = agreement_report['agreement_matrix']

stability_report = identify_stable_unstable_features(
    triangulation_compare_df,
    top_k=20,
    rank_spread_threshold=20,
    base_var_threshold=None,
    shap_var_threshold=None,
)
stability_df = stability_report['feature_stability_df']
stable_features = stability_report['stable_features']
unstable_features = stability_report['unstable_features']

fn_shap_report = fn_specific_shap_analysis(
    shap_report=shap_report,
    fn_mask=fn_mask,
    top_k=20,
)
fn_shap_df = fn_shap_report['fn_shap_df']
fn_shap_top_df = fn_shap_report['fn_top_features']
fn_shap_delta_df = fn_shap_report['positive_delta_features']

triangulation_compare_df.head(20)
```

```python
# Cell 31: One-call full triangulation report (same outputs bundled)
triangulation_report = build_triangulation_report(
    X_val=X_val,
    y_val=y_val,
    ensemble_model=ensemble_model,
    base_models=base_models,
    feature_names=feature_names_val,
    fn_mask=fn_mask,
    top_k=20,
    n_repeats_perm=10,
    scoring='balanced_accuracy',
    random_state=1945,
    rank_spread_threshold=20,
)

triangulation_report['comparison_df'].head(20)
```

```python
# Cell 32: Inspect agreement, stability, and FN-focused SHAP outputs
agreement_table_df
```

```python
agreement_matrix_df
```

```python
stability_df[['feature', 'stability_count', 'is_stable', 'rank_spread', 'is_unstable']].head(30)
```

```python
fn_shap_top_df
```

```python
fn_shap_delta_df
```

```python
# Cell 33: Optional exports for feature importance triangulation outputs
triangulation_compare_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Feature_Triangulation_Comparison.xlsx', index=False)
agreement_table_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Feature_Triangulation_Agreement.xlsx', index=False)
stability_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Feature_Triangulation_Stability.xlsx', index=False)
base_model_details_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Feature_Triangulation_Base_Model_Details.xlsx', index=False)
if not fn_shap_df.empty:
    fn_shap_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Feature_Triangulation_FN_SHAP.xlsx', index=False)
```

```python
# Cell 34: Build evaluation dataframe for segment/temporal robustness
df_eval = all_rows_fn[all_rows_fn['Is_Test']].copy()
df_eval = df_eval.dropna(subset=['At Risk (0/1)', 'Pred_Prob'])

# Use scored test labels/probabilities.
df_eval['y_true'] = df_eval['At Risk (0/1)'].astype(int)
df_eval['y_proba'] = df_eval['Pred_Prob'].astype(float)

# Prefer Depart_Date as time column (fallback to Arrival/Dispersal Date).
time_col_eval = 'Depart_Date' if 'Depart_Date' in df_eval.columns else 'Arrival/Dispersal Date'
route_col_eval = 'Route' if 'Route' in df_eval.columns else None

df_eval[['y_true', 'y_proba', time_col_eval]].head(5)
```

```python
# Cell 35: Run full segment + temporal robustness report
segment_report = build_segment_temporal_report(
    df=df_eval,
    y_true_col='y_true',
    y_proba_col='y_proba',
    time_col=time_col_eval,
    route_col=route_col_eval,
    feature_cols=None,  # infer numeric features automatically
    threshold=0.25,
    threshold_grid=np.arange(0.15, 0.351, 0.01),
    early_late_split={'type': 'year', 'cutoff_year': 1942},
    min_segment_n=10,
    min_pos_n=3,
    multiple_test_method='fdr_bh',
    objective='max_mcc_with_recall_constraint',
    constraints={'current_threshold': 0.25, 'recall_min': 0.70, 'fp_increase_bound': 0.20},
    delta_threshold=0.03,
    plot=False,
)
```

```python
# Cell 36: Inspect segment report outputs
segment_metrics_df = segment_report['segment_metrics']
fn_distribution_df = segment_report['fn_distribution']
shift_results_df = segment_report['shift_results']
per_segment_opt_threshold_df = segment_report['per_segment_opt_threshold']
stability_summary_df = segment_report['stability_summary']

segment_metrics_df.head(20)
```

```python
fn_distribution_df.head(20)
```

```python
shift_results_df.head(20)
```

```python
per_segment_opt_threshold_df.head(20)
```

```python
stability_summary_df
```

```python
# Cell 37: Optional exports for segment/temporal robustness outputs
segment_metrics_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Segment_Metrics.xlsx', index=False)
fn_distribution_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Segment_FN_Distribution.xlsx', index=False)
shift_results_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Segment_Shift_Tests.xlsx', index=False)
per_segment_opt_threshold_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Segment_Threshold_Stability.xlsx', index=False)
stability_summary_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Segment_Threshold_Stability_Summary.xlsx', index=False)
```

```python
# Cell 38: Leakage and split-integrity checks
leakage_flags_df = flag_leakage_columns(
    X_train,
    feature_metadata=None,  # pass metadata dict/DataFrame if available
    patterns=None,  # uses default suspicious leakage patterns
)

split_integrity_report = check_split_integrity(
    X_train=X_train,
    X_test=X_test,
    id_col=None,  # set if you keep an ID column in model matrix
    convoy_id_col=None,  # set if convoy/group id is present in matrix
)

leakage_flags_df.head(20), split_integrity_report
```

```python
# Cell 39: Alignment, confusion groups, missingness, and outlier concentration
alignment_train_report = audit_alignment(X_train, y_train, df_raw=None, id_col=None)
alignment_test_report = audit_alignment(X_test, y_test, df_raw=None, id_col=None)

test_groups = confusion_groups(y_test, y_proba, threshold=0.25)

missingness_group_df = missingness_by_confusion_group(
    X_test=X_test,
    groups=test_groups,
    multiple_test_method='fdr_bh',
    missingness_test={'method': 'chi2'},
)

numeric_cols_quality = X_train.select_dtypes(include=[np.number]).columns.tolist()
train_outlier_bounds = outlier_bounds_from_train(
    X_train=X_train,
    numeric_cols=numeric_cols_quality,
    method='iqr',
    params={'k': 1.5},
)
outliers_group_df = outliers_by_confusion_group(
    X_test=X_test,
    groups=test_groups,
    bounds=train_outlier_bounds,
)

alignment_train_report, alignment_test_report
```

```python
missingness_group_df.head(20)
```

```python
outliers_group_df.head(20)
```

```python
# Cell 40: Preprocessing audit (train-only fit consistency heuristics)
# If you have a fitted pipeline object available, set preprocess_pipeline_ref to it.
preprocess_pipeline_ref = globals().get('preprocess_pipeline', None)

preprocessing_audit_report = audit_preprocessing(
    preprocess_pipeline=preprocess_pipeline_ref,
    X_train=X_train,
    X_test=X_test,
    feature_groups=None,  # optional: {'numeric': [...], 'categorical': [...]}
    tolerance=1e-6,
)
preprocessing_audit_report
```

```python
# Cell 41: Ranked risk summary table
risk_summary_df = build_risk_summary(
    leakage_flags=leakage_flags_df,
    split_integrity=split_integrity_report,
    alignment_train=alignment_train_report,
    alignment_test=alignment_test_report,
    missingness_df=missingness_group_df,
    outliers_df=outliers_group_df,
    preprocessing_audit=preprocessing_audit_report,
    alpha=0.05,
    report_top_n=30,
)
risk_summary_df
```

```python
# Cell 42: Optional exports for leakage/data-quality checks
leakage_flags_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Leakage_Flags.xlsx', index=False)
missingness_group_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Missingness_By_Confusion_Group.xlsx', index=False)
outliers_group_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Outliers_By_Confusion_Group.xlsx', index=False)
risk_summary_df.to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Leakage_Data_Quality_Risk_Summary.xlsx', index=False)
pd.DataFrame([split_integrity_report]).to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Split_Integrity_Report.xlsx', index=False)
pd.DataFrame([alignment_train_report]).to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Alignment_Train_Report.xlsx', index=False)
pd.DataFrame([alignment_test_report]).to_excel('/Users/matthewplambeck/Desktop/Convoy Predictor/results/Alignment_Test_Report.xlsx', index=False)
```
