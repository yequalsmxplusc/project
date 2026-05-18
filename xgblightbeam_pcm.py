# -*- coding: utf-8 -*-
"""
xgblightbeam_pcm.py
====================
XGBoost Thermal Prediction — PCM Battery Cooling System
Datasets: 0_3.xlsx, 0_4.xlsx, 0_5.xlsx  (aspect ratios 0.3, 0.4, 0.5)

Physical context:
  - PCM (Phase Change Material) absorbs battery heat via latent heat
  - T_battery: battery temperature (K)  — prediction target
  - T_pcm:     PCM temperature (K)
  - liquid_frac: PCM melt fraction (0 = solid, 1 = fully melted)
  - Nu:        Nusselt number — convective heat transfer intensity
  - aspect_ratio: PCM enclosure geometry

DATA LEAKAGE DESIGN
────────────────────
All features are computed from lagged / non-target quantities.
T_battery(t) is NEVER used as an input — only T_bat_lag1 = T_battery(t-1)
and further lags/rolling stats. See pcm_data.py for full rationale.

FEATURE GROUPS (all leak-free, via pcm_data.py)
──────────────────────────────────────────────────
  Core (8):      liquid_frac, Nu, T_pcm, T_bat_lag1, dT_lag1,
                 heat_flux_lag1, melting_rate, T_pcm_rate
  Extended (8):  lf_sq, Nu_x_lf, Nu_abs, thermal_res, phase_flag,
                 T_bat_rate_lag, Nu_rate, T_pcm_lf
  Cumulative (1): cumul_heat  (computed after temporal split)
  Lag (35):      {7 cols} x lag{1,5,10,25,50}
  Roll-mean (35): {7 cols} x rollmean_{1,5,10,25,50}
  Roll-std (35):  {7 cols}  x rollstd_{1,5,10,25,50}
  Total: ~122+ features

VALIDATION SPLIT
─────────────────
The validation set is carved from the END of the training window (last 10%)
and NEVER from the test set — eliminates eval_set leakage from original code.
"""

import sys
import os

# Ensure pcm_data.py in same directory is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from pcm_data import (
    load_raw, clean, engineer_features,
    add_cumulative_features, add_lag_rolling_features,
    get_xgb_feature_cols, temporal_split,
    prepare_ar_xgb,
    evaluate_predictions, print_metrics,
    FEATURE_COLS_EXTENDED, LAG_BASE_COLS, LAG_STEPS, TARGET_COL,
)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("✅ Environment Ready: XGBoost + pcm_data pipeline loaded.")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATHS = {
    '0.3': '0_3.xlsx',
    '0.4': '0_4.xlsx',
    '0.5': '0_5.xlsx',
}

PREDICTION_HORIZON = 50      # predict T_battery N timesteps ahead
TRAIN_RATIO        = 0.85
VAL_FRACTION       = 0.10    # fraction of TRAINING set used for validation

XGB_PARAMS = dict(
    n_estimators     = 600,
    learning_rate    = 0.04,
    max_depth        = 7,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    objective        = 'reg:squarederror',
    n_jobs           = -1,
    random_state     = 42,
    early_stopping_rounds = 30,   # uses VAL set, not test
)

OUTPUT_DIR = './'

# ============================================================================
# DATA LOADING & FEATURE ENGINEERING  (via pcm_data.py)
# ============================================================================

print("\n📂 Loading and engineering features for all aspect ratios...")
all_data   = {}   # raw dataframes (for EDA)
all_xgb    = {}   # prepared dicts from prepare_ar_xgb

for ar, path in DATASET_PATHS.items():
    print(f"\n  AR {ar} — {path}")

    # Load and clean
    df_raw = load_raw(path)
    df_raw = clean(df_raw, ar)
    all_data[ar] = df_raw

    # Full XGBoost pipeline (leak-free)
    data = prepare_ar_xgb(
        path=path,
        ar_label=ar,
        prediction_horizon=PREDICTION_HORIZON,
        train_ratio=TRAIN_RATIO,
        val_fraction=VAL_FRACTION,
    )
    all_xgb[ar] = data

    print(f"     Train : {data['X_train'].shape[0]} samples | "
          f"Val: {data['X_val'].shape[0]} | "
          f"Test: {data['X_test'].shape[0]}")
    print(f"     Features: {len(data['feature_cols'])}")
    print(f"     Target range (test): "
          f"{data['y_test'].min():.2f} – {data['y_test'].max():.2f} K")

print("\n✅ All datasets prepared — no data leakage.")

# ============================================================================
# EXPLORATORY VISUALISATION
# ============================================================================

print("\n📊 Generating exploratory plots...")

fig, axes = plt.subplots(3, 4, figsize=(22, 14))
for row, ar in enumerate(DATASET_PATHS.keys()):
    df = all_data[ar].copy()
    df['dT'] = df['T_battery'] - df['T_pcm']   # raw dT for EDA only (not a feature)

    axes[row, 0].plot(df['time'], df['T_battery'], color='firebrick', lw=0.6)
    axes[row, 0].set_title(f'AR {ar} — T_battery (K)')
    axes[row, 0].set_xlabel('time (s)')

    axes[row, 1].plot(df['time'], df['T_pcm'], color='steelblue', lw=0.6)
    axes[row, 1].set_title(f'AR {ar} — T_pcm (K)')

    axes[row, 2].plot(df['time'], df['liquid_frac'], color='green', lw=0.6)
    axes[row, 2].set_title(f'AR {ar} — liquid_frac')

    axes[row, 3].plot(df['time'], df['dT'], color='purple', lw=0.6)
    axes[row, 3].set_title(f'AR {ar} — ΔT = T_bat − T_pcm (K)')

plt.suptitle('PCM Thermal Dataset Overview (all aspect ratios)', fontsize=14,
             fontweight='bold')
plt.tight_layout()
eda_path = os.path.join(OUTPUT_DIR, 'pcm_eda.png')
plt.savefig(eda_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  EDA plot → {os.path.basename(eda_path)}")

# Correlation heatmap — use leak-free features from AR 0.4
df_corr_raw  = load_raw('0_4.xlsx')
df_corr_raw  = clean(df_corr_raw, '0.4')
df_corr_feat = engineer_features(df_corr_raw, extended=True)
df_corr_feat = add_cumulative_features(df_corr_feat)
df_corr_feat = add_lag_rolling_features(df_corr_feat)
df_corr_num  = df_corr_feat.select_dtypes(include='number')
corr = (df_corr_num.corr()[['T_battery']]
        .sort_values('T_battery', ascending=False)
        .head(25))
plt.figure(figsize=(5, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Top 25 Feature Correlations with T_battery (AR 0.4)\n'
          '[features are leak-free — all use lagged T_battery]')
plt.tight_layout()
corr_path = os.path.join(OUTPUT_DIR, 'pcm_correlation.png')
plt.savefig(corr_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Correlation plot → {os.path.basename(corr_path)}")

# ============================================================================
# TRAIN — IN-DOMAIN (one XGBoost model per AR, validation from training window)
# ============================================================================

print("\n" + "=" * 70)
print("  IN-DOMAIN TRAINING  (one XGBoost model per aspect ratio)")
print("  Validation set = last 10% of training window  [NO test leakage]")
print("=" * 70)

models           = {}
results          = {}
predictions_dict = {}

for ar in DATASET_PATHS.keys():
    d = all_xgb[ar]

    X_train, y_train = d['X_train'], d['y_train']
    X_val,   y_val   = d['X_val'],   d['y_val']
    X_test,  y_test  = d['X_test'],  d['y_test']
    feature_cols     = d['feature_cols']

    print(f"\n🔧 AR {ar} | train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
    print(f"   Features: {len(feature_cols)}")

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],   # validation from TRAINING window only
        verbose=False,
    )

    best_iter = model.best_iteration
    print(f"   Best iteration: {best_iter}")

    preds    = model.predict(X_test)
    metrics  = evaluate_predictions(y_test, preds)

    models[ar]          = model
    predictions_dict[ar] = (y_test, preds)
    results[ar] = dict(
        **{k: metrics[k] for k in ('mae', 'rmse', 'r2')},
        y_true=y_test, y_pred=preds,
        errors=np.abs(y_test - preds),
        metrics=metrics,
        feature_cols=feature_cols,
    )

    print(f"   MAE : {metrics['mae']:.4f} K  |  "
          f"RMSE: {metrics['rmse']:.4f} K  |  "
          f"R²: {metrics['r2']:.4f}")

print("\n✅ Training complete — all evaluations on held-out test set.")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print(f"  RESULTS SUMMARY  (horizon = {PREDICTION_HORIZON} timesteps ahead)")
print("=" * 70)
print(f"\n{'AR':<8} {'MAE (K)':<14} {'RMSE (K)':<14} "
      f"{'R²':<10} {'MAE 60% (K)':<14} {'Eff. (%)':<10}")
print("-" * 70)

summary_rows = []
for ar, r in results.items():
    m = r['metrics']
    print(f"{ar:<8} {m['mae']:<14.4f} {m['rmse']:<14.4f} "
          f"{m['r2']:<10.4f} {m['mae_p60']:<14.4f} "
          f"{m['prediction_eff_pct']:.2f}")
    summary_rows.append(dict(
        aspect_ratio=ar,
        T_min=r['y_true'].min(), T_max=r['y_true'].max(),
        T_range=m['T_range'], T_mean=m['T_mean'],
        mae=m['mae'], rmse=m['rmse'], r2=m['r2'],
        mae_p60=m['mae_p60'], rmse_p60=m['rmse_p60'], r2_p60=m['r2_p60'],
        relative_error_pct=m['relative_error_pct'],
        prediction_efficiency_pct=m['prediction_eff_pct'],
    ))

avg_mae  = np.mean([r['mae']  for r in results.values()])
avg_rmse = np.mean([r['rmse'] for r in results.values()])
avg_r2   = np.mean([r['r2']   for r in results.values()])
print("-" * 70)
print(f"{'AVG':<8} {avg_mae:<14.4f} {avg_rmse:<14.4f} {avg_r2:.4f}")

# ============================================================================
# VISUALISATION — PREDICTIONS
# ============================================================================

fig, axes = plt.subplots(len(DATASET_PATHS), 2, figsize=(18, 14))

for row, ar in enumerate(DATASET_PATHS.keys()):
    y_true = results[ar]['y_true']
    y_pred = results[ar]['y_pred']
    errors = results[ar]['errors']
    show   = min(500, len(y_true))

    # Time series
    axes[row, 0].plot(y_true[:show], label='Actual T_battery',
                      color='black', lw=1.5)
    axes[row, 0].plot(y_pred[:show], label='XGB Predicted',
                      color='cyan', lw=1.5, ls='--')
    axes[row, 0].set_title(
        f'AR {ar} — Predicted vs Actual T_battery (first {show} test pts)')
    axes[row, 0].set_ylabel('T_battery (K)')
    axes[row, 0].legend()
    axes[row, 0].grid(True, alpha=0.3)

    # Scatter
    axes[row, 1].scatter(y_true, y_pred, alpha=0.3, s=6, color='steelblue')
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[row, 1].plot(lim, lim, 'r--', lw=2, label='Perfect')
    axes[row, 1].set_xlabel('Actual (K)')
    axes[row, 1].set_ylabel('Predicted (K)')
    axes[row, 1].set_title(
        f'AR {ar} — MAE={results[ar]["mae"]:.4f} K  R²={results[ar]["r2"]:.4f}')
    axes[row, 1].legend()
    axes[row, 1].grid(True, alpha=0.3)

plt.suptitle(
    f'XGBoost PCM Thermal Prediction  (horizon={PREDICTION_HORIZON} steps)',
    fontsize=14, fontweight='bold')
plt.tight_layout()
pred_path = os.path.join(OUTPUT_DIR, 'pcm_predictions.png')
plt.savefig(pred_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Prediction plot → {os.path.basename(pred_path)}")

# ============================================================================
# FEATURE IMPORTANCE (per AR)
# ============================================================================

fig, axes = plt.subplots(1, len(DATASET_PATHS), figsize=(21, 8))
for i, ar in enumerate(DATASET_PATHS.keys()):
    model  = models[ar]
    feats  = results[ar]['feature_cols']
    imp    = pd.Series(model.feature_importances_, index=feats)
    imp    = imp.sort_values(ascending=True).tail(20)
    imp.plot(kind='barh', ax=axes[i], color='teal')
    axes[i].set_title(f'AR {ar} — Top 20 Feature Importances')
    axes[i].set_xlabel('Gain')
    axes[i].grid(True, alpha=0.3, axis='x')

plt.suptitle('XGBoost Feature Importance by Aspect Ratio',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fi_path = os.path.join(OUTPUT_DIR, 'pcm_feature_importance.png')
plt.savefig(fi_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Feature importance plot → {os.path.basename(fi_path)}")

# ============================================================================
# ERROR DISTRIBUTION
# ============================================================================

fig, axes = plt.subplots(1, len(DATASET_PATHS), figsize=(18, 5))
for i, ar in enumerate(DATASET_PATHS.keys()):
    err = results[ar]['errors']
    p60 = np.percentile(err, 60)
    axes[i].hist(err, bins=60, alpha=0.75, color='coral', edgecolor='k')
    axes[i].axvline(p60, color='red', ls='--', lw=2,
                    label=f'60th pctile: {p60:.4f} K')
    axes[i].set_title(f'AR {ar} — Error Distribution')
    axes[i].set_xlabel('|Error| (K)')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Absolute Error Distributions', fontsize=13, fontweight='bold')
plt.tight_layout()
err_path = os.path.join(OUTPUT_DIR, 'pcm_error_dist.png')
plt.savefig(err_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Error distribution plot → {os.path.basename(err_path)}")

# ============================================================================
# INTERACTIVE PLOTLY
# ============================================================================

fig_plotly = make_subplots(
    rows=len(DATASET_PATHS), cols=1,
    subplot_titles=[f'Aspect Ratio {ar}' for ar in DATASET_PATHS],
    shared_xaxes=False,
)

for row_idx, ar in enumerate(DATASET_PATHS.keys(), start=1):
    y_true = results[ar]['y_true']
    y_pred = results[ar]['y_pred']
    show   = min(800, len(y_true))
    t      = np.arange(show)

    fig_plotly.add_trace(
        go.Scatter(x=t, y=y_true[:show], name=f'Actual AR{ar}',
                   line=dict(color='black', width=1.5)),
        row=row_idx, col=1)
    fig_plotly.add_trace(
        go.Scatter(x=t, y=y_pred[:show], name=f'Predicted AR{ar}',
                   line=dict(color='cyan', width=1.5, dash='dash')),
        row=row_idx, col=1)

fig_plotly.update_layout(
    title_text=(f'PCM Battery Temperature Prediction — XGBoost '
                f'(horizon={PREDICTION_HORIZON} steps)'),
    height=900, template='plotly_white', hovermode='x unified',
)
fig_plotly.update_yaxes(title_text='T_battery (K)')
plotly_path = os.path.join(OUTPUT_DIR, 'pcm_predictions_interactive.html')
fig_plotly.write_html(plotly_path)
print(f"  Interactive plot → {os.path.basename(plotly_path)}")

# ============================================================================
# EFFICIENCY ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("  EFFICIENCY ANALYSIS")
print("=" * 70)

for ar, r in results.items():
    m = r['metrics']
    print(f"\n📊 AR {ar}:")
    print(f"   T_battery range : {r['y_true'].min():.2f} – "
          f"{r['y_true'].max():.2f} K  (Δ={m['T_range']:.2f} K)")
    print(f"   MAE             : {m['mae']:.4f} K   "
          f"Relative error: {m['relative_error_pct']:.3f}%")
    print(f"   MAE (60th pct)  : {m['mae_p60']:.4f} K")
    print(f"   Prediction eff. : {m['prediction_eff_pct']:.2f}%")
    print(f"   R²              : {m['r2']:.4f}")

# ============================================================================
# EXPORT — models + results
# ============================================================================

print("\n" + "=" * 70)
print("  EXPORTING MODELS & RESULTS")
print("=" * 70)

detailed_rows = []
for ar, r in results.items():
    for i, (yt, yp) in enumerate(zip(r['y_true'], r['y_pred'])):
        detailed_rows.append(
            dict(aspect_ratio=ar, idx=i, y_true=yt, y_pred=yp,
                 error=abs(yt - yp)))

detail_path = os.path.join(OUTPUT_DIR, 'pcm_detailed_results.csv')
pd.DataFrame(detailed_rows).to_csv(detail_path, index=False)
print(f"✅ {os.path.basename(detail_path)}")

df_summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_DIR, 'pcm_summary_results.csv')
df_summary.to_csv(summary_path, index=False)
print(f"✅ {os.path.basename(summary_path)}")
print("\n", df_summary.to_string(index=False))

for ar, model in models.items():
    json_path = os.path.join(OUTPUT_DIR, f'xgb_pcm_ar{ar}.json')
    model.save_model(json_path)
    print(f"✅ Model saved: {os.path.basename(json_path)}")

    feat_path = os.path.join(OUTPUT_DIR, f'features_ar{ar}.txt')
    with open(feat_path, 'w') as f:
        for feat in results[ar]['feature_cols']:
            f.write(feat + '\n')
    print(f"✅ Feature list: {os.path.basename(feat_path)}")

# ============================================================================
# LaTeX TABLE
# ============================================================================

print("\n" + "=" * 70)
print("  LaTeX TABLE")
print("=" * 70)
print("\\begin{table}[h]\\centering")
print("\\caption{XGBoost PCM thermal prediction "
      "(in-domain, per aspect ratio, no data leakage)}")
print("\\begin{tabular}{lcccc}")
print("\\hline")
print("AR & $T$ range (K) & MAE (K) & $R^2$ & Eff. (\\%) \\\\\\hline")
for r in summary_rows:
    print(f"{r['aspect_ratio']} & {r['T_min']:.1f}–{r['T_max']:.1f} & "
          f"{r['mae']:.4f} & {r['r2']:.4f} & "
          f"{r['prediction_efficiency_pct']:.1f} \\\\")
print("\\hline")
print(f"Avg & — & {avg_mae:.4f} & {avg_r2:.4f} & — \\\\\\hline")
print("\\end{tabular}\\end{table}")

print("\n✅ Done. All outputs saved.")
