# -*- coding: utf-8 -*-
"""
PCM.py — Hybrid SOTA: XGBoost + Stateful LSTM+Attention Ensemble
=================================================================
Trains two specialist branches per aspect ratio, then stacks them
with a Ridge meta-learner for the final T_battery prediction.

XGBoost branch : extended physics features + lag/rolling (122+ features)
LSTM branch    : core 8 leak-free sequential features (300-step chunks)
Meta-learner   : Ridge regression on [xgb_pred, lstm_pred]

DATA LEAKAGE POLICY
────────────────────
• T_battery(t) NEVER appears as a feature — only T_bat_lag1 = T_battery(t-1)
• Scalers fit on training split only
• Validation set carved from END of training window — not from test set
• cumul_heat computed separately on train and test after temporal split

Depends on: pcm_data.py (must be in same directory)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import random
from datetime import datetime
import pytz

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Masking, MultiHeadAttention, LayerNormalization, Input
)
from tensorflow.keras.optimizers import Adam

from pcm_data import (
    load_raw, clean, engineer_features,
    add_cumulative_features, add_lag_rolling_features,
    temporal_split, fit_scalers, apply_scalers,
    build_sequences, prepare_ar, prepare_ar_xgb,
    get_xgb_feature_cols,
    evaluate_predictions, print_metrics,
    FEATURE_COLS_CORE, TARGET_COL,
)

sns.set_style('whitegrid')
IST = pytz.timezone('Asia/Kolkata')

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATHS = {'0.3': '0_3.xlsx', '0.4': '0_4.xlsx', '0.5': '0_5.xlsx'}

# XGBoost
XGB_PARAMS = dict(
    n_estimators=600, learning_rate=0.04, max_depth=7,
    subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror',
    n_jobs=-1, random_state=42, early_stopping_rounds=30,
)
PREDICTION_HORIZON = 0      # current-step prediction so both branches match
VAL_FRACTION       = 0.10   # carved from end of training window

# LSTM
SEQ_STEPS   = 300
EPOCHS      = 200
PATIENCE    = 30
LSTM_UNITS  = 256
DENSE_UNITS = [128, 64]
ATT_HEADS   = 4
ATT_KEY_DIM = 64
LR          = 1e-4
TRAIN_RATIO = 0.85

SEED = 42
OUTPUT_DIR = './'

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

FEATURE_COLS_LSTM = FEATURE_COLS_CORE   # 8 leak-free sequential features


# ─────────────────────────────────────────────────────────────────────────────
# TF HELPER
# ─────────────────────────────────────────────────────────────────────────────

def reset_lstm_states(model: keras.Model) -> None:
    for layer in model.layers:
        if hasattr(layer, 'reset_states'):
            layer.reset_states()


# ─────────────────────────────────────────────────────────────────────────────
# LSTM BRANCH: BUILD / TRAIN / PREDICT
# ─────────────────────────────────────────────────────────────────────────────

def build_lstm(batch_size: int, n_features: int) -> keras.Model:
    inp   = Input(batch_shape=(batch_size, SEQ_STEPS, n_features), name='inp')
    x     = Masking(mask_value=0.0)(inp)
    x     = LSTM(LSTM_UNITS, activation='tanh', return_sequences=True,
                 stateful=True, name='lstm_1')(x)
    x     = LSTM(LSTM_UNITS, activation='tanh', return_sequences=True,
                 stateful=True, name='lstm_2')(x)
    x     = LSTM(LSTM_UNITS, activation='tanh', return_sequences=True,
                 stateful=True, name='lstm_3')(x)
    attn  = MultiHeadAttention(num_heads=ATT_HEADS, key_dim=ATT_KEY_DIM,
                               name='attention')(x, x)
    x     = LayerNormalization(name='ln')(x + attn)
    x     = Dense(DENSE_UNITS[0], activation='selu')(x)
    x     = Dense(DENSE_UNITS[1], activation='selu')(x)
    out   = Dense(1, activation='linear', name='output')(x)
    mdl   = Model(inputs=inp, outputs=out, name='LSTM_Attention')
    mdl.compile(optimizer=Adam(LR), loss='huber',
                metrics=['mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return mdl


def train_lstm(model, X_train, y_train, epochs, patience, name):
    n    = X_train.shape[0]
    hist = {'loss': [], 'mse': [], 'rmse': []}
    best_loss, pat_cnt, best_w = float('inf'), 0, None

    for epoch in range(epochs):
        avg = {'loss': [], 'mse': [], 'rmse': []}
        for ci in random.sample(range(n), n):
            res = model.fit(X_train[ci:ci+1], y_train[ci:ci+1],
                            shuffle=False, verbose=0, batch_size=1)
            avg['loss'].append(res.history['loss'][0])
            avg['mse'].append(res.history['mse'][0])
            avg['rmse'].append(res.history['rmse'][0])
            print(f'\r  chunk {ci+1}/{n} loss={np.mean(avg["loss"]):.4e}',
                  end='', flush=True)

        el = float(np.mean(avg['loss']))
        em = float(np.mean(avg['mse']))
        er = float(np.mean(avg['rmse']))
        hist['loss'].append(el); hist['mse'].append(em); hist['rmse'].append(er)
        print(f'\rEpoch {epoch+1:>3}/{epochs} loss={el:.6f} mse={em:.6f} rmse={er:.6f}')
        reset_lstm_states(model)

        if el < best_loss - 1e-6:
            best_loss, pat_cnt, best_w = el, 0, model.get_weights()
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                print(f'  Early stop at epoch {epoch+1}')
                break

    if best_w is not None:
        model.set_weights(best_w)
    return hist


def predict_lstm_raw(model, X) -> np.ndarray:
    """Returns scaled predictions (n_chunks*SEQ_STEPS, 1)."""
    preds = []
    reset_lstm_states(model)
    for ci in range(X.shape[0]):
        p = model.predict(X[ci:ci+1], verbose=0)
        preds.append(p[0])
    return np.concatenate(preds, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost BRANCH: BUILD / TRAIN / PREDICT
# (horizon=0 so targets align with LSTM current-step targets)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_xgb_horizon0(path, ar_label, train_ratio=0.85, val_fraction=0.10):
    """
    Like prepare_ar_xgb but with horizon=0 (predict current T_battery).
    The target is T_battery itself (not shifted) — still leak-free because
    T_battery does NOT appear anywhere in the feature set.
    """
    df_raw  = load_raw(path)
    df_raw  = clean(df_raw, ar_label)
    df_feat = engineer_features(df_raw, extended=True)

    train_df, test_df = temporal_split(df_feat, train_ratio)

    train_df = add_cumulative_features(train_df)
    test_df  = add_cumulative_features(test_df)
    train_df = add_lag_rolling_features(train_df)
    test_df  = add_lag_rolling_features(test_df)

    # horizon=0: target is current T_battery (T_battery NOT in features)
    train_df['target'] = train_df['T_battery']
    test_df['target']  = test_df['T_battery']

    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    feature_cols = get_xgb_feature_cols()
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    X_full = train_df[feature_cols].values
    y_full = train_df['target'].values
    val_n  = int(len(X_full) * val_fraction)

    return {
        'X_train':      X_full[:-val_n],
        'y_train':      y_full[:-val_n],
        'X_val':        X_full[-val_n:],
        'y_val':        y_full[-val_n:],
        'X_test':       test_df[feature_cols].values,
        'y_test':       test_df['target'].values,
        'feature_cols': feature_cols,
        'train_df':     train_df,
        'test_df':      test_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# META-LEARNER ALIGNMENT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def align_predictions(xgb_preds: np.ndarray,
                      lstm_preds_K: np.ndarray,
                      xgb_y: np.ndarray) -> tuple:
    """
    Both branches must predict the same number of test samples.
    XGBoost produces one flat vector; LSTM produces multiples of SEQ_STEPS.
    We truncate to the shorter of the two to guarantee alignment.
    """
    n = min(len(xgb_preds), len(lstm_preds_K), len(xgb_y))
    return xgb_preds[:n], lstm_preds_K[:n], xgb_y[:n]


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(hist, ar, name):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, key, col in zip(axes, ['loss', 'mse', 'rmse'],
                            ['steelblue', 'coral', 'green']):
        ax.plot(hist[key], color=col, lw=1.5)
        ax.set_title(f'AR {ar} — {key.upper()}')
        ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3)
    plt.suptitle(f'LSTM Training History | {name}', fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, f'{name}_lstm_history.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  History → {os.path.basename(p)}')


def plot_hybrid_results(all_results):
    ars = list(all_results.keys())
    fig, axes = plt.subplots(3, len(ars), figsize=(7 * len(ars), 15))
    labels = ['XGBoost', 'LSTM', 'Ensemble']
    colors = ['dodgerblue', 'tomato', 'gold']

    for col, ar in enumerate(ars):
        r  = all_results[ar]
        yt = r['y_true']
        show = min(600, len(yt))

        for row, (key, lbl, clr) in enumerate(
                zip(['xgb_pred', 'lstm_pred', 'ens_pred'], labels, colors)):
            axes[row, col].plot(yt[:show], color='black', lw=1, label='Actual')
            axes[row, col].plot(r[key][:show], color=clr, lw=1.5,
                                ls='--', label=lbl)
            m = r[f'{key.split("_")[0]}_metrics'] if key != 'ens_pred' \
                else r['ens_metrics']
            axes[row, col].set_title(
                f'AR {ar} | {lbl}\nMAE={m["mae"]:.4f} K  R²={m["r2"]:.4f}',
                fontweight='bold')
            axes[row, col].set_xlabel('Step')
            axes[row, col].set_ylabel('T_battery (K)')
            axes[row, col].legend(fontsize=8)
            axes[row, col].grid(True, alpha=0.3)

    plt.suptitle('Hybrid SOTA — XGBoost · LSTM · Ensemble (Test Set)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'pcm_hybrid_results.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Hybrid results plot → {os.path.basename(p)}')


def plot_scatter_trio(all_results):
    ars = list(all_results.keys())
    fig, axes = plt.subplots(len(ars), 3, figsize=(18, 5 * len(ars)))
    labels = ['XGBoost', 'LSTM', 'Ensemble']

    for row, ar in enumerate(ars):
        r  = all_results[ar]
        yt = r['y_true']
        for col, (key, lbl) in enumerate(
                zip(['xgb_pred', 'lstm_pred', 'ens_pred'], labels)):
            yp = r[key]
            lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
            axes[row, col].scatter(yt, yp, alpha=0.3, s=5, color='steelblue')
            axes[row, col].plot(lim, lim, 'r--', lw=1.5)
            axes[row, col].set_xlabel('Actual (K)')
            axes[row, col].set_ylabel('Predicted (K)')
            axes[row, col].set_title(f'AR {ar} — {lbl}')
            axes[row, col].grid(True, alpha=0.3)

    plt.suptitle('Scatter: Actual vs Predicted (all branches)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'pcm_hybrid_scatter.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Scatter plot → {os.path.basename(p)}')


def plot_error_comparison(all_results):
    ars = list(all_results.keys())
    fig, axes = plt.subplots(1, len(ars), figsize=(7 * len(ars), 5))

    for col, ar in enumerate(ars):
        r  = all_results[ar]
        yt = r['y_true']
        for key, lbl, clr in zip(
                ['xgb_pred', 'lstm_pred', 'ens_pred'],
                ['XGBoost', 'LSTM', 'Ensemble'],
                ['dodgerblue', 'tomato', 'gold']):
            err = np.abs(yt - r[key])
            axes[col].hist(err, bins=50, alpha=0.55, color=clr,
                           edgecolor='k', linewidth=0.3, label=lbl)
        axes[col].set_title(f'AR {ar} — Error Distributions')
        axes[col].set_xlabel('|Error| (K)')
        axes[col].legend(); axes[col].grid(True, alpha=0.3)

    plt.suptitle('Absolute Error: XGBoost vs LSTM vs Ensemble',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'pcm_hybrid_error_dist.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Error dist plot → {os.path.basename(p)}')


def plot_feature_importance(xgb_models, xgb_feature_cols):
    ars = list(xgb_models.keys())
    fig, axes = plt.subplots(1, len(ars), figsize=(7 * len(ars), 8))
    for i, ar in enumerate(ars):
        imp = pd.Series(xgb_models[ar].feature_importances_,
                        index=xgb_feature_cols[ar])
        imp = imp.sort_values(ascending=True).tail(20)
        imp.plot(kind='barh', ax=axes[i], color='teal')
        axes[i].set_title(f'AR {ar} — XGBoost Top-20 Features')
        axes[i].set_xlabel('Gain'); axes[i].grid(True, alpha=0.3, axis='x')
    plt.suptitle('XGBoost Feature Importances', fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'pcm_hybrid_feature_importance.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Feature importance → {os.path.basename(p)}')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

print('=' * 70)
print('  PCM.py — HYBRID SOTA MODEL')
print('  XGBoost (extended features) + LSTM+Attention (sequential)')
print('  Stacked via Ridge meta-learner → final T_battery prediction')
print('=' * 70)
print(f'  LSTM features ({len(FEATURE_COLS_LSTM)}): {FEATURE_COLS_LSTM}')
print(f'  Data leakage : NONE (T_bat_lag1 used, never T_battery(t))')

all_results      = {}
xgb_models_dict  = {}
xgb_feat_dict    = {}

for ar, path in DATASET_PATHS.items():
    print(f'\n{"=" * 60}')
    print(f'  ASPECT RATIO {ar}')
    print(f'{"=" * 60}')
    ts = datetime.now(IST).strftime('%Y%m%d_%H%M%S')
    name = f'{ts}_pcm_ar{ar}'

    # ── XGBoost branch ──────────────────────────────────────────────────────
    print('\n[XGBoost branch]')
    xd = prepare_xgb_horizon0(path, ar,
                               train_ratio=TRAIN_RATIO,
                               val_fraction=VAL_FRACTION)
    xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_model.fit(xd['X_train'], xd['y_train'],
                  eval_set=[(xd['X_val'], xd['y_val'])],
                  verbose=False)
    xgb_test_pred = xgb_model.predict(xd['X_test'])
    xgb_metrics   = evaluate_predictions(xd['y_test'], xgb_test_pred)
    print(f'  [XGB]  MAE={xgb_metrics["mae"]:.4f}  '
          f'RMSE={xgb_metrics["rmse"]:.4f}  R²={xgb_metrics["r2"]:.4f}')
    xgb_models_dict[ar] = xgb_model
    xgb_feat_dict[ar]   = xd['feature_cols']

    # ── LSTM branch ─────────────────────────────────────────────────────────
    print('\n[LSTM+Attention branch]')
    ld = prepare_ar(path, ar, FEATURE_COLS_LSTM, SEQ_STEPS,
                    train_ratio=TRAIN_RATIO, extended_features=False)
    X_tr, y_tr = ld['X_train'], ld['y_train']
    X_te, y_te = ld['X_test'],  ld['y_test']
    y_sc       = ld['y_scaler']

    lstm_model = build_lstm(batch_size=1, n_features=ld['n_features'])
    hist = train_lstm(lstm_model, X_tr, y_tr,
                      epochs=EPOCHS, patience=PATIENCE, name=name)
    plot_training_history(hist, ar, name)

    lstm_pred_sc   = predict_lstm_raw(lstm_model, X_te)
    lstm_true_sc   = y_te.reshape(-1, 1)
    lstm_pred_K    = y_sc.inverse_transform(lstm_pred_sc).flatten()
    lstm_true_K    = y_sc.inverse_transform(lstm_true_sc).flatten()
    lstm_metrics   = evaluate_predictions(lstm_true_K, lstm_pred_K)
    print(f'  [LSTM] MAE={lstm_metrics["mae"]:.4f}  '
          f'RMSE={lstm_metrics["rmse"]:.4f}  R²={lstm_metrics["r2"]:.4f}')

    # Save LSTM
    lstm_model.save(os.path.join(OUTPUT_DIR, f'{name}_lstm.keras'))
    pd.DataFrame(hist).to_csv(
        os.path.join(OUTPUT_DIR, f'{name}_lstm_history.csv'), index=False)

    del lstm_model
    tf.keras.backend.clear_session()

    # ── Alignment ────────────────────────────────────────────────────────────
    # XGBoost: one prediction per timestep (all test rows)
    # LSTM:    one prediction per SEQ_STEPS*n_chunks timesteps
    # Align by truncating to shorter vector
    xgb_p, lstm_p, y_true_aligned = align_predictions(
        xgb_test_pred, lstm_pred_K, xd['y_test']
    )

    # ── Ridge meta-learner ──────────────────────────────────────────────────
    # Split aligned test block: first half → meta-train, second half → meta-test
    # (Both halves are UNSEEN by XGBoost and LSTM — they are test predictions)
    n_meta = len(xgb_p)
    n_half = n_meta // 2
    meta_X_tr = np.column_stack([xgb_p[:n_half], lstm_p[:n_half]])
    meta_y_tr = y_true_aligned[:n_half]
    meta_X_te = np.column_stack([xgb_p[n_half:], lstm_p[n_half:]])
    meta_y_te = y_true_aligned[n_half:]

    ridge = Ridge(alpha=1.0)
    ridge.fit(meta_X_tr, meta_y_tr)
    ens_pred  = ridge.predict(meta_X_te)
    ens_metrics = evaluate_predictions(meta_y_te, ens_pred)
    print(f'\n  [ENS]  MAE={ens_metrics["mae"]:.4f}  '
          f'RMSE={ens_metrics["rmse"]:.4f}  R²={ens_metrics["r2"]:.4f}')
    print(f'  Ridge weights — XGB: {ridge.coef_[0]:.4f}  '
          f'LSTM: {ridge.coef_[1]:.4f}  bias: {ridge.intercept_:.4f}')

    # Save results
    all_results[ar] = {
        'y_true':       meta_y_te,
        'xgb_pred':     xgb_p[n_half:],
        'lstm_pred':    lstm_p[n_half:],
        'ens_pred':     ens_pred,
        'xgb_metrics':  xgb_metrics,
        'lstm_metrics': lstm_metrics,
        'ens_metrics':  ens_metrics,
        'ridge_coef':   ridge.coef_.tolist(),
        'ridge_bias':   float(ridge.intercept_),
    }

    # Save XGBoost model
    xgb_model.save_model(os.path.join(OUTPUT_DIR, f'{name}_xgb.json'))
    with open(os.path.join(OUTPUT_DIR, f'{name}_xgb_features.txt'), 'w') as f:
        for feat in xd['feature_cols']:
            f.write(feat + '\n')

print('\n' + '=' * 70)
print('  FINAL SUMMARY — ALL ASPECT RATIOS')
print('=' * 70)

header = f'{"AR":<6} {"Branch":<10} {"MAE (K)":<12} {"RMSE (K)":<12} ' \
         f'{"R²":<8} {"Eff %":<8} {"MAE 60%":<10}'
print(header)
print('-' * 70)

summary_rows = []
for ar, r in all_results.items():
    for branch, key, mkey in [
        ('XGBoost',  'xgb_pred',  'xgb_metrics'),
        ('LSTM',     'lstm_pred', 'lstm_metrics'),
        ('Ensemble', 'ens_pred',  'ens_metrics'),
    ]:
        m = r[mkey]
        print(f'{ar:<6} {branch:<10} {m["mae"]:<12.4f} {m["rmse"]:<12.4f} '
              f'{m["r2"]:<8.4f} {m["prediction_eff_pct"]:<8.2f} '
              f'{m["mae_p60"]:<10.4f}')
        summary_rows.append(dict(
            aspect_ratio=ar, branch=branch,
            mae=m['mae'], rmse=m['rmse'], r2=m['r2'],
            mae_p60=m['mae_p60'], rmse_p60=m['rmse_p60'], r2_p60=m['r2_p60'],
            relative_error_pct=m['relative_error_pct'],
            prediction_eff_pct=m['prediction_eff_pct'],
        ))
    print()

# Plots
plot_hybrid_results(all_results)
plot_scatter_trio(all_results)
plot_error_comparison(all_results)
plot_feature_importance(xgb_models_dict, xgb_feat_dict)

# CSV
df_sum = pd.DataFrame(summary_rows)
df_sum.to_csv(os.path.join(OUTPUT_DIR, 'pcm_hybrid_summary.csv'), index=False)
print(f'\nSummary CSV → pcm_hybrid_summary.csv')
print(df_sum.to_string(index=False))

# LaTeX
print('\n' + '=' * 70)
print('  LaTeX TABLE')
print('=' * 70)
print('\\begin{table}[h]\\centering')
print('\\caption{Hybrid SOTA (XGBoost + LSTM + Ensemble) PCM thermal prediction}')
print('\\begin{tabular}{llcccc}\\hline')
print('AR & Branch & MAE (K) & RMSE (K) & $R^2$ & Eff (\\%) \\\\ \\hline')
for r in summary_rows:
    print(f"{r['aspect_ratio']} & {r['branch']} & {r['mae']:.4f} & "
          f"{r['rmse']:.4f} & {r['r2']:.4f} & {r['prediction_eff_pct']:.1f} \\\\")
print('\\hline')
print('\\end{tabular}\\end{table}')

print('\n✅ PCM.py Done — all outputs saved.')
