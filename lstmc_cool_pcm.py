# -*- coding: utf-8 -*-
"""
lstmc_cool_pcm.py
=================
Stateful LSTM + Multi-Head Self-Attention for PCM battery thermal prediction.

Architecture
────────────
  Input (batch=1, SEQ_STEPS=300, 8 features — all leak-free)
    → Masking
    → LSTM(256, stateful) x 3
    → MultiHeadAttention(num_heads=4, key_dim=64)
    → LayerNormalization (residual)
    → Dense(128, selu) → Dense(64, selu) → Dense(1, linear)

DATA LEAKAGE
─────────────
All 8 input features are drawn from pcm_data.FEATURE_COLS_CORE.
None uses T_battery(t) directly — only T_bat_lag1 = T_battery(t-1).

TF 2.x COMPATIBILITY
──────────────────────
  model.reset_states() removed from Sequential API in TF ≥ 2.13.
  Fixed via reset_lstm_states() helper that iterates layers.
  InputLayer(batch_shape=...) used inside Functional API.

Depends on: pcm_data.py (must be in same directory)
"""

import sys
import os

# Ensure pcm_data.py in same directory is importable without os.chdir()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import warnings
from datetime import datetime
import pytz

import plotly.graph_objects as go

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Masking, MultiHeadAttention,
    LayerNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from pcm_data import (
    load_raw, clean, engineer_features,
    prepare_ar, evaluate_predictions, print_metrics,
    FEATURE_COLS_CORE, TARGET_COL,
)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

DATASET_PATHS = {
    '0.3': '0_3.xlsx',
    '0.4': '0_4.xlsx',
    '0.5': '0_5.xlsx',
}

SEQ_STEPS   = 300
EPOCHS      = 200
PATIENCE    = 30
LSTM_UNITS  = 256
DENSE_UNITS = [128, 64]
ATT_HEADS   = 4
ATT_KEY_DIM = 64
LR          = 1e-4
TRAIN_RATIO = 0.85
OUTPUT_DIR  = './'
SEED        = 42

IST = pytz.timezone('Asia/Kolkata')

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

FEATURE_COLS = FEATURE_COLS_CORE   # 8 leak-free features from pcm_data.py


# ============================================================================
# TF 2.x STATEFUL RESET HELPER
# ============================================================================

def reset_lstm_states(model: keras.Model) -> None:
    """Reset hidden states of all stateful LSTM layers in the model."""
    for layer in model.layers:
        if hasattr(layer, 'reset_states'):
            layer.reset_states()


# ============================================================================
# MODEL — stateful LSTM + multi-head attention
# ============================================================================

def build_model(batch_size: int, n_features: int) -> keras.Model:
    """
    Functional API — batch_size=1 for stateful chunk-by-chunk training.

    Architecture: 3 stateful LSTMs → self-attention → residual+norm → dense
    Loss: Huber (robust to phase-transition outliers)
    """
    inp = Input(batch_shape=(batch_size, SEQ_STEPS, n_features), name='input')

    # Masking: ignore zero-padded positions (safety for variable-length data)
    x = Masking(mask_value=0.0, name='masking')(inp)

    # 3-layer stateful LSTM
    x = LSTM(LSTM_UNITS, activation='tanh',
              return_sequences=True, stateful=True, name='lstm_1')(x)
    x = LSTM(LSTM_UNITS, activation='tanh',
              return_sequences=True, stateful=True, name='lstm_2')(x)
    x = LSTM(LSTM_UNITS, activation='tanh',
              return_sequences=True, stateful=True, name='lstm_3')(x)

    # Multi-head self-attention over the LSTM output sequence
    attn_out = MultiHeadAttention(
        num_heads=ATT_HEADS,
        key_dim=ATT_KEY_DIM,
        name='attention',
    )(x, x)

    # Residual + LayerNorm (standard transformer block pattern)
    x = LayerNormalization(name='layer_norm')(x + attn_out)

    # Dense head applied at every timestep (seq-to-seq prediction)
    x   = Dense(DENSE_UNITS[0], activation='selu', name='dense_1')(x)
    x   = Dense(DENSE_UNITS[1], activation='selu', name='dense_2')(x)
    out = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=inp, outputs=out, name='LSTM_Attention_PCM')
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='huber',
        metrics=['mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')],
    )
    model.summary()
    return model


# ============================================================================
# TRAINING LOOP  (stateful, chunk-shuffled per epoch)
# ============================================================================

def train_stateful(model: keras.Model,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   epochs: int,
                   patience: int,
                   experiment_name: str) -> dict:
    """
    Stateful training: each chunk fed one at a time (batch_size=1).
    Chunk order shuffled each epoch; LSTM states reset at epoch boundary.
    Early stopping monitors training loss (no separate val — same AR).
    Best weights restored at end.
    """
    n_chunks = X_train.shape[0]
    hist = {'loss': [], 'mse': [], 'rmse': []}
    best_loss      = float('inf')
    patience_count = 0
    best_weights   = None
    ckpt_path      = os.path.join(OUTPUT_DIR, f'{experiment_name}_best.weights.h5')

    for epoch in range(epochs):
        avg = {'loss': [], 'mse': [], 'rmse': []}
        chunk_order = random.sample(range(n_chunks), n_chunks)

        for ci, chunk_idx in enumerate(chunk_order):
            bx  = X_train[chunk_idx: chunk_idx + 1]   # (1, SEQ_STEPS, n_feat)
            by  = y_train[chunk_idx: chunk_idx + 1]   # (1, SEQ_STEPS, 1)
            res = model.fit(bx, by, shuffle=False, verbose=0, batch_size=1)
            avg['loss'].append(res.history['loss'][0])
            avg['mse'].append(res.history['mse'][0])
            avg['rmse'].append(res.history['rmse'][0])
            print(f'\r  Chunk {ci+1}/{n_chunks} | '
                  f'loss={np.mean(avg["loss"]):.4e}', end='', flush=True)

        epoch_loss = float(np.mean(avg['loss']))
        epoch_mse  = float(np.mean(avg['mse']))
        epoch_rmse = float(np.mean(avg['rmse']))
        hist['loss'].append(epoch_loss)
        hist['mse'].append(epoch_mse)
        hist['rmse'].append(epoch_rmse)

        print(f'\rEpoch {epoch+1:>3}/{epochs} | '
              f'loss={epoch_loss:.6f}  mse={epoch_mse:.6f}  '
              f'rmse={epoch_rmse:.6f}')

        # Reset stateful LSTM hidden states at epoch boundary
        reset_lstm_states(model)

        # Early stopping on training loss
        if epoch_loss < best_loss - 1e-6:
            best_loss      = epoch_loss
            patience_count = 0
            best_weights   = model.get_weights()
            model.save_weights(ckpt_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f'  Early stopping at epoch {epoch+1}')
                break

    if best_weights is not None:
        model.set_weights(best_weights)

    return hist


# ============================================================================
# INFERENCE  (stateful: feed chunks sequentially, reset before pass)
# ============================================================================

def predict_stateful(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Sequential stateful inference.
    Returns (n_chunks * SEQ_STEPS, 1).
    """
    preds = []
    reset_lstm_states(model)
    for ci in range(X.shape[0]):
        bx = X[ci: ci + 1]                     # (1, SEQ_STEPS, n_feat)
        p  = model.predict(bx, verbose=0)       # (1, SEQ_STEPS, 1)
        preds.append(p[0])                      # (SEQ_STEPS, 1)
    return np.concatenate(preds, axis=0)         # (n_chunks*SEQ_STEPS, 1)


# ============================================================================
# VISUALISATION
# ============================================================================

def plot_training_history(hist: dict, ar: str, experiment_name: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, key, colour in zip(axes, ['loss', 'mse', 'rmse'],
                                ['steelblue', 'coral', 'green']):
        ax.plot(hist[key], color=colour, linewidth=1.5)
        ax.set_title(f'AR {ar} — {key.upper()}')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'Training History  |  {experiment_name}', fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f'{experiment_name}_history.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  History plot → {os.path.basename(out)}')


def plot_predictions(y_true_K: np.ndarray, y_pred_K: np.ndarray,
                     ar: str, split: str, experiment_name: str,
                     show_n: int = 1000):
    n   = min(show_n, len(y_true_K))
    idx = np.arange(n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=idx, y=y_pred_K[:n], mode='lines', name='Predicted',
        line=dict(color='cyan', width=1.5)))
    fig.add_trace(go.Scatter(
        x=idx, y=y_true_K[:n], mode='lines', name='Actual',
        line=dict(color='black', width=1.5)))
    fig.update_layout(
        title=f'AR {ar} — {split} (first {n} steps)',
        xaxis_title='Step', yaxis_title='T_battery (K)',
        template='plotly_white', hovermode='x unified',
    )
    out = os.path.join(OUTPUT_DIR, f'{experiment_name}_{split}_predictions.html')
    fig.write_html(out)
    print(f'  {split.capitalize()} prediction plot → {os.path.basename(out)}')


def plot_all_summary(all_results: dict):
    ars = list(all_results.keys())
    fig, axes = plt.subplots(2, len(ars), figsize=(7 * len(ars), 10))

    for col, ar in enumerate(ars):
        r      = all_results[ar]
        y_true = r['y_true_test_K']
        y_pred = r['y_pred_test_K']
        show   = min(1000, len(y_true))

        axes[0, col].plot(y_true[:show], color='black', lw=1, label='Actual')
        axes[0, col].plot(y_pred[:show], color='cyan', lw=1.5,
                          ls='--', label='Predicted')
        axes[0, col].set_title(
            f'AR {ar}\nMAE={r["metrics"]["mae"]:.4f} K  '
            f'R²={r["metrics"]["r2"]:.4f}',
            fontweight='bold')
        axes[0, col].set_xlabel('Step')
        axes[0, col].set_ylabel('T_battery (K)')
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].scatter(y_true, y_pred, alpha=0.3, s=6,
                             color='steelblue')
        lim = [min(y_true.min(), y_pred.min()),
               max(y_true.max(), y_pred.max())]
        axes[1, col].plot(lim, lim, 'r--', lw=2)
        axes[1, col].set_xlabel('Actual (K)')
        axes[1, col].set_ylabel('Predicted (K)')
        axes[1, col].set_title(f'AR {ar} — Scatter', fontweight='bold')
        axes[1, col].grid(True, alpha=0.3)

    plt.suptitle('LSTM+Attention PCM Prediction — Test Set',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'lstm_pcm_summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Summary plot → {os.path.basename(out)}')


def eda_plots(df_raw: pd.DataFrame, ar: str):
    df   = engineer_features(df_raw, extended=False)
    cols = [
        ('liquid_frac',    'green'),
        ('Nu',             'orange'),
        ('T_pcm',          'steelblue'),
        ('T_bat_lag1',     'brown'),
        ('dT_lag1',        'red'),
        ('heat_flux_lag1', 'purple'),
        ('melting_rate',   'teal'),
        ('T_pcm_rate',     'olive'),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(22, 8))
    for ax, (col, colour) in zip(axes.flat, cols):
        if col in df.columns:
            sns.kdeplot(data=df, x=col, ax=ax, color=colour,
                        fill=True, alpha=0.4)
            ax.set_title(f'AR {ar} — {col}')
            ax.grid(True, alpha=0.3)
    plt.suptitle(f'Feature Distributions — AR {ar}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f'eda_ar{ar}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  EDA plot → {os.path.basename(out)}')


# ============================================================================
# MAIN
# ============================================================================

print('=' * 70)
print('  LSTM + ATTENTION  PCM THERMAL PREDICTION')
print('=' * 70)
print(f'  Features  : {FEATURE_COLS}')
print(f'  Target    : {TARGET_COL}')
print(f'  Seq steps : {SEQ_STEPS}')
print(f'  LSTM units: {LSTM_UNITS}  (3 stateful layers)')
print(f'  Attention : {ATT_HEADS} heads, key_dim={ATT_KEY_DIM}')
print(f'  Epochs    : {EPOCHS}  (patience={PATIENCE})')
print(f'  LR        : {LR}')
print(f'  Data leakage: NONE (all derived features use lagged T_battery)')

all_results   = {}
all_histories = {}

for ar, path in DATASET_PATHS.items():
    print(f'\n{"="*60}')
    print(f'  ASPECT RATIO {ar}')
    print(f'{"="*60}')

    # EDA
    df_raw = load_raw(path)
    df_raw = clean(df_raw, ar)
    eda_plots(df_raw, ar)

    # Data pipeline (leak-free, via pcm_data.prepare_ar)
    data     = prepare_ar(path, ar, FEATURE_COLS, SEQ_STEPS,
                          train_ratio=TRAIN_RATIO, extended_features=False)
    X_train  = data['X_train']
    y_train  = data['y_train']
    X_test   = data['X_test']
    y_test   = data['y_test']
    y_scaler = data['y_scaler']

    print(f'  Train chunks : {X_train.shape[0]} × '
          f'({SEQ_STEPS} steps, {data["n_features"]} features)')
    print(f'  Test  chunks : {X_test.shape[0]} × '
          f'({SEQ_STEPS} steps, {data["n_features"]} features)')

    experiment_name = (datetime.now(IST).strftime('%Y%m%d_%H%M%S')
                       + f'_lstm_pcm_ar{ar}')
    print(f'  Experiment   : {experiment_name}')

    # Build model (batch_size=1 for chunk-by-chunk stateful training)
    model = build_model(batch_size=1, n_features=data['n_features'])

    # Train
    hist = train_stateful(model, X_train, y_train,
                          epochs=EPOCHS, patience=PATIENCE,
                          experiment_name=experiment_name)
    all_histories[ar] = hist
    plot_training_history(hist, ar, experiment_name)

    # Predict — stateful pass over both splits
    y_pred_train_sc = predict_stateful(model, X_train)
    y_pred_test_sc  = predict_stateful(model, X_test)

    y_true_train_sc = y_train.reshape(-1, 1)
    y_true_test_sc  = y_test.reshape(-1, 1)

    y_pred_train_K = y_scaler.inverse_transform(y_pred_train_sc).flatten()
    y_true_train_K = y_scaler.inverse_transform(y_true_train_sc).flatten()
    y_pred_test_K  = y_scaler.inverse_transform(y_pred_test_sc).flatten()
    y_true_test_K  = y_scaler.inverse_transform(y_true_test_sc).flatten()

    # Metrics
    metrics = evaluate_predictions(y_true_test_K, y_pred_test_K)
    print(f'\n  ── Test metrics ──')
    print_metrics(metrics)

    # Plots
    plot_predictions(y_true_train_K, y_pred_train_K, ar,
                     'train', experiment_name)
    plot_predictions(y_true_test_K,  y_pred_test_K,  ar,
                     'test',  experiment_name)

    # Save model
    model_path = os.path.join(OUTPUT_DIR, f'{experiment_name}.keras')
    model.save(model_path)
    print(f'  Model saved → {os.path.basename(model_path)}')

    pd.DataFrame(hist).to_csv(
        os.path.join(OUTPUT_DIR, f'{experiment_name}_history.csv'),
        index=False)

    all_results[ar] = {
        'metrics':        metrics,
        'y_true_train_K': y_true_train_K,
        'y_pred_train_K': y_pred_train_K,
        'y_true_test_K':  y_true_test_K,
        'y_pred_test_K':  y_pred_test_K,
    }

    del model
    tf.keras.backend.clear_session()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print('\n' + '=' * 70)
print('  FINAL RESULTS SUMMARY')
print('=' * 70)
print(f'\n{"AR":<8} {"MAE (K)":<14} {"RMSE (K)":<14} {"R²":<10} '
      f'{"MAE 60% (K)":<14} {"R² 60%":<10}')
print('-' * 70)

summary_rows = []
for ar, r in all_results.items():
    m = r['metrics']
    print(f'{ar:<8} {m["mae"]:<14.4f} {m["rmse"]:<14.4f} {m["r2"]:<10.4f} '
          f'{m["mae_p60"]:<14.4f} {m["r2_p60"]:.4f}')
    summary_rows.append(dict(
        aspect_ratio=ar,
        T_min=r['y_true_test_K'].min(),
        T_max=r['y_true_test_K'].max(),
        **{k: m[k] for k in ('mae', 'rmse', 'r2',
                              'mae_p60', 'rmse_p60', 'r2_p60',
                              'relative_error_pct', 'prediction_eff_pct')},
    ))

avg_mae  = np.mean([r['metrics']['mae']  for r in all_results.values()])
avg_rmse = np.mean([r['metrics']['rmse'] for r in all_results.values()])
avg_r2   = np.mean([r['metrics']['r2']   for r in all_results.values()])
print('-' * 70)
print(f'{"AVG":<8} {avg_mae:<14.4f} {avg_rmse:<14.4f} {avg_r2:.4f}')

plot_all_summary(all_results)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUTPUT_DIR, 'lstm_pcm_summary.csv'), index=False)
print(f'\nSummary CSV saved → lstm_pcm_summary.csv')
print(df_summary.to_string(index=False))

# LaTeX
print('\n' + '=' * 70)
print('  LaTeX TABLE')
print('=' * 70)
print('\\begin{table}[h]\\centering')
print('\\caption{Stateful LSTM+Attention PCM thermal prediction '
      '(per aspect ratio, no data leakage)}')
print('\\begin{tabular}{lcccc}\\hline')
print('AR & $T$ range (K) & MAE (K) & $R^2$ & Eff. (\\%) \\\\ \\hline')
for r in summary_rows:
    print(f"{r['aspect_ratio']} & {r['T_min']:.1f}–{r['T_max']:.1f} & "
          f"{r['mae']:.4f} & {r['r2']:.4f} & "
          f"{r['prediction_eff_pct']:.1f} \\\\")
print(f'\\hline Avg & — & {avg_mae:.4f} & {avg_r2:.4f} & — \\\\ \\hline')
print('\\end{tabular}\\end{table}')

print('\nDone.')
