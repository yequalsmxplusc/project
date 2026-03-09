# CELL 1

# Cell: setup imports and config
import os, sys, json, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_recall_fscore_support
%matplotlib inline

# File names / module names - adjust if your filenames differ
MODULE_PRIMARY = "CHT.ipynb"          # expects CHT.ipynb with predict_all()
MODULE_SECONDARY = "newanalysis.ipynb"        # expects newanalysis.ipynb with predict_all()

# CSV fallbacks (if modules not importable)
CSV_PRIMARY = "predictions_CHT.csv"   # columns: actual,predicted
CSV_SECONDARY = "predictions_newanalysis.csv"

# Hybrid thresholds & params
# Temperature ranges as requested (Low, High, Critical, Super-critical, Hyper-critical)
RANGES = {
    "Low": (None, 60),
    "High": (60, 100),
    "Critical": (100, 250),
    "Super-critical": (250, 400),
    "Hyper-critical": (400, None)
}

# Tolerance for classification-style metrics
TOLERANCE = 0.05  # ±5%
# Hybrid weighting when disagreement (give more weight to primary model)
PRIMARY_WEIGHT = 0.7
SECONDARY_WEIGHT = 0.3

print('Setup complete. Adjust module names or CSV filenames at the top if needed.')

# CELL 2

# Cell: helper functions to load predictions from module or CSV
import importlib.util, importlib, inspect

def load_from_module(mod_name):
    """Try to import module mod_name (python file without .py) and call predict_all() or get_predictions().
       Returns (actuals, preds) as numpy arrays or None if not available."""
    try:
        # if it's a file path
        if os.path.exists(mod_name + ".py"):
            spec = importlib.util.spec_from_file_location(mod_name, mod_name + ".py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            mod = importlib.import_module(mod_name)
    except Exception as e:
        # import failed
        # print("Module import failed:", e)
        return None
    # Look for predict_all or get_predictions
    for fn in ("predict_all", "get_predictions", "predict"):
        if hasattr(mod, fn) and callable(getattr(mod, fn)):
            try:
                out = getattr(mod, fn)()
                # allow (y_true, y_pred) or dict / dataframe
                if isinstance(out, tuple) and len(out) >= 2:
                    y_true, y_pred = np.array(out[0]), np.array(out[1])
                    return y_true, y_pred
                if isinstance(out, dict):
                    y_true = np.array(out.get('actual') or out.get('y_true') or out.get('y_test'))
                    y_pred = np.array(out.get('predicted') or out.get('y_pred') or out.get('y_hat'))
                    return y_true, y_pred
                # dataframe
                if hasattr(out, 'shape') and out.shape[1] >= 2:
                    arr = np.array(out)
                    return arr[:,0], arr[:,1]
            except Exception as e:
                # print('predict fn failed:', e)
                return None
    # fallback: look for variables y_test, y_pred in module
    for var in ("y_test","y_true","actuals","y_actual"):
        for varp in ("y_pred","y_hat","predicted"):
            if hasattr(mod, var) and hasattr(mod, varp):
                try:
                    return np.array(getattr(mod,var)), np.array(getattr(mod,varp))
                except Exception:
                    pass
    return None

def load_from_csv(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    # Accept columns names in variety
    col_actual = None
    for c in ('actual','y_true','y_test','true'):
        if c in df.columns:
            col_actual = c; break
    col_pred = None
    for c in ('predicted','y_pred','y_hat','pred'):
        if c in df.columns:
            col_pred = c; break
    if col_actual is None or col_pred is None:
        return None
    return df[col_actual].values, df[col_pred].values

def load_predictions(primary_module, secondary_module, csv_primary, csv_secondary):
    # Try modules first
    primary = load_from_module(primary_module)
    secondary = load_from_module(secondary_module)
    if primary is None:
        primary = load_from_csv(csv_primary)
    if secondary is None:
        secondary = load_from_csv(csv_secondary)
    return primary, secondary

# quick test load (won't error if files missing)
p, s = load_predictions(MODULE_PRIMARY, MODULE_SECONDARY, CSV_PRIMARY, CSV_SECONDARY)
print('Loaded primary:', bool(p), 'secondary:', bool(s))

# CELL 3

# Cell: multithreaded prediction runner
def run_models_multithread(primary_loader, secondary_loader):
    """primary_loader and secondary_loader are callables that return (y_true, y_pred).
       They are run in parallel threads and results returned."""
    results = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {}
        futures[ex.submit(primary_loader)] = 'primary'
        futures[ex.submit(secondary_loader)] = 'secondary'
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                out = fut.result()
                results[key] = out  # may be None
            except Exception as e:
                results[key] = None
    return results

# Define loader wrappers
def primary_loader():
    return load_from_module(MODULE_PRIMARY) or load_from_csv(CSV_PRIMARY)

def secondary_loader():
    return load_from_module(MODULE_SECONDARY) or load_from_csv(CSV_SECONDARY)

res = run_models_multithread(primary_loader, secondary_loader)
print('Primary available:', bool(res.get('primary')), 'Secondary available:', bool(res.get('secondary')))

# CELL 4

# Cell: hybrid combination logic (range-based routing + weighted averaging on disagreement)
# After multithreaded run, combine results into hybrid prediction array

res_primary = res.get('primary')
res_secondary = res.get('secondary')

if res_primary is None and res_secondary is None:
    raise RuntimeError('No predictions available from primary or secondary loaders. Provide CSVs or importable modules.')

# Choose the source of ground-truth actuals: prefer primary then secondary
actuals = None
if res_primary is not None and len(res_primary)>=2:
    actuals = np.array(res_primary[0])
elif res_secondary is not None and len(res_secondary)>=2:
    actuals = np.array(res_secondary[0])
else:
    raise RuntimeError('Could not determine actuals from model outputs.')

# align lengths
if res_primary is not None:
    y_pred_primary = np.array(res_primary[1])[:len(actuals)]
else:
    y_pred_primary = None
if res_secondary is not None:
    y_pred_secondary = np.array(res_secondary[1])[:len(actuals)]
else:
    y_pred_secondary = None

n = len(actuals)
hybrid_pred = np.full(n, np.nan)

def get_range_label(t):
    for r,(lo,hi) in RANGES.items():
        if lo is None and t < hi: return r
        if hi is None and t >= lo: return r
        if lo is not None and hi is not None and lo <= t < hi: return r
    return 'Unknown'

# combine per-sample
for i, a in enumerate(actuals):
    r = get_range_label(a)
    # simple rule: prefer primary unless range is critical/hyper-critical and secondary exists and differs significantly
    if y_pred_primary is None:
        hybrid_pred[i] = y_pred_secondary[i]
        continue
    if y_pred_secondary is None:
        hybrid_pred[i] = y_pred_primary[i]
        continue
    p = y_pred_primary[i]
    s = y_pred_secondary[i]
    # if both agree within 3% of actual -> take weighted avg (favor primary)
    if abs(p - s) <= 0.03 * max(abs(a),1):
        hybrid_pred[i] = PRIMARY_WEIGHT * p + SECONDARY_WEIGHT * s
        continue
    # If in higher-risk ranges (Critical, Super-critical, Hyper-critical), trust secondary more if it predicts higher safety-critical response
    if r in ('Critical','Super-critical','Hyper-critical'):
        # choose the prediction closer to actual if known historic accuracy; else favor secondary by small margin
        # Here we use weighted average favoring secondary slightly in risky ranges
        hybrid_pred[i] = 0.4 * p + 0.6 * s
    else:
        # Low/High ranges: favor primary
        hybrid_pred[i] = PRIMARY_WEIGHT * p + SECONDARY_WEIGHT * s

print('Hybrid prediction array constructed, length =', len(hybrid_pred))

# CELL 5

# Cell: compute metrics per model and for hybrid, plus per-range metrics and breaking points
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_recall_fscore_support

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def within_tol(y_true, y_pred, tol=TOLERANCE):
    return np.abs(y_true - y_pred) <= (tol * np.maximum(np.abs(y_true),1e-6))

models_results = {}
# primary
if y_pred_primary is not None:
    models_results['primary'] = {'pred': y_pred_primary, 'reg': regression_metrics(actuals, y_pred_primary),
                                 'acc5': float(np.mean(within_tol(actuals, y_pred_primary)))}
# secondary
if y_pred_secondary is not None:
    models_results['secondary'] = {'pred': y_pred_secondary, 'reg': regression_metrics(actuals, y_pred_secondary),
                                   'acc5': float(np.mean(within_tol(actuals, y_pred_secondary)))}
# hybrid
models_results['hybrid'] = {'pred': hybrid_pred, 'reg': regression_metrics(actuals, hybrid_pred),
                            'acc5': float(np.mean(within_tol(actuals, hybrid_pred)))}

# per-range metrics and breaking points
per_range = {k: {} for k in RANGES.keys()}
breaking_points = {k: {} for k in ['primary','secondary','hybrid']}

for name, info in models_results.items():
    y_pred = info['pred']
    per_range[name] = {}
    for r,(lo,hi) in RANGES.items():
        idx = [i for i,t in enumerate(actuals) if (lo is None or t>=lo) and (hi is None or t<hi)]
        if not idx:
            per_range[name][r] = None
            continue
        yt = actuals[idx]
        yp = y_pred[idx]
        reg = regression_metrics(yt, yp)
        acc5 = float(np.mean(within_tol(yt, yp)))
        per_range[name][r] = {'regression': reg, 'acc5': acc5}
        # breaking points: error > mean + 2*std
        errs = np.abs(yp - yt)
        thr = errs.mean() + 2*errs.std()
        breaks = [{'index': int(idx[i]), 'actual': float(yt[i]), 'predicted': float(yp[i]), 'error': float(errs[i])}
                  for i in range(len(errs)) if errs[i] > thr]
        breaking_points[name][r] = breaks

# summary print
import pandas as pd
rows = []
for name, info in models_results.items():
    reg = info['reg']
    rows.append({'Model': name, 'MAE': reg['MAE'], 'RMSE': reg['RMSE'], 'R2': reg['R2'], 'Accuracy±5%': info['acc5']})
df_summary = pd.DataFrame(rows).sort_values('MAE')
print(df_summary.to_string(index=False))

# save summary and breaking points
os.makedirs('hybrid_report', exist_ok=True)
df_summary.to_csv('hybrid_report/summary_metrics.csv', index=False)
with open('hybrid_report/breaking_points.json','w',encoding='utf-8') as f:
    json.dump(breaking_points, f, indent=2)

# CELL 6

# Cell: Visualizations (actual vs predictions + error histograms + per-range MAE)
plt.figure(figsize=(12,6))
plt.plot(actuals, label='Actual', linewidth=2)
if y_pred_primary is not None:
    plt.plot(y_pred_primary, label='Primary (CHT_updated)', alpha=0.8)
if y_pred_secondary is not None:
    plt.plot(y_pred_secondary, label='Secondary (newanalysis)', alpha=0.8)
plt.plot(hybrid_pred, label='Hybrid', linestyle='--', linewidth=2)
plt.legend()
plt.title('Actual vs Predictions (Primary, Secondary, Hybrid)')
plt.xlabel('Sample index')
plt.ylabel('Temperature (°C)')
plt.tight_layout()
plt.savefig('hybrid_report/actual_vs_pred.png', dpi=200)
plt.show()

# Error histograms
for name,info in [('primary', y_pred_primary), ('secondary', y_pred_secondary), ('hybrid', hybrid_pred)]:
    if info is None:
        continue
    errs = np.abs(info - actuals)
    plt.figure(figsize=(8,4))
    plt.hist(errs, bins=40)
    plt.title(f'Absolute Error Distribution - {name}')
    plt.xlabel('Absolute error (°C)')
    plt.tight_layout()
    plt.savefig(f'hybrid_report/error_dist_{name}.png', dpi=150)
    plt.show()

# Per-range MAE chart
for r,(lo,hi) in RANGES.items():
    names = []
    maes = []
    for name,info in models_results.items():
        idx = [i for i,t in enumerate(actuals) if (lo is None or t>=lo) and (hi is None or t<hi)]
        if not idx: continue
        yt = actuals[idx]; yp = info['pred'][idx]
        names.append(name); maes.append(mean_absolute_error(yt, yp))
    if names:
        plt.figure(figsize=(8,4))
        plt.bar(names, maes)
        plt.title(f'MAE in range {r}')
        plt.ylabel('MAE (°C)')
        plt.tight_layout()
        plt.savefig(f'hybrid_report/mae_{r}.png', dpi=150)
        plt.show()

# CELL 7

# Cell: display report assets and final remarks
from IPython.display import Image, display
print('Summary table saved to hybrid_report/summary_metrics.csv')
display(Image('hybrid_report/actual_vs_pred.png'))
for img in sorted([p for p in os.listdir('hybrid_report') if p.startswith('error_dist') or p.startswith('mae_')]):
    display(Image(os.path.join('hybrid_report', img)))

print('\nBreaking points saved to hybrid_report/breaking_points.json')

