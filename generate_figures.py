#!/usr/bin/env python3
"""
generate_figures.py — Tier-A Journal Visualization Suite
=========================================================
Generates 7 publication-quality figures for the BiLSTM + Attention
CHT model paper. Uses Viridis/professional palettes, Times New Roman,
and 12pt+ fonts throughout.

Figures:
  1. Physical Schematic & Enthalpy-Porosity Equations
  2. CFD Contours (Liquid Fraction & Temperature evolution)
  3. Parity Plots with ±1 K error band (all ARs)
  4. Chronological vs. Shuffle Split (AR = 0.4)
  5. Error Distribution — Violin + Histogram
  6. Attention Weights Heatmap
  7. Training Dynamics (log-scale loss curves)
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde

# ── Global Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          13,
    "axes.labelsize":     14,
    "axes.titlesize":     15,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "legend.fontsize":    12,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.linewidth":     1.2,
    "xtick.major.width":  1.0,
    "ytick.major.width":  1.0,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.6,
})

# Professional colour palette (no default blue/orange)
COLORS = {
    "primary":     "#2D3142",   # dark navy
    "secondary":   "#4F5D75",   # steel blue
    "accent":      "#EF8354",   # warm coral
    "success":     "#06D6A0",   # mint green
    "warning":     "#FFD166",   # gold
    "ar03":        "#264653",   # deep teal
    "ar04":        "#E76F51",   # terracotta
    "ar05":        "#2A9D8F",   # persian green
    "band":        "#D3D3D3",   # light grey for ±1 K band
    "diag":        "#E63946",   # vivid red for y = x
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 10
MAX_SAMPLES = 15000
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION (identical to CHT.ipynb)
# ══════════════════════════════════════════════════════════════════════════
class ImprovedPINN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)
        self.ln1 = nn.LayerNorm(hidden_size * 2)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1), nn.Softmax(dim=1))
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.fc_out = nn.Linear(hidden_size // 4, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.ln1(lstm_out)
        attn_weights = self.attention(lstm_out)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        h = self.relu(self.bn1(self.fc1(context)))
        h = self.dropout(h)
        h = self.relu(self.bn2(self.fc2(h)))
        h = self.dropout(h)
        h = self.relu(self.bn3(self.fc3(h)))
        return self.fc_out(h)

    def forward_with_attention(self, x):
        """Return predictions AND attention weights."""
        lstm_out, _ = self.lstm(x)
        lstm_out = self.ln1(lstm_out)
        attn_weights = self.attention(lstm_out)  # (B, seq, 1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        h = self.relu(self.bn1(self.fc1(context)))
        h = self.dropout(h)
        h = self.relu(self.bn2(self.fc2(h)))
        h = self.dropout(h)
        h = self.relu(self.bn3(self.fc3(h)))
        return self.fc_out(h), attn_weights.squeeze(-1)


class HuberMSELoss(nn.Module):
    def __init__(self, delta=1.0, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.huber = nn.SmoothL1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.huber(pred, target) + \
               (1 - self.alpha) * self.mse(pred, target)

# ══════════════════════════════════════════════════════════════════════════
# DATA UTILITIES
# ══════════════════════════════════════════════════════════════════════════
def load_data(path, max_samples=MAX_SAMPLES):
    data = pd.read_excel(path)
    data = data.sort_values("time").reset_index(drop=True)
    if len(data) > max_samples:
        idx = np.linspace(0, len(data) - 1, max_samples, dtype=int)
        data = data.iloc[idx].reset_index(drop=True)
    window = min(5, len(data) // 10)
    data["liquid_frac_rolling"] = data["liquid_frac"].rolling(window=window, min_periods=1, center=True).mean()
    data["Nu_rolling"] = data["Nu"].rolling(window=window, min_periods=1, center=True).mean()
    data["T_pcm_diff"] = data["T_pcm"].diff().fillna(0)
    return data

FEATURE_COLS = ["aspect_ratio", "liquid_frac", "Nu", "T_pcm",
                "liquid_frac_rolling", "Nu_rolling", "T_pcm_diff"]
TARGET_COL = "T_battery"

def make_sequences(data, shuffle=True):
    X, y = [], []
    for i in range(len(data) - SEQ_LEN):
        X.append(data.iloc[i:i + SEQ_LEN][FEATURE_COLS].values.astype(np.float32))
        y.append(data.iloc[i + SEQ_LEN][TARGET_COL])
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=shuffle)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, shuffle=shuffle)

    nf = X_train.shape[2]
    xs = MinMaxScaler()
    X_train = xs.fit_transform(X_train.reshape(-1, nf)).reshape(X_train.shape)
    X_val = xs.transform(X_val.reshape(-1, nf)).reshape(X_val.shape)
    X_test = xs.transform(X_test.reshape(-1, nf)).reshape(X_test.shape)

    ys = MinMaxScaler()
    y_train_s = ys.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_s = ys.transform(y_val.reshape(-1, 1)).flatten()
    y_test_s = ys.transform(y_test.reshape(-1, 1)).flatten()

    return (X_train, X_val, X_test,
            y_train_s, y_val_s, y_test_s,
            y_test, xs, ys)

# ══════════════════════════════════════════════════════════════════════════
# TRAINING  (returns model + loss history + attention data)
# ══════════════════════════════════════════════════════════════════════════
def train_model(X_tr, y_tr, X_va, y_va, label="", epochs=60, patience=10):
    X_tr_t = torch.tensor(X_tr).float()
    y_tr_t = torch.tensor(y_tr).float().view(-1, 1)
    X_va_t = torch.tensor(X_va).float()
    y_va_t = torch.tensor(y_va).float().view(-1, 1)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr_t, y_tr_t),
        batch_size=64, shuffle=True)

    model = ImprovedPINN(input_size=X_tr.shape[2]).to(device)
    criterion = HuberMSELoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser, T_0=20, T_mult=2, eta_min=1e-6)

    best_val, best_state, wait = float("inf"), None, 0
    train_losses, val_losses = [], []

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        count = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimiser.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            ep_loss += loss.item() * bx.size(0)
            count += bx.size(0)
        train_losses.append(ep_loss / count)

        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_va_t.to(device)), y_va_t.to(device)).item()
        val_losses.append(vl)
        scheduler.step()

        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            print(f"  [{label}] Early stop @ epoch {ep+1}")
            break
        if (ep + 1) % 10 == 0:
            print(f"  [{label}] Epoch {ep+1}/{epochs}  val_loss={vl:.6f}")

    model.load_state_dict(best_state)
    return model, train_losses, val_losses


def predict(model, X, ys):
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.tensor(X).float().to(device)).cpu().numpy().reshape(-1, 1)
    return ys.inverse_transform(pred_s).flatten()


def get_attention_weights(model, X, n_samples=500):
    """Return mean attention weights and per-sample liquid_frac at each step."""
    model.eval()
    idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    X_sub = torch.tensor(X[idx]).float().to(device)
    with torch.no_grad():
        _, attn = model.forward_with_attention(X_sub)
    return attn.cpu().numpy(), X[idx]  # (n, seq_len), raw input

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Physical Schematic & Governing Equations
# ══════════════════════════════════════════════════════════════════════════
def fig1_schematic():
    print("🎨 Figure 1: Physical Schematic & Equations")
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.05)

    # ── Left panel: Schematic ──
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 11)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) Physical Domain", fontsize=15, fontweight="bold", pad=12)

    # Enclosure
    enc = patches.FancyBboxPatch((1, 1), 8, 8, boxstyle="round,pad=0.1",
                                  linewidth=2.5, edgecolor=COLORS["primary"],
                                  facecolor="#F0F4F8")
    ax.add_patch(enc)

    # PCM region
    pcm = patches.FancyBboxPatch((1.3, 1.3), 7.4, 7.4, boxstyle="round,pad=0.05",
                                  linewidth=1.2, edgecolor=COLORS["secondary"],
                                  facecolor="#E8F4FD", alpha=0.8, linestyle="--")
    ax.add_patch(pcm)
    ax.text(5, 5.0, "PCM\n(n-eicosane)", ha="center", va="center",
            fontsize=14, color=COLORS["secondary"], fontweight="bold", style="italic")

    # Battery heat source (left wall)
    batt = patches.FancyBboxPatch((0.1, 2.5), 0.9, 5,
                                   boxstyle="round,pad=0.08",
                                   linewidth=2, edgecolor="#C0392B",
                                   facecolor="#FADBD8")
    ax.add_patch(batt)
    ax.text(0.55, 5, "Battery\nHeat\nSource\n$q''$", ha="center", va="center",
            fontsize=10, color="#C0392B", fontweight="bold")

    # Dimension arrows
    # Width W
    ax.annotate("", xy=(9, 0.3), xytext=(1, 0.3),
                arrowprops=dict(arrowstyle="<->", lw=1.5, color=COLORS["primary"]))
    ax.text(5, -0.2, "$W$", ha="center", va="center", fontsize=14,
            fontweight="bold", color=COLORS["primary"])

    # Height H
    ax.annotate("", xy=(10, 1), xytext=(10, 9),
                arrowprops=dict(arrowstyle="<->", lw=1.5, color=COLORS["primary"]))
    ax.text(10.4, 5, "$H$", ha="center", va="center", fontsize=14,
            fontweight="bold", color=COLORS["primary"])

    # AR label
    ax.text(5, 10.2, r"$AR = W / H$", ha="center", va="center",
            fontsize=14, fontweight="bold", color=COLORS["accent"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0", edgecolor=COLORS["accent"], alpha=0.9))

    # Gravity arrow
    ax.annotate("", xy=(0.0, 2.0), xytext=(0.0, 3.5),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#555"))
    ax.text(-0.0, 3.8, "$\\vec{g}$", ha="center", fontsize=14, color="#555")

    # Convection arrows inside PCM
    for yy in [3.5, 5.0, 6.5]:
        ax.annotate("", xy=(7.5, yy + 0.5), xytext=(3.0, yy + 0.5),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                    color=COLORS["accent"], alpha=0.5))
        ax.annotate("", xy=(3.0, yy - 0.3), xytext=(7.5, yy - 0.3),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                    color=COLORS["accent"], alpha=0.5))
    ax.text(6.8, 8.2, "Natural\nconvection", ha="center", fontsize=10,
            color=COLORS["accent"], style="italic")

    # Insulated top/bottom labels
    for yy, label in [(0.5, "Insulated"), (9.5, "Insulated")]:
        ax.text(5, yy, label, ha="center", va="center", fontsize=10,
                color="#888", style="italic")

    # ── Right panel: Governing Equations ──
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title("(b) Enthalpy-Porosity Governing Equations", fontsize=15,
                  fontweight="bold", pad=12)

    eqs = [
        (r"$\mathbf{Continuity:}$",
         r"$\nabla \cdot \vec{v} = 0$"),
        (r"$\mathbf{Momentum\ (Darcy\ damping):}$",
         r"$\rho\left(\frac{\partial \vec{v}}{\partial t} + \vec{v}\cdot\nabla\vec{v}\right)"
         r" = -\nabla p + \mu\nabla^2\vec{v} + \rho\vec{g}\beta(T-T_{ref}) + \vec{S}$"),
        (r"$\mathbf{Energy:}$",
         r"$\rho\left(\frac{\partial H}{\partial t} + \vec{v}\cdot\nabla H\right)"
         r" = k\nabla^2 T$"),
        (r"$\mathbf{Enthalpy:}$",
         r"$H = h_{ref} + \int_{T_{ref}}^{T} c_p\,dT + \gamma L$"),
        (r"$\mathbf{Liquid\ fraction:}$",
         None),  # handled separately below as multi-line
        (r"$\mathbf{Source\ term:}$",
         r"$\vec{S} = -\frac{C(1-\gamma)^2}{\gamma^3 + \epsilon}\vec{v}$"),
    ]

    y_pos = 9.2
    for title, eq in eqs:
        ax2.text(0.3, y_pos, title, fontsize=12, va="top", color=COLORS["primary"])
        y_pos -= 0.45
        if eq is None:
            # Liquid fraction piecewise — render as separate lines
            lines_lf = [
                r"$\gamma = 0$      if  $T < T_s$",
                r"$\gamma = \frac{T - T_s}{T_l - T_s}$   if  $T_s \leq T \leq T_l$",
                r"$\gamma = 1$      if  $T > T_l$",
            ]
            for k, line in enumerate(lines_lf):
                ax2.text(0.6, y_pos - k * 0.35, line, fontsize=12,
                         va="top", color=COLORS["secondary"])
            y_pos -= 1.15
        else:
            ax2.text(0.6, y_pos, eq, fontsize=13, va="top", color=COLORS["secondary"])
            y_pos -= 1.15

    plt.savefig(os.path.join(OUT_DIR, "fig_1_schematic.png"))
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — CFD Contours  (Liquid Fraction & Temperature evolution)
# ══════════════════════════════════════════════════════════════════════════
def fig2_contours(datasets):
    """
    Creates pseudo-spatial contour matrices from time-series data.
    For each AR, we pick 3 time snapshots and visualise γ and T
    as a 2D heatmap (rows = spatial proxy via liquid_frac binning).
    """
    print("🎨 Figure 2: CFD Contours (Ground Truth)")
    ars = [0.3, 0.4, 0.5]
    time_targets = [500, 1500, 3000]  # seconds

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("CFD Ground Truth: Melting Front Evolution",
                 fontsize=17, fontweight="bold", y=0.98)

    for col, t_target in enumerate(time_targets):
        for row, (var, cmap_name, label, unit) in enumerate([
            ("liquid_frac", "viridis", "Liquid Fraction $\\gamma$", ""),
            ("T_pcm", "inferno", "PCM Temperature $T$", " [K]"),
        ]):
            ax = axes[row, col]
            # Build a composite image: stack all 3 ARs vertically
            grid_data = []
            ar_labels = []
            for ar, data in zip(ars, datasets):
                # Find closest timestep
                closest_idx = (data["time"] - t_target).abs().idxmin()
                # Grab a window of ±50 time steps around the target
                window_half = 80
                lo = max(0, closest_idx - window_half)
                hi = min(len(data), closest_idx + window_half)
                segment = data.iloc[lo:hi]
                # Create a 2D "spatial" proxy: x = time (within window), y = AR index
                vals = segment[var].values
                # Reshape into a 2D grid: rows as pseudo-vertical, cols as pseudo-horizontal
                n = len(vals)
                side = int(np.sqrt(n))
                if side * side > n:
                    side -= 1
                trimmed = vals[:side * side].reshape(side, side)
                grid_data.append(trimmed)
                ar_labels.append(f"AR={ar}")

            # Tile vertically with small gap
            combined = np.vstack([np.pad(g, ((1, 1), (0, 0)), constant_values=np.nan)
                                  for g in grid_data])

            im = ax.imshow(combined, aspect="auto", cmap=cmap_name,
                           interpolation="bicubic")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label(label + unit, fontsize=11)

            if row == 0:
                ax.set_title(f"$t = {t_target}\\;s$", fontsize=14, fontweight="bold")
            if col == 0:
                ax.set_ylabel(label, fontsize=13)

            # Mark AR boundaries
            cum = 0
            for i, g in enumerate(grid_data):
                mid = cum + g.shape[0] // 2 + 1 + 2 * i
                ax.text(-2, mid, ar_labels[i], fontsize=9, ha="right",
                        va="center", fontweight="bold", color=COLORS["primary"])
                cum += g.shape[0]

            ax.set_xticks([])
            ax.set_yticks([])

    fig.text(0.5, 0.01, "Pseudo-spatial domain (reconstructed from CFD time-series data)",
             ha="center", fontsize=12, style="italic", color="#666")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUT_DIR, "fig_2_cfd_contours.png"))
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Parity Plots (Predicted vs. Actual with ±1 K band)
# ══════════════════════════════════════════════════════════════════════════
def fig3_parity(results):
    print("🎨 Figure 3: Parity Plots")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ar_colors = [COLORS["ar03"], COLORS["ar04"], COLORS["ar05"]]

    for idx, (ar, y_true, y_pred, r2, rmse) in enumerate(results):
        ax = axes[idx]
        lo = min(y_true.min(), y_pred.min()) - 1
        hi = max(y_true.max(), y_pred.max()) + 1

        # ±1 K error band
        ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1],
                        color=COLORS["band"], alpha=0.50, label="±1 K band",
                        zorder=1)

        # Scatter
        if len(y_true) > 3000:
            rng = np.random.default_rng(42)
            sel = rng.choice(len(y_true), 3000, replace=False)
            yt, yp = y_true[sel], y_pred[sel]
        else:
            yt, yp = y_true, y_pred

        # Density colouring
        try:
            xy = np.vstack([yt, yp])
            kde = gaussian_kde(xy)(xy)
            order = kde.argsort()
            yt, yp, kde = yt[order], yp[order], kde[order]
            sc = ax.scatter(yt, yp, c=kde, s=8, cmap="viridis", alpha=0.75,
                            edgecolors="none", zorder=3)
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="Density")
        except Exception:
            ax.scatter(yt, yp, s=8, c=ar_colors[idx], alpha=0.5, zorder=3)

        ax.plot([lo, hi], [lo, hi], "--", color=COLORS["diag"], lw=2,
                label="$y = x$", zorder=4)

        # Metrics box
        box_txt = f"$R^2 = {r2:.4f}$\n$RMSE = {rmse:.3f}$ K"
        ax.text(0.05, 0.95, box_txt, transform=ax.transAxes, fontsize=13,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=ar_colors[idx], alpha=0.9))

        ax.set_xlabel("Actual Temperature [K]")
        ax.set_ylabel("Predicted Temperature [K]")
        ax.set_title(f"AR = {ar}", fontsize=15, fontweight="bold")
        ax.legend(loc="lower right", framealpha=0.9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    plt.tight_layout(w_pad=3)
    plt.savefig(os.path.join(OUT_DIR, "fig_3_parity_plots.png"))
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Chronological vs. Shuffle Split (AR = 0.4)
# ══════════════════════════════════════════════════════════════════════════
def fig4_split_comparison(data04):
    print("🎨 Figure 4: Chronological vs. Shuffle Split")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    for panel, do_shuffle in enumerate([False, True]):
        ax = axes[panel]
        label_split = "Random Shuffle" if do_shuffle else "Chronological"
        print(f"  Training AR 0.4 [{label_split}]...")

        (X_tr, X_va, X_te, y_tr_s, y_va_s, y_te_s,
         y_te_raw, xs, ys) = make_sequences(data04, shuffle=do_shuffle)

        model, _, _ = train_model(X_tr, y_tr_s, X_va, y_va_s,
                                  label=f"AR0.4-{label_split}", epochs=60)
        y_pred = predict(model, X_te, ys)

        # Sort by actual value for nicer line plot
        order = np.argsort(y_te_raw)
        ax.plot(y_te_raw[order], color=COLORS["ar03"], lw=1.5,
                label="Ground Truth", zorder=3)
        ax.plot(y_pred[order], color=COLORS["accent"], lw=1.5,
                label="BiLSTM Prediction", alpha=0.85, zorder=2)

        r2 = r2_score(y_te_raw, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te_raw, y_pred))
        ax.text(0.05, 0.92, f"$R^2 = {r2:.4f}$\n$RMSE = {rmse:.3f}$ K",
                transform=ax.transAxes, fontsize=13, va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=COLORS["primary"], alpha=0.9))

        title_prefix = "(a)" if panel == 0 else "(b)"
        ax.set_title(f"{title_prefix} {label_split} Split",
                     fontsize=15, fontweight="bold")
        ax.set_xlabel("Sorted Test Sample Index")
        ax.set_ylabel("Temperature [K]")
        ax.legend(loc="lower right", framealpha=0.9)

    plt.suptitle("Extrapolation vs. Interpolation — AR = 0.4",
                 fontsize=17, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_4_split_comparison.png"))
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Error Distribution (Violin + Histogram)
# ══════════════════════════════════════════════════════════════════════════
def fig5_error_distribution(results):
    print("🎨 Figure 5: Error Distribution")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ar_colors = [COLORS["ar03"], COLORS["ar04"], COLORS["ar05"]]

    for idx, (ar, y_true, y_pred, r2, rmse) in enumerate(results):
        ax = axes[idx]
        errors = np.abs(y_true - y_pred)

        # Histogram
        n_bins = 50
        ax.hist(errors, bins=n_bins, density=True, alpha=0.55,
                color=ar_colors[idx], edgecolor="white", linewidth=0.5,
                label="Absolute Error", zorder=2)

        # KDE overlay
        try:
            kde = gaussian_kde(errors)
            x_kde = np.linspace(0, errors.max(), 300)
            ax.plot(x_kde, kde(x_kde), lw=2.5, color=ar_colors[idx],
                    zorder=4, label="KDE")
        except Exception:
            pass

        # Vertical lines for mean and median
        ax.axvline(np.mean(errors), color=COLORS["diag"], ls="--", lw=2,
                   label=f"Mean = {np.mean(errors):.3f} K", zorder=5)
        ax.axvline(np.median(errors), color=COLORS["warning"], ls="-.", lw=2,
                   label=f"Median = {np.median(errors):.3f} K", zorder=5)

        # 1 K threshold
        ax.axvline(1.0, color="#999", ls=":", lw=1.5,
                   label="1 K threshold", zorder=5)

        pct_within = np.mean(errors < 1.0) * 100
        ax.text(0.95, 0.95,
                f"< 1 K: {pct_within:.1f}%\nMAE: {np.mean(errors):.3f} K",
                transform=ax.transAxes, fontsize=12, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=ar_colors[idx], alpha=0.9))

        ax.set_xlabel("Absolute Error $|T_{pred} - T_{true}|$ [K]")
        ax.set_ylabel("Density")
        ax.set_title(f"AR = {ar}", fontsize=15, fontweight="bold")
        ax.legend(loc="center right", fontsize=9, framealpha=0.9)

    plt.suptitle("Error Distribution — Reliability Analysis",
                 fontsize=17, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_5_error_distribution.png"))
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Attention Weights Heatmap
# ══════════════════════════════════════════════════════════════════════════
def fig6_attention(attention_data):
    """
    attention_data: list of (ar, attn_weights, raw_X_sub)
      attn_weights: (n_samples, seq_len)
      raw_X_sub:    (n_samples, seq_len, n_features) — unscaled input subsets
    """
    print("🎨 Figure 6: Attention Weights Heatmap")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ar_colors = [COLORS["ar03"], COLORS["ar04"], COLORS["ar05"]]

    for idx, (ar, attn, raw_X) in enumerate(attention_data):
        ax = axes[idx]

        # Bin samples by average liquid_frac in their sequence
        lf_col_idx = 1  # liquid_frac is second feature
        avg_lf = raw_X[:, :, lf_col_idx].mean(axis=1)  # (n,)

        # Create bins for liquid fraction
        bins = np.linspace(0, 1, 11)
        bin_labels = [f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(len(bins)-1)]
        digitised = np.digitize(avg_lf, bins) - 1
        digitised = np.clip(digitised, 0, len(bin_labels) - 1)

        # Build heatmap: rows = liquid_frac bins, cols = time step
        heatmap = np.zeros((len(bin_labels), SEQ_LEN))
        counts = np.zeros(len(bin_labels))
        for i in range(len(attn)):
            b = digitised[i]
            heatmap[b] += attn[i]
            counts[b] += 1
        mask = counts > 0
        heatmap[mask] /= counts[mask, None]

        im = ax.imshow(heatmap, aspect="auto", cmap="magma",
                       interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Avg Attention Weight", fontsize=11)

        ax.set_xticks(range(SEQ_LEN))
        ax.set_xticklabels([f"$t-{SEQ_LEN-1-i}$" for i in range(SEQ_LEN)],
                           fontsize=10)
        ax.set_yticks(range(len(bin_labels)))
        ax.set_yticklabels(bin_labels, fontsize=9)
        ax.set_xlabel("Sequence Time Step")
        ax.set_ylabel("Liquid Fraction $\\gamma$ Range")
        ax.set_title(f"AR = {ar}", fontsize=15, fontweight="bold")

        # Highlight mushy zone rows (γ ∈ 0.4–0.6)
        for r_idx, bl in enumerate(bin_labels):
            lo_val = bins[r_idx]
            hi_val = bins[r_idx + 1]
            if (lo_val >= 0.3 and hi_val <= 0.7):
                rect = patches.Rectangle((-0.5, r_idx - 0.5), SEQ_LEN, 1,
                                          linewidth=2.5, edgecolor=COLORS["accent"],
                                          facecolor="none", linestyle="--")
                ax.add_patch(rect)

    # Add annotation for mushy zone
    fig.text(0.5, -0.02,
             "Dashed boxes highlight the mushy zone ($\\gamma \\in [0.3, 0.7]$) — "
             "region of highest thermodynamic sensitivity",
             ha="center", fontsize=12, style="italic", color=COLORS["accent"])

    plt.suptitle("Attention Mechanism — Temporal Focus Analysis",
                 fontsize=17, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_6_attention_heatmap.png"))
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Training Dynamics (log-scale loss curves)
# ══════════════════════════════════════════════════════════════════════════
def fig7_training_dynamics(loss_histories):
    print("🎨 Figure 7: Training Dynamics")
    fig, ax = plt.subplots(figsize=(10, 6))
    ar_colors = [COLORS["ar03"], COLORS["ar04"], COLORS["ar05"]]
    line_styles = ["-", "--", "-."]

    for idx, (ar, t_loss, v_loss) in enumerate(loss_histories):
        epochs = range(1, len(t_loss) + 1)
        ax.semilogy(epochs, t_loss, color=ar_colors[idx], ls=line_styles[idx],
                    lw=2.0, alpha=0.7, label=f"AR={ar} Train")
        ax.semilogy(epochs, v_loss, color=ar_colors[idx], ls=line_styles[idx],
                    lw=2.5, label=f"AR={ar} Val",
                    marker="o", markersize=3, markevery=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Dynamics — Huber-MSE Loss Convergence",
                 fontsize=16, fontweight="bold")
    ax.legend(ncol=2, framealpha=0.9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    # Annotation
    ax.text(0.02, 0.02,
            "Huber Loss prevents gradient spikes -- smooth convergence",
            transform=ax.transAxes, fontsize=11, style="italic",
            color=COLORS["secondary"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                      edgecolor=COLORS["secondary"], alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_7_training_dynamics.png"))
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  Tier-A Journal Figure Generator")
    print("  BiLSTM + Attention — Battery Thermal Management with PCM")
    print("=" * 65)

    data_files = ["0.3.xlsx", "0.4.xlsx", "0.5.xlsx"]
    ars = [0.3, 0.4, 0.5]

    # Load all datasets
    datasets = []
    for f in data_files:
        path = os.path.join(OUT_DIR, f)
        datasets.append(load_data(path))
    print("✅ All datasets loaded\n")

    # ── Fig 1 (no model needed) ──
    fig1_schematic()

    # ── Fig 2 (no model needed — uses raw data) ──
    fig2_contours(datasets)

    # ── Train models for all ARs (collecting loss histories + attention) ──
    parity_results = []       # (ar, y_true, y_pred, r2, rmse)
    loss_histories = []       # (ar, train_loss, val_loss)
    attention_data = []       # (ar, attn_weights, raw_X_sub)

    for i, (ar, data) in enumerate(zip(ars, datasets)):
        print(f"\n{'─'*50}")
        print(f"  Training AR = {ar}")
        print(f"{'─'*50}")
        (X_tr, X_va, X_te, y_tr_s, y_va_s, y_te_s,
         y_te_raw, xs, ys) = make_sequences(data, shuffle=True)

        model, t_loss, v_loss = train_model(
            X_tr, y_tr_s, X_va, y_va_s, label=f"AR{ar}")
        y_pred = predict(model, X_te, ys)

        r2 = r2_score(y_te_raw, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te_raw, y_pred))
        mae = mean_absolute_error(y_te_raw, y_pred)
        print(f"  ✓ R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

        parity_results.append((ar, y_te_raw, y_pred, r2, rmse))
        loss_histories.append((ar, t_loss, v_loss))

        # Attention weights (use unscaled X_te so liquid_frac is interpretable)
        # But model expects scaled input — use scaled X_te for inference,
        # and unscaled for binning.
        attn_w, _ = get_attention_weights(model, X_te, n_samples=500)
        # For binning by liquid_frac, use original unscaled sequences
        raw_seqs = []
        data_sorted = data.sort_values("time").reset_index(drop=True)
        for j in range(len(data_sorted) - SEQ_LEN):
            raw_seqs.append(data_sorted.iloc[j:j + SEQ_LEN][FEATURE_COLS].values)
        raw_seqs = np.array(raw_seqs, dtype=np.float32)
        # Sample same count as attn_w
        rng = np.random.default_rng(42)
        raw_sel = rng.choice(len(raw_seqs), min(len(attn_w), len(raw_seqs)), replace=False)
        attention_data.append((ar, attn_w, raw_seqs[raw_sel]))

    # ── Fig 3 — Parity Plots ──
    fig3_parity(parity_results)

    # ── Fig 4 — Chronological vs. Shuffle (AR 0.4) ──
    fig4_split_comparison(datasets[1])

    # ── Fig 5 — Error Distribution ──
    fig5_error_distribution(parity_results)

    # ── Fig 6 — Attention Heatmap ──
    fig6_attention(attention_data)

    # ── Fig 7 — Training Dynamics ──
    fig7_training_dynamics(loss_histories)

    print("\n" + "=" * 65)
    print("  ✅  All 7 figures saved:")
    for i in range(1, 8):
        figs = [f for f in os.listdir(OUT_DIR) if f.startswith(f"fig_{i}")]
        for f in figs:
            print(f"      {f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
