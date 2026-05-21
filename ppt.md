---
marp: true
theme: default
paginate: true
backgroundColor: #f8f9fa
color: #1a1a2e
size: 16:9
---

# PCM-Enhanced Battery Thermal Management
## A Hybrid SOTA Ensemble Approach

**System**: Conjugate Heat Transfer (CHT) Hybrid Ensemble Model
**Target**: Panasonic NCR18650B NCA EV Cell (3.7V)
**Cooling**: RT-45 Nano-enhanced PCM (NePCM)

---

# The Physical System & Challenges

**The Heat Transfer Pathway:**
Battery Cell ➔ Heat Generation ➔ PCM Enclosure ➔ Environment

**PCM Phase-Transition Regimes:**
- **Solid Phase ($\alpha = 0$)**: Solid conduction dominant; battery temp rises rapidly.
- **Mushy/Melting ($0 < \alpha < 1$)**: Latent heat absorption; phase-transition buffers battery heating.
- **Liquid Phase ($\alpha = 1$)**: Natural convection; no latent reserve; temp rise accelerates.

**The Core Challenge:** 
Capturing non-linear thresholding (tree models) AND long-term temporal dependencies (deep learning) across shifting phase regimes.

---

# Physics-Informed Feature Engineering

Embedding thermodynamics directly into the data pipeline:

- **Temperature Driving Force**: $\Delta T = T_{battery} - T_{pcm}$
- **Heat Flux Proxy**: $q_{proxy} = Nu \times \Delta T$ *(Surrogate for convective transfer)*
- **PCM Melting Rate**: $\dot{\alpha} = \frac{d\alpha}{dt}$ *(Latent heat absorption rate)*
- **Cumulative Heat Proxy**: $Q_{cum}(t) = \sum Nu(\tau)\Delta T(\tau)$ *(PCM charging progress)*
- **Thermal Resistance Proxy**: $R_{th} = \frac{\Delta T}{|Nu|+\varepsilon}$ *(Identifies thermal bottlenecks)*
- **Nonlinear Interactions**: $Nu \times \alpha$, $T_{pcm} \times \alpha$ *(Convection conditioned on melt state)*

---

# Hybrid AI Architecture

```text
[Physics Features] ➔ Feature Engineering ➔ Leak-Free Split
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
  XGBoost Branch               Stateful LSTM + Attention
  (Tabular/Lag Features)       (3D Tensors [batch, 300, feat])
  - High-freq non-linear       - Long-term temporal propagation
  - Instant interactions       - Multi-Head Attention (4 heads)
         │                               │
         └───────────────┬───────────────┘
                         ▼
              Ridge Meta-Learner ($\alpha=1.0$)
                         │
                         ▼
           Future Battery Temperature ($T_{battery}$)
```

---

# Branch 1: XGBoost Thermal Regression

**Role**: Strong baseline predictor capturing instantaneous, non-linear interactions.

- **Input**: Extended lag/rolling variants (up to 50 timesteps) + Physics features
- **Key Hyperparameters**: 
  - `n_estimators` = 600
  - `max_depth` = 7
  - `learning_rate` = 0.04
  - `subsample` = 0.8
- **Objective**: `reg:squarederror`
- **Advantage**: Highly robust on structured physics features, low computational cost, excellent at thermodynamic interpolation.

---

# Branch 2: Stateful LSTM + Attention

**Role**: Mapping long-term temporal propagation and thermal inertia.

- **Core Engine**: 3 Stacked Stateful LSTM layers (256 units, `tanh` activation)
- **Attention Mechanism**: Multi-Head Attention (4 heads, key dim 64)
  - Differentially weights past critical thermal events (e.g., phase transition boundaries)
  - Improves hotspot recognition and transient stability
- **Dense Head**: Layer Normalization ➔ Dense [128, 64]
- **Sequence Length**: 300 timesteps

---

# Robust Optimization & Loss Function

**Training Objective: Huber Loss**

$$\mathcal{L}_{\delta}(e)=\begin{cases}\frac{1}{2}e^2, & |e|\leq\delta\\\delta(|e|-\frac{\delta}{2}), & |e|>\delta\end{cases}$$

**Why Huber Loss?**
- Robust to thermal spikes and noise
- Stable near phase transitions
- Less sensitive to outliers than MSE

**Training Strategy:**
- Gradient clipping to prevent unstable recurrent optimization
- Chronological train/test splits with early stopping

---

# Strict Data Leakage Prevention

A fundamental pillar ensuring true out-of-sample predictive robustness:

1. 🚫 **Target Isolation**: $T_{battery}(t)$ is **never** an input. Strictly uses $T_{battery}(t-1)$.
2. 📏 **Train-Only Scaling**: `MinMaxScaler` fitted exclusively on training data; statically applied to val/test.
3. ⏱️ **Temporal Splitting**: Validation carved sequentially from the *end* of the training window (no shuffle).
4. 🧮 **Causal Cumulatives**: Cumulative sums (`cumul_heat`) calculated *after* temporal splits to prevent future data interpolation.

---

# System Strengths

- **Thread-Safe Parallelism**: Processes multiple geometrical aspect ratios concurrently using `ThreadPoolExecutor` without Keras graph fragmentation.
- **Hardware-Agnostic Scaling**: Memory footprints carefully tuned (`MAX_PARALLEL_AR = 2`) to prevent RAM overflow.
- **Hybrid Efficacy**: Successfully bridges gradient tree boosting (tabular) and deep recurrent networks (temporal).
- **Comprehensive Diagnostics**: Built-in Pearson correlations, error distributions, feature importance, and LaTeX table synthesis.

---

# Current Limitations

- **Static CFD Dependency**: Relies on upstream Nusselt/Liquid Fraction data. Confidence degrades when extrapolating heavily (e.g., $Re \gg 1000$).
- **Fixed Temporal Resolution**: Stateful LSTM is sensitive to sampling frequency; irregular datasets require aggressive interpolation.
- **Cold Start Penalty**: The 300-step sequence requirement means the first 300 timesteps of inference operate with limited historical memory.

---

# Future Horizons: Toward Thermodynamic AI

Bridging the remaining scientific gaps:

1. **Physics-Informed Neural Networks (PINNs)**: Penalize violations of Navier-Stokes or Energy Conservation PDEs in the loss function.
2. **Electrochemical-Thermal Coupling**: Integrate real-time SoC degradation and internal resistance ($R_{int}$) estimators.
3. **Adaptive Sequence Lengths**: Implement Transformer-XL or TCNs to remove the rigid 300-step constraint.
4. **Multi-Chemistry Generalization**: Use Transfer Learning to map existing NCA model weights to LFP, NMC, or LCO thermal profiles.

---

# Thank You
## Questions?