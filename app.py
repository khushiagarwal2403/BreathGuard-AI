"""
Multi-Gas Sensor Fusion: Robustness & Uncertainty Simulator — Interactive Demo
--------------------------------------------------------------------------------
Streamlit wrapper around the core simulation/modeling pipeline. Lets a visitor
move noise/drift sliders and watch robustness curves, confusion matrices, and
uncertainty behavior update live.

SCOPE NOTE: all data is synthetic. This is an engineering methodology demo
(robustness, uncertainty, explainability), not a diagnostic tool.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

st.set_page_config(page_title="Multi-Gas Sensor Fusion Simulator", layout="wide")

RANDOM_STATE = 42
N_SAMPLES = 10000
GASES = ["H2", "CH4", "H2S", "NH3", "Acetone"]
LABEL_NOISE_RATE = 0.08

GAS_RANGES = {"H2": (0, 100), "CH4": (0, 50), "H2S": (0, 20), "NH3": (0, 20), "Acetone": (0, 50)}
ENV_RANGES = {"CO2": (2.0, 5.5), "Flow": (0.2, 1.0), "Humidity": (30, 95), "Temperature": (20, 40)}

CROSS_SENSITIVITY_MATRIX = np.array([
    [1.00, 0.05, 0.02, 0.03, 0.01],
    [0.04, 1.00, 0.03, 0.02, 0.01],
    [0.02, 0.04, 1.00, 0.08, 0.03],
    [0.03, 0.02, 0.06, 1.00, 0.05],
    [0.01, 0.01, 0.03, 0.04, 1.00],
])

GAS_FEATURES = [f"Sensor_{g}" for g in GASES]
QUALITY_FEATURES = ["CO2", "Flow", "Humidity", "Temperature", "SignalStability", "BQS"]
ALL_FEATURES = GAS_FEATURES + QUALITY_FEATURES


# ---------------------------------------------------------------------------
# Core simulation functions (same logic as the standalone script)
# ---------------------------------------------------------------------------

def generate_true_gases(n, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"True_{g}": rng.uniform(lo, hi, n) for g, (lo, hi) in GAS_RANGES.items()})


def generate_environment(n, seed=42):
    rng = np.random.default_rng(seed + 100)
    data = {v: rng.uniform(lo, hi, n) for v, (lo, hi) in ENV_RANGES.items()}
    data["SignalStability"] = rng.uniform(0.4, 1.0, n)
    return pd.DataFrame(data)


def simulate_sensors(true_gases, environment, noise_level, drift_strength,
                      poor_capture_prob=0.20, seed=42):
    rng = np.random.default_rng(seed + 200)
    n = len(true_gases)
    true_matrix = true_gases[[f"True_{g}" for g in GASES]].values
    observed = true_matrix @ CROSS_SENSITIVITY_MATRIX.T

    observed *= (1 + 0.002 * (environment["Humidity"].values - 60))[:, None]
    observed *= (1 + 0.005 * (environment["Temperature"].values - 30))[:, None]

    t = np.linspace(0, 1, n)
    observed *= (1 + drift_strength * t)[:, None]
    observed *= (1 + rng.normal(0, noise_level, size=observed.shape))

    poor = rng.random(n) < poor_capture_prob
    capture_factor = np.ones(n)
    capture_factor[poor] = rng.uniform(0.40, 0.85, poor.sum())
    observed *= capture_factor[:, None]

    return pd.DataFrame(observed, columns=GAS_FEATURES), poor


def create_label(true_gases, seed=42):
    rng = np.random.default_rng(seed + 300)
    score = (0.45 * true_gases["True_H2"] + 0.35 * true_gases["True_CH4"]
              + 0.10 * true_gases["True_H2S"] + 0.05 * true_gases["True_NH3"]
              + 0.05 * true_gases["True_Acetone"])
    threshold = np.percentile(score, 60)
    label = (score >= threshold).astype(int)
    flip = rng.random(len(label)) < LABEL_NOISE_RATE
    return np.where(flip, 1 - label, label)


def clip01(x):
    return np.clip(x, 0, 1)


def breath_quality_score(data):
    co2 = clip01((data["CO2"].values - 2.0) / 3.0)
    flow = clip01((data["Flow"].values - 0.3) / 0.5)
    humidity = 1 - clip01(np.abs(data["Humidity"].values - 60) / 40)
    temperature = 1 - clip01(np.abs(data["Temperature"].values - 30) / 15)
    stability = data["SignalStability"].values
    return clip01(0.35 * co2 + 0.20 * flow + 0.15 * humidity + 0.10 * temperature + 0.20 * stability)


@st.cache_resource(show_spinner="Generating base dataset and training models (first load only)...")
def build_base_pipeline():
    true_gases = generate_true_gases(N_SAMPLES)
    environment = generate_environment(N_SAMPLES)
    sensors, poor = simulate_sensors(true_gases, environment, noise_level=0.05, drift_strength=0.10)
    label = create_label(true_gases)

    df = pd.concat([true_gases, sensors, environment], axis=1)
    df["PoorCapture"] = poor.astype(int)
    df["TargetClass"] = label
    df["BQS"] = breath_quality_score(df)

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE,
                                            stratify=df["TargetClass"])

    baseline = Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))])
    baseline.fit(df.loc[train_idx, GAS_FEATURES], df.loc[train_idx, "TargetClass"])

    quality = RandomForestClassifier(n_estimators=500, min_samples_split=5, min_samples_leaf=2,
                                      random_state=RANDOM_STATE, n_jobs=-1)
    quality.fit(df.loc[train_idx, ALL_FEATURES], df.loc[train_idx, "TargetClass"])

    return df, train_idx, test_idx, baseline, quality


def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred), "F1": f1_score(y, pred),
        "ROC-AUC": roc_auc_score(y, proba),
    }, pred, proba


def apply_noise_drift(test_df, noise_level, drift_strength, seed=777):
    rng = np.random.default_rng(seed)
    out = test_df.copy()
    for f in GAS_FEATURES:
        std = out[f].std()
        out[f] = out[f] * (1 + drift_strength) + rng.normal(0, noise_level * std, len(out))
    out["BQS"] = breath_quality_score(out)
    return out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🧪 Multi-Gas Sensor Fusion: Robustness & Uncertainty Simulator")
st.caption(
    "Interactive demo — all data is **synthetic**. This is an engineering methodology study "
    "of sensor-fusion robustness, uncertainty, and explainability, not a diagnostic tool."
)

df, train_idx, test_idx, baseline_model, quality_model = build_base_pipeline()
X_test_base = df.loc[test_idx, GAS_FEATURES]
X_test_quality = df.loc[test_idx, ALL_FEATURES]
y_test = df.loc[test_idx, "TargetClass"]

tab1, tab2, tab3, tab4 = st.tabs([
    "⚖️ Live Robustness Comparison", "📊 Ablation Study", "🎯 Uncertainty", "ℹ️ About This Project"
])

# --- TAB 1: Live noise/drift sliders ---
with tab1:
    st.subheader("Drag the sliders to degrade the sensors and watch both models respond")
    col_a, col_b = st.columns(2)
    with col_a:
        noise_level = st.slider("Sensor noise level", 0.0, 0.30, 0.0, 0.01)
    with col_b:
        drift_strength = st.slider("Sensor drift strength", 0.0, 0.30, 0.0, 0.01)

    degraded_test = apply_noise_drift(df.loc[test_idx], noise_level, drift_strength)

    base_metrics, base_pred, base_proba = evaluate(baseline_model, degraded_test[GAS_FEATURES], y_test)
    qual_metrics, qual_pred, qual_proba = evaluate(quality_model, degraded_test[ALL_FEATURES], y_test)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Conventional Gas-Only Fusion**")
        st.metric("ROC-AUC", f"{base_metrics['ROC-AUC']:.4f}")
        st.write(pd.DataFrame([base_metrics]).T.rename(columns={0: "Value"}))
    with m2:
        st.markdown("**Quality-Aware Fusion**")
        delta = qual_metrics["ROC-AUC"] - base_metrics["ROC-AUC"]
        st.metric("ROC-AUC", f"{qual_metrics['ROC-AUC']:.4f}", delta=f"{delta:+.4f} vs. baseline")
        st.write(pd.DataFrame([qual_metrics]).T.rename(columns={0: "Value"}))

    if qual_metrics["ROC-AUC"] < base_metrics["ROC-AUC"]:
        st.warning(
            "At this degradation level, the quality-aware model's advantage has disappeared — "
            "this is the key finding of the project: quality-aware fusion helps at moderate "
            "degradation but is not a substitute for a hardware fix once degradation is severe."
        )
    elif delta < 0.005:
        st.info("The two models are converging — the quality-aware advantage is shrinking as degradation increases.")
    else:
        st.success("Quality-aware fusion is maintaining a clear advantage at this degradation level.")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, base_pred)).plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Conventional Fusion")
    ConfusionMatrixDisplay(confusion_matrix(y_test, qual_pred)).plot(ax=axes[1], colorbar=False)
    axes[1].set_title("Quality-Aware Fusion")
    plt.tight_layout()
    st.pyplot(fig)

# --- TAB 2: Ablation ---
with tab2:
    st.subheader("Which quality features actually help?")
    st.caption("Precomputed on clean (non-degraded) data, same held-out test set throughout.")

    @st.cache_data(show_spinner="Running ablation study...")
    def run_ablation(_df, _train_idx, _test_idx):
        configs = {
            "Gas Only": GAS_FEATURES,
            "Gas + CO2": GAS_FEATURES + ["CO2"],
            "Gas + CO2 + Flow": GAS_FEATURES + ["CO2", "Flow"],
            "Gas + Environment": GAS_FEATURES + ["CO2", "Flow", "Humidity", "Temperature"],
            "Full Quality-Aware": ALL_FEATURES,
        }
        rows = []
        for name, feats in configs.items():
            m = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
            m.fit(_df.loc[_train_idx, feats], _df.loc[_train_idx, "TargetClass"])
            metrics, _, _ = evaluate(m, _df.loc[_test_idx, feats], _df.loc[_test_idx, "TargetClass"])
            rows.append({"Feature Set": name, **metrics})
        return pd.DataFrame(rows)

    ablation_df = run_ablation(df, train_idx, test_idx)
    st.dataframe(ablation_df.style.format({c: "{:.4f}" for c in ablation_df.columns if c != "Feature Set"}),
                 use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.bar(ablation_df["Feature Set"], ablation_df["ROC-AUC"])
    ax2.set_ylabel("ROC-AUC")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    st.pyplot(fig2)

    st.info(
        "**Insight:** CO2 and flow alone barely move the needle. The improvement only shows up "
        "once humidity and temperature (full environmental context) are included — that's what "
        "corrects for the sensor-biasing confounds, not the derived Breath Quality Score alone."
    )

# --- TAB 3: Uncertainty ---
with tab3:
    st.subheader("Does the model know when to distrust itself?")

    @st.cache_data(show_spinner="Training bootstrap ensemble...")
    def run_uncertainty(_df, _train_idx, _test_idx):
        rng = np.random.default_rng(RANDOM_STATE)
        probs = []
        for i in range(10):
            boot_idx = rng.choice(_train_idx, size=len(_train_idx), replace=True)
            m = RandomForestClassifier(n_estimators=200, random_state=i, n_jobs=-1)
            m.fit(_df.loc[boot_idx, ALL_FEATURES], _df.loc[boot_idx, "TargetClass"])
            probs.append(m.predict_proba(_df.loc[_test_idx, ALL_FEATURES])[:, 1])
        probs = np.array(probs)
        mean_p = probs.mean(axis=0)
        unc = probs.std(axis=0)
        return pd.DataFrame({
            "Uncertainty": unc, "BQS": _df.loc[_test_idx, "BQS"].values,
            "DistanceFromBoundary": np.abs(mean_p - 0.5),
        })

    unc_df = run_uncertainty(df, train_idx, test_idx)
    corr_bqs = unc_df["Uncertainty"].corr(unc_df["BQS"])
    corr_boundary = unc_df["Uncertainty"].corr(unc_df["DistanceFromBoundary"])

    c1, c2 = st.columns(2)
    with c1:
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.scatter(unc_df["BQS"], unc_df["Uncertainty"], alpha=0.3, s=10)
        ax3.set_xlabel("Breath Quality Score"); ax3.set_ylabel("Ensemble Uncertainty")
        ax3.set_title(f"vs. Sample Quality  (r = {corr_bqs:.3f})")
        st.pyplot(fig3)
    with c2:
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        ax4.scatter(unc_df["DistanceFromBoundary"], unc_df["Uncertainty"], alpha=0.3, s=10, color="darkorange")
        ax4.set_xlabel("Distance from decision boundary"); ax4.set_ylabel("Ensemble Uncertainty")
        ax4.set_title(f"vs. Decision Confidence  (r = {corr_boundary:.3f})")
        st.pyplot(fig4)

    st.warning(
        f"**Honest finding:** uncertainty barely correlates with input sample quality "
        f"(r = {corr_bqs:.3f}), but correlates strongly with decision-boundary distance "
        f"(r = {corr_boundary:.3f}). This ensemble measures *classification-boundary confidence*, "
        f"not *input-quality confidence* — a real distinction, reported here rather than glossed over."
    )

# --- TAB 4: About ---
with tab4:
    st.markdown("""
### What this demo shows
A synthetic simulation testbed for multi-gas sensor fusion: does giving a model access to
environmental/quality metadata (humidity, flow, a composite Breath Quality Score) make it more
robust to sensor noise and drift than a model using raw gas readings alone?

### Key finding
Quality-aware fusion wins under clean-to-moderate conditions (ROC-AUC 0.885 vs. 0.874 baseline),
but that advantage **shrinks and disappears under severe degradation** — move the sliders in the
first tab to see this happen live. That's the central, defensible result of this project: software
fusion helps up to a point; beyond that, you need a hardware fix.

### Why the label has injected noise
An earlier version of the classification label was a near-deterministic function of the same
latent values the sensors measure — which would have made the task artificially easy. This
version injects 8% irreducible label noise so the reported metrics are honest.

### Scope
All data is synthetic. No real sensor hardware or breath/gas samples were used. This is a
methodology and robustness study, not a diagnostic system.

**Full write-up, code, and report:** see the linked GitHub repository.
""")
