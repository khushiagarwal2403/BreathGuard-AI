# ============================================================
# MULTI-GAS SENSOR FUSION: ROBUSTNESS & UNCERTAINTY SIMULATOR
# ------------------------------------------------------------
# A synthetic-data testbed for evaluating multi-sensor gas
# fusion pipelines under noise, drift, and low-quality capture
# conditions.
#
# SCOPE AND HONESTY NOTE (read before reusing numbers anywhere):
# - All gas concentrations, sensor readings, and class labels
#   are SYNTHETIC. Nothing here is measured from real hardware
#   or represents a real clinical/biological signal.
# - This script exists to test ENGINEERING QUESTIONS: does a
#   quality-aware fusion model degrade more gracefully than a
#   naive one under sensor noise and drift? Does an ensemble's
#   uncertainty estimate track breath-capture quality? Is the
#   model's reasoning explainable (SHAP)?
# - The classification label includes injected, non-recoverable
#   noise (see LABEL_NOISE_RATE) specifically so that accuracy/
#   ROC-AUC numbers are not trivially close to 1.0. Earlier
#   versions of this script derived the label as a clean linear
#   function of the same latent gases the sensors measure, which
#   made the task easier than it looked and would not survive
#   scrutiny in an interview. Don't remove LABEL_NOISE_RATE to
#   make the numbers prettier -- that reintroduces the problem.
# - Do not use clinical/diagnostic language when describing this
#   project (it does not diagnose SIBO, IMO, or anything else).
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error,
    r2_score, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP is not installed. Explainability section will be skipped.")

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

N_SAMPLES = 10000
OUTPUT_FOLDER = "results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

GASES = ["H2", "CH4", "H2S", "NH3", "Acetone"]

# Fraction of labels randomly flipped after threshold assignment.
# This simulates irreducible real-world uncertainty (e.g. biological
# variability, measurement error not captured by the sensor noise
# model) and prevents the classifier from achieving unrealistically
# clean separation just by inverting the sensor transform.
LABEL_NOISE_RATE = 0.08

GAS_RANGES = {
    "H2": (0, 100), "CH4": (0, 50), "H2S": (0, 20),
    "NH3": (0, 20), "Acetone": (0, 50),
}

ENV_RANGES = {
    "CO2": (2.0, 5.5), "Flow": (0.2, 1.0),
    "Humidity": (30, 95), "Temperature": (20, 40),
}

CROSS_SENSITIVITY_MATRIX = np.array([
    [1.00, 0.05, 0.02, 0.03, 0.01],
    [0.04, 1.00, 0.03, 0.02, 0.01],
    [0.02, 0.04, 1.00, 0.08, 0.03],
    [0.03, 0.02, 0.06, 1.00, 0.05],
    [0.01, 0.01, 0.03, 0.04, 1.00],
])

# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

def generate_true_gases(n_samples, random_state=42):
    rng = np.random.default_rng(random_state)
    data = {f"True_{gas}": rng.uniform(lo, hi, n_samples)
            for gas, (lo, hi) in GAS_RANGES.items()}
    return pd.DataFrame(data)


def generate_environment(n_samples, random_state=42):
    rng = np.random.default_rng(random_state + 100)
    data = {var: rng.uniform(lo, hi, n_samples)
            for var, (lo, hi) in ENV_RANGES.items()}
    data["SignalStability"] = rng.uniform(0.4, 1.0, n_samples)
    return pd.DataFrame(data)


def simulate_sensor_signals(true_gases, environment, noise_level=0.05,
                             drift_strength=0.10, poor_capture_probability=0.20,
                             random_state=42):
    """Applies cross-sensitivity, humidity/temperature confounds, drift,
    Gaussian noise, and poor-capture events to the latent 'true' gas
    concentrations to produce realistic noisy sensor readings."""
    rng = np.random.default_rng(random_state + 200)
    n_samples = len(true_gases)

    true_matrix = true_gases[[f"True_{g}" for g in GASES]].values
    observed = true_matrix @ CROSS_SENSITIVITY_MATRIX.T

    humidity_factor = 1 + 0.002 * (environment["Humidity"].values - 60)
    observed *= humidity_factor[:, None]

    temperature_factor = 1 + 0.005 * (environment["Temperature"].values - 30)
    observed *= temperature_factor[:, None]

    time = np.linspace(0, 1, n_samples)
    drift_factor = 1 + drift_strength * time
    observed *= drift_factor[:, None]

    noise = rng.normal(0, noise_level, size=observed.shape)
    observed *= (1 + noise)

    poor_capture = rng.random(n_samples) < poor_capture_probability
    capture_factor = np.ones(n_samples)
    capture_factor[poor_capture] = rng.uniform(0.40, 0.85, poor_capture.sum())
    observed *= capture_factor[:, None]

    sensor_df = pd.DataFrame(observed, columns=[f"Sensor_{g}" for g in GASES])
    return sensor_df, poor_capture


def create_synthetic_label(true_gases, random_state=42):
    """
    Builds a binary target from a weighted, thresholded combination of
    the latent gas concentrations, then injects irreducible label noise.

    IMPORTANT: this is a synthetic pipeline-validation label, not a
    clinical phenotype. It exists to test whether the fusion model can
    recover a noisy latent pattern through noisy sensors -- not to
    detect any real biological condition.
    """
    rng = np.random.default_rng(random_state + 300)

    score = (
        0.45 * true_gases["True_H2"]
        + 0.35 * true_gases["True_CH4"]
        + 0.10 * true_gases["True_H2S"]
        + 0.05 * true_gases["True_NH3"]
        + 0.05 * true_gases["True_Acetone"]
    )

    threshold = np.percentile(score, 60)
    label = (score >= threshold).astype(int)

    # Inject irreducible noise: randomly flip a fraction of labels so the
    # task cannot be solved by simply inverting the sensor transform.
    flip_mask = rng.random(len(label)) < LABEL_NOISE_RATE
    label = np.where(flip_mask, 1 - label, label)

    return label


def min_max_score(values, low, high):
    return np.clip((values - low) / (high - low), 0, 1)


def calculate_breath_quality_score(data):
    co2_score = min_max_score(data["CO2"].values, 2.0, 5.0)
    flow_score = min_max_score(data["Flow"].values, 0.3, 0.8)

    humidity_distance = np.abs(data["Humidity"].values - 60)
    humidity_score = 1 - np.clip(humidity_distance / 40, 0, 1)

    temperature_distance = np.abs(data["Temperature"].values - 30)
    temperature_score = 1 - np.clip(temperature_distance / 15, 0, 1)

    stability_score = data["SignalStability"].values

    bqs = (0.35 * co2_score + 0.20 * flow_score + 0.15 * humidity_score
           + 0.10 * temperature_score + 0.20 * stability_score)
    return np.clip(bqs, 0, 1)


# ============================================================
# BUILD DATASET
# ============================================================

print("\nGenerating synthetic dataset...")
true_gases = generate_true_gases(N_SAMPLES)
environment = generate_environment(N_SAMPLES)
sensor_data, poor_capture = simulate_sensor_signals(true_gases, environment)
label = create_synthetic_label(true_gases)

df = pd.concat([true_gases, sensor_data, environment], axis=1)
df["PoorCapture"] = poor_capture.astype(int)
df["TargetClass"] = label
df["BQS"] = calculate_breath_quality_score(df)

print("Dataset shape:", df.shape)
print(df.head())
df.to_csv(f"{OUTPUT_FOLDER}/synthetic_sensor_dataset.csv", index=False)

gas_features = [f"Sensor_{g}" for g in GASES]
quality_features = ["CO2", "Flow", "Humidity", "Temperature", "SignalStability", "BQS"]
all_features = gas_features + quality_features
target = "TargetClass"

indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices, test_size=0.20, random_state=RANDOM_STATE, stratify=df[target]
)

X_train_baseline = df.loc[train_idx, gas_features]
X_test_baseline = df.loc[test_idx, gas_features]
X_train_quality = df.loc[train_idx, all_features]
X_test_quality = df.loc[test_idx, all_features]
y_train = df.loc[train_idx, target]
y_test = df.loc[test_idx, target]

# ============================================================
# MODELS: BASELINE VS QUALITY-AWARE FUSION
# ============================================================

print("\nTraining baseline (gas-only) model...")
baseline_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
])
baseline_model.fit(X_train_baseline, y_train)

print("Training quality-aware fusion model...")
quality_model = RandomForestClassifier(
    n_estimators=500, min_samples_split=5, min_samples_leaf=2,
    random_state=RANDOM_STATE, n_jobs=-1,
)
quality_model.fit(X_train_quality, y_train)


def evaluate_model(model, X_test, y_test, model_name):
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction),
        "Recall": recall_score(y_test, prediction),
        "F1": f1_score(y_test, prediction),
        "ROC_AUC": roc_auc_score(y_test, probability),
    }


baseline_results = evaluate_model(baseline_model, X_test_baseline, y_test, "Conventional Gas Fusion")
quality_results = evaluate_model(quality_model, X_test_quality, y_test, "Quality-Aware Multi-Gas Fusion")
comparison_df = pd.DataFrame([baseline_results, quality_results])
print("\nMODEL COMPARISON")
print(comparison_df)
comparison_df.to_csv(f"{OUTPUT_FOLDER}/model_comparison.csv", index=False)

for model, X, name in [
    (baseline_model, X_test_baseline, "Conventional_Fusion"),
    (quality_model, X_test_quality, "Quality_Aware_Fusion"),
]:
    prediction = model.predict(X)
    cm = confusion_matrix(y_test, prediction)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title(name.replace("_", " "))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/{name}_confusion_matrix.png", dpi=300)
    plt.close()

importance_df = pd.DataFrame({
    "Feature": all_features, "Importance": quality_model.feature_importances_
}).sort_values("Importance", ascending=False)
print("\nFEATURE IMPORTANCE")
print(importance_df)

plt.figure(figsize=(9, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Feature Importance in Quality-Aware Fusion")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/feature_importance.png", dpi=300)
plt.close()

# ============================================================
# ROBUSTNESS EXPERIMENTS
# ============================================================

def add_sensor_noise(data, noise_level, random_state=42):
    rng = np.random.default_rng(random_state)
    noisy_data = data.copy()
    for feature in gas_features:
        feature_std = noisy_data[feature].std()
        noisy_data[feature] += rng.normal(0, noise_level * feature_std, len(noisy_data))
    return noisy_data


def add_sensor_drift(data, drift_strength):
    drifted_data = data.copy()
    drift = np.linspace(1, 1 + drift_strength, len(drifted_data))
    for feature in gas_features:
        drifted_data[feature] = drifted_data[feature] * drift
    return drifted_data


print("\nRunning noise robustness experiment...")
noise_levels = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
noise_results = []
test_data = df.loc[test_idx].copy()

for noise_level in noise_levels:
    noisy_test = add_sensor_noise(test_data, noise_level)
    noisy_test["BQS"] = calculate_breath_quality_score(noisy_test)
    b = evaluate_model(baseline_model, noisy_test[gas_features], y_test, "Baseline")
    q = evaluate_model(quality_model, noisy_test[all_features], y_test, "Quality")
    noise_results.append({
        "NoiseLevel": noise_level,
        "Baseline_AUC": b["ROC_AUC"], "Quality_Aware_AUC": q["ROC_AUC"],
        "Baseline_F1": b["F1"], "Quality_Aware_F1": q["F1"],
    })

noise_df = pd.DataFrame(noise_results)
print(noise_df)
noise_df.to_csv(f"{OUTPUT_FOLDER}/noise_robustness.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(noise_df["NoiseLevel"], noise_df["Baseline_AUC"], marker="o", label="Conventional Fusion")
plt.plot(noise_df["NoiseLevel"], noise_df["Quality_Aware_AUC"], marker="o", label="Quality-Aware Fusion")
plt.xlabel("Noise Level"); plt.ylabel("ROC-AUC")
plt.title("Robustness Under Increasing Sensor Noise")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/noise_robustness.png", dpi=300)
plt.close()

print("\nRunning drift robustness experiment...")
drift_levels = [0.00, 0.05, 0.10, 0.20, 0.30]
drift_results = []

for drift_level in drift_levels:
    drifted_test = add_sensor_drift(test_data, drift_level)
    drifted_test["BQS"] = calculate_breath_quality_score(drifted_test)
    b = evaluate_model(baseline_model, drifted_test[gas_features], y_test, "Baseline")
    q = evaluate_model(quality_model, drifted_test[all_features], y_test, "Quality")
    drift_results.append({
        "DriftLevel": drift_level, "Baseline_AUC": b["ROC_AUC"], "Quality_Aware_AUC": q["ROC_AUC"],
    })

drift_df = pd.DataFrame(drift_results)
drift_df.to_csv(f"{OUTPUT_FOLDER}/drift_robustness.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(drift_df["DriftLevel"], drift_df["Baseline_AUC"], marker="o", label="Conventional Fusion")
plt.plot(drift_df["DriftLevel"], drift_df["Quality_Aware_AUC"], marker="o", label="Quality-Aware Fusion")
plt.xlabel("Drift Strength"); plt.ylabel("ROC-AUC")
plt.title("Robustness Under Sensor Drift")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/drift_robustness.png", dpi=300)
plt.close()

# ============================================================
# ABLATION STUDY
# ============================================================

print("\nRunning ablation study...")
ablation_configurations = {
    "Gas Only": gas_features,
    "Gas + CO2": gas_features + ["CO2"],
    "Gas + CO2 + Flow": gas_features + ["CO2", "Flow"],
    "Gas + Environment": gas_features + ["CO2", "Flow", "Humidity", "Temperature"],
    "Full Quality-Aware": all_features,
}

ablation_results = []
for model_name, features in ablation_configurations.items():
    model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(df.loc[train_idx, features], y_train)
    ablation_results.append(evaluate_model(model, df.loc[test_idx, features], y_test, model_name))

ablation_df = pd.DataFrame(ablation_results)
print(ablation_df)
ablation_df.to_csv(f"{OUTPUT_FOLDER}/ablation_study.csv", index=False)

plt.figure(figsize=(9, 5))
plt.bar(ablation_df["Model"], ablation_df["ROC_AUC"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("ROC-AUC"); plt.title("Ablation Study")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/ablation_study.png", dpi=300)
plt.close()

# ============================================================
# GAS CONCENTRATION REGRESSION (no label-noise issue -- clean task)
# ============================================================

print("\nRunning gas concentration estimation...")
gas_regression_results = []
gas_regressors = {}

for gas in GASES:
    target_gas = f"True_{gas}"
    y_train_gas = df.loc[train_idx, target_gas]
    y_test_gas = df.loc[test_idx, target_gas]

    regressor = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    regressor.fit(X_train_quality, y_train_gas)
    prediction = regressor.predict(X_test_quality)

    gas_regression_results.append({
        "Gas": gas,
        "MAE": mean_absolute_error(y_test_gas, prediction),
        "RMSE": np.sqrt(mean_squared_error(y_test_gas, prediction)),
        "R2": r2_score(y_test_gas, prediction),
    })
    gas_regressors[gas] = regressor

gas_regression_df = pd.DataFrame(gas_regression_results)
print(gas_regression_df)
gas_regression_df.to_csv(f"{OUTPUT_FOLDER}/gas_regression_results.csv", index=False)

# ============================================================
# UNCERTAINTY ESTIMATION (bootstrap ensemble)
# ============================================================

print("\nTraining uncertainty ensemble...")
N_ENSEMBLE = 10
ensemble_models = []
rng = np.random.default_rng(RANDOM_STATE)

for i in range(N_ENSEMBLE):
    bootstrap_indices = rng.choice(train_idx, size=len(train_idx), replace=True)
    model = RandomForestClassifier(n_estimators=200, random_state=i, n_jobs=-1)
    model.fit(df.loc[bootstrap_indices, all_features], df.loc[bootstrap_indices, target])
    ensemble_models.append(model)

ensemble_probabilities = np.array([
    model.predict_proba(X_test_quality)[:, 1] for model in ensemble_models
])
mean_probability = ensemble_probabilities.mean(axis=0)
uncertainty = ensemble_probabilities.std(axis=0)

uncertainty_df = pd.DataFrame({
    "PredictionProbability": mean_probability,
    "Uncertainty": uncertainty,
    "TrueLabel": y_test.values,
    "BQS": X_test_quality["BQS"].values,
})
uncertainty_df.to_csv(f"{OUTPUT_FOLDER}/uncertainty_results.csv", index=False)

plt.figure(figsize=(8, 5))
plt.scatter(uncertainty_df["BQS"], uncertainty_df["Uncertainty"], alpha=0.4)
plt.xlabel("Breath Quality Score"); plt.ylabel("Prediction Uncertainty")
plt.title("Prediction Uncertainty vs Sample Quality")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/uncertainty_vs_bqs.png", dpi=300)
plt.close()

# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

if SHAP_AVAILABLE:
    print("\nRunning SHAP explainability...")
    shap_sample = X_test_quality.sample(min(500, len(X_test_quality)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(quality_model)
    shap_values = explainer.shap_values(shap_sample)

    plt.figure()
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], shap_sample, show=False)
    else:
        shap.summary_plot(shap_values, shap_sample, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# SAVE MODELS
# ============================================================

joblib.dump(baseline_model, f"{OUTPUT_FOLDER}/baseline_model.pkl")
joblib.dump(quality_model, f"{OUTPUT_FOLDER}/quality_aware_model.pkl")
joblib.dump(gas_regressors, f"{OUTPUT_FOLDER}/gas_regressors.pkl")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SENSOR FUSION ROBUSTNESS & UNCERTAINTY SIMULATION COMPLETE")
print("=" * 70)
print("\n1. MODEL COMPARISON\n", comparison_df)
print("\n2. NOISE ROBUSTNESS\n", noise_df)
print("\n3. DRIFT ROBUSTNESS\n", drift_df)
print("\n4. ABLATION STUDY\n", ablation_df)
print("\n5. GAS CONCENTRATION ESTIMATION\n", gas_regression_df)
print(f"\nAll results saved in: {OUTPUT_FOLDER}")
print("=" * 70)
