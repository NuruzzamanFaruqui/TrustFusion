# =========================
# 📦 Install Dependencies
# =========================
!pip install lightgbm xgboost catboost shap tensorflow scikit-learn pandas matplotlib seaborn psutil --quiet

# =========================
# 📥 Imports
# =========================
import os
import time
import threading
import psutil
import tracemalloc
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from IPython.display import display

# =========================
# 📂 Load Data
# =========================
output_dir = "/content/drive/MyDrive/TrustFusion/AllModels_with_ResourceTracking"
os.makedirs(output_dir, exist_ok=True)

original_df = pd.read_csv('/content/drive/MyDrive/TrustFusion/preprocessed/preprocessed-dataset.csv')
rl_df = pd.read_csv('/content/drive/MyDrive/TrustFusion/LightGBM-RL/rl-generated-data.csv')

target_col = 'user_trust_tier'
y_original = original_df[target_col].astype(int)
X_original = original_df.drop(columns=[target_col])
y_rl = y_original.sample(n=len(rl_df), random_state=42).reset_index(drop=True)

X_combined = pd.concat([X_original, rl_df], ignore_index=True)
y_combined = pd.concat([y_original, y_rl], ignore_index=True)
weights = np.concatenate([np.ones(len(X_original)), np.full(len(rl_df), 3.0)])

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_combined, y_combined, weights, test_size=0.2, stratify=y_combined, random_state=42
)

# =========================
# 🧠 Feature Groups for SHAP
# =========================
behavioral_feats = ['usage_frequency', 'peer_rating', 'violation_count', 'complaint_count']
contextual_feats = ['location', 'term', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
profile_feats = ['account_age_days', 'total_interactions', 'admin_flag']
modality_groups = {
    "Behavioral": behavioral_feats,
    "Contextual": contextual_feats,
    "Profile Priors": profile_feats,
    "Others": list(set(X_combined.columns) - set(behavioral_feats + contextual_feats + profile_feats))
}

# =========================
# 🤖 Define Models
# =========================
models = {
    "LightGBM": lgb.LGBMClassifier(objective='multiclass', num_class=4, learning_rate=0.05, n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(objective='multi:softprob', num_class=4, learning_rate=0.05, n_estimators=100, eval_metric='mlogloss', use_label_encoder=False, random_state=42),
    "CatBoost": cb.CatBoostClassifier(loss_function='MultiClass', learning_rate=0.05, iterations=100, verbose=0, random_seed=42),
    "DeepNN": "keras_model"
}

results = []
shap_attributions = []

# =========================
# ⚙️ Train, Explain, Track Resources
# =========================
def evaluate_and_explain(model, name):
    print(f"\n🔧 Training: {name}")
    cpu_usage = []

    def track_cpu():
        while tracking:
            usage = psutil.cpu_percent(interval=0.5)
            cpu_usage.append(usage)

    tracking = True
    tracemalloc.start()
    cpu_thread = threading.Thread(target=track_cpu)
    cpu_thread.start()
    start_time = time.time()

    # ========== Training ==========
    if name == "DeepNN":
        y_train_cat = to_categorical(y_train, num_classes=4)
        y_test_cat = to_categorical(y_test, num_classes=4)

        model = Sequential([
            Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(4, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train_cat, sample_weight=w_train, epochs=10, batch_size=64, verbose=1, validation_split=0.1)
        train_time = time.time() - start_time
        y_pred = np.argmax(model.predict(X_test), axis=1)
    else:
        model.fit(X_train, y_train, sample_weight=w_train)
        train_time = time.time() - start_time
        y_pred = model.predict(X_test)

    # ========== SHAP Explanation ==========
    shap_start = time.time()
    if name == "DeepNN":
        background = X_train.sample(n=100, random_state=42).values
        explainer = shap.DeepExplainer(model, background)
        sample = X_test.sample(n=100, random_state=42).values
        shap_vals = explainer.shap_values(sample)
        mean_shap = np.abs(np.stack(shap_vals)).mean(axis=(0, 1))
    else:
        explainer = shap.TreeExplainer(model)
        sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
        shap_vals = explainer.shap_values(sample)
        mean_shap = np.abs(np.stack(shap_vals)).mean(axis=(0, 1))
    shap_time = time.time() - shap_start

    # ========== Stop Tracking ==========
    tracking = False
    cpu_thread.join()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_cpu = round(np.mean(cpu_usage), 2)
    peak_ram = round(peak / 1e6, 2)

    # ========== Metrics ==========
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    results.append([
        name, acc, prec, rec, f1,
        round(train_time, 2), round(shap_time, 2),
        peak_ram, avg_cpu
    ])

    shap_dict = {f: float(np.mean(v)) for f, v in zip(X_test.columns.tolist(), mean_shap)}
    total = sum(shap_dict.values()) or 1.0
    shap_by_modality = {
        group: round(100 * sum(shap_dict.get(f, 0.0) for f in feats) / total, 2)
        for group, feats in modality_groups.items()
    }
    shap_by_modality["Model"] = name
    shap_attributions.append(shap_by_modality)

# =========================
# 🚀 Run All Models
# =========================
for name, model in models.items():
    if name == "DeepNN":
        model = None  # Placeholder
    evaluate_and_explain(model, name)

# =========================
# 💾 Save Results
# =========================
df_metrics = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1-Score",
    "Train Time (s)", "SHAP Time (s)", "Peak RAM (MB)", "Avg CPU (%)"
])
df_shap = pd.DataFrame(shap_attributions).set_index("Model")

df_metrics.to_csv(f"{output_dir}/metrics.csv", index=False)
df_shap.to_csv(f"{output_dir}/shap_attribution.csv")

print("\n📊 Model Performance:")
display(df_metrics)

print("\n🔍 SHAP Attribution by Modality:")
display(df_shap)
