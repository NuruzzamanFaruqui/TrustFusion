# =========================
# 📦 Imports
# =========================
import os
import numpy as np
import pandas as pd
import joblib
import shap
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

# =========================
# 📁 Define Output Directory
# =========================
output_dir = "/content/drive/MyDrive/TrustFusion/Comparison"

# =========================
# 📂 Load and Prepare Data
# =========================
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
# 🧪 Evasion Attack Function
# =========================
def run_evasion_attack(model_name, model, X_test, y_test, scaled=False, scaler=None):
    print(f"\n🔍 Evasion Attack on {model_name}")

    # 1. Select low-trust users (Tier 0 or 1)
    mask = y_test <= 1
    X_attack = X_test[mask].copy()
    y_true = y_test[mask].copy()

    # 2. Modify features (simulate evasion)
    X_evasion = X_attack.copy()

    # Behavioral spoofing
    if 'peer_rating' in X_evasion.columns:
        X_evasion['peer_rating'] = np.clip(X_evasion['peer_rating'] + 1.0, 0, 5)
    if 'violation_count' in X_evasion.columns:
        X_evasion['violation_count'] = 0
    if 'complaint_count' in X_evasion.columns:
        X_evasion['complaint_count'] = 0

    # Contextual spoofing: copy from trusted user
    trusted_context = X_train[y_train == 3].sample(1, random_state=42)
    for col in ['location', 'term', 'hour_sin', 'hour_cos']:
        if col in X_evasion.columns:
            X_evasion[col] = trusted_context[col].values[0]

    # 3. Predict before/after evasion
    if scaled:
        X_attack_scaled = scaler.transform(X_attack)
        X_evasion_scaled = scaler.transform(X_evasion)
        original_preds = np.argmax(model.predict(X_attack_scaled), axis=1)
        evasion_preds = np.argmax(model.predict(X_evasion_scaled), axis=1)
    else:
        original_preds = model.predict(X_attack)
        evasion_preds = model.predict(X_evasion)

    # 4. Report results
    upgrades = (evasion_preds > original_preds)
    upgrade_rate = upgrades.sum() / len(upgrades)
    print(f"🟢 Upgrade Rate: {upgrade_rate:.2%}")
    print("\n📊 Classification Report (Before Evasion):")
    print(classification_report(y_true, original_preds))
    print("\n📊 Classification Report (After Evasion):")
    print(classification_report(y_true, evasion_preds))

# =========================
# 🔁 Load and Evaluate Each Model
# =========================

# LightGBM
lgb_model = joblib.load(f"{output_dir}/lightgbm_model.pkl")
run_evasion_attack("LightGBM", lgb_model, X_test, y_test)

# XGBoost
xgb_model = joblib.load(f"{output_dir}/xgboost_model.pkl")
run_evasion_attack("XGBoost", xgb_model, X_test, y_test)

# CatBoost
cat_model = cb.CatBoostClassifier()
cat_model.load_model(f"{output_dir}/catboost_model.cbm")
run_evasion_attack("CatBoost", cat_model, X_test, y_test)

# DNN
dnn_model = load_model(f"{output_dir}/dnn_model.h5")
scaler = joblib.load(f"{output_dir}/dnn_scaler.pkl")
run_evasion_attack("DNN", dnn_model, X_test, y_test, scaled=True, scaler=scaler)
