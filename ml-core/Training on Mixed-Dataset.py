# ✅ Install required packages
!pip install lightgbm shap scikit-learn pandas matplotlib seaborn --quiet

# 📥 Imports
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# 📁 Output Directory
output_dir = '/content/drive/MyDrive/TrustFusion/LightGBM-Mixed'
os.makedirs(output_dir, exist_ok=True)

# 📂 Load datasets
original_df = pd.read_csv('/content/drive/MyDrive/TrustFusion/preprocessed/preprocessed-dataset.csv')
rl_df = pd.read_csv('/content/drive/MyDrive/TrustFusion/LightGBM-RL/rl-generated-data.csv')

# 🎯 Extract targets and features
target_col = 'user_trust_tier'
y_original = original_df[target_col].astype(int)
X_original = original_df.drop(columns=[target_col])

# 🧪 Sample pseudo-labels for RL-enhanced data
y_rl = y_original.sample(n=len(rl_df), random_state=42).reset_index(drop=True)

# ➕ Combine datasets
X_combined = pd.concat([X_original, rl_df], ignore_index=True)
y_combined = pd.concat([y_original, y_rl], ignore_index=True)

# 💾 Save Combined Mixed Dataset (Features + Labels)
mixed_df = X_combined.copy()
mixed_df[target_col] = y_combined
mixed_df.to_csv(os.path.join(output_dir, 'mixed-dataset.csv'), index=False)
print(f"✅ Mixed dataset saved to: {os.path.join(output_dir, 'mixed-dataset.csv')}")


# 🔢 Assign weights: higher for RL-enhanced samples
weights = np.concatenate([np.ones(len(X_original)), np.full(len(rl_df), 3.0)])

# 🔀 Train-test split
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_combined, y_combined, weights, test_size=0.2, stratify=y_combined, random_state=42
)

# 🚀 LightGBM Training
params = {
    'objective': 'multiclass',
    'num_class': len(np.unique(y_combined)),
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
    'random_state': 42
}
evals_result = {}
train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
test_data = lgb.Dataset(X_test, label=y_test, weight=w_test, reference=train_data)

model = lgb.train(params, train_data, valid_sets=[train_data, test_data],
                  valid_names=['train', 'valid'], num_boost_round=100,
                  callbacks=[lgb.record_evaluation(evals_result), lgb.log_evaluation(period=10)])

# 💾 Save Model
model_path = os.path.join(output_dir, 'lightgbm_mixed_model.txt')
model.save_model(model_path)

# 📈 Save & Plot Learning Curve
learning_curve_df = pd.DataFrame({
    'Iteration': list(range(1, len(evals_result['train']['multi_logloss']) + 1)),
    'Train_Loss': evals_result['train']['multi_logloss'],
    'Validation_Loss': evals_result['valid']['multi_logloss']
})
learning_curve_df.to_csv(os.path.join(output_dir, 'learning-curve.csv'), index=False)

plt.figure(figsize=(8, 5))
plt.plot(learning_curve_df['Iteration'], learning_curve_df['Train_Loss'], label='Train Loss')
plt.plot(learning_curve_df['Iteration'], learning_curve_df['Validation_Loss'], label='Validation Loss')
plt.xlabel('Boosting Iterations')
plt.ylabel('Multi-class Log Loss')
plt.title('Learning Curve (Mixed Model)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'learning-curve.pdf'))
plt.show()

# 📊 Evaluation
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\n📊 Final Mixed Model Performance:\nAccuracy : {acc:.4f}\nPrecision: {prec:.4f}\nRecall   : {rec:.4f}\nF1-Score : {f1:.4f}")

# 🧾 Save Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, 'classification-report.csv'))
with open(os.path.join(output_dir, 'classification-report.json'), 'w') as f:
    json.dump(report, f, indent=4)
display(pd.DataFrame(report).transpose())

# 🔢 Save Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm)
cm_df.to_csv(os.path.join(output_dir, 'confusion-matrix.csv'), index=False)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Mixed Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion-matrix.pdf'))
plt.show()

# 🔍 SHAP Attribution by Modality
behavioral_feats = ['usage_frequency', 'peer_rating', 'violation_count', 'complaint_count']
contextual_feats = ['location', 'term', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
profile_feats = ['account_age_days', 'total_interactions', 'admin_flag']
modality_groups = {
    "Behavioral": behavioral_feats,
    "Contextual": contextual_feats,
    "Profile Priors": profile_feats,
    "Others": list(set(X_combined.columns) - set(behavioral_feats + contextual_feats + profile_feats))
}

explainer = shap.TreeExplainer(model)
sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
shap_values = explainer.shap_values(sample)
mean_shap = np.abs(np.stack(shap_values)).mean(axis=(0, 1))
shap_dict = {f: float(np.mean(v)) for f, v in zip(sample.columns.tolist(), mean_shap)}
total = sum(shap_dict.values()) or 1.0

attribution = {
    group: round(100 * sum(shap_dict.get(f, 0.0) for f in feats) / total, 2)
    for group, feats in modality_groups.items()
}

print("\n🔍 SHAP Attribution by Modality (%):")
for k, v in attribution.items():
    print(f"{k:<15}: {v:.2f}%")

# 💾 Save SHAP Attribution
pd.DataFrame(list(attribution.items()), columns=["Modality", "Attribution (%)"]).to_csv(os.path.join(output_dir, 'shap-attribution-modality.csv'), index=False)
with open(os.path.join(output_dir, 'shap-attribution-modality.json'), 'w') as f:
    json.dump(attribution, f, indent=4)

plt.figure(figsize=(8, 4))
sns.barplot(x=list(attribution.keys()), y=list(attribution.values()))
plt.title("SHAP Attribution by Modality (Mixed Model)")
plt.ylabel("Attribution (%)")
plt.xticks(rotation=15)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap-attribution-barplot.pdf'))
plt.show()

# 📋 Save Summary
summary = {
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1_score": round(f1, 4),
    "dataset_combination": "original + RL-generated",
    "rl_data_source": "/content/drive/MyDrive/TrustFusion/LightGBM-RL/rl-generated-data.csv",
    "shap_sample_size": min(500, len(X_test)),
    "model_path": model_path
}
with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=4)
print(f"\n📄 Summary saved to: {os.path.join(output_dir, 'summary.json')}")
