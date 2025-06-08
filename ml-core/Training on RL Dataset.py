# 📦 Required Imports
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
output_dir = '/content/drive/MyDrive/TrustFusion/LightGBM-RL'
os.makedirs(output_dir, exist_ok=True)

# 📂 Load RL-enhanced dataset
df = pd.read_csv(f'{output_dir}/rl-generated-data.csv')

# 📌 Load original dataset to assign pseudo-labels (simulate real user feedback)
original_df = pd.read_csv('/content/drive/MyDrive/TrustFusion/preprocessed/preprocessed-dataset.csv')
target_col = 'user_trust_tier'
y = original_df[target_col].astype(int).sample(n=len(df), random_state=42).reset_index(drop=True)

# 🎯 Train/test split
X_train, X_test, y_train, y_test = train_test_split(df, y, stratify=y, test_size=0.2, random_state=42)

# 🚀 Train LightGBM model on RL-enhanced data
params = {
    'objective': 'multiclass',
    'num_class': len(np.unique(y)),
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
    'random_state': 42
}
evals_result = {}
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

model_rl = lgb.train(params, train_data, valid_sets=[train_data, test_data],
                     valid_names=['train', 'valid'],
                     num_boost_round=100,
                     callbacks=[
                         lgb.record_evaluation(evals_result),
                         lgb.log_evaluation(period=10)
                     ])

# 💾 Save trained model
model_path = os.path.join(output_dir, 'lightgbm_rl_model.txt')
model_rl.save_model(model_path)
print(f"✅ Model saved to: {model_path}")

# 📈 Save and plot learning curve
learning_curve_df = pd.DataFrame({
    'Iteration': list(range(1, len(evals_result['train']['multi_logloss']) + 1)),
    'Train_Loss': evals_result['train']['multi_logloss'],
    'Validation_Loss': evals_result['valid']['multi_logloss']
})
curve_csv_path = os.path.join(output_dir, 'learning-curve.csv')
learning_curve_df.to_csv(curve_csv_path, index=False)
print(f"✅ Learning curve saved to: {curve_csv_path}")

plt.figure(figsize=(8, 5))
plt.plot(learning_curve_df['Iteration'], learning_curve_df['Train_Loss'], label='Train Loss')
plt.plot(learning_curve_df['Iteration'], learning_curve_df['Validation_Loss'], label='Validation Loss')
plt.xlabel('Boosting Iterations')
plt.ylabel('Multi-class Log Loss')
plt.title('Learning Curve (RL-Retrained Model)')
plt.legend()
plt.grid(True)
plt.tight_layout()
curve_fig_path = os.path.join(output_dir, 'learning-curve.pdf')
plt.savefig(curve_fig_path)
plt.show()

# 📊 Model evaluation
y_pred_proba = model_rl.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\n📊 RL-Retrained Model Performance:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")

# 🧾 Classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_json_path = os.path.join(output_dir, 'classification-report.json')
with open(report_json_path, 'w') as f:
    json.dump(report, f, indent=4)
report_csv_path = os.path.join(output_dir, 'classification-report.csv')
pd.DataFrame(report).transpose().to_csv(report_csv_path)
print(f"✅ Classification report saved to: {report_csv_path}")
display(pd.DataFrame(report).transpose())

# 🔢 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm)
cm_csv_path = os.path.join(output_dir, 'confusion-matrix.csv')
cm_df.to_csv(cm_csv_path, index=False)
print(f"✅ Confusion matrix saved to: {cm_csv_path}")
display(cm_df)

# 🎨 Confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - RL-Retrained Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
cm_fig_path = os.path.join(output_dir, 'confusion-matrix.pdf')
plt.savefig(cm_fig_path)
plt.show()

# 🔍 SHAP Attribution
behavioral_feats = ['usage_frequency', 'peer_rating', 'violation_count', 'complaint_count']
contextual_feats = ['location', 'term', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
profile_feats = ['account_age_days', 'total_interactions', 'admin_flag']
modality_groups = {
    "Behavioral": behavioral_feats,
    "Contextual": contextual_feats,
    "Profile Priors": profile_feats,
    "Others": list(set(df.columns) - set(behavioral_feats + contextual_feats + profile_feats))
}

explainer = shap.TreeExplainer(model_rl)
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

# 💾 Save SHAP Attribution (JSON + CSV)
shap_json_path = os.path.join(output_dir, 'shap-attribution-modality.json')
with open(shap_json_path, 'w') as f:
    json.dump(attribution, f, indent=4)
shap_csv_path = os.path.join(output_dir, 'shap-attribution-modality.csv')
pd.DataFrame(list(attribution.items()), columns=["Modality", "Attribution (%)"]).to_csv(shap_csv_path, index=False)
print(f"✅ SHAP attribution saved to: {shap_csv_path}")

# 📈 SHAP attribution bar chart
plt.figure(figsize=(8, 4))
sns.barplot(x=list(attribution.keys()), y=list(attribution.values()))
plt.title("SHAP Attribution by Modality (RL-Retrained Model)")
plt.ylabel("Attribution (%)")
plt.xticks(rotation=15)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap-attribution-barplot.pdf'))
plt.show()

# 📋 Summary file
summary = {
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1_score": round(f1, 4),
    "rl_dataset_used": f"{output_dir}/rl-generated-data.csv",
    "labels_sampled_from": "original dataset with pseudo-labeling",
    "shap_sample_size": min(500, len(X_test)),
    "model_path": model_path
}
summary_path = os.path.join(output_dir, 'summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=4)
print(f"📄 Summary saved to: {summary_path}")
