# 📦 Install LightGBM and SHAP
!pip install lightgbm shap --quiet

# 📌 Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 📁 Output Directory
output_dir = '/content/drive/MyDrive/TrustFusion/LightGBM-Original'
os.makedirs(output_dir, exist_ok=True)

# 📥 Load Dataset
csv_path = '/content/drive/MyDrive/TrustFusion/preprocessed/preprocessed-dataset.csv'
df = pd.read_csv(csv_path)

# 🎯 Features and Target
target_col = 'user_trust_tier'
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# 🔀 Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 🌲 LightGBM Dataset Format
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# ⚙️ LightGBM Parameters
params = {
    'objective': 'multiclass',
    'num_class': len(np.unique(y)),
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
    'random_state': 42
}

# 📊 Store Evaluation Results
evals_result = {}

# 🚀 Train Model with Learning Curve Recording
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'valid'],
    num_boost_round=100,
    callbacks=[
        lgb.record_evaluation(evals_result),
        lgb.log_evaluation(period=10)
    ]
)

# 💾 Save the trained model
model_path = os.path.join(output_dir, 'lightgbm-original.txt')
model.save_model(model_path)
print(f"✅ Model saved to: {model_path}")

# 📈 Save learning curve values and figure
learning_curve_df = pd.DataFrame({
    'Iteration': list(range(1, len(evals_result['train']['multi_logloss']) + 1)),
    'Train_Loss': evals_result['train']['multi_logloss'],
    'Validation_Loss': evals_result['valid']['multi_logloss']
})
curve_csv_path = os.path.join(output_dir, 'learning-curve.csv')
learning_curve_df.to_csv(curve_csv_path, index=False)
print(f"✅ Learning curve data saved to: {curve_csv_path}")

# 🎨 Plot and show learning curve
plt.figure(figsize=(8, 5))
plt.plot(learning_curve_df['Iteration'], learning_curve_df['Train_Loss'], label='Train Loss')
plt.plot(learning_curve_df['Iteration'], learning_curve_df['Validation_Loss'], label='Validation Loss')
plt.xlabel('Boosting Iterations')
plt.ylabel('Multi-class Log Loss')
plt.title('Learning Curve (Original Dataset)')
plt.legend()
plt.grid(True)
plt.tight_layout()
curve_fig_path = os.path.join(output_dir, 'learning-curve.pdf')
plt.savefig(curve_fig_path)
plt.show()

# 🧪 Predict
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# 🧾 Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
report_json_path = os.path.join(output_dir, 'classification-report.json')
with open(report_json_path, 'w') as f:
    json.dump(report, f, indent=4)
print(f"✅ Classification report saved to: {report_json_path}")

# 💾 Save classification report as CSV + Show in Colab
report_df = pd.DataFrame(report).transpose()
report_csv_path = os.path.join(output_dir, 'classification-report.csv')
report_df.to_csv(report_csv_path)
print(f"✅ Classification report saved to: {report_csv_path}")

print("\n📋 Classification Report:")
display(report_df.round(4))

# 🔢 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm)
cm_csv_path = os.path.join(output_dir, 'confusion-matrix.csv')
cm_df.to_csv(cm_csv_path, index=False)
print(f"✅ Confusion matrix saved to: {cm_csv_path}")

# 🎨 Plot and show confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - user_trust_tier")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
cm_fig_path = os.path.join(output_dir, 'confusion-matrix.pdf')
plt.savefig(cm_fig_path)
plt.show()

print("\n📋 Confusion Matrix:")
display(cm_df)

# ⚡ SHAP Explainability
print("\n🔍 Running SHAP Explainability...")

# 🎯 Define modality groups
behavioral_feats = ['usage_frequency', 'peer_rating', 'violation_count', 'complaint_count']
contextual_feats = ['location', 'term', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
profile_feats = ['account_age_days', 'total_interactions', 'admin_flag']
modality_groups = {
    "Behavioral": behavioral_feats,
    "Contextual": contextual_feats,
    "Profile Priors": profile_feats,
    "Others": list(set(X.columns) - set(behavioral_feats + contextual_feats + profile_feats))
}

# 🧠 SHAP Explainer and Values
explainer = shap.TreeExplainer(model)
sample = X_test.sample(n=500, random_state=42)
shap_values = explainer.shap_values(sample)

# ✅ Multiclass handling
if isinstance(shap_values, list):
    shap_values = np.mean(np.abs(np.array(shap_values)), axis=0)
else:
    shap_values = np.abs(shap_values)

# 🔄 Safe scalar conversion
def to_scalar(x):
    if isinstance(x, np.ndarray):
        return x.item() if x.size == 1 else float(np.mean(x))
    return float(x)

mean_shap = shap_values.mean(axis=0)
shap_dict = {f: to_scalar(v) for f, v in zip(sample.columns.tolist(), mean_shap)}

# 📊 Aggregate SHAP scores by modality
group_scores = {}
total = sum(shap_dict.values())
for group, feats in modality_groups.items():
    score = sum(shap_dict.get(f, 0.0) for f in feats)
    group_scores[group] = round(100 * score / total, 2) if total > 0 else 0

# 💾 Save SHAP attribution (JSON + CSV) + Show
shap_json_path = os.path.join(output_dir, 'shap-attribution-modality.json')
with open(shap_json_path, 'w') as f:
    json.dump(group_scores, f, indent=4)
print(f"✅ SHAP attribution saved to: {shap_json_path}")

shap_csv_path = os.path.join(output_dir, 'shap-attribution-modality.csv')
shap_df = pd.DataFrame(list(group_scores.items()), columns=['Modality', 'Attribution (%)'])
shap_df.to_csv(shap_csv_path, index=False)
print(f"✅ SHAP attribution saved to: {shap_csv_path}")

print("\n📊 SHAP Attribution by Modality (%):")
display(shap_df)
