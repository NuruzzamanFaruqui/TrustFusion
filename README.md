# TrustFusion Framework

A modular trust-based decision system for secure IoT environments with machine learning, reinforcement learning, and explainable AI.

> 🔒 The full dataset used in this system is currently under embargo until publication. A sample dataset is included for framework testing.

## 📦 Modules Overview

- `api-server/`: Token-authenticated Flask server for trust tier prediction.
- `edge-device/`: Raspberry Pi GPIO interface and local access control.
- `ui-dashboard/`: Streamlit-based admin interface with SHAP explainability.
- `ml-core/`: Model training (LightGBM, DNN) and SHAP evaluation scripts.
- `retraining-loop/`: Merges feedback + retrains with RL/real-world data.
- `deployment/`: Systemd service and install script for edge automation.
- `configs/`: Centralized YAML/.env config for model, paths, auth.
- `logs/`: Prediction logs, model audit trail, evasion alerts.
- `data/`: Sample dataset + schema with embargo notice.

## ⚙️ Getting Started

```bash
git clone <repo-url>
cd TrustFusion
pip install -r requirements.txt
```

## 🔁 Running the Framework

- Start the backend server: `python api-server/app.py`
- Launch the UI dashboard: `streamlit run ui-dashboard/app.py`
- To enable edge-device service: see `deployment/install-edge-service.sh`

## 📬 Dataset Access

A synthetic sample is provided. Contact the author for access to the full dataset after publication.