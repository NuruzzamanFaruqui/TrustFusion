# utils.py
import requests
import json
from config import API_URL, ADMIN_TOKEN, LOG_FILE
from datetime import datetime
import pandas as pd

def send_prediction_request(user_data):
    headers = {"Authorization": f"Bearer super-secret-access-token"}
    response = requests.post(API_URL, json=user_data, headers=headers)
    if response.status_code == 200:
        result = response.json()
        log_decision(user_data, result)
        return result
    else:
        return {"error": "API failed", "status_code": response.status_code}

def log_decision(user_input, prediction):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": user_input,
        "trust_tier": prediction.get("trust_tier"),
        "decision": prediction.get("decision")
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

def load_logs():
    try:
        with open(LOG_FILE, "r") as f:
            lines = [json.loads(line) for line in f]
        return pd.DataFrame(lines)
    except FileNotFoundError:
        return pd.DataFrame()


def get_shap_html(shap_id):
    shap_url = f"http://localhost:5000/shap/{shap_id}"
    headers = {"Authorization": "Bearer super-secret-access-token"}
    try:
        response = requests.get(shap_url, headers=headers)
        if response.status_code == 200:
            return response.text
        else:
            return None
    except Exception:
        return None

