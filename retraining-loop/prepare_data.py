# prepare_data.py
import pandas as pd
import json
import os

LOG_PATH = "../logs/trust_decisions.jsonl"
PROCESSED_PATH = "processed/new_samples.csv"

def extract_new_labeled_samples():
    rows = []
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()

    with open(LOG_PATH, "r") as f:
        for line in f:
            entry = json.loads(line)
            input_data = entry.get("input", {})
            trust_tier = entry.get("trust_tier", None)
            decision = entry.get("decision", None)

            if trust_tier is not None and decision in ["grant", "deny"]:
                input_data["user_trust_tier"] = trust_tier
                rows.append(input_data)

    df = pd.DataFrame(rows)
    os.makedirs("processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Extracted {len(df)} new labeled samples.")
    return df
