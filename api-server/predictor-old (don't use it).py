# predictor.py
import lightgbm as lgb
import numpy as np
import pandas as pd
from config import MODEL_PATH

model = lgb.Booster(model_file=MODEL_PATH)

def predict_trust_tier(features: dict) -> dict:
    df = pd.DataFrame([features])
    preds = model.predict(df)
    tier = int(np.argmax(preds))
    decision = "grant" if tier >= 2 else "deny"
    return {
        "trust_tier": tier,
        "decision": decision
    }
