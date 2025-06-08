# predictor.py (append after predict_trust_tier logic)
import shap
import os

explainer = shap.TreeExplainer(model)
SHAP_DIR = "shap_outputs"
os.makedirs(SHAP_DIR, exist_ok=True)

def save_shap_force_plot(input_row, uid):
    shap_values = explainer.shap_values(input_row)
    base_value = explainer.expected_value
    force_plot = shap.force_plot(base_value[np.argmax(shap_values)],
                                 shap_values[np.argmax(shap_values)],
                                 input_row,
                                 matplotlib=False)
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    with open(f"{SHAP_DIR}/{uid}.html", "w") as f:
        f.write(shap_html)


import uuid

def predict_trust_tier(features: dict) -> dict:
    df = pd.DataFrame([features])
    preds = model.predict(df)
    tier = int(np.argmax(preds))
    decision = "grant" if tier >= 2 else "deny"

    shap_id = str(uuid.uuid4())
    save_shap_force_plot(df, shap_id)

    return {
        "trust_tier": tier,
        "decision": decision,
        "shap_id": shap_id
    }
