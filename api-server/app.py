# app.py
from flask import Flask, request, jsonify
from auth import require_auth
from predictor import predict_trust_tier

app = Flask(__name__)

@app.route("/")
def health_check():
    return "TrustFusion Prediction API is running."

@app.route("/predict", methods=["POST"])
@require_auth
def predict():
    try:
        user_input = request.get_json(force=True)
        if not isinstance(user_input, dict):
            return jsonify({"error": "Invalid input format"}), 400
        prediction = predict_trust_tier(user_input)
        return jsonify(prediction), 200
    except Exception as e:
        return jsonify({"error": f"Failed to process input: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

@app.route("/shap/<shap_id>")
@require_auth
def get_shap_plot(shap_id):
    path = f"shap_outputs/{shap_id}.html"
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    with open(path, "r") as f:
        return f.read()


#If you want SHAP, you must include the followings

from utils import get_shap_html
import streamlit.components.v1 as components

if "result" in st.session_state and "shap_id" in st.session_state.result:
    st.subheader("Explanation (SHAP Force Plot)")
    shap_html = get_shap_html(st.session_state.result["shap_id"])
    if shap_html:
        components.html(shap_html, height=300, scrolling=True)
    else:
        st.info("SHAP explanation not available.")


