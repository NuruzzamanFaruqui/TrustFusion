# app.py
import streamlit as st
from auth import check_auth
from utils import send_prediction_request, load_logs

st.set_page_config(page_title="TrustFusion Admin Dashboard", layout="wide")

if not check_auth():
    st.stop()

st.title("🔐 TrustFusion Admin Dashboard")

# Left panel: Submit user request manually
with st.sidebar:
    st.header("Manual Trust Evaluation")
    user_data = {
        "usage_frequency": st.slider("Usage Frequency", 0.0, 1.0, 0.5),
        "peer_rating": st.slider("Peer Rating", 0.0, 5.0, 4.0),
        "violation_count": st.number_input("Violation Count", 0, 10, 0),
        "complaint_count": st.number_input("Complaint Count", 0, 10, 0),
        "account_age_days": st.number_input("Account Age (days)", 1, 1000, 100),
        "total_interactions": st.number_input("Total Interactions", 1, 10000, 500),
        "admin_flag": st.selectbox("Admin Flag", [0, 1]),
        "location": st.text_input("Location", "Entrance_Gate_A"),
        "term": st.text_input("Term", "Spring_2025"),
        "hour": st.slider("Hour", 0, 23, 12),
        "day_of_week": st.slider("Day of Week", 0, 6, 2),
        "hour_sin": 0.0, "hour_cos": 0.0,
        "day_sin": 0.0, "day_cos": 0.0
    }

    import numpy as np
    user_data["hour_sin"] = np.sin(2 * np.pi * user_data["hour"] / 24)
    user_data["hour_cos"] = np.cos(2 * np.pi * user_data["hour"] / 24)
    user_data["day_sin"] = np.sin(2 * np.pi * user_data["day_of_week"] / 7)
    user_data["day_cos"] = np.cos(2 * np.pi * user_data["day_of_week"] / 7)

    if st.button("Submit Prediction"):
        result = send_prediction_request(user_data)
        st.session_state.result = result

# Center panel: Display result
if "result" in st.session_state:
    result = st.session_state.result
    st.subheader("Prediction Result")
    st.write(f"**Trust Tier:** {result.get('trust_tier')}")
    st.write(f"**Access Decision:** {result.get('decision').upper()}")

# Right panel: Show logs
st.subheader("Recent Logs")
log_df = load_logs()
if not log_df.empty:
    st.dataframe(log_df.tail(10).sort_values("timestamp", ascending=False))
else:
    st.info("No logs found yet.")
