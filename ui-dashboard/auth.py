# auth.py
import streamlit as st
from config import ADMIN_TOKEN

def check_auth():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        token = st.text_input("Enter Admin Token:", type="password")
        if st.button("Login"):
            if token == ADMIN_TOKEN:
                st.session_state.authenticated = True
                st.success("Access granted.")
            else:
                st.error("Invalid token.")
        return False
    return True
