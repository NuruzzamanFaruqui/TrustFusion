# api_client.py
import requests

API_ENDPOINT = "http://<server-ip>:5000/predict"  # Replace with your backend IP

def send_to_backend(user_data):
    try:
        response = requests.post(API_ENDPOINT, json=user_data)
        if response.status_code == 200:
            result = response.json()
            return result.get("trust_tier", -1), result.get("decision", "deny")
        else:
            print("API error:", response.status_code)
            return -1, "error"
    except requests.exceptions.RequestException as e:
        print("Connection failed:", e)
        return -1, "error"
