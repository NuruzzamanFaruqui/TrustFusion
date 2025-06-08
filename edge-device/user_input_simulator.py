# user_input_simulator.py
import random

def get_user_sample():
    # Replace with actual biometric data extraction logic
    sample = {
        "usage_frequency": round(random.uniform(0, 1), 3),
        "peer_rating": round(random.uniform(0, 5), 2),
        "violation_count": random.randint(0, 5),
        "complaint_count": random.randint(0, 5),
        "account_age_days": random.randint(30, 365),
        "total_interactions": random.randint(10, 1000),
        "admin_flag": random.choice([0, 1])
    }
    return sample
