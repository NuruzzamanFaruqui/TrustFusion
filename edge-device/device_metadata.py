# device_metadata.py
import socket
from datetime import datetime
import random

def get_device_metadata():
    metadata = {
        "device_id": socket.gethostname(),
        "timestamp": datetime.utcnow().isoformat(),
        "hour": datetime.utcnow().hour,
        "day_of_week": datetime.utcnow().weekday(),
        "location": "Entrance_Gate_A",
        "term": "Spring_2025"
    }
    return metadata
