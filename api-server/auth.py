# auth.py
from flask import request, jsonify
from functools import wraps
from config import API_SECRET_KEY

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or token != f"Bearer {API_SECRET_KEY}":
            return jsonify({"error": "Unauthorized"}), 403
        return f(*args, **kwargs)
    return wrapper
