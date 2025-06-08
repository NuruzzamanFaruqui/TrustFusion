headers = {"Authorization": "Bearer super-secret-access-token"}
response = requests.post(API_ENDPOINT, json=user_data, headers=headers)
