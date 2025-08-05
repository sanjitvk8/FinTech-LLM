import requests
import json

# Test configuration
BASE_URL = "http://localhost:8000/api/v1"
AUTH_TOKEN = "d5f4ea72e4cce0351417c18d19b196b40e7dbe8f29d2ece3c9b01f6680160650"

headers = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

# Test payload
payload = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment?",
        "What are the policy benefits?"
    ]
}

try:
    response = requests.post(f"{BASE_URL}/hackrx/run", headers=headers, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
