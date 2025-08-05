import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
AUTH_TOKEN = "d5f4ea72e4cce0351417c18d19b196b40e7dbe8f29d2ece3c9b01f6680160650"

headers = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_main_endpoint():
    """Test main query endpoint"""
    print("Testing main endpoint...")
    
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?"
        ]
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/hackrx/run", headers=headers, json=payload)
    end_time = time.time()
    
    print(f"Status: {response.status_code}")
    print(f"Response time: {end_time - start_time:.2f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Processing time: {result.get('processing_time', 0):.2f} seconds")
        print(f"Number of answers: {len(result.get('answers', []))}")
        print("\nAnswers:")
        for i, answer in enumerate(result.get('answers', []), 1):
            print(f"{i}. {answer}")
            print()
    else:
        print(f"Error: {response.text}")

def test_structured_query():
    """Test structured query endpoint"""
    print("Testing structured query endpoint...")
    
    payload = {
        "query": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
        "documents_url": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    }
    
    response = requests.post(f"{BASE_URL}/process-structured-query", headers=headers, params=payload)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Decision: {result.get('decision', {}).get('decision', 'N/A')}")
        print(f"Justification: {result.get('decision', {}).get('justification', 'N/A')}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    print("Starting API tests...\n")
    
    test_health_check()
    test_main_endpoint()
    # test_structured_query()  # Uncomment to test structured queries
    
    print("Tests completed!")

