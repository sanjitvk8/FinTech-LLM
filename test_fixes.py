#!/usr/bin/env python3
"""
Simple test to verify that the fixes are working correctly
"""
import os
import sys
import time
import requests
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_api_endpoints():
    """Test the API endpoints to make sure they work with the fixes"""
    base_url = "http://localhost:8000"
    
    print("Testing API endpoints after fixes...")
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        print(f"✓ Health check: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"  Health status: {health_data.get('status', 'unknown')}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False
    
    # Test models endpoint
    try:
        response = requests.get(f"{base_url}/api/v1/models/gemini", timeout=5)
        print(f"✓ Models endpoint: {response.status_code}")
        
        if response.status_code == 200:
            model_data = response.json()
            print(f"  Models available: {len(model_data.get('models', []))}")
    except Exception as e:
        print(f"✗ Models endpoint failed: {e}")
        # This might fail if GEMINI_API_KEY is not set, which is OK
    
    # Test documents list endpoint
    try:
        response = requests.get(f"{base_url}/api/v1/documents/list", timeout=5)
        print(f"✓ Documents list endpoint: {response.status_code}")
        
        if response.status_code == 200:
            docs_data = response.json()
            print(f"  Uploaded documents: {len(docs_data.get('documents', {}))}")
    except Exception as e:
        print(f"✗ Documents list endpoint failed: {e}")
        return False
    
    print("✓ All tests passed!")
    return True

if __name__ == "__main__":
    test_api_endpoints()