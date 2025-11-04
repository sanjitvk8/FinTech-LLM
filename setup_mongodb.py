#!/usr/bin/env python3
"""
Setup and verification script for MongoDB in FinTech-LLM project
"""

import subprocess
import sys
import os
import logging
from pymongo import MongoClient
from app.core.config import settings

def install_missing_packages():
    """Install missing Python packages required for MongoDB functionality."""
    required_packages = ["pymongo", "PyPDF2"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {missing_packages}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}: {e}")
                return False
        return True
    else:
        print("All required packages are already installed.")
        return True

def test_mongodb_connection():
    """Test MongoDB connection directly."""
    try:
        print(f"Testing MongoDB connection to: {settings.MONGODB_URL}")
        print(f"Database: {settings.MONGODB_DB_NAME}")
        
        client = MongoClient(settings.MONGODB_URL, serverSelectionTimeoutMS=5000)
        # Test the connection
        client.admin.command('ping')
        
        print("✓ MongoDB connection successful!")
        
        # Access the database and collection
        db = client[settings.MONGODB_DB_NAME]
        collection = db[settings.CHATS_COLLECTION_NAME]
        
        # Test inserting and retrieving a document
        test_doc = {
            "test_field": "connection_test",
            "timestamp": "test_time"
        }
        
        result = collection.insert_one(test_doc)
        print(f"✓ Test document inserted with ID: {result.inserted_id}")
        
        # Retrieve the test document
        retrieved = collection.find_one({"_id": result.inserted_id})
        if retrieved:
            print("✓ Test document retrieved successfully")
        
        # Clean up
        collection.delete_one({"_id": result.inserted_id})
        print("✓ Test document cleaned up")
        
        # Check document count
        count = collection.count_documents({})
        print(f"✓ Collection '{settings.CHATS_COLLECTION_NAME}' exists with {count} documents")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"✗ MongoDB connection failed: {e}")
        return False

def create_indexes():
    """Create indexes on the MongoDB collection."""
    try:
        client = MongoClient(settings.MONGODB_URL)
        db = client[settings.MONGODB_DB_NAME]
        collection = db[settings.CHATS_COLLECTION_NAME]
        
        # Create indexes for better query performance
        collection.create_index("timestamp")
        collection.create_index([("document_id", 1)])
        collection.create_index([("user_query", "text"), ("response", "text")])
        
        print("✓ MongoDB indexes created successfully")
        client.close()
        return True
    except Exception as e:
        print(f"✗ Failed to create MongoDB indexes: {e}")
        return False

def verify_config():
    """Verify MongoDB configuration settings."""
    print(f"MongoDB URL: {settings.MONGODB_URL}")
    print(f"Database Name: {settings.MONGODB_DB_NAME}")
    print(f"Collection Name: {settings.CHATS_COLLECTION_NAME}")
    
    # Check if settings are using defaults
    if settings.MONGODB_URL == "mongodb://localhost:27017":
        print("! Using default MongoDB URL. Make sure MongoDB is running on localhost:27017")
    return True

def main():
    print("FinTech-LLM MongoDB Setup and Verification")
    print("=" * 50)
    
    print("\n1. Checking and installing required packages...")
    packages_ok = install_missing_packages()
    
    if not packages_ok:
        print("Could not install required packages. Please install them manually.")
        return False
    
    print("\n2. Verifying configuration...")
    config_ok = verify_config()
    
    if not config_ok:
        print("Configuration verification failed.")
        return False
    
    print("\n3. Testing MongoDB connection...")
    connection_ok = test_mongodb_connection()
    
    if not connection_ok:
        print("MongoDB connection test failed. Make sure MongoDB is running.")
        return False
    
    print("\n4. Creating MongoDB indexes...")
    indexes_ok = create_indexes()
    
    if not indexes_ok:
        print("Could not create MongoDB indexes.")
        return False
    
    print("\n" + "=" * 50)
    print("✓ MongoDB setup completed successfully!")
    print("You can now run the FinTech-LLM application with MongoDB support.")
    print("\nTo start the application: python run_project.py")
    print("To test the application: http://localhost:8000/api/v1/mongodb-health")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)