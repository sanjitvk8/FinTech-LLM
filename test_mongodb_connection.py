#!/usr/bin/env python3
"""
Test script to verify MongoDB connection and check if data is being inserted properly.
"""

import os
import sys
from datetime import datetime
import logging
from pymongo import MongoClient

# Add the project root to the path so we can import the MongoDB manager
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_mongodb_connection():
    """Test the MongoDB connection directly"""
    try:
        from pymongo import MongoClient
        from app.utils.mongodb import mongodb_manager
        
        # Test basic connection
        print("Testing MongoDB connection...")
        
        # Connect using the MongoDB manager
        mongodb_manager.connect()
        print("✓ MongoDB connection successful!")
        
        # Check if database and collection exist by inserting a test document
        test_doc = {
            "test_query": "Test connection",
            "test_response": "Connection successful",
            "test_document_id": "test_doc",
            "sources": ["test_source"],
            "context": ["test_context"],
            "timestamp": datetime.utcnow()
        }
        
        # Insert the test document
        result = mongodb_manager.chats_collection.insert_one(test_doc)
        print(f"✓ Test document inserted with ID: {result.inserted_id}")
        
        # Retrieve the test document
        retrieved_doc = mongodb_manager.chats_collection.find_one({"_id": result.inserted_id})
        if retrieved_doc:
            print("✓ Test document retrieved successfully")
            print(f"  Retrieved: {retrieved_doc['test_query']}")
        else:
            print("✗ Failed to retrieve test document")
        
        # Count total documents in the collection
        total_docs = mongodb_manager.chats_collection.count_documents({})
        print(f"✓ Total documents in collection: {total_docs}")
        
        # Show the most recent document (if any exist beyond our test)
        recent_docs = list(mongodb_manager.chats_collection.find().sort("timestamp", -1).limit(5))
        if len(recent_docs) > 1:  # More than just our test doc
            print(f"✓ Found {len(recent_docs)-1} existing documents (excluding test)")
            for doc in recent_docs[1:]:  # Skip our test doc
                print(f"  - Query: {doc.get('user_query', 'N/A')[:50]}...")
        
        # Clean up: remove the test document
        mongodb_manager.chats_collection.delete_one({"_id": result.inserted_id})
        print("✓ Test document cleaned up")
        
        # Disconnect
        mongodb_manager.disconnect()
        print("✓ MongoDB connection closed")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ MongoDB connection test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_with_env_vars():
    """Test using environment variables"""
    print("\nTesting with environment variables...")
    
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    mongodb_db_name = os.getenv("MONGODB_DB_NAME", "fintech_llm")
    chats_collection_name = os.getenv("CHATS_COLLECTION_NAME", "chats")
    
    print(f"MongoDB URL: {mongodb_url}")
    print(f"Database: {mongodb_db_name}")
    print(f"Collection: {chats_collection_name}")
    
    try:
        client = MongoClient(mongodb_url)
        db = client[mongodb_db_name]
        collection = db[chats_collection_name]
        
        # Test connection
        client.admin.command('ping')
        print("✓ Direct MongoDB connection successful!")
        
        # Count documents
        doc_count = collection.count_documents({})
        print(f"✓ Document count: {doc_count}")
        
        # Show if there are any documents
        if doc_count > 0:
            sample_docs = list(collection.find().limit(3))
            print(f"Sample documents:")
            for i, doc in enumerate(sample_docs, 1):
                print(f"  {i}. Query: {doc.get('user_query', 'N/A')[:50]}...")
        
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"✗ Environment variable test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_app_logging():
    """Check app logs for MongoDB-related messages"""
    print("\nChecking application logs for MongoDB messages...")
    
    log_file = "app.log"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        mongodb_logs = [line for line in lines if "MongoDB" in line or "mongodb" in line or "pymongo" in line]
        
        if mongodb_logs:
            print(f"Found {len(mongodb_logs)} MongoDB-related log entries:")
            for log in mongodb_logs[-10:]:  # Show last 10 entries
                print(f"  {log.strip()}")
        else:
            print("No MongoDB-related log entries found in app.log")
    else:
        print("No app.log file found")

if __name__ == "__main__":
    print("MongoDB Connection Test")
    print("="*50)
    
    success = True
    
    # Test 1: Basic MongoDB connection with our manager
    success &= test_mongodb_connection()
    
    # Test 2: Test with environment variables
    success &= test_with_env_vars()
    
    # Test 3: Check application logs
    check_app_logging()
    
    print("\n" + "="*50)
    if success:
        print("✓ All tests passed! MongoDB is properly configured.")
    else:
        print("✗ Some tests failed. Please check the output above for details.")
        sys.exit(1)