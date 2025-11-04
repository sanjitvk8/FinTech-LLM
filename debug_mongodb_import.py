#!/usr/bin/env python3
"""
Debug script to check MongoDB manager import
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Current working directory: {os.getcwd()}")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths

# Check if app directory exists
app_dir = os.path.join(project_root, 'app')
print(f"App directory exists: {os.path.exists(app_dir)}")

# Check if utils directory exists
utils_dir = os.path.join(app_dir, 'utils')
print(f"Utils directory exists: {os.path.exists(utils_dir)}")

# Check if mongodb.py exists
mongodb_file = os.path.join(utils_dir, 'mongodb.py')
print(f"MongoDB file exists: {os.path.exists(mongodb_file)}")

# Try to import step by step
try:
    print("\nTrying: from app.utils.mongodb import mongodb_manager")
    from app.utils.mongodb import mongodb_manager
    print(f"Success! mongodb_manager = {mongodb_manager}")
    print(f"mongodb_manager type: {type(mongodb_manager)}")
    
    if mongodb_manager:
        print("MongoDB manager is available!")
    else:
        print("MongoDB manager is None")
        
except ImportError as e:
    print(f"ImportError: {e}")
    
    try:
        print("\nTrying alternative: import sys, os and modify path")
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(current_dir, 'app'))
        sys.path.insert(0, os.path.join(current_dir, 'app', 'utils'))
        
        from mongodb import mongodb_manager
        print(f"Alternative import success! mongodb_manager = {mongodb_manager}")
    except Exception as e2:
        print(f"Alternative import also failed: {e2}")
        
        # Let's check if we can at least import pymongo and create manually
        print("\nTrying to create MongoDB manager directly...")
        try:
            from pymongo import MongoClient
            print("PyMongo imported successfully")
            
            # Try to create a client directly to test connection
            from app.core.config import settings
            print(f"MongoDB URL from settings: {settings.MONGODB_URL}")
            
            client = MongoClient(settings.MONGODB_URL, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')  # Test the connection
            print("Direct MongoDB connection successful!")
            
            # If direct connection works, there's an import issue
            print("The issue is with the import, not the MongoDB connection itself")
            
        except Exception as conn_error:
            print(f"Direct MongoDB connection failed: {conn_error}")