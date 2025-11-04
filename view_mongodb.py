#!/usr/bin/env python3
"""
Database viewer script for FinTech-LLM project.
This script connects to MongoDB and displays stored chat conversations.
"""

import os
import sys
from datetime import datetime
import argparse
from pymongo import MongoClient

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def get_mongodb_config():
    """Get MongoDB configuration from environment variables or defaults."""
    config = {
        "MONGODB_URL": os.getenv("MONGODB_URL", "mongodb://localhost:27017"),
        "MONGODB_DB_NAME": os.getenv("MONGODB_DB_NAME", "fintech_llm"),
        "CHATS_COLLECTION_NAME": os.getenv("CHATS_COLLECTION_NAME", "chats")
    }
    return config

def connect_to_mongodb():
    """Connect to MongoDB and return the database reference."""
    config = get_mongodb_config()
    try:
        client = MongoClient(config["MONGODB_URL"])
        db = client[config["MONGODB_DB_NAME"]]
        collection = db[config["CHATS_COLLECTION_NAME"]]
        return client, db, collection
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None, None, None

def display_chat(chat):
    """Display a single chat conversation in a formatted way."""
    print("=" * 80)
    print(f"Chat ID: {chat.get('_id')}")
    print(f"Timestamp: {chat.get('timestamp', 'N/A')}")
    
    if chat.get('document_id'):
        print(f"Document ID: {chat.get('document_id')}")
    
    print("-" * 40)
    print(f"User Query: {chat.get('user_query', 'N/A')}")
    print("-" * 40)
    print(f"AI Response: {chat.get('response', 'N/A')}")
    
    if chat.get('sources'):
        print(f"Sources: {', '.join(chat.get('sources', []))}")
    
    if chat.get('context'):
        print("Context:")
        for i, ctx in enumerate(chat.get('context', []), 1):
            print(f"  {i}. {ctx}")
    
    print("=" * 80)
    print()

def display_all_chats(limit=100):
    """Display all chat conversations from the database."""
    client, db, collection = connect_to_mongodb()
    if not client:
        return
    
    try:
        # Get all chats sorted by timestamp (newest first)
        chats = list(collection.find().sort("timestamp", -1).limit(limit))
        
        if not chats:
            print("No chat conversations found in the database.")
            return
        
        print(f"Found {len(chats)} chat conversation(s):\n")
        
        for chat in chats:
            # Convert ObjectId to string for display
            chat["_id"] = str(chat["_id"])
            if isinstance(chat.get("timestamp"), datetime):
                chat["timestamp"] = chat["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            
            display_chat(chat)
        
    except Exception as e:
        print(f"Error retrieving chats: {e}")
    finally:
        client.close()

def display_chats_by_document(document_id, limit=100):
    """Display chat conversations related to a specific document."""
    client, db, collection = connect_to_mongodb()
    if not client:
        return
    
    try:
        # Get chats for a specific document sorted by timestamp (newest first)
        chats = list(collection.find({"document_id": document_id}).sort("timestamp", -1).limit(limit))
        
        if not chats:
            print(f"No chat conversations found for document ID: {document_id}")
            return
        
        print(f"Found {len(chats)} chat conversation(s) for document {document_id}:\n")
        
        for chat in chats:
            # Convert ObjectId to string for display
            chat["_id"] = str(chat["_id"])
            if isinstance(chat.get("timestamp"), datetime):
                chat["timestamp"] = chat["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            
            display_chat(chat)
        
    except Exception as e:
        print(f"Error retrieving chats for document {document_id}: {e}")
    finally:
        client.close()

def display_recent_chats(count=10):
    """Display the most recent chat conversations."""
    client, db, collection = connect_to_mongodb()
    if not client:
        return
    
    try:
        # Get recent chats sorted by timestamp (newest first)
        chats = list(collection.find().sort("timestamp", -1).limit(count))
        
        if not chats:
            print("No chat conversations found in the database.")
            return
        
        print(f"Showing the {len(chats)} most recent chat conversation(s):\n")
        
        for chat in chats:
            # Convert ObjectId to string for display
            chat["_id"] = str(chat["_id"])
            if isinstance(chat.get("timestamp"), datetime):
                chat["timestamp"] = chat["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            
            display_chat(chat)
        
    except Exception as e:
        print(f"Error retrieving recent chats: {e}")
    finally:
        client.close()

def display_db_stats():
    """Display database statistics."""
    client, db, collection = connect_to_mongodb()
    if not client:
        return
    
    try:
        # Get collection stats
        count = collection.count_documents({})
        print(f"Database: {db.name}")
        print(f"Collection: {collection.name}")
        print(f"Total chat conversations: {count}")
        
        # Get the date range of conversations
        first_doc = collection.find_one(sort=[("timestamp", 1)])
        last_doc = collection.find_one(sort=[("timestamp", -1)])
        
        if first_doc and last_doc:
            first_date = first_doc.get("timestamp").strftime("%Y-%m-%d %H:%M:%S") if isinstance(first_doc.get("timestamp"), datetime) else str(first_doc.get("timestamp"))
            last_date = last_doc.get("timestamp").strftime("%Y-%m-%d %H:%M:%S") if isinstance(last_doc.get("timestamp"), datetime) else str(last_doc.get("timestamp"))
            print(f"Date range: {first_date} to {last_date}")
        
    except Exception as e:
        print(f"Error getting database stats: {e}")
    finally:
        client.close()

def main():
    parser = argparse.ArgumentParser(description="FinTech-LLM Database Viewer")
    parser.add_argument("--all", action="store_true", help="Display all chat conversations")
    parser.add_argument("--recent", type=int, default=10, help="Display recent chat conversations (default: 10)")
    parser.add_argument("--document", type=str, help="Display chats for a specific document ID")
    parser.add_argument("--stats", action="store_true", help="Display database statistics")
    parser.add_argument("--limit", type=int, default=100, help="Limit the number of results (default: 100)")
    
    args = parser.parse_args()
    
    # Check if no arguments were provided
    if not any([args.all, args.recent != 10, args.document, args.stats]):
        print("Welcome to the FinTech-LLM Database Viewer!")
        print("\nUsage examples:")
        print("  python view_mongodb.py --all            # Display all conversations")
        print("  python view_mongodb.py --recent 5       # Display 5 most recent conversations")
        print("  python view_mongodb.py --document ID    # Display conversations for specific document")
        print("  python view_mongodb.py --stats          # Display database statistics")
        print("\nDefault action is to show 10 recent conversations:")
        print()
    
    if args.stats:
        display_db_stats()
    elif args.all:
        display_all_chats(args.limit)
    elif args.document:
        display_chats_by_document(args.document, args.limit)
    else:
        # Default to showing recent chats or if --recent was specified
        display_recent_chats(args.recent)

if __name__ == "__main__":
    main()