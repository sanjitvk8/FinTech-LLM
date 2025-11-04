import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from ..core.config import settings

logger = logging.getLogger(__name__)

class MongoDBManager:
    """
    MongoDB Manager for handling database operations related to chat conversations
    """
    
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.chats_collection: Optional[Collection] = None
    
    def connect(self):
        """
        Connect to MongoDB and initialize the database and collection
        """
        try:
            self.client = MongoClient(settings.MONGODB_URL)
            self.db = self.client[settings.MONGODB_DB_NAME]
            self.chats_collection = self.db[settings.CHATS_COLLECTION_NAME]
            
            # Create indexes for better query performance
            self.chats_collection.create_index("timestamp")
            self.chats_collection.create_index([("document_id", 1)])
            self.chats_collection.create_index([("user_query", "text"), ("response", "text")])
            
            logger.info(f"Successfully connected to MongoDB: {settings.MONGODB_URL}")
            logger.info(f"Database: {settings.MONGODB_DB_NAME}")
            logger.info(f"Collection: {settings.CHATS_COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            raise
    
    def disconnect(self):
        """
        Close the MongoDB connection
        """
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def save_chat_conversation(self, user_query: str, response: str, document_id: Optional[str] = None, sources: Optional[List[str]] = None, context: Optional[List[str]] = None) -> Optional[str]:
        """
        Save a chat conversation to the database
        
        Args:
            user_query: The question asked by the user
            response: The response from the AI
            document_id: Optional document ID if the query was related to a specific document
            sources: Optional list of sources used in the response
            context: Optional list of context chunks used in the response
        
        Returns:
            The ID of the inserted document, or None if insertion failed
        """
        try:
            chat_doc = {
                "user_query": user_query,
                "response": response,
                "document_id": document_id,
                "sources": sources or [],
                "context": context or [],
                "timestamp": datetime.utcnow()
            }
            
            result = self.chats_collection.insert_one(chat_doc)
            logger.info(f"Chat conversation saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving chat conversation: {str(e)}")
            return None
    
    def get_all_chats(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve all chat conversations from the database
        
        Args:
            limit: Maximum number of chats to retrieve (default 100)
        
        Returns:
            List of chat conversations
        """
        try:
            chats = list(self.chats_collection.find().sort("timestamp", -1).limit(limit))
            # Convert ObjectId to string for JSON serialization
            for chat in chats:
                chat["_id"] = str(chat["_id"])
            return chats
        except Exception as e:
            logger.error(f"Error retrieving chat conversations: {str(e)}")
            return []
    
    def get_chats_by_document_id(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve chat conversations related to a specific document
        
        Args:
            document_id: ID of the document to filter by
        
        Returns:
            List of chat conversations related to the document
        """
        try:
            chats = list(self.chats_collection.find({"document_id": document_id}).sort("timestamp", -1))
            # Convert ObjectId to string for JSON serialization
            for chat in chats:
                chat["_id"] = str(chat["_id"])
            return chats
        except Exception as e:
            logger.error(f"Error retrieving chats for document {document_id}: {str(e)}")
            return []
    
    def get_recent_chats(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent chat conversations
        
        Args:
            count: Number of recent chats to retrieve (default 10)
        
        Returns:
            List of recent chat conversations
        """
        try:
            chats = list(self.chats_collection.find().sort("timestamp", -1).limit(count))
            # Convert ObjectId to string for JSON serialization
            for chat in chats:
                chat["_id"] = str(chat["_id"])
            return chats
        except Exception as e:
            logger.error(f"Error retrieving recent chats: {str(e)}")
            return []

# Global MongoDB manager instance
mongodb_manager = MongoDBManager()