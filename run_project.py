#!/usr/bin/env python3
"""
Single file runner for the FinTech-LLM project.
Simply run: python run_project.py
"""

import os
import sys
import subprocess
import threading
import time
import logging
from pathlib import Path
import importlib.util
import signal
import atexit
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import time as time_module
import google.generativeai as genai  # Using the current stable format
from typing import Optional
import tempfile
import mimetypes

# Initialize MongoDB manager
import sys
import os
import logging

logger = logging.getLogger(__name__)

# Add the project root to sys.path to ensure proper imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

mongodb_manager = None

try:
    # Try to import the pre-built MongoDB manager
    from app.utils.mongodb import mongodb_manager
    if mongodb_manager is not None:
        print("MongoDB manager imported successfully")
    else:
        print("Warning: MongoDB manager was imported but is None")
except ImportError as e:
    # If the import fails, try to dynamically create one
    print(f"Could not import MongoDB manager from app.utils.mongodb: {e}")
    print("Attempting to create MongoDB manager dynamically...")
    
    try:
        # Import required modules directly
        from pymongo import MongoClient
        from datetime import datetime
        from typing import Optional, List, Dict, Any
        from app.core.config import settings
        
        # Create a simple MongoDB manager class
        class DynamicMongoDBManager:
            def __init__(self):
                self.client = None
                self.db = None
                self.chats_collection = None
            
            def connect(self):
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
                    return True
                except Exception as e:
                    logger.error(f"Error connecting to MongoDB: {str(e)}")
                    return False
            
            def disconnect(self):
                if self.client:
                    self.client.close()
                    logger.info("MongoDB connection closed")
            
            def save_chat_conversation(self, user_query: str, response: str, document_id: Optional[str] = None, sources: Optional[List[str]] = None, context: Optional[List[str]] = None) -> Optional[str]:
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
                try:
                    chats = list(self.chats_collection.find().sort("timestamp", -1).limit(limit))
                    for chat in chats:
                        chat["_id"] = str(chat["_id"])
                    return chats
                except Exception as e:
                    logger.error(f"Error retrieving chat conversations: {str(e)}")
                    return []
        
        mongodb_manager = DynamicMongoDBManager()
        print("Dynamically created MongoDB manager successfully")
        
    except ImportError as mongo_import_error:
        print(f"Could not create MongoDB manager: missing dependencies - {mongo_import_error}")
        print("Please install pymongo: pip install pymongo")
        mongodb_manager = None
    except Exception as e:
        print(f"Unexpected error creating MongoDB manager: {e}")
        mongodb_manager = None
except Exception as e:
    print(f"Unexpected error importing MongoDB manager: {e}")
    mongodb_manager = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

# Document storage
uploaded_documents = {}  # In-memory storage - in production, use a proper database

# Configuration class (simplified version of the config in the original project)
class Settings:
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "LLM Document Processing System"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    
    # Groq Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "llama3-8b-8192"
    
    # Google Gemini Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-2.5-pro"  # Model with strong document processing
    
    # Select which provider to use (openai, groq, gemini, auto)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    PINECONE_INDEX_NAME: str = "document-retrieval"
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./llm_document_processing.db")
    
    # Authentication
    SECRET_KEY: str = os.getenv("SECRET_KEY", "Kj8#mN2$pQ9@rT6&yU3!eW7*zA1%cV5^bX4(dF8)gH0+lM9-nP2=qS6~tY3<zA7>")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Processing Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_TOKENS_PER_REQUEST: int = 4000
    SIMILARITY_THRESHOLD: float = 0.3
    MAX_RETRIEVED_CHUNKS: int = 10
    
    # File Processing
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".txt", ".eml"}
    
    # Cache Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL: int = 3600  # 1 hour

    @property
    def has_openai_key(self):
        return bool(self.OPENAI_API_KEY and self.OPENAI_API_KEY != "")

settings = Settings()

# Document processing functions
def get_gemini_client():
    """Initialize and return Gemini client if API key is available."""
    if settings.GEMINI_API_KEY:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        return genai
    return None


def get_compatible_gemini_model_instance():
    """Return a GenerativeModel instance that's compatible with generateContent.

    Tries the configured model first; on failure queries available models and
    picks the first one that supports 'generateContent', updates
    settings.GEMINI_MODEL, and returns the instance.
    """
    try:
        import google.generativeai as genai_module
        genai_module.configure(api_key=settings.GEMINI_API_KEY)

        # Try the configured model first
        try:
            return genai_module.GenerativeModel(model_name=settings.GEMINI_MODEL)
        except Exception as e:
            logger.warning(f"Configured Gemini model '{settings.GEMINI_MODEL}' not usable: {e}")

        # Fall back: list available models and pick one that supports generateContent
        available = list_available_gemini_models()
        for m in available:
            methods = m.get('supported_generation_methods', [])
            if 'generateContent' in methods:
                candidate = m.get('name')
                try:
                    model = genai_module.GenerativeModel(model_name=candidate)
                    settings.GEMINI_MODEL = candidate
                    logger.info(f"Switched Gemini model to compatible model: {candidate}")
                    return model
                except Exception as e2:
                    logger.warning(f"Failed to instantiate candidate model {candidate}: {e2}")

        raise RuntimeError("No compatible Gemini model found that supports generateContent")
    except Exception:
        raise

def upload_file_to_gemini(file_path: str, mime_type: Optional[str] = None):
    """Upload a file to Gemini and return the file object."""
    try:
        import google.generativeai as genai_module
        genai_module.configure(api_key=settings.GEMINI_API_KEY)
        import mimetypes
        from pathlib import Path
        converted_temp_path = None

        # Determine MIME type if not provided
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "application/octet-stream"  # Default fallback

        # If the file is a .docx (which Gemini may reject), convert to plain text first
        suffix = Path(file_path).suffix.lower()
        if suffix == ".docx":
            try:
                # Local import so missing dependency only affects this branch
                from docx import Document
                import tempfile

                doc = Document(file_path)
                text_parts = [p.text for p in doc.paragraphs if p.text]
                text_content = "\n\n".join(text_parts) or ""

                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmpf:
                    tmpf.write(text_content.encode("utf-8"))
                    converted_temp_path = tmpf.name

                # Upload the converted plain-text file instead
                upload_path = converted_temp_path
                upload_mime = "text/plain"
            except Exception as e:
                logger.warning(f"Failed to convert .docx to text, will attempt raw upload: {e}")
                upload_path = file_path
                upload_mime = mime_type
        else:
            upload_path = file_path
            upload_mime = mime_type

        # Upload the file
        uploaded_file = genai_module.upload_file(path=upload_path, mime_type=upload_mime)
        
        # Wait for the file to be processed
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = genai_module.get_file(uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            # Clean up converted file if present
            if converted_temp_path and os.path.exists(converted_temp_path):
                try:
                    os.unlink(converted_temp_path)
                except Exception:
                    pass
            raise ValueError(f"File processing failed: {uploaded_file.state.name}")

        # Clean up converted file if present
        if converted_temp_path and os.path.exists(converted_temp_path):
            try:
                os.unlink(converted_temp_path)
            except Exception:
                pass

        return uploaded_file
    except Exception as e:
        logger.error(f"Error uploading file to Gemini: {str(e)}")
        raise

def query_with_gemini(file_path: str, query: str):
    """Query a document using Gemini API."""
    try:
        import google.generativeai as genai_module
        genai_module.configure(api_key=settings.GEMINI_API_KEY)
        
        # Upload the file to Gemini
        uploaded_file = upload_file_to_gemini(file_path)
        
        # Get a compatible model (tries configured model first, then auto-selects)
        model = get_compatible_gemini_model_instance()
        
        # Generate content with the uploaded file and query
        response = model.generate_content([uploaded_file, query])
        
        # Clean up the uploaded file (optional, depending on use case)
        # genai_module.delete_file(uploaded_file.name)
        
        return response.text
    except Exception as e:
        logger.error(f"Error querying with Gemini: {str(e)}")
        raise

def process_document_with_gemini(uploaded_file, query: str):
    """Process an uploaded document with Gemini and return the response."""
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.filename}") as temp_file:
        temp_file.write(uploaded_file.file.read())
        temp_file_path = temp_file.name

    try:
        # Query the document using Gemini
        response_text = query_with_gemini(temp_file_path, query)
        
        return {
            "answer": response_text,
            "context": ["Document processed using Gemini API"],
            "sources": [uploaded_file.filename]
        }
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


def list_available_gemini_models():
    """List all available Gemini models."""
    try:
        import google.generativeai as genai_module
        genai_module.configure(api_key=settings.GEMINI_API_KEY)
        
        # List all available models
        models = genai_module.list_models()
        
        available_models = []
        for model in models:
            # Filter for models that support the use case
            supported_generation_methods = set(['generateContent', 'countTokens', 'embedContent'])
            compatible_methods = set(model.supported_generation_methods)
            
            if supported_generation_methods.intersection(compatible_methods):
                available_models.append({
                    'name': model.name,
                    'display_name': getattr(model, 'display_name', 'N/A'),
                    'description': getattr(model, 'description', 'N/A'),
                    'input_token_limit': getattr(model, 'input_token_limit', 'N/A'),
                    'output_token_limit': getattr(model, 'output_token_limit', 'N/A'),
                    'supported_generation_methods': model.supported_generation_methods
                })
        
        return available_models
    except Exception as e:
        logger.error(f"Error listing Gemini models: {str(e)}")
        return []

def check_and_install_dependencies():
    """Check if required packages are installed, and install them if needed."""
    required_packages = [
        "fastapi",
        "uvicorn", 
        "python-multipart",
        "pydantic",
        "openai",
        "tiktoken",
        "sentence-transformers",
        "faiss-cpu",
        "PyPDF2",
        "python-docx",
        "requests",
        "python-jose[cryptography]",
        "passlib[bcrypt]",
        "aiofiles",
        "pandas",
        "numpy",
        "scikit-learn",
        "sqlalchemy",
        "python-dotenv",
        "httpx",
        "pydantic-settings",
        "groq",
        "streamlit",  # Add streamlit to the required packages
        "google-generativeai"  # Add Google Generative AI
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if '[' in package:
                # Handle packages with extras like python-jose[cryptography]
                base_package = package.split('[')[0]
                importlib.util.find_spec(base_package)
            else:
                importlib.util.find_spec(package)
        except (ImportError, AttributeError, ValueError):
            missing_packages.append(package)
    
    if missing_packages:
        logger.info(f"Installing missing packages: {missing_packages}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"Successfully installed {package}")
            except subprocess.CalledProcessError:
                logger.error(f"Failed to install {package}")
                return False
        return True
    else:
        logger.info("All required packages are already installed.")
        return True

def setup_directories():
    """Create necessary directories for the application."""
    directories = ["logs", "data", "temp"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Directory {directory}/ created or already exists")

def create_sample_env_file():
    """Create a sample .env file if it doesn't exist."""
    env_file = Path(".env")
    if not env_file.exists():
        sample_content = """# FinTech-LLM Environment Variables
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1-aws
SECRET_KEY=your_secret_key_here
DATABASE_URL=postgresql://user:password@localhost/dbname
REDIS_URL=redis://localhost:6379
"""
        with open(env_file, 'w') as f:
            f.write(sample_content)
        logger.info("Sample .env file created. Please update it with your actual API keys.")
    else:
        logger.info(".env file already exists.")

# Simulated API endpoints (simplified version)
from fastapi import File, UploadFile
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    document_id: str = None

class QueryResponse(BaseModel):
    answer: str
    context: list = []
    sources: list = []
    document_id: str = None

def create_app():
    """Create and configure the FastAPI application."""
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan context manager"""
        # Startup
        logger.info("Starting LLM Document Processing System...")
        logger.info(f"Version: {settings.VERSION}")
        
        # Verify configuration
        if not settings.has_openai_key:
            logger.warning("OpenAI API key not configured! The application will run with limited functionality.")
        
        # Initialize MongoDB connection
        if mongodb_manager is not None:
            try:
                mongodb_manager.connect()
                logger.info("MongoDB connection established successfully")
                logger.info(f"Connected to: {settings.MONGODB_URL}")
                logger.info(f"Using database: {settings.MONGODB_DB_NAME}")
                logger.info(f"Using collection: {settings.CHATS_COLLECTION_NAME}")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {str(e)}")
                logger.warning("Application will continue without MongoDB functionality")
        else:
            logger.warning("MongoDB manager not available, skipping MongoDB initialization")
            logger.info("To enable MongoDB functionality, ensure pymongo is installed: pip install pymongo")
        
        yield
        
        # Shutdown
        logger.info("Shutting down LLM Document Processing System...")
        # Close MongoDB connection
        mongodb_manager.disconnect()

    # Create FastAPI application
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="LLM-powered intelligent document query and retrieval system",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time_module.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        
        response = await call_next(request)
        
        # Log response
        process_time = time_module.time() - start_time
        logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
        
        return response

    # Health check endpoint
    @app.get(f"{settings.API_V1_STR}/health")
    async def health_check():
        return {"status": "healthy", "version": settings.VERSION}

    # Document upload endpoint for processing
    @app.post(f"{settings.API_V1_STR}/documents/upload")
    async def upload_document(file: UploadFile = File(...)):
        try:
            # Save the uploaded file temporarily
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, file.filename)
            
            with open(temp_file_path, "wb") as buffer:
                import shutil
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"File saved to: {temp_file_path}")
            
            # Generate a unique ID for this document
            import uuid
            document_id = str(uuid.uuid4())
            
            # Store document info in memory (in production, use a proper database)
            uploaded_documents[document_id] = {
                "filename": file.filename,
                "file_path": temp_file_path,
                "size": file.size if hasattr(file, 'size') and file.size is not None else 0,
                "upload_time": time.time(),
                "provider": "gemini" if settings.LLM_PROVIDER.lower() == "gemini" and settings.GEMINI_API_KEY else settings.LLM_PROVIDER
            }
            
            # Process the document based on the configured LLM provider
            if settings.LLM_PROVIDER.lower() == "gemini" and settings.GEMINI_API_KEY:
                # For Gemini, we'll return success and let queries be processed directly
                return {
                    "message": f"Document {file.filename} received and ready for Gemini processing",
                    "filename": file.filename,
                    "document_id": document_id,
                    "size": file.size if hasattr(file, 'size') and file.size is not None else 0,
                    "status": "ready",
                    "provider": "gemini"
                }
            else:
                # Default response for other providers
                return {
                    "message": f"Document {file.filename} received and queued for processing",
                    "filename": file.filename,
                    "document_id": document_id,
                    "size": file.size if hasattr(file, 'size') and file.size is not None else 0,
                    "status": "processing"
                }
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "FILE_PROCESSING_ERROR",
                    "message": f"Error processing file: {str(e)}"
                }
            )

    # Document processing endpoint for direct processing with query
    @app.post(f"{settings.API_V1_STR}/documents/process")
    async def process_document_with_query(file: UploadFile = File(...), query: str = ""):
        try:
            # Save the uploaded file temporarily
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, file.filename)
            
            with open(temp_file_path, "wb") as buffer:
                import shutil
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"File saved to: {temp_file_path} for processing with query: {query}")
            
            # Generate a unique ID for this document
            import uuid
            document_id = str(uuid.uuid4())
            
            # Store document info in memory (in production, use a proper database)
            uploaded_documents[document_id] = {
                "filename": file.filename,
                "file_path": temp_file_path,
                "size": file.size if hasattr(file, 'size') and file.size is not None else 0,
                "upload_time": time.time(),
                "provider": "gemini" if settings.LLM_PROVIDER.lower() == "gemini" and settings.GEMINI_API_KEY else settings.LLM_PROVIDER
            }
            
            # Process the document with the requested LLM provider
            if settings.LLM_PROVIDER.lower() == "gemini" and settings.GEMINI_API_KEY:
                # Process using Gemini API with document and query
                try:
                    # Use the query or provide a default summary query
                    actual_query = query if query else "Provide a summary of this document"
                    # Create a temporary file object to pass to the processing function
                    file.file.seek(0)  # Reset file pointer
                    result = process_document_with_gemini(file, actual_query)
                    # Add document_id to the result
                    if isinstance(result, dict) and 'sources' in result:
                        result['document_id'] = document_id
                    
                    response = QueryResponse(**result)
                    
                    # Store the conversation in MongoDB if connection is available
                    if mongodb_manager is not None:
                        try:
                            chat_id = mongodb_manager.save_chat_conversation(
                                user_query=actual_query,
                                response=response.answer,
                                document_id=document_id,
                                sources=response.sources,
                                context=response.context
                            )
                            if chat_id:
                                logger.info(f"Document processing chat successfully saved to MongoDB with ID: {chat_id}")
                            else:
                                logger.warning("Document processing chat was not saved to MongoDB (returned None ID)")
                        except Exception as e:
                            logger.error(f"Error saving document processing chat to MongoDB: {str(e)}", exc_info=True)
                    else:
                        logger.info("MongoDB not available, skipping chat storage for document processing")
                    
                    return response
                except Exception as e:
                    logger.error(f"Error processing document with Gemini: {str(e)}")
                    response = QueryResponse(
                        answer=f"Error processing with Gemini: {str(e)}",
                        context=[],
                        sources=[],
                        document_id=document_id
                    )
                    
                    # Store the error in MongoDB as well
                    if mongodb_manager is not None:
                        try:
                            chat_id = mongodb_manager.save_chat_conversation(
                                user_query=query if query else "Provide a summary of this document",
                                response=response.answer,
                                document_id=document_id,
                                sources=response.sources,
                                context=response.context
                            )
                            if chat_id:
                                logger.info(f"Document processing error chat successfully saved to MongoDB with ID: {chat_id}")
                            else:
                                logger.warning("Document processing error chat was not saved to MongoDB (returned None ID)")
                        except Exception as mongo_e:
                            logger.error(f"Error saving document processing error chat to MongoDB: {str(mongo_e)}", exc_info=True)
                    else:
                        logger.info("MongoDB not available, skipping error chat storage for document processing")
                    
                    return response
            else:
                # Return a default response if no provider is configured
                response = {
                    "message": f"Document {file.filename} received. Configure an LLM provider for processing.",
                    "filename": file.filename,
                    "document_id": document_id,
                    "size": file.size if hasattr(file, 'size') and file.size is not None else 0,
                    "status": "uploaded",
                    "provider": settings.LLM_PROVIDER
                }
                
                # Store the conversation in MongoDB if connection is available
                if mongodb_manager is not None:
                    try:
                        chat_id = mongodb_manager.save_chat_conversation(
                            user_query=query if query else "Provide a summary of this document",
                            response=response["message"],
                            document_id=document_id,
                            sources=[file.filename],
                            context=["Document upload without processing"]
                        )
                        if chat_id:
                            logger.info(f"Document upload chat successfully saved to MongoDB with ID: {chat_id}")
                        else:
                            logger.warning("Document upload chat was not saved to MongoDB (returned None ID)")
                    except Exception as e:
                        logger.error(f"Error saving document upload chat to MongoDB: {str(e)}", exc_info=True)
                else:
                    logger.info("MongoDB not available, skipping chat storage for document upload")
                
                return response
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "FILE_PROCESSING_ERROR",
                    "message": f"Error processing file: {str(e)}"
                }
            )

    # Add endpoint to list uploaded documents
    @app.get(f"{settings.API_V1_STR}/documents/list")
    async def list_documents():
        """List all uploaded documents"""
        return {"documents": uploaded_documents}

    # Add endpoint to list Gemini models
    @app.get(f"{settings.API_V1_STR}/models/gemini")
    async def list_gemini_models():
        """List all available Gemini models"""
        if not settings.GEMINI_API_KEY:
            return {"error": "GEMINI_API_KEY not configured", "models": []}
        
        models = list_available_gemini_models()
        return {"models": models, "default_model": settings.GEMINI_MODEL}

    # Add endpoint to get stored chat conversations
    @app.get(f"{settings.API_V1_STR}/chats")
    async def get_chats(limit: int = 100):
        """Retrieve stored chat conversations from MongoDB"""
        if mongodb_manager is not None:
            try:
                chats = mongodb_manager.get_all_chats(limit=limit)
                return {"chats": chats, "total": len(chats)}
            except Exception as e:
                logger.error(f"Error retrieving chats: {str(e)}")
                return {"chats": [], "total": 0, "error": str(e)}
        else:
            logger.warning("MongoDB not available, returning empty chat list")
            return {"chats": [], "total": 0, "error": "MongoDB not available"}

    # Add endpoint to get chats by document ID
    @app.get(f"{settings.API_V1_STR}/chats/document/{{document_id}}")
    async def get_chats_by_document(document_id: str):
        """Retrieve chat conversations related to a specific document"""
        if mongodb_manager is not None:
            try:
                chats = mongodb_manager.get_chats_by_document_id(document_id)
                return {"chats": chats, "total": len(chats), "document_id": document_id}
            except Exception as e:
                logger.error(f"Error retrieving chats for document {document_id}: {str(e)}")
                return {"chats": [], "total": 0, "document_id": document_id, "error": str(e)}
        else:
            logger.warning("MongoDB not available, returning empty chat list for document")
            return {"chats": [], "total": 0, "document_id": document_id, "error": "MongoDB not available"}

    # Query endpoint with document processing support
    @app.post(f"{settings.API_V1_STR}/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        logger.info(f"Received query: {request.query}")
        
        # If we have a specific document to query and Gemini is configured, use it
        if settings.LLM_PROVIDER.lower() == "gemini" and settings.GEMINI_API_KEY:
            response = await query_documents_with_gemini(request)
        elif settings.LLM_PROVIDER.lower() == "openai" and settings.OPENAI_API_KEY:
            response = await query_documents_with_openai(request)
        elif settings.LLM_PROVIDER.lower() == "groq" and settings.GROQ_API_KEY:
            response = await query_documents_with_groq(request)
        else:
            # Default response if no provider is configured
            response = QueryResponse(
                answer=f"This is a simulated response to your query: '{request.query}'. Configured provider: {settings.LLM_PROVIDER}",
                context=["Simulated context from document chunks"],
                sources=["Simulated document source"]
            )
        
        # Store the conversation in MongoDB if connection is available
        if mongodb_manager is not None:
            try:
                chat_id = mongodb_manager.save_chat_conversation(
                    user_query=request.query,
                    response=response.answer,
                    document_id=request.document_id,
                    sources=response.sources,
                    context=response.context
                )
                if chat_id:
                    logger.info(f"Chat conversation successfully saved to MongoDB with ID: {chat_id}")
                else:
                    logger.warning("Chat conversation was not saved to MongoDB (returned None ID)")
            except Exception as e:
                logger.error(f"Error saving chat to MongoDB: {str(e)}", exc_info=True)
        else:
            logger.info("MongoDB not available, skipping chat storage")
        
        return response

    # Specific query endpoints for each provider
    async def query_documents_with_gemini(request: QueryRequest):
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = get_compatible_gemini_model_instance()
            
            # Check if document_id was provided
            if request.document_id and request.document_id in uploaded_documents:
                # Get the document file path
                doc_info = uploaded_documents[request.document_id]
                file_path = doc_info['file_path']
                
                # Use the centralized upload helper so .docx and other conversions
                # are handled consistently (upload_file_to_gemini will convert
                # .docx -> .txt when needed and wait for processing).
                uploaded_file = upload_file_to_gemini(file_path)

                # Generate content with the uploaded file and query
                response = model.generate_content([uploaded_file, request.query])
                response_text = response.text if response else f"Unable to process query: {request.query}"
                
                # Clean up the uploaded file from Gemini (optional, depending on use case)
                # genai_module.delete_file(uploaded_file.name)
                
                return QueryResponse(
                    answer=response_text,
                    context=[f"Response based on document: {doc_info['filename']}"],
                    sources=[doc_info['filename']],
                    document_id=request.document_id
                )
            else:
                # If no document is specified, just respond to the query generically
                response = model.generate_content(request.query)
                response_text = response.text if response else f"Unable to process query: {request.query}"
                
                return QueryResponse(
                    answer=response_text,
                    context=["General Gemini response context"],
                    sources=["Gemini API"],
                    document_id=request.document_id
                )
        except Exception as e:
            logger.error(f"Error querying with Gemini: {str(e)}")
            return QueryResponse(
                answer=f"Error with Gemini: {str(e)}",
                context=[],
                sources=[],
                document_id=request.document_id
            )
    
    async def query_documents_with_openai(request: QueryRequest):
        try:
            # For this simplified version, return a placeholder response
            # In a real implementation, you'd connect the document to the query
            context_info = []
            if request.document_id and request.document_id in uploaded_documents:
                doc_info = uploaded_documents[request.document_id]
                context_info.append(f"Query related to document: {doc_info['filename']}")
            
            return QueryResponse(
                answer=f"OpenAI response: Processing query '{request.query}' using OpenAI.",
                context=context_info,
                sources=["OpenAI API"],
                document_id=request.document_id
            )
        except Exception as e:
            logger.error(f"Error querying with OpenAI: {str(e)}")
            return QueryResponse(
                answer=f"Error with OpenAI: {str(e)}",
                context=[],
                sources=[],
                document_id=request.document_id
            )
    
    async def query_documents_with_groq(request: QueryRequest):
        try:
            # For this simplified version, return a placeholder response
            # In a real implementation, you'd connect the document to the query
            context_info = []
            if request.document_id and request.document_id in uploaded_documents:
                doc_info = uploaded_documents[request.document_id]
                context_info.append(f"Query related to document: {doc_info['filename']}")
            
            return QueryResponse(
                answer=f"Groq response: Processing query '{request.query}' using Groq.",
                context=context_info,
                sources=["Groq API"],
                document_id=request.document_id
            )
        except Exception as e:
            logger.error(f"Error querying with Groq: {str(e)}")
            return QueryResponse(
                answer=f"Error with Groq: {str(e)}",
                context=[],
                sources=[],
                document_id=request.document_id
            )

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "LLM Document Processing System",
            "version": settings.VERSION,
            "docs_url": "/docs",
            "health_check": f"{settings.API_V1_STR}/health"
        }

    # MongoDB health check endpoint
    @app.get(f"{settings.API_V1_STR}/mongodb-health")
    async def mongodb_health():
        """Check MongoDB connection status"""
        if mongodb_manager is not None:
            try:
                if mongodb_manager.client:
                    # Try a simple operation to verify the connection
                    mongodb_manager.client.admin.command('ping')
                    # Check if collection exists by attempting to count documents
                    count = mongodb_manager.chats_collection.count_documents({}) if mongodb_manager.chats_collection else 0
                    return {
                        "status": "healthy",
                        "message": "MongoDB connection is active",
                        "database": settings.MONGODB_DB_NAME,
                        "collection": settings.CHATS_COLLECTION_NAME,
                        "document_count": count
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "message": "MongoDB manager initialized but client not connected"
                    }
            except Exception as e:
                logger.error(f"MongoDB health check failed: {str(e)}")
                return {
                    "status": "unhealthy",
                    "message": f"MongoDB connection error: {str(e)}"
                }
        else:
            return {
                "status": "unhealthy",
                "message": "MongoDB manager not initialized - check if pymongo is installed"
            }

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Global exception handler caught: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": str(exc) if settings.DEBUG else None
            }
        )

    return app

def run_backend():
    """Run the FastAPI backend."""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=False)

def run_frontend():
    """Run the Streamlit frontend."""
    # Check if streamlit is available, install if not
    try:
        import streamlit
        logger.info("Streamlit is available")
    except ImportError:
        logger.info("Installing Streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Run Streamlit app using subprocess for better reliability
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ])

def get_log_content(log_file, lines=50):
    """Get the last N lines from a log file."""
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines_list = f.readlines()
                return ''.join(lines_list[-lines:])
        else:
            return f"Log file {log_file} does not exist."
    except Exception as e:
        return f"Error reading log file: {str(e)}"

def display_logs():
    """Display logs from both backend and frontend."""
    print("\n" + "="*80)
    print("LOGS FOR DEBUGGING")
    print("="*80)
    
    # Backend logs
    print("\n📋 BACKEND (FastAPI) LOGS:")
    print("-" * 40)
    backend_logs = get_log_content("app.log")
    print(backend_logs if backend_logs else "No backend logs found.")
    
    # Frontend logs would be handled differently in a real scenario
    print("\n🖥️  FRONTEND (Streamlit) LOGS:")
    print("-" * 40)
    print("For Streamlit logs, check the terminal where run_project.py is running.")
    
    # System info
    print("\n🔧 SYSTEM INFORMATION:")
    print("-" * 40)
    print(f"LLM Provider: {settings.LLM_PROVIDER}")
    print(f"Gemini API configured: {'Yes' if settings.GEMINI_API_KEY else 'No'}")
    print(f"OpenAI API configured: {'Yes' if settings.has_openai_key else 'No'}")
    print(f"Groq API configured: {'Yes' if settings.GROQ_API_KEY else 'No'}")
    
    print("\n" + "="*80)
    print("To view logs in real-time, use: tail -f app.log")
    print("Press Ctrl+C to return to the application")
    print("="*80)

def main():
    """Main function to run the application."""
    logger.info("Starting FinTech-LLM project setup...")
    
    # Check for --logs flag to display logs
    if "--logs" in sys.argv:
        display_logs()
        return
    
    # Check and install dependencies
    logger.info("Checking dependencies...")
    if not check_and_install_dependencies():
        logger.error("Failed to install required dependencies. Exiting.")
        sys.exit(1)
    
    # Setup directories
    logger.info("Setting up directories...")
    setup_directories()
    
    # Create sample .env file if needed
    logger.info("Checking environment file...")
    create_sample_env_file()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Update settings with loaded environment variables
    settings.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", settings.OPENAI_API_KEY)
    settings.GROQ_API_KEY = os.getenv("GROQ_API_KEY", settings.GROQ_API_KEY)
    settings.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", settings.GEMINI_API_KEY)
    settings.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", settings.PINECONE_API_KEY)
    settings.SECRET_KEY = os.getenv("SECRET_KEY", settings.SECRET_KEY)
    settings.DATABASE_URL = os.getenv("DATABASE_URL", settings.DATABASE_URL)
    settings.REDIS_URL = os.getenv("REDIS_URL", settings.REDIS_URL)
    settings.LLM_PROVIDER = os.getenv("LLM_PROVIDER", settings.LLM_PROVIDER)
    
    # Print startup message
    logger.info("")
    logger.info("="*60)
    logger.info("✅ FinTech-LLM Project starting...")
    logger.info("="*60)
    logger.info("📊 Backend API will be available at: http://localhost:8000")
    logger.info("🖥️  Frontend Interface will be available at: http://localhost:8501")
    logger.info("📄 API Documentation at: http://localhost:8000/docs")
    logger.info("🔍 Health check at: http://localhost:8000/api/v1/health")
    logger.info("📜 To view logs: python run_project.py --logs")
    logger.info("="*60)
    
    # Check API keys
    providers_configured = []
    if settings.has_openai_key:
        providers_configured.append("OpenAI")
    if settings.GEMINI_API_KEY:
        providers_configured.append("Gemini")
    if settings.GROQ_API_KEY:
        providers_configured.append("Groq")
    
    if providers_configured:
        logger.info(f"✅ AI Providers configured: {', '.join(providers_configured)}")
    else:
        logger.warning("⚠️  No AI provider configured. Please set API keys in .env file.")
    
    logger.info(f"🎯 Using provider: {settings.LLM_PROVIDER}")
    
    # Start backend in a separate thread
    import threading
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Give backend a moment to start
    time.sleep(2)
    
    # Run frontend (this will block, so it needs to be last)
    logger.info("Starting Streamlit frontend...")
    run_frontend()

if __name__ == "__main__":
    main()