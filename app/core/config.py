import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "LLM Document Processing System"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "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")
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    
    # Groq Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "llama3-8b-8192"
    
    # Select which provider to use (groq, openai, auto)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "pcsk_74W9e2_Ph5MYgYX8q4BsBdhVkjaMHFeUVLsVq2JM8HNyvsEsPRPEC3i5aWGgdcq3wpugHo")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    PINECONE_INDEX_NAME: str = "document-retrieval"
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")
    
    # MongoDB Configuration
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "fintech_llm")
    CHATS_COLLECTION_NAME: str = os.getenv("CHATS_COLLECTION_NAME", "chats")
    
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
    
    # Cache Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL: int = 3600  # 1 hour
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"
    }

settings = Settings()

# Keep ALLOWED_EXTENSIONS as an internal constant (not read from environment).
# Attach it to the settings instance for backward compatibility with code
# that references `settings.ALLOWED_EXTENSIONS`.
# Pydantic BaseSettings disallows setting unknown fields via attribute assignment
# after construction. Use object.__setattr__ to attach this runtime-only
# constant to the settings instance without invoking pydantic's validators.
object.__setattr__(settings, 'ALLOWED_EXTENSIONS', {".pdf", ".docx", ".txt", ".eml"})

