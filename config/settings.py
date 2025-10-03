import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    
    # Google Gemini (for LLM)
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    
    # Milvus
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_user: Optional[str] = os.getenv("MILVUS_USER")
    milvus_password: Optional[str] = os.getenv("MILVUS_PASSWORD")
    milvus_db_name: str = os.getenv("MILVUS_DB_NAME", "llamaindex")
    milvus_collection_name: str = os.getenv("MILVUS_COLLECTION_NAME", "documents")
    
    # Redis
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    
    # Application
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1024"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    
    # Cache
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    enable_cache: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Allow extra fields in .env file

settings = Settings()