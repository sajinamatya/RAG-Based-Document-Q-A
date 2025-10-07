import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings."""
    
    # Google AI Configuration
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "models/embedding-001"
    
    
    # Milvus Configuration
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "documents"
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    redis_ttl: int = 3600
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600
    
    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    max_file_size_mb: int = 50
    
    # Retrieval Configuration
    similarity_top_k: int = 5
    response_mode: str = "compact"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

# Create global settings instance
settings = Settings()