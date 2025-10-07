"""
Alternative RAG system using only Google Gemini (both LLM and embeddings)
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings as LlamaSettings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from src.milvus_manager import MilvusManager
from src.redis_cache import RedisCache
from src.file_handler import FileUploadHandler
from config.settings import settings

logger = logging.getLogger(__name__)

class GeminiRAGSystem:
    """RAG system using Gemini for both LLM and embeddings."""
    
    def __init__(self):
        self.settings = settings
        
        # Initialize components
        self.milvus_manager = None
        self.redis_cache = None
        self.file_handler = None
        self.llm = None
        self.embed_model = None
        self.index = None
        self.query_engine = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all RAG components."""
        try:
            logger.info("Initializing Gemini RAG system...")
            
            # Initialize Gemini LLM
            self.llm = Gemini(
                model=self.settings.gemini_model,
                api_key=self.settings.google_api_key,
                temperature=0.1
            )
            
            # Initialize Gemini embeddings
            self.embed_model = GeminiEmbedding(
                model_name=self.settings.gemini_embedding_model,
                api_key=self.settings.google_api_key
            )
            
            # Set global LlamaIndex settings
            LlamaSettings.llm = self.llm
            LlamaSettings.embed_model = self.embed_model
            LlamaSettings.chunk_size = self.settings.chunk_size
            LlamaSettings.chunk_overlap = self.settings.chunk_overlap
            
            # Initialize other components
            self.milvus_manager = MilvusManager()
            self.redis_cache = RedisCache()
            self.file_handler = FileUploadHandler()
            
            # Initialize index
            self._initialize_index()
            
            logger.info("✅ Gemini RAG system initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini RAG system: {e}")
            raise
    
    def _initialize_index(self):
        """Initialize vector store index."""
        try:
            vector_store = self.milvus_manager.get_vector_store()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Try to load existing index or create new one
            try:
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context
                )
                logger.info("Loaded existing vector index")
            except:
                # Create empty index
                self.index = VectorStoreIndex(
                    nodes=[],
                    storage_context=storage_context
                )
                logger.info("Created new vector index")
            
            # Initialize query engine
            self._initialize_query_engine()
            
        except Exception as e:
            logger.error(f"Failed to initialize index: {e}")
            raise
    
    def _initialize_query_engine(self):
        """Initialize the query engine."""
        try:
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=self.settings.similarity_top_k
            )
            
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=None  # Use default
            )
            
            logger.info("Query engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize query engine: {e}")
            raise
    
    def upload_document(self, file_path: str, metadata: Optional[Dict] = None) -> bool:
        """Upload and process a document."""
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Check cache first
            cache_key = f"doc_processed:{file_path}"
            if self.redis_cache and self.redis_cache.get_cache(cache_key):
                logger.info("Document already processed (found in cache)")
                return True
            
            # Process document
            documents = self.file_handler.process_document(Path(file_path))
            if not documents:
                logger.error("Failed to process document")
                return False
            
            # Create embeddings and add to index
            text_splitter = SentenceSplitter(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap
            )
            
            nodes = text_splitter.get_nodes_from_documents(documents)
            
            # Add nodes to index (this will generate embeddings using Gemini)
            self.index.insert_nodes(nodes)
            
            # Cache the processing result
            if self.redis_cache:
                self.redis_cache.set_cache(cache_key, True, ttl=3600)
            
            logger.info(f"✅ Successfully processed document with {len(nodes)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload document: {e}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system."""
        try:
            logger.info(f"Processing query: {question}")
            
            # Check cache first
            cache_key = f"query:{question}"
            if self.redis_cache:
                cached_result = self.redis_cache.get_cache(cache_key)
                if cached_result:
                    logger.info("Query result found in cache")
                    return cached_result
            
            # Generate response using Gemini
            response = self.query_engine.query(question)
            
            # Prepare result
            result = {
                "answer": str(response),
                "sources": [],
                "metadata": {
                    "model": "gemini-flash-1.5",
                    "embedding_model": "gemini-embedding-001",
                    "cached": False
                }
            }
            
            # Extract source information
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    source_info = {
                        "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "score": getattr(node, 'score', 0.0),
                        "metadata": node.metadata
                    }
                    result["sources"].append(source_info)
            
            # Cache the result
            if self.redis_cache:
                self.redis_cache.set_cache(cache_key, result, ttl=self.settings.redis_ttl)
            
            logger.info("✅ Query processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "sources": [],
                "metadata": {"error": True}
            }
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        try:
            logger.info(f"Searching documents for: {query}")
            
            # Check cache
            cache_key = f"search:{query}:{top_k}"
            if self.redis_cache:
                cached_result = self.redis_cache.get_cache(cache_key)
                if cached_result:
                    return cached_result
            
            # Generate query embedding using Gemini
            query_embedding = self.embed_model.get_query_embedding(query)
            
            # Search in Milvus
            similar_docs = self.milvus_manager.search_similar(query_embedding, top_k)
            
            # Cache results
            if self.redis_cache:
                self.redis_cache.set_cache(cache_key, similar_docs, ttl=1800)
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "backend": "Gemini (Full)",
            "llm_model": self.settings.gemini_model,
            "embedding_model": self.settings.gemini_embedding_model,
            "milvus_connected": self.milvus_manager.is_connected() if self.milvus_manager else False,
            "redis_connected": self.redis_cache.is_connected() if self.redis_cache else False,
            "cache_stats": {}
        }
        
        # Add Milvus stats
        if self.milvus_manager:
            milvus_stats = self.milvus_manager.get_collection_stats()
            if milvus_stats:
                stats["milvus_stats"] = milvus_stats
        
        # Add Redis stats
        if self.redis_cache:
            cache_stats = self.redis_cache.get_stats()
            if cache_stats:
                stats["cache_stats"] = cache_stats
        
        return stats
    
    def is_ready(self) -> bool:
        """Check if the system is ready to use."""
        return (
            self.llm is not None and
            self.embed_model is not None and
            self.index is not None and
            self.query_engine is not None and
            self.milvus_manager is not None  # Allow fallback vector store
        )