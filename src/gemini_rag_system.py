"""
Alternative RAG system using only Google Gemini (both LLM and embeddings)
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import Document, Settings

from src.milvus_manager import MilvusManager
from src.redis_cache import RedisCache
from src.file_handler import FileUploadHandler
from config.settings import settings

logger = logging.getLogger(__name__)


class GeminiRAGSystem:
    """RAG System using Google Gemini for both LLM and embeddings."""
    
    def __init__(self):
        self.settings = settings
        
        # Initialize Milvus vector store
        self.milvus_manager = MilvusManager()
        self.cache = RedisCache()
        self.file_handler = FileUploadHandler()
        
        # Initialize Gemini LLM and embeddings
        self.llm = Gemini(
            api_key=self.settings.google_api_key,
            model=self.settings.gemini_model,
            temperature=0.1
        )
        
        # Use Gemini for embeddings too
        self.embed_model = GeminiEmbedding(
            api_key=self.settings.google_api_key,
            model_name="models/text-embedding-004"  # Latest Gemini embedding model
        )
        
        # Configure global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        # Initialize index
        self._initialize_index()
        
        # Initialize query engine
        self.query_engine = None
        self._setup_query_engine()
    
    def _initialize_index(self):
        """Initialize Milvus vector store index."""
        try:
            vector_store = self.milvus_manager.get_vector_store()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
            logger.info("Milvus vector store index initialized successfully with Gemini")
        except Exception as e:
            logger.error(f"Error initializing Milvus vector store index: {e}")
            raise
    
    def _setup_query_engine(self):
        """Setup the query engine with retriever and post-processor."""
        try:
            # Create retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=10
            )
            
            # Create post-processor for similarity filtering
            postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
            
            # Create query engine
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                node_postprocessors=[postprocessor]
            )
            
            logger.info("Gemini query engine setup completed")
        except Exception as e:
            logger.error(f"Error setting up query engine: {e}")
            raise
    
    # The rest of the methods remain the same as the main RAG system
    def upload_and_process_file(self, filename: str, file_content: bytes) -> Dict[str, Any]:
        """Upload and process a file for RAG."""
        try:
            # Save uploaded file
            file_path = self.file_handler.save_uploaded_file(filename, file_content)
            logger.info(f"File saved: {file_path}")
            
            # Process document
            documents = self.file_handler.process_document(file_path)
            logger.info(f"Processed {len(documents)} document chunks")
            
            # Add documents to Milvus vector store
            node_ids = self.milvus_manager.add_documents(documents)
            
            # Cache document metadata
            file_metadata = {
                "filename": filename,
                "file_path": str(file_path),
                "document_count": len(documents),
                "node_ids": node_ids
            }
            
            self.cache.cache_document_metadata(str(file_path), file_metadata)
            
            # Refresh index to include new documents
            self._initialize_index()
            self._setup_query_engine()
            
            return {
                "success": True,
                "message": f"Successfully processed {filename} with Gemini",
                "file_path": str(file_path),
                "document_count": len(documents),
                "node_ids": node_ids
            }
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def query(self, question: str, use_cache: bool = True) -> Dict[str, Any]:
        """Query the RAG system using Gemini."""
        try:
            # Check cache first
            if use_cache and self.cache.is_connected():
                cached_results = self.cache.get_cached_search_results(question)
                if cached_results:
                    logger.info("Retrieved results from cache")
                    return {
                        "success": True,
                        "response": cached_results.get("response", ""),
                        "source_nodes": cached_results.get("source_nodes", []),
                        "cached": True,
                        "model": "Gemini (cached)"
                    }
            
            # Get embedding for the question (using Gemini embeddings)
            question_embedding = None
            if self.cache.is_connected():
                question_embedding = self.cache.get_cached_embedding(question)
            
            if not question_embedding:
                question_embedding = self.embed_model.get_text_embedding(question)
                if self.cache.is_connected():
                    self.cache.cache_embedding(question, question_embedding)
            
            # Query the engine (will use Gemini LLM)
            if not self.query_engine:
                raise ValueError("Query engine not initialized")
            
            response = self.query_engine.query(question)
            
            # Process response
            result = {
                "success": True,
                "response": str(response),
                "source_nodes": [],
                "cached": False,
                "model": f"Gemini {self.settings.gemini_model}"
            }
            
            # Extract source information
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    source_info = {
                        "text": node.node.text[:500] + "..." if len(node.node.text) > 500 else node.node.text,
                        "score": getattr(node, 'score', 0.0),
                        "metadata": node.node.metadata
                    }
                    result["source_nodes"].append(source_info)
            
            # Cache the results
            if use_cache and self.cache.is_connected():
                cache_data = {
                    "response": result["response"],
                    "source_nodes": result["source_nodes"]
                }
                self.cache.cache_search_results(question, cache_data)
            
            logger.info(f"Gemini query processed successfully: {question[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query with Gemini: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed documents."""
        try:
            milvus_stats = self.milvus_manager.get_collection_stats()
            cache_stats = self.cache.get_cache_stats()
            
            return {
                "milvus": milvus_stats,
                "cache": cache_stats,
                "uploaded_files": len(self.file_handler.get_uploaded_files()),
                "backend": "Milvus + Gemini",
                "llm_model": self.settings.gemini_model,
                "embedding_model": "text-embedding-004"
            }
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {"error": str(e)}