import os
import logging
import time
from typing import List, Dict, Any, Optional
from threading import Lock
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from config.settings import settings

logger = logging.getLogger(__name__)

class MilvusManager:
    """Manages Milvus vector database operations with improved error handling and reconnection logic."""
    
    def __init__(self):
        self.settings = settings
        self.collection_name = "documents"
        self.collection = None
        self.vector_store = None
        self.connection_alias = "default"
        self._connection_lock = Lock()
        self._max_retries = 3
        self._retry_delay = 2  # seconds
        self._batch_size = 1000
        self._connect()
    
    def _connect(self):
        """Connect to Milvus with improved connection handling."""
        with self._connection_lock:
            try:
                # Check if connection already exists and is alive
                existing_connections = connections.list_connections()
                
                if self.connection_alias in existing_connections:
                    try:
                        utility.get_server_version(using=self.connection_alias)
                        logger.info(f"Using existing Milvus connection: {self.connection_alias}")
                    except Exception:
                        # Connection exists but is dead, disconnect and reconnect
                        logger.warning("Existing connection is dead, reconnecting...")
                        connections.disconnect(self.connection_alias)
                        self._create_new_connection()
                else:
                    # Create new connection
                    self._create_new_connection()
                
                # Verify connection
                utility.get_server_version(using=self.connection_alias)
                logger.info("Connected to Milvus successfully")
                
                # Initialize collection and vector store
                self._initialize_collection()
                self._initialize_vector_store()
                
            except Exception as e:
                logger.error(f"Failed to connect to Milvus: {e}")
                raise
    
    def _create_new_connection(self):
        """Create a new Milvus connection."""
        connections.connect(
            alias=self.connection_alias,
            host=self.settings.milvus_host,
            port=self.settings.milvus_port
        )
        logger.info("Created new Milvus connection successfully")
    
    def _reconnect(self) -> bool:
        """Attempt to reconnect to Milvus."""
        try:
            logger.info("Attempting to reconnect to Milvus...")
            # Disconnect existing connection if any
            if self.connection_alias in connections.list_connections():
                try:
                    connections.disconnect(self.connection_alias)
                except Exception:
                    pass
            
            # Reconnect
            self._connect()
            return True
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False
    
    def _initialize_vector_store(self):
        """Initialize vector store with existing connection."""
        # Skip MilvusVectorStore initialization due to async connection issues in Streamlit
        # The fallback vector store works perfectly with our established pymilvus connection
        logger.info("Using fallback vector store with direct collection access")
        self.vector_store = None
    
    def _initialize_collection(self):
        """Initialize or get existing collection."""
        try:
            if utility.has_collection(self.collection_name, using=self.connection_alias):
                self.collection = Collection(self.collection_name, using=self.connection_alias)
                logger.info(f"Using existing collection: {self.collection_name}")
            else:
                self._create_collection()
            
            # Create index if it doesn't exist
            if not self.collection.has_index():
                self._create_index()
            
            # Load collection to memory
            self.collection.load()
            logger.info(f"Collection {self.collection_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def _create_collection(self):
        """Create collection with schema."""
        try:
            # Define schema with optimized field sizes
            # Using embedding_dim from settings (384 for HuggingFace, 1536 for Gemini)
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.settings.embedding_dim),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="RAG document collection",
                enable_dynamic_field=True
            )
            
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using=self.connection_alias
            )
            
            logger.info(f"Created collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def _create_index(self):
        """Create index for the collection with adaptive parameters."""
        try:
            # Determine optimal nlist based on collection size
            num_entities = self.collection.num_entities
            nlist = max(128, min(16384, int((num_entities ** 0.5) * 4))) if num_entities > 0 else 128
            
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": nlist}
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info(f"Created collection index with nlist={nlist}")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def _execute_with_retry(self, operation, *args, **kwargs):
        """Execute an operation with retry logic."""
        last_exception = None
        
        for attempt in range(self._max_retries):
            try:
                # Check connection before operation
                if not self.is_connected():
                    logger.warning(f"Connection lost, attempting reconnection (attempt {attempt + 1}/{self._max_retries})")
                    if not self._reconnect():
                        raise ConnectionError("Failed to reconnect to Milvus")
                
                # Execute operation
                return operation(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Operation failed (attempt {attempt + 1}/{self._max_retries}): {e}")
                
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay)
                    # Try to reconnect on failure
                    self._reconnect()
        
        # All retries exhausted
        logger.error(f"Operation failed after {self._max_retries} attempts")
        raise last_exception
    
    def is_connected(self) -> bool:
        """Check if Milvus is connected and responsive."""
        try:
            if self.connection_alias not in connections.list_connections():
                return False
            utility.get_server_version(using=self.connection_alias)
            return self.collection is not None
        except Exception:
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the collection with batch processing."""
        def _add_batch(batch):
            if not self.collection:
                raise RuntimeError("Collection not initialized")
            
            # Prepare data for insertion
            ids = [doc["id"] for doc in batch]
            texts = [doc["text"] for doc in batch]
            embeddings = [doc["embedding"] for doc in batch]
            metadata = [doc.get("metadata", {}) for doc in batch]
            
            # Insert data
            self.collection.insert([ids, texts, embeddings, metadata])
            self.collection.flush()
            
            logger.info(f"Added {len(batch)} documents to Milvus")
        
        try:
            # Process in batches
            for i in range(0, len(documents), self._batch_size):
                batch = documents[i:i + self._batch_size]
                self._execute_with_retry(_add_batch, batch)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents with optional filtering."""
        def _search():
            if not self.collection:
                raise RuntimeError("Collection not initialized")
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Build filter expression if provided
            expr = None
            if filters:
                filter_expressions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_expressions.append(f'metadata["{key}"] == "{value}"')
                    else:
                        filter_expressions.append(f'metadata["{key}"] == {value}')
                expr = " && ".join(filter_expressions)
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "metadata"],
                expr=expr
            )
            
            # Process results
            similar_docs = []
            for hits in results:
                for hit in hits:
                    similar_docs.append({
                        "id": hit.id,
                        "text": hit.entity.get("text"),
                        "metadata": hit.entity.get("metadata"),
                        "score": hit.score
                    })
            
            return similar_docs
        
        try:
            return self._execute_with_retry(_search)
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs."""
        def _delete():
            if not self.collection:
                raise RuntimeError("Collection not initialized")
            
            # Build expression for deletion
            id_list = ', '.join([f'"{doc_id}"' for doc_id in document_ids])
            expr = f"id in [{id_list}]"
            
            self.collection.delete(expr)
            self.collection.flush()
            logger.info(f"Deleted {len(document_ids)} documents")
        
        try:
            self._execute_with_retry(_delete)
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def update_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Update documents by deleting and re-inserting."""
        try:
            # Extract IDs
            doc_ids = [doc["id"] for doc in documents]
            
            # Delete existing documents
            if not self.delete_documents(doc_ids):
                return False
            
            # Insert updated documents
            return self.add_documents(documents)
            
        except Exception as e:
            logger.error(f"Failed to update documents: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            if self.collection:
                # Delete all entities
                self.collection.delete(expr="id != ''")
                self.collection.flush()
                logger.info("Cleared all documents from collection")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    def get_collection_stats(self) -> Optional[Dict]:
        """Get collection statistics."""
        def _get_stats():
            if not self.collection:
                return None
            
            # Refresh collection stats
            self.collection.flush()
            
            stats = {
                "entity_count": self.collection.num_entities,
                "collection_name": self.collection_name,
                "connection_status": "Connected" if self.is_connected() else "Disconnected",
                "has_index": self.collection.has_index(),
                "is_loaded": self.collection.is_loaded if hasattr(self.collection, 'is_loaded') else True
            }
            return stats
        
        try:
            return self._execute_with_retry(_get_stats)
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return None
    
    def get_vector_store(self):
        """Get the vector store for LlamaIndex."""
        if self.vector_store:
            return self.vector_store
        else:
            logger.warning("Using fallback vector store")
            return self._create_fallback_vector_store()
    
    def _create_fallback_vector_store(self):
        """Create a fallback vector store that uses the collection directly."""
        class FallbackVectorStore:
            def __init__(self, milvus_manager):
                self.manager = milvus_manager
                # Required attributes for LlamaIndex compatibility
                self.stores_text = True
                self.is_embedding_query = True
            
            def add(self, nodes):
                """Add nodes to the vector store."""
                documents = []
                node_ids = []
                for node in nodes:
                    node_ids.append(node.node_id)
                    documents.append({
                        "id": node.node_id,
                        "text": node.text,
                        "embedding": node.embedding,
                        "metadata": node.metadata
                    })
                # LlamaIndex expects a list of node IDs to be returned
                success = self.manager.add_documents(documents)
                return node_ids if success else []
            
            def query(self, query_embedding, top_k=5):
                """Query the vector store."""
                return self.manager.search_similar(query_embedding, top_k)
            
            def delete(self, node_ids):
                """Delete nodes from the vector store."""
                return self.manager.delete_documents(node_ids)
        
        return FallbackVectorStore(self)
    
    def reconnect_if_needed(self):
        """Public method to manually trigger reconnection if needed."""
        if not self.is_connected():
            return self._reconnect()
        return True
    
    def disconnect(self):
        """Disconnect from Milvus."""
        with self._connection_lock:
            try:
                if self.connection_alias in connections.list_connections():
                    connections.disconnect(self.connection_alias)
                    logger.info("Disconnected from Milvus")
                self.collection = None
                self.vector_store = None
            except Exception as e:
                logger.error(f"Error disconnecting from Milvus: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()