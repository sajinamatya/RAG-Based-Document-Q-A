import os
import logging
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from config.settings import settings

logger = logging.getLogger(__name__)

class MilvusManager:
    """Manages Milvus vector database operations."""
    
    def __init__(self):
        self.settings = settings
        self.collection_name = "documents"
        self.collection = None
        self.vector_store = None
        self.connection_alias = "default"
        self._connect()
    
    def _connect(self):
        """Connect to Milvus with improved connection handling."""
        try:
            # Check if connection already exists
            existing_connections = connections.list_connections()
            
            if self.connection_alias in existing_connections:
                logger.info(f"Using existing Milvus connection: {self.connection_alias}")
            else:
                # Create new connection
                connections.connect(
                    alias=self.connection_alias,
                    host=self.settings.milvus_host,
                    port=self.settings.milvus_port
                )
                logger.info("Created new Milvus connection successfully")
            
            # Test connection
            utility.get_server_version(using=self.connection_alias)
            logger.info("Connected to Milvus successfully")
            
            # Initialize collection and vector store
            self._initialize_collection()
            self._initialize_vector_store()
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _initialize_vector_store(self):
        """Initialize vector store with existing connection."""
        try:
            # Use proper URI format with tcp protocol for remote Milvus connection
            milvus_uri = f"tcp://{self.settings.milvus_host}:{self.settings.milvus_port}"
            
            self.vector_store = MilvusVectorStore(
                uri=milvus_uri,
                collection_name=self.collection_name,
                dim=1536,
                overwrite=False,
                token=""  # Empty token for non-Zilliz connections
            )
            logger.info(f"Vector store initialized with URI: {milvus_uri}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            # Fall back to direct collection usage
            self.vector_store = None
    
    def _initialize_collection(self):
        """Initialize or get existing collection."""
        try:
            if utility.has_collection(self.collection_name, using=self.connection_alias):
                self.collection = Collection(self.collection_name, using=self.connection_alias)
                logger.info(f"Using existing collection: {self.collection_name}")
            else:
                self._create_collection()
            
            # Load collection to memory
            if not self.collection.has_index():
                self._create_index()
            
            self.collection.load()
            logger.info(f"Collection {self.collection_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def _create_collection(self):
        """Create collection with schema."""
        try:
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=65535, is_primary=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
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
        """Create index for the collection."""
        try:
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info("Created collection index")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def is_connected(self) -> bool:
        """Check if Milvus is connected."""
        try:
            if self.connection_alias not in connections.list_connections():
                return False
            utility.get_server_version(using=self.connection_alias)
            return True
        except:
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the collection."""
        try:
            if not self.is_connected() or not self.collection:
                logger.error("Milvus not connected or collection not initialized")
                return False
            
            # Prepare data for insertion
            ids = [doc["id"] for doc in documents]
            texts = [doc["text"] for doc in documents]
            embeddings = [doc["embedding"] for doc in documents]
            metadata = [doc.get("metadata", {}) for doc in documents]
            
            # Insert data
            self.collection.insert([ids, texts, embeddings, metadata])
            self.collection.flush()
            
            logger.info(f"Added {len(documents)} documents to Milvus")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        try:
            if not self.is_connected() or not self.collection:
                logger.error("Milvus not connected or collection not initialized")
                return []
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "metadata"]
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
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def get_collection_stats(self):
        """Get collection statistics."""
        try:
            if not self.is_connected() or not self.collection:
                return None
                
            # Refresh collection stats
            self.collection.flush()
            
            stats = {
                "entity_count": self.collection.num_entities,
                "collection_name": self.collection_name,
                "connection_status": "Connected",
                "has_index": self.collection.has_index(),
                "is_loaded": True
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return None
    
    def get_vector_store(self):
        """Get the vector store for LlamaIndex."""
        if self.vector_store:
            return self.vector_store
        else:
            # Return a simple wrapper if vector store initialization failed
            logger.warning("Using fallback vector store")
            return self._create_fallback_vector_store()
    
    def _create_fallback_vector_store(self):
        """Create a fallback vector store that uses the collection directly."""
        class FallbackVectorStore:
            def __init__(self, milvus_manager):
                self.manager = milvus_manager
            
            def add(self, nodes):
                # Convert nodes to documents format
                documents = []
                for node in nodes:
                    documents.append({
                        "id": node.node_id,
                        "text": node.text,
                        "embedding": node.embedding,
                        "metadata": node.metadata
                    })
                return self.manager.add_documents(documents)
        
        return FallbackVectorStore(self)
    
    def disconnect(self):
        """Disconnect from Milvus."""
        try:
            if self.connection_alias in connections.list_connections():
                connections.disconnect(self.connection_alias)
                logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")