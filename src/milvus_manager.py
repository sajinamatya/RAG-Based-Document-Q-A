from typing import List, Optional, Dict, Any
import logging

from llama_index.core import Document
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.openai import OpenAIEmbedding
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

from config.settings import settings

logger = logging.getLogger(__name__)


class MilvusManager:
    """Manages Milvus vector database operations."""
    
    def __init__(self):
        self.settings = settings
        self.connection_alias = "default"
        self.collection_name = self.settings.milvus_collection_name
        self.dim = self.settings.embedding_dimension
        self.vector_store = None
        self._connect()
        self._create_collection_if_not_exists()
    
    def _connect(self):
        """Connect to Milvus database."""
        try:
            # Check if connection already exists
            if self.connection_alias in connections.list_connections():
                connections.remove_connection(self.connection_alias)
            
            connections.connect(
                alias=self.connection_alias,
                host=self.settings.milvus_host,
                port=self.settings.milvus_port,
                user=self.settings.milvus_user or "",
                password=self.settings.milvus_password or "",
                secure=False  # Set to False for local connections
            )
            logger.info("Connected to Milvus successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _create_collection_if_not_exists(self):
        """Create collection if it doesn't exist."""
        if not utility.has_collection(self.collection_name):
            # Define collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]
            
            schema = CollectionSchema(fields=fields, description="LlamaIndex documents")
            
            # Create collection
            collection = Collection(
                name=self.collection_name,
                schema=schema,
                using=self.connection_alias
            )
            
            # Create index for vector field
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            logger.info(f"Created collection '{self.collection_name}' with index")
        
        # Initialize vector store using URI (for Docker Milvus)
        self.vector_store = MilvusVectorStore(
            uri=f"http://{self.settings.milvus_host}:{self.settings.milvus_port}",
            collection_name=self.collection_name,
            dim=self.dim,
            overwrite=False
        )
    
    def get_vector_store(self) -> MilvusVectorStore:
        """Get the vector store instance."""
        return self.vector_store
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        try:
            node_ids = []
            for doc in documents:
                # Add document to vector store
                self.vector_store.add([doc])
                node_ids.append(doc.node_id if hasattr(doc, 'node_id') else str(id(doc)))
            
            logger.info(f"Added {len(documents)} documents to Milvus")
            return node_ids
        except Exception as e:
            logger.error(f"Error adding documents to Milvus: {e}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
        """Search for similar documents."""
        try:
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k
            )
            
            result = self.vector_store.query(query)
            return [
                {
                    "node": node,
                    "score": score
                }
                for node, score in zip(result.nodes, result.similarities)
            ]
        except Exception as e:
            logger.error(f"Error searching Milvus: {e}")
            raise
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents by IDs."""
        try:
            collection = Collection(self.collection_name)
            collection.delete(expr=f'id in {doc_ids}')
            logger.info(f"Deleted {len(doc_ids)} documents from Milvus")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents from Milvus: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            collection = Collection(self.collection_name)
            collection.load()
            stats = {
                "total_documents": collection.num_entities,
                "collection_name": self.collection_name,
                "dimension": self.dim
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            collection = Collection(self.collection_name)
            collection.drop()
            self._create_collection_if_not_exists()
            logger.info("Cleared collection successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False