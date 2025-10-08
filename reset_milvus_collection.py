"""
Script to drop existing Milvus collection and recreate with new embedding dimensions
Run this script when switching embedding models (e.g., from Gemini to HuggingFace)
"""

from pymilvus import connections, utility
from config.settings import settings

def reset_collection():
    """Drop existing collection to allow recreation with new dimensions."""
    try:
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=settings.milvus_host,
            port=settings.milvus_port
        )
        print(f"✅ Connected to Milvus at {settings.milvus_host}:{settings.milvus_port}")
        
        # Check if collection exists
        collection_name = "documents"
        if utility.has_collection(collection_name):
            print(f"📦 Found existing collection: {collection_name}")
            
            # Drop the collection
            utility.drop_collection(collection_name)
            print(f"🗑️  Dropped collection: {collection_name}")
            print(f"✅ Collection will be recreated with new dimensions ({settings.embedding_dim}D) on next startup")
        else:
            print(f"ℹ️  Collection '{collection_name}' does not exist. Nothing to drop.")
        
        # Disconnect
        connections.disconnect("default")
        print("✅ Disconnected from Milvus")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Milvus Collection Reset Script")
    print("=" * 60)
    print(f"Current embedding dimension: {settings.embedding_dim}D")
    print(f"Using HuggingFace: {settings.use_huggingface_embeddings}")
    print(f"Model: {settings.huggingface_model if settings.use_huggingface_embeddings else settings.gemini_embedding_model}")
    print("=" * 60)
    
    confirm = input("\n⚠️  This will delete all existing documents. Continue? (yes/no): ")
    if confirm.lower() == 'yes':
        reset_collection()
    else:
        print("❌ Operation cancelled")
