#!/usr/bin/env python3
"""
Main CLI interface for Gemini RAG system with full Gemini embeddings.
"""

import argparse
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.gemini_rag_system import GeminiRAGSystem
from src.logging_config import setup_logging
import logging

logger = logging.getLogger(__name__)

class GeminiCLI:
    """Command-line interface for Gemini RAG system."""
    
    def __init__(self):
        setup_logging()
        self.rag_system = None
        self._initialize_rag_system()
    
    def _initialize_rag_system(self):
        """Initialize the Gemini RAG system."""
        try:
            print("🤖 Initializing Gemini RAG System...")
            self.rag_system = GeminiRAGSystem()
            
            if self.rag_system.is_ready():
                print("✅ Gemini RAG System initialized successfully!")
                print(f"🧠 LLM: {self.rag_system.settings.gemini_model}")
                print(f"🔍 Embeddings: {self.rag_system.settings.gemini_embedding_model}")
            else:
                print("❌ RAG system initialization incomplete")
                
        except Exception as e:
            print(f"❌ Failed to initialize RAG system: {e}")
            sys.exit(1)
    
    def upload_document(self, file_path: str):
        """Upload a document."""
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        print(f"📄 Uploading document: {file_path}")
        success = self.rag_system.upload_document(file_path)
        
        if success:
            print("✅ Document uploaded successfully!")
        else:
            print("❌ Failed to upload document")
    
    def query_system(self, question: str):
        """Query the RAG system."""
        print(f"❓ Question: {question}")
        print("🤔 Thinking with Gemini...")
        
        result = self.rag_system.query(question)
        
        print("\n" + "="*50)
        print("🤖 Gemini's Answer:")
        print("="*50)
        print(result["answer"])
        
        if result.get("sources"):
            print("\n📚 Sources:")
            for i, source in enumerate(result["sources"], 1):
                print(f"\n{i}. {source['text']}")
                if source.get("score"):
                    print(f"   Relevance: {source['score']:.3f}")
        
        print("\n" + "="*50)
        
        # Show model info
        metadata = result.get("metadata", {})
        print(f"🧠 Model: {metadata.get('model', 'Unknown')}")
        print(f"🔍 Embeddings: {metadata.get('embedding_model', 'Unknown')}")
        if metadata.get("cached"):
            print("⚡ Result was cached")
    
    def search_documents(self, query: str, top_k: int = 5):
        """Search for similar documents."""
        print(f"🔍 Searching for: {query}")
        
        results = self.rag_system.search_documents(query, top_k)
        
        if results:
            print(f"\n📋 Found {len(results)} similar documents:")
            for i, doc in enumerate(results, 1):
                print(f"\n{i}. Score: {doc.get('score', 0):.3f}")
                print(f"   Text: {doc.get('text', 'N/A')[:200]}...")
        else:
            print("❌ No similar documents found")
    
    def show_stats(self):
        """Show system statistics."""
        stats = self.rag_system.get_system_stats()
        
        print("\n📊 System Statistics")
        print("="*30)
        print(f"Backend: {stats.get('backend', 'Unknown')}")
        print(f"LLM Model: {stats.get('llm_model', 'Unknown')}")
        print(f"Embedding Model: {stats.get('embedding_model', 'Unknown')}")
        print(f"Milvus Connected: {'✅' if stats.get('milvus_connected') else '❌'}")
        print(f"Redis Connected: {'✅' if stats.get('redis_connected') else '❌'}")
        
        if stats.get("milvus_stats"):
            milvus_stats = stats["milvus_stats"]
            print(f"\n🗄️ Milvus Statistics:")
            print(f"  Documents: {milvus_stats.get('entity_count', 0)}")
            print(f"  Collection: {milvus_stats.get('collection_name', 'N/A')}")
        
        if stats.get("cache_stats"):
            cache_stats = stats["cache_stats"]
            print(f"\n⚡ Cache Statistics:")
            print(f"  Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
            print(f"  Total Keys: {cache_stats.get('total_keys', 0)}")
    
    def interactive_mode(self):
        """Start interactive mode."""
        print("\n🎯 Gemini RAG Interactive Mode")
        print("Commands: upload <file>, search <query>, stats, quit")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n💬 Enter your question or command: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                
                if user_input.lower().startswith('upload '):
                    file_path = user_input[7:].strip()
                    self.upload_document(file_path)
                    continue
                
                if user_input.lower().startswith('search '):
                    query = user_input[7:].strip()
                    self.search_documents(query)
                    continue
                
                # Default: treat as question
                self.query_system(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Gemini RAG System CLI")
    parser.add_argument("--upload", help="Upload a document")
    parser.add_argument("--query", help="Ask a question")
    parser.add_argument("--search", help="Search documents")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    
    args = parser.parse_args()
    
    cli = GeminiCLI()
    
    if args.upload:
        cli.upload_document(args.upload)
    elif args.query:
        cli.query_system(args.query)
    elif args.search:
        cli.search_documents(args.search)
    elif args.stats:
        cli.show_stats()
    elif args.interactive:
        cli.interactive_mode()
    else:
        # Default to interactive mode
        cli.interactive_mode()

if __name__ == "__main__":
    main()