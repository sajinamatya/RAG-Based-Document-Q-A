"""
Main application with Gemini Flash 2.0 as the primary LLM backend
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_system import RAGSystem
from src.gemini_rag_system import GeminiRAGSystem
from config.settings import settings


class GeminiCLIInterface:
    """Command-line interface using Gemini Flash 2.0 by default."""
    
    def __init__(self, use_full_gemini: bool = True):
        print(f"🤖 Initializing RAG System with Gemini Flash 2.0...")
        
        if use_full_gemini:
            # Use Gemini for both LLM and embeddings
            self.rag_system = GeminiRAGSystem()
            self.backend = "Gemini Flash 2.0 (Full)"
        else:
            # Use Gemini for LLM, OpenAI for embeddings
            self.rag_system = RAGSystem()
            self.backend = "Hybrid (Gemini Flash 2.0 LLM + OpenAI Embeddings)"
        
        print(f"✅ RAG System ready with {self.backend}")
    
    def upload_file(self, file_path: str):
        """Upload and process a file."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"Error: File {file_path} does not exist.")
                return
            
            with open(file_path, 'rb') as f:
                content = f.read()
            
            result = self.rag_system.upload_and_process_file(file_path.name, content)
            
            if result["success"]:
                print(f"✅ Successfully uploaded and processed: {file_path.name}")
                print(f"   Documents created: {result['document_count']}")
                print(f"   Backend: {self.backend}")
            else:
                print(f"❌ Error processing file: {result['error']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def query(self, question: str):
        """Query the RAG system."""
        try:
            print(f"\n🔍 Querying with {self.backend}: {question}")
            print("=" * 60)
            
            result = self.rag_system.query(question)
            
            if result["success"]:
                print(f"\n📝 Response:")
                print(result["response"])
                
                # Show model information
                model_info = result.get("model", self.backend)
                print(f"\n🤖 Model: {model_info}")
                
                if result["source_nodes"]:
                    print(f"\n📚 Sources ({len(result['source_nodes'])}):")
                    for i, source in enumerate(result["source_nodes"], 1):
                        print(f"\n{i}. (Score: {source['score']:.3f})")
                        print(f"   Text: {source['text']}")
                        if source.get('metadata'):
                            print(f"   File: {source['metadata'].get('file_name', 'Unknown')}")
                
                if result.get("cached"):
                    print("\n💾 (Result retrieved from cache)")
                    
            else:
                print(f"❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def show_stats(self):
        """Show system statistics."""
        try:
            stats = self.rag_system.get_document_stats()
            
            print(f"\n📊 System Statistics - {self.backend}")
            print("=" * 50)
            
            if "milvus" in stats:
                milvus_stats = stats["milvus"]
                print(f"📁 Total documents: {milvus_stats.get('total_documents', 0)}")
                print(f"🗄️  Collection: {milvus_stats.get('collection_name', 'N/A')}")
                print(f"📐 Vector dimension: {milvus_stats.get('dimension', 0)}")
            
            print(f"🎯 Vector Backend: {stats.get('backend', 'Milvus')}")
            
            if "llm_model" in stats:
                print(f"🤖 LLM Model: {stats.get('llm_model', 'N/A')}")
                print(f"📊 Embedding Model: {stats.get('embedding_model', 'N/A')}")
            
            if "cache" in stats:
                cache_stats = stats["cache"]
                if cache_stats.get("connected"):
                    print(f"💾 Cache keys: {cache_stats.get('total_keys', 0)}")
                    print(f"🎯 Hit rate: {cache_stats.get('hit_rate', 0):.1f}%")
                    print(f"💿 Memory used: {cache_stats.get('used_memory', 'N/A')}")
                else:
                    print("💾 Cache: Not connected")
            
            print(f"📄 Uploaded files: {stats.get('uploaded_files', 0)}")
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
    
    def search_similar(self, query: str, top_k: int = 5):
        """Search for similar documents."""
        try:
            print(f"\n🔍 Searching similar documents for: {query}")
            print("=" * 50)
            
            results = self.rag_system.search_similar_documents(query, top_k)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. (Similarity: {result['score']:.3f})")
                    print(f"   Text: {result['text']}")
                    if result.get('metadata'):
                        print(f"   File: {result['metadata'].get('file_name', 'Unknown')}")
            else:
                print("No similar documents found.")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def clear_data(self):
        """Clear all data from the system."""
        try:
            confirm = input("⚠️  This will delete all uploaded files, vectors, and cache. Continue? (y/N): ")
            if confirm.lower() == 'y':
                result = self.rag_system.clear_all_data()
                if result["success"]:
                    print("✅ All data cleared successfully!")
                    print(f"   Files deleted: {result['files_deleted']}")
                    print(f"   Milvus cleared: {result['milvus_cleared']}")
                    print(f"   Cache cleared: {result['cache_cleared']}")
                else:
                    print(f"❌ Error clearing data: {result['error']}")
            else:
                print("Operation cancelled.")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def interactive_mode(self):
        """Interactive mode for querying."""
        print(f"\n🤖 Interactive RAG System - {self.backend}")
        print("Commands: 'upload <file>', 'query <question>', 'search <query>', 'stats', 'clear', 'quit'")
        print("=" * 80)
        
        while True:
            try:
                user_input = input("\n💬 > ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("Goodbye! 👋")
                    break
                
                parts = user_input.split(' ', 1)
                command = parts[0].lower()
                
                if command == 'upload' and len(parts) > 1:
                    self.upload_file(parts[1])
                elif command == 'query' and len(parts) > 1:
                    self.query(parts[1])
                elif command == 'search' and len(parts) > 1:
                    self.search_similar(parts[1])
                elif command == 'stats':
                    self.show_stats()
                elif command == 'clear':
                    self.clear_data()
                else:
                    print("❓ Available commands:")
                    print("   upload <file_path>  - Upload and process a file")
                    print("   query <question>    - Ask a question")
                    print("   search <query>      - Search similar documents")
                    print("   stats               - Show system statistics")
                    print("   clear               - Clear all data")
                    print("   quit                - Exit the program")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='RAG System with Gemini Flash 2.0 and Milvus')
    parser.add_argument('--upload', type=str, help='Upload and process a file')
    parser.add_argument('--query', type=str, help='Query the RAG system')
    parser.add_argument('--search', type=str, help='Search similar documents')
    parser.add_argument('--stats', action='store_true', help='Show system statistics')
    parser.add_argument('--clear', action='store_true', help='Clear all data')
    parser.add_argument('--interactive', action='store_true', help='Enter interactive mode')
    parser.add_argument('--backend', choices=['gemini', 'hybrid'], default='gemini',
                       help='Choose backend: gemini (full Gemini) or hybrid (Gemini LLM + OpenAI embeddings)')
    
    args = parser.parse_args()
    
    use_gemini = args.backend == 'gemini'
    cli = GeminiCLIInterface(use_full_gemini=use_gemini)
    
    if args.upload:
        cli.upload_file(args.upload)
    elif args.query:
        cli.query(args.query)
    elif args.search:
        cli.search_similar(args.search)
    elif args.stats:
        cli.show_stats()
    elif args.clear:
        cli.clear_data()
    elif args.interactive:
        cli.interactive_mode()
    else:
        cli.interactive_mode()


if __name__ == "__main__":
    main()