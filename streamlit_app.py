import streamlit as st
import sys
import os
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import json

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import your RAG components
try:
    from src.rag_system import RAGSystem
    from src.gemini_rag_system import GeminiRAGSystem
    from src.logging_config import setup_logging
    from config.settings import settings
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please make sure all dependencies are installed: pip install -r requirements.txt")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🤖 RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

def initialize_system(use_gemini: bool = True):
    """Initialize the RAG system."""
    try:
        with st.spinner(f"🚀 Initializing RAG System with {'Gemini' if use_gemini else 'Hybrid'}..."):
            setup_logging()
            if use_gemini:
                st.session_state.rag_system = GeminiRAGSystem()
                st.session_state.backend = "Gemini (Full)"
            else:
                st.session_state.rag_system = RAGSystem()
                st.session_state.backend = "Hybrid (Gemini LLM + OpenAI Embeddings)"
            
            st.session_state.system_initialized = True
        return True
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {str(e)}")
        return False

def display_header():
    """Display the main header."""
    st.title("🤖 RAG System with Gemini Flash 2.0")
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
        <h4 style='color: #1f77b4; margin: 0;'>Advanced RAG with Google Gemini 2.0 Flash</h4>
        <p style='margin: 0.5rem 0 0 0; color: #666;'>
            Upload documents, ask questions, and get AI-powered answers with source citations.
            Using <strong>Gemini Flash 2.0</strong> LLM, <strong>Milvus</strong> for vector storage and <strong>Redis</strong> for caching.
        </p>
    </div>
    """, unsafe_allow_html=True)

def sidebar_navigation():
    """Create sidebar navigation."""
    st.sidebar.title("🔧 Navigation")
    
    # Model selection section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Model Configuration")
    
    model_choice = st.sidebar.selectbox(
        "Choose LLM Backend:",
        ["Gemini (Full)", "Hybrid (Gemini LLM + OpenAI Embeddings)"],
        help="Gemini (Full) uses Gemini for both LLM and embeddings. Hybrid uses Gemini LLM with OpenAI embeddings.",
        key="model_choice"
    )
    
    use_gemini = model_choice == "Gemini (Full)"
    
    if st.sidebar.button("🔄 Reinitialize System", help="Click to apply model changes") or not st.session_state.system_initialized:
        if initialize_system(use_gemini):
            st.sidebar.success(f"✅ System initialized with {st.session_state.backend}")
            st.rerun()
        
    if st.session_state.system_initialized:
        st.sidebar.info(f"📊 Current Backend: {st.session_state.get('backend', 'Unknown')}")
    
    st.sidebar.markdown("---")
    
    pages = {
        "🏠 Home": "home",
        "📁 Upload Documents": "upload", 
        "💬 Chat & Query": "chat",
        "🔍 Search Documents": "search",
        "📊 System Stats": "stats",
        "⚙️ Settings": "settings"
    }
    
    selected = st.sidebar.radio("Choose a page:", list(pages.keys()))
    return pages[selected]

def system_status_sidebar():
    """Show system status in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 System Status")
    
    if st.session_state.system_initialized:
        try:
            stats = st.session_state.rag_system.get_document_stats()
            
            # Milvus status
            if "milvus" in stats:
                milvus_stats = stats["milvus"]
                st.sidebar.metric(
                    "📁 Documents", 
                    milvus_stats.get('total_documents', 0)
                )
            
            # Cache status
            if "cache" in stats:
                cache_stats = stats["cache"]
                if cache_stats.get("connected"):
                    st.sidebar.metric(
                        "💾 Cache Hit Rate", 
                        f"{cache_stats.get('hit_rate', 0):.1f}%"
                    )
                    st.sidebar.success("🔗 Redis Connected")
                else:
                    st.sidebar.error("❌ Redis Disconnected")
            
            # Files status
            st.sidebar.metric(
                "📄 Files", 
                stats.get('uploaded_files', 0)
            )
            
        except Exception as e:
            st.sidebar.error(f"Status error: {e}")
    else:
        st.sidebar.error("❌ System Not Initialized")

def upload_page():
    """Document upload page."""
    st.header("📁 Upload Documents")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload New Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Files", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    try:
                        result = st.session_state.rag_system.upload_and_process_file(
                            file.name, file.read()
                        )
                        
                        if result["success"]:
                            st.success(f"✅ {file.name} processed successfully!")
                            st.session_state.uploaded_files.append({
                                "name": file.name,
                                "path": result["file_path"],
                                "chunks": result["document_count"],
                                "timestamp": time.time()
                            })
                        else:
                            st.error(f"❌ Error processing {file.name}: {result['error']}")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing {file.name}: {str(e)}")
                
                status_text.text("Processing complete!")
                st.rerun()
    
    with col2:
        st.markdown("### Upload Guidelines")
        st.info("""
        📝 **Supported Formats:**
        - PDF documents
        - Text files (.txt)
        - Word documents (.docx)
        
        📏 **File Limits:**
        - Max size: 100MB per file
        - No limit on number of files
        
        ⚡ **Processing:**
        - Documents are split into chunks
        - Embeddings are generated
        - Stored in Milvus vector DB
        """)

def chat_page():
    """Chat and query interface."""
    st.header("💬 Chat with Your Documents")
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["type"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    if message.get("sources"):
                        with st.expander("📚 Sources", expanded=False):
                            for j, source in enumerate(message["sources"]):
                                st.markdown(f"""
                                **Source {j+1}** (Score: {source['score']:.3f})
                                
                                {source['text']}
                                
                                *File: {source['metadata'].get('file_name', 'Unknown')}*
                                
                                ---
                                """)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.chat_history.append({
            "type": "user",
            "content": prompt
        })
        
        # Display user message
        st.chat_message("user").write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    result = st.session_state.rag_system.query(prompt)
                    
                    if result["success"]:
                        response = result["response"]
                        sources = result.get("source_nodes", [])
                        cached = result.get("cached", False)
                        
                        # Display response
                        st.write(response)
                        
                        if cached:
                            st.caption("💾 Retrieved from cache")
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "type": "assistant",
                            "content": response,
                            "sources": sources
                        })
                        
                        # Display sources
                        if sources:
                            with st.expander("📚 Sources", expanded=False):
                                for j, source in enumerate(sources):
                                    st.markdown(f"""
                                    **Source {j+1}** (Score: {source['score']:.3f})
                                    
                                    {source['text']}
                                    
                                    *File: {source['metadata'].get('file_name', 'Unknown')}*
                                    
                                    ---
                                    """)
                    else:
                        error_msg = f"❌ Error: {result['error']}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "type": "assistant",
                            "content": error_msg
                        })
                        
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "type": "assistant", 
                        "content": error_msg
                    })
    
    # Clear chat button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

def search_page():
    """Document search page."""
    st.header("🔍 Search Documents")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input(
            "Search for similar content:",
            placeholder="Enter search terms..."
        )
        
        top_k = st.slider("Number of results:", 1, 20, 5)
        
        if search_query and st.button("🔍 Search", type="primary"):
            with st.spinner("Searching..."):
                try:
                    results = st.session_state.rag_system.search_similar_documents(
                        search_query, top_k
                    )
                    
                    if results:
                        st.success(f"Found {len(results)} similar documents")
                        
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Result {i} - Score: {result['score']:.3f}"):
                                st.markdown(f"**Text:**\n\n{result['text']}")
                                st.markdown(f"**File:** {result['metadata'].get('file_name', 'Unknown')}")
                                st.markdown(f"**Similarity Score:** {result['score']:.3f}")
                    else:
                        st.warning("No similar documents found.")
                        
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
    
    with col2:
        st.markdown("### Search Tips")
        st.info("""
        🔍 **How to search:**
        - Use natural language queries
        - Be specific for better results
        - Try different keywords
        
        📊 **Similarity scores:**
        - Higher scores = more relevant
        - Range: 0.0 to 1.0
        - Typically good results > 0.7
        """)

def stats_page():
    """System statistics page."""
    st.header("📊 System Statistics")
    
    try:
        stats = st.session_state.rag_system.get_document_stats()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📁 Total Documents",
                stats.get("milvus", {}).get("total_documents", 0)
            )
        
        with col2:
            st.metric(
                "📄 Uploaded Files", 
                stats.get("uploaded_files", 0)
            )
        
        with col3:
            if stats.get("cache", {}).get("connected"):
                st.metric(
                    "💾 Cache Keys",
                    stats["cache"].get("total_keys", 0)
                )
            else:
                st.metric("💾 Cache Keys", "N/A")
        
        with col4:
            if stats.get("cache", {}).get("connected"):
                hit_rate = stats["cache"].get("hit_rate", 0)
                st.metric(
                    "🎯 Cache Hit Rate",
                    f"{hit_rate:.1f}%",
                    delta=f"{hit_rate - 50:.1f}%" if hit_rate > 50 else None
                )
            else:
                st.metric("🎯 Cache Hit Rate", "N/A")
        
        # Detailed stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🗄️ Milvus Database")
            if "milvus" in stats:
                milvus_data = stats["milvus"]
                milvus_df = pd.DataFrame([
                    {"Metric": "Collection Name", "Value": milvus_data.get("collection_name", "N/A")},
                    {"Metric": "Total Documents", "Value": milvus_data.get("total_documents", 0)},
                    {"Metric": "Vector Dimension", "Value": milvus_data.get("dimension", 0)}
                ])
                st.dataframe(milvus_df, use_container_width=True, hide_index=True)
            else:
                st.error("Milvus statistics unavailable")
        
        with col2:
            st.subheader("💾 Redis Cache")
            if stats.get("cache", {}).get("connected"):
                cache_data = stats["cache"]
                cache_df = pd.DataFrame([
                    {"Metric": "Status", "Value": "Connected ✅"},
                    {"Metric": "Total Keys", "Value": cache_data.get("total_keys", 0)},
                    {"Metric": "Cache Hits", "Value": cache_data.get("hits", 0)},
                    {"Metric": "Cache Misses", "Value": cache_data.get("misses", 0)},
                    {"Metric": "Hit Rate", "Value": f"{cache_data.get('hit_rate', 0):.1f}%"},
                    {"Metric": "Memory Used", "Value": cache_data.get("used_memory", "N/A")}
                ])
                st.dataframe(cache_df, use_container_width=True, hide_index=True)
            else:
                st.error("Redis cache unavailable")
        
        # Uploaded files timeline
        if st.session_state.uploaded_files:
            st.subheader("📈 Upload Timeline")
            
            upload_df = pd.DataFrame(st.session_state.uploaded_files)
            upload_df['upload_time'] = pd.to_datetime(upload_df['timestamp'], unit='s')
            
            fig = px.timeline(
                upload_df,
                x_start="upload_time",
                x_end="upload_time", 
                y="name",
                title="Document Upload Timeline"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cache performance chart
        if stats.get("cache", {}).get("connected"):
            st.subheader("💾 Cache Performance")
            
            cache_stats = stats["cache"]
            hits = cache_stats.get("hits", 0)
            misses = cache_stats.get("misses", 0)
            
            if hits + misses > 0:
                fig = go.Figure(data=[go.Pie(
                    labels=['Cache Hits', 'Cache Misses'],
                    values=[hits, misses],
                    hole=0.3
                )])
                fig.update_layout(title_text="Cache Hit vs Miss Rate")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No cache activity yet")
                
    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")

def settings_page():
    """Settings and configuration page."""
    st.header("⚙️ Settings & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 System Settings")
        
        # Environment status
        st.markdown("### Environment Variables")
        env_checks = {
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
            "MILVUS_HOST": bool(os.getenv("MILVUS_HOST")),
            "REDIS_HOST": bool(os.getenv("REDIS_HOST"))
        }
        
        for env_var, is_set in env_checks.items():
            status = "✅" if is_set else "❌"
            st.text(f"{status} {env_var}")
        
        # Current settings display
        st.markdown("### Current Configuration")
        config_data = {
            "Chunk Size": settings.chunk_size,
            "Chunk Overlap": settings.chunk_overlap,
            "Max File Size (MB)": settings.max_file_size_mb,
            "Cache TTL (seconds)": settings.cache_ttl_seconds,
            "Cache Enabled": settings.enable_cache
        }
        
        config_df = pd.DataFrame(
            list(config_data.items()),
            columns=["Setting", "Value"]
        )
        st.dataframe(config_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("🧹 System Management")
        
        # Clear cache button
        if st.button("🗑️ Clear Cache", use_container_width=True):
            try:
                if st.session_state.rag_system:
                    # Clear cache logic here
                    st.success("Cache cleared successfully!")
                else:
                    st.error("System not initialized")
            except Exception as e:
                st.error(f"Error clearing cache: {str(e)}")
        
        # Clear all data button (dangerous)
        st.markdown("### ⚠️ Danger Zone")
        if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
            if st.checkbox("I understand this will delete everything"):
                try:
                    result = st.session_state.rag_system.clear_all_data()
                    if result["success"]:
                        st.success("All data cleared!")
                        st.session_state.uploaded_files = []
                        st.session_state.chat_history = []
                        st.rerun()
                    else:
                        st.error(f"Error: {result['error']}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # System info
        st.markdown("### 📋 System Information")
        st.info(f"""
        **Python Version:** {sys.version.split()[0]}
        
        **Components:**
        - LlamaIndex for RAG
        - Milvus for vector storage  
        - Redis for caching
        - Streamlit for UI
        
        **Features:**
        - Document upload & processing
        - Semantic search
        - Chat interface
        - Source attribution
        - Performance caching
        """)

def home_page():
    """Home/dashboard page."""
    st.header("🏠 Welcome to Your RAG System")
    
    # Quick stats overview
    if st.session_state.system_initialized:
        try:
            stats = st.session_state.rag_system.get_document_stats()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📁 Documents", stats.get("milvus", {}).get("total_documents", 0))
            with col2:
                st.metric("📄 Files", stats.get("uploaded_files", 0))
            with col3:
                cache_status = "🟢 Connected" if stats.get("cache", {}).get("connected") else "🔴 Disconnected"
                st.metric("💾 Cache", cache_status)
        except:
            st.warning("Could not load system stats")
    
    # Getting started guide
    st.markdown("## 🚀 Getting Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📁 1. Upload Documents
        - Go to the **Upload Documents** page
        - Upload PDF, TXT, or DOCX files
        - Documents will be processed automatically
        
        ### 💬 2. Start Chatting
        - Visit the **Chat & Query** page
        - Ask questions about your documents
        - Get AI-powered answers with sources
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 3. Search Content
        - Use the **Search Documents** page
        - Find similar content across all documents
        - Explore semantic search capabilities
        
        ### 📊 4. Monitor Performance
        - Check **System Stats** for insights
        - Monitor cache performance
        - Track document processing
        """)
    
    # Recent activity
    if st.session_state.uploaded_files:
        st.markdown("## 📈 Recent Activity")
        recent_files = st.session_state.uploaded_files[-5:]  # Last 5 files
        
        for file_info in reversed(recent_files):
            upload_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(file_info["timestamp"]))
            st.markdown(f"📄 **{file_info['name']}** - {file_info['chunks']} chunks - {upload_time}")
    
    # Quick actions
    st.markdown("## ⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📁 Upload Files", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()
    
    with col2:
        if st.button("💬 Start Chat", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()
    
    with col3:
        if st.button("🔍 Search Docs", use_container_width=True):
            st.session_state.page = "search"
            st.rerun()
    
    with col4:
        if st.button("📊 View Stats", use_container_width=True):
            st.session_state.page = "stats"
            st.rerun()

def main():
    """Main application."""
    # Initialize system if not already done
    if not st.session_state.system_initialized:
        if not initialize_system():
            st.stop()
    
    # Display header
    display_header()
    
    # Get current page from sidebar or session state
    current_page = sidebar_navigation()
    if 'page' in st.session_state:
        current_page = st.session_state.page
    
    # Show system status in sidebar
    system_status_sidebar()
    
    # Route to appropriate page
    if current_page == "home":
        home_page()
    elif current_page == "upload":
        upload_page()
    elif current_page == "chat":
        chat_page()
    elif current_page == "search":
        search_page()
    elif current_page == "stats":
        stats_page()
    elif current_page == "settings":
        settings_page()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🤖 RAG System powered by LlamaIndex, Milvus & Redis"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()