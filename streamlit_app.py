import streamlit as st
import os
import sys
import logging
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.gemini_rag_system import GeminiRAGSystem
from src.logging_config import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Gemini RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_rag_system():
    """Initialize the Gemini RAG system."""
    if 'rag_system' not in st.session_state:
        with st.spinner("🤖 Initializing Gemini RAG System..."):
            try:
                st.session_state.rag_system = GeminiRAGSystem()
                
                if st.session_state.rag_system.is_ready():
                    st.success("✅ Gemini RAG System initialized successfully!")
                else:
                    st.error("❌ RAG system initialization incomplete")
                    
            except Exception as e:
                st.error(f"❌ Failed to initialize RAG system: {e}")
                st.session_state.rag_system = None

def display_header():
    """Display the application header."""
    st.title("🤖 Gemini RAG System")
    st.markdown("### 🧠 Powered by Google Gemini Flash 1.5 & Gemini Embeddings")
    
    # System status
    col1, col2, col3, col4 = st.columns(4)
    
    if 'rag_system' in st.session_state and st.session_state.rag_system:
        rag_system = st.session_state.rag_system
        
        with col1:
            st.metric("🧠 LLM", "Gemini Flash 1.5")
        
        with col2:
            st.metric("🔍 Embeddings", "Gemini Embedding")
        
        with col3:
            milvus_status = "✅ Connected" if rag_system.milvus_manager and rag_system.milvus_manager.is_connected() else "❌ Disconnected"
            st.metric("🗄️ Milvus", milvus_status)
        
        with col4:
            redis_status = "✅ Connected" if rag_system.redis_cache and rag_system.redis_cache.is_connected() else "❌ Disconnected"
            st.metric("⚡ Redis", redis_status)
    else:
        st.warning("⚠️ System not initialized")

def display_upload_page():
    """Display document upload interface."""
    st.header("📁 Upload Documents")
    
    if 'rag_system' not in st.session_state or not st.session_state.rag_system:
        st.error("❌ Please initialize the system first")
        return
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Upload PDF, TXT, or DOCX files to add to your knowledge base"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.expander(f"📄 {uploaded_file.name}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**File:** {uploaded_file.name}")
                    st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
                    st.write(f"**Type:** {uploaded_file.type}")
                
                with col2:
                    if st.button(f"Upload", key=f"upload_{uploaded_file.name}"):
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            try:
                                # Save file temporarily
                                os.makedirs("uploads", exist_ok=True)
                                file_path = f"uploads/{uploaded_file.name}"
                                
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                
                                # Process with RAG system
                                success = st.session_state.rag_system.upload_document(file_path)
                                
                                if success:
                                    st.success(f"✅ {uploaded_file.name} processed successfully!")
                                else:
                                    st.error(f"❌ Failed to process {uploaded_file.name}")
                                
                                # Clean up
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                                    
                            except Exception as e:
                                st.error(f"❌ Error processing {uploaded_file.name}: {e}")

def display_query_page():
    """Display question answering interface."""
    st.header("💬 Ask Questions")
    
    if 'rag_system' not in st.session_state or not st.session_state.rag_system:
        st.error("❌ Please initialize the system first")
        return
    
    # Question input
    question = st.text_area(
        "What would you like to know?",
        placeholder="Enter your question here...",
        height=100
    )
    
    # Query button
    if st.button("🤖 Ask Gemini", type="primary"):
        if question.strip():
            with st.spinner("🤔 Gemini is thinking..."):
                try:
                    result = st.session_state.rag_system.query(question)
                    
                    # Display answer
                    st.markdown("### 🤖 Gemini's Answer:")
                    st.markdown(result["answer"])
                    
                    # Display sources
                    if result.get("sources"):
                        st.markdown("### 📚 Sources:")
                        for i, source in enumerate(result["sources"], 1):
                            with st.expander(f"Source {i} (Relevance: {source.get('score', 0):.3f})"):
                                st.text(source["text"])
                                if source.get("metadata"):
                                    st.json(source["metadata"])
                    
                    # Display metadata
                    metadata = result.get("metadata", {})
                    with st.expander("🔧 Response Details"):
                        st.json(metadata)
                        
                except Exception as e:
                    st.error(f"❌ Error processing question: {e}")
        else:
            st.warning("⚠️ Please enter a question")

def display_search_page():
    """Display document search interface."""
    st.header("🔍 Search Documents")
    
    if 'rag_system' not in st.session_state or not st.session_state.rag_system:
        st.error("❌ Please initialize the system first")
        return
    
    # Search input
    search_query = st.text_input(
        "Search for similar content:",
        placeholder="Enter search terms..."
    )
    
    # Number of results
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
    
    # Search button
    if st.button("🔍 Search", type="primary"):
        if search_query.strip():
            with st.spinner("🔍 Searching with Gemini embeddings..."):
                try:
                    results = st.session_state.rag_system.search_documents(search_query, top_k)
                    
                    if results:
                        st.markdown(f"### 📋 Found {len(results)} similar documents:")
                        
                        for i, doc in enumerate(results, 1):
                            with st.expander(f"Result {i} - Similarity: {doc.get('score', 0):.3f}"):
                                st.text(doc.get('text', 'N/A'))
                                if doc.get('metadata'):
                                    st.json(doc['metadata'])
                    else:
                        st.info("No similar documents found")
                        
                except Exception as e:
                    st.error(f"❌ Error searching documents: {e}")
        else:
            st.warning("⚠️ Please enter a search query")

def display_stats_page():
    """Display system statistics."""
    st.header("📊 System Statistics")
    
    if 'rag_system' not in st.session_state or not st.session_state.rag_system:
        st.error("❌ Please initialize the system first")
        return
    
    try:
        stats = st.session_state.rag_system.get_system_stats()
        
        # System overview
        st.subheader("🤖 System Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Backend", stats.get('backend', 'Unknown'))
            st.metric("LLM Model", stats.get('llm_model', 'Unknown'))
        
        with col2:
            st.metric("Embedding Model", stats.get('embedding_model', 'Unknown'))
            st.metric("System Ready", "✅ Yes" if st.session_state.rag_system.is_ready() else "❌ No")
        
        # Milvus stats
        if stats.get('milvus_stats'):
            st.subheader("🗄️ Milvus Database")
            milvus_stats = stats['milvus_stats']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents", milvus_stats.get('entity_count', 0))
            with col2:
                st.metric("Collection", milvus_stats.get('collection_name', 'N/A'))
            with col3:
                st.metric("Status", "✅ Connected" if stats.get('milvus_connected') else "❌ Disconnected")
        
        # Redis cache stats
        if stats.get('cache_stats'):
            st.subheader("⚡ Redis Cache")
            cache_stats = stats['cache_stats']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                hit_rate = cache_stats.get('hit_rate', 0)
                st.metric("Hit Rate", f"{hit_rate:.1%}")
            with col2:
                st.metric("Total Keys", cache_stats.get('total_keys', 0))
            with col3:
                st.metric("Status", "✅ Connected" if stats.get('redis_connected') else "❌ Disconnected")
            
            # Cache hit rate chart
            if hit_rate > 0:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = hit_rate * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Cache Hit Rate (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error getting statistics: {e}")

def sidebar_navigation():
    """Create sidebar navigation."""
    st.sidebar.title("🤖 Gemini RAG Navigation")
    
    pages = [
        "🏠 Home",
        "📁 Upload Documents", 
        "💬 Ask Questions",
        "🔍 Search Documents",
        "📊 System Statistics"
    ]
    
    return st.sidebar.selectbox("Choose a page", pages)

def main():
    """Main application."""
    
    # Initialize RAG system
    initialize_rag_system()
    
    # Display header
    display_header()
    
    # Navigation
    page = sidebar_navigation()
    
    # Page routing
    if page == "🏠 Home":
        st.markdown("""
        ## Welcome to the Gemini RAG System! 🤖
        
        This system uses **Google Gemini Flash 1.5** for both language understanding and embeddings,
        providing a fully Google-powered AI experience.
        
        ### 🎯 Features:
        - **📄 Document Upload**: Upload PDF, TXT, and DOCX files
        - **💬 Intelligent Q&A**: Ask questions and get contextual answers
        - **🔍 Semantic Search**: Find relevant information across your documents
        - **⚡ Fast Caching**: Redis-powered caching for instant responses
        - **🗄️ Vector Storage**: Milvus database for scalable vector search
        
        ### 🚀 Powered By:
        - **🧠 LLM**: Google Gemini Flash 1.5
        - **🔍 Embeddings**: Google Gemini Embedding Model
        - **🗄️ Vector DB**: Milvus
        - **⚡ Cache**: Redis
        - **🎨 UI**: Streamlit
        
        ### 📋 Getting Started:
        1. **Upload Documents** - Add your knowledge base
        2. **Ask Questions** - Get AI-powered answers
        3. **Search Content** - Find relevant information
        4. **Monitor Performance** - Check system statistics
        
        Navigate using the sidebar to explore all features! 👈
        """)
        
    elif page == "📁 Upload Documents":
        display_upload_page()
        
    elif page == "💬 Ask Questions":
        display_query_page()
        
    elif page == "🔍 Search Documents":
        display_search_page()
        
    elif page == "📊 System Statistics":
        display_stats_page()

if __name__ == "__main__":
    main()