# 🤖 RAG System with Gemini Flash 2.0, Milvus & Redis

A comprehensive Retrieval-Augmented Generation (RAG) system featuring **Google Gemini Flash 2.0** LLM, document upload, intelligent caching, and AI-powered question answering with a beautiful **Streamlit interface**.

## 🌟 Features

- **🤖 Gemini Flash 2.0**: Latest Google AI model for superior performance
- **📁 Document Processing**: Support for PDF, TXT, DOCX files
- **🔍 Smart Q&A**: Ask questions and get answers with source citations
- **� Redis Caching**: Lightning-fast response caching
- **�️ Milvus Vector DB**: High-performance vector similarity search
- **🎨 Modern Web UI**: Interactive **Streamlit** interface
- **💻 CLI Interface**: Command-line tools for advanced users
- **� Real-time Analytics**: System statistics and performance monitoring

## 📋 Prerequisites

### Required Services

1. **Docker Desktop** 
   - Download from [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Make sure Docker is running before starting

2. **Google API Key** (Required)
   - Get your API key from [Google AI Studio](https://ai.google.dev/)
   - The key is needed for Gemini Flash 2.0 access

### System Requirements

- Python 3.8+
- 4GB+ RAM
- 2GB+ disk space
- Docker Desktop

## 🚀 Quick Start

### 1. One-Click Setup (Windows)

Simply run the optimized startup script:

```bash
start.bat
```

This script will:
- ✅ Set up Python virtual environment
- ✅ Install all dependencies
- ✅ Start Docker services (Milvus + Redis)
- ✅ Create .env configuration file
- ✅ Launch your chosen interface (Web or CLI)

1. **Clone and Setup**
```bash
# Navigate to project directory
cd e:\lamaindex

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Configuration**
```bash
# Copy environment template
copy .env.example .env
```

Edit `.env` file with your Google API key:
```env
# Google Gemini Configuration (Required)
GOOGLE_API_KEY=your_google_api_key_here

# Optional: OpenAI for hybrid mode
OPENAI_API_KEY=your_openai_api_key_here

# Service Configuration (defaults work fine)
MILVUS_HOST=localhost
MILVUS_PORT=19530
REDIS_HOST=localhost
REDIS_PORT=6379
GEMINI_MODEL=gemini-2.0-flash-exp
```bash
docker-compose -f docker-compose-milvus.yml up -d
```

4. **Run the Application**

**Web Interface (Recommended):**
```bash
streamlit run streamlit_app.py
```

**Command Line Interface:**
```bash
python main_gemini.py --interactive
```

**Or simply use the startup script:**
```bash
start.bat  # Windows
```

## 🎯 Usage

### 📱 Web Interface (Streamlit)

1. **Start the app**: Run `start.bat` or `streamlit run streamlit_app.py`
2. **Open browser**: Navigate to `http://localhost:8501`
3. **Choose model**: Select between "Gemini (Full)" or "Hybrid" mode
4. **Upload documents**: Use the sidebar to upload PDF, TXT, or DOCX files
5. **Ask questions**: Type your questions in the chat interface
6. **View sources**: See which documents were used to generate answers

### 💻 Command Line Interface

```bash
# Interactive chat mode
python main_gemini.py --interactive

# Single query
python main_gemini.py --query "What is this document about?"

# Upload a file
python main_gemini.py --upload "path/to/your/document.pdf"

# Show help
python main_gemini.py --help
```

## 🎨 Interface Features

### 🏠 Main Dashboard
- **Model Selection**: Choose between full Gemini or hybrid mode
- **System Status**: Real-time monitoring of services
- **Quick Stats**: Document count, cache status, and performance metrics

### � Document Upload
- **Multi-file Upload**: Drag & drop or select multiple files
- **Real-time Progress**: Visual progress bars during processing
- **Format Support**: PDF, TXT, DOCX with validation
- **Upload Guidelines**: Tips and file limits

### 💬 Chat Interface  
- **Conversational AI**: Natural language question answering
- **Chat History**: Persistent conversation memory
- **Source Citations**: Expandable source references
- **Cache Indicators**: Shows when results are cached

### 🔍 Document Search
- **Semantic Search**: Find similar content using AI
- **Adjustable Results**: Slider for number of results (1-20)  
- **Relevance Scores**: Similarity scores for each result
- **Content Preview**: Truncated text with full metadata

### 📊 Statistics Dashboard
- **System Metrics**: Document count, cache performance
- **Visual Charts**: Pie charts for cache hit/miss rates  
- **Upload Timeline**: Interactive timeline of document uploads
- **Performance Monitoring**: Memory usage, hit rates

### ⚙️ Settings & Management
- **Environment Status**: Check required API keys and services
- **System Configuration**: Current settings display
- **Data Management**: Clear cache or all data options
- **Troubleshooting**: System information and diagnostics

## 🚀 Running Different Interfaces

### 🎨 Streamlit UI (Recommended)
```bash
# Windows
run_streamlit.bat

# Linux/Mac  
./run_streamlit.sh

# Manual
streamlit run streamlit_app.py
```
Access at: http://localhost:8501

### 💻 Command Line Interface
```bash
# Interactive mode (great for testing)
python main.py --interactive

# Direct commands
python main.py --upload "documents/sample.pdf"
python main.py --query "What is this document about?"
python main.py --search "artificial intelligence" 
python main.py --stats
```

### 🌐 FastAPI Web Interface
```bash
python web_app.py
```
Access at: http://localhost:8000

## 📖 Usage Guide

### 1. Upload Documents

**CLI:**
```bash
# Interactive mode
> upload documents/sample.pdf

# Direct command
python main.py --upload "documents/sample.pdf"
```

**Web Interface:**
1. Open http://localhost:8000
2. Use the "Upload Document" section
3. Select your file and click "Upload & Process"

### 2. Query Documents

**CLI:**
```bash
# Interactive mode
> query What are the main topics in the document?

# Direct command
python main.py --query "What are the main topics?"
```

**Web Interface:**
1. Use the "Ask a Question" section
2. Enter your question and click "Ask Question"
3. View response with source attributions

### 3. Search Similar Content

**CLI:**
```bash
# Interactive mode
> search machine learning concepts

# Direct command
python main.py --search "machine learning"
```

### 4. System Statistics

**CLI:**
```bash
# Interactive mode
> stats

# Direct command
python main.py --stats
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │────│   RAG System     │────│   Response      │
│ (CLI/Web)       │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ File Handler    │    │   Query Engine   │    │ Cache Manager   │
│ - PDF Reader    │    │ - Retriever      │    │ - Redis         │
│ - DOCX Reader   │    │ - Post-processor │    │ - Search Cache  │
│ - Text Splitter │    │ - LLM (OpenAI)   │    │ - Embeddings    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Vector Database  │
                    │ - Milvus         │
                    │ - Embeddings     │
                    │ - Similarity     │
                    └──────────────────┘
```

## 🔧 Configuration Options

### Document Processing
- `CHUNK_SIZE`: Size of text chunks (default: 1024)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `MAX_FILE_SIZE_MB`: Maximum file size (default: 100MB)

### Vector Database
- `MILVUS_HOST`: Milvus server host
- `MILVUS_PORT`: Milvus server port
- `EMBEDDING_DIMENSION`: Vector dimensions (default: 1536 for OpenAI)

### Caching
- `CACHE_TTL_SECONDS`: Cache expiration time (default: 3600s)
- `ENABLE_CACHE`: Enable/disable caching (default: true)

## 🚀 API Endpoints (Web Interface)

- `GET /`: Main web interface
- `POST /upload/`: Upload and process documents
- `POST /query/`: Query the RAG system
- `POST /search/`: Search similar documents
- `GET /stats/`: Get system statistics
- `DELETE /clear/`: Clear all data

## 📊 Monitoring & Statistics

The system provides comprehensive statistics:

- **Document Statistics**: Total documents, chunks, collection info
- **Cache Performance**: Hit rate, memory usage, key counts
- **File Management**: Uploaded files, processing status

## 🔍 Troubleshooting

### Common Issues

1. **Milvus Connection Error**
   ```bash
   # Check if Milvus is running
   docker ps | grep milvus
   # Or check local installation
   ```

2. **Redis Connection Error**
   ```bash
   # Check if Redis is running
   redis-cli ping
   # Should return "PONG"
   ```

3. **Docker Issues**

   **Docker not found:**
   ```bash
   # Install Docker Desktop first
   # Windows/Mac: https://www.docker.com/products/docker-desktop/
   # Linux: https://docs.docker.com/engine/install/
   ```

   **Services won't start:**
   ```bash
   # Check if ports are in use
   netstat -an | findstr "19530\|6379"  # Windows
   netstat -an | grep "19530\|6379"     # Linux/Mac

   # Free up ports or change configuration
   ```

   **Docker Compose errors:**
   ```bash
   # Update Docker Compose
   docker-compose --version

   # Recreate containers
   docker-compose down -v
   docker-compose up -d
   ```

   **Container health issues:**
   ```bash
   # Check container logs
   docker-compose logs milvus
   docker-compose logs redis

   # Restart specific service
   docker-compose restart milvus
   ```

   **Permission issues (Linux):**
   ```bash
   # Add user to docker group
   sudo usermod -aG docker $USER
   # Logout and login again
   ```

4. **OpenAI API Errors**
   - Verify API key in `.env`
   - Check API quotas and billing
   - Ensure internet connectivity

5. **File Processing Errors**
   - Check file format (PDF, TXT, DOCX only)
   - Verify file size limit
   - Ensure file is not corrupted

### Performance Optimization

1. **Adjust chunk size** for better retrieval
2. **Enable caching** for faster responses
3. **Use SSD storage** for Milvus data
4. **Increase RAM** for better vector operations

## 🔒 Security Notes

- Keep your OpenAI API key secure
- Use authentication for production deployments
- Configure firewall rules for Milvus/Redis
- Regular backup of vector data

## 📁 Project Structure (Optimized)

```
e:\lamaindex\
├── 📁 src/                     # Core application code
│   ├── rag_system.py          # Main RAG orchestrator (hybrid mode)
│   ├── gemini_rag_system.py   # Full Gemini implementation
│   ├── milvus_manager.py      # Milvus vector database manager
│   ├── redis_cache.py         # Redis caching layer
│   ├── file_handler.py        # Document processing utilities
│   └── logging_config.py      # Logging configuration
├── 📁 config/                 # Configuration files
│   └── settings.py           # Application settings
├── 📁 uploads/               # Uploaded documents storage
├── 📁 data/                  # Processed data cache
├── 🚀 start.bat              # Optimized one-click startup script
├── 🌐 streamlit_app.py       # Streamlit web interface
├── 💻 main_gemini.py         # Command-line interface
├── 🐳 docker-compose-milvus.yml # Docker services configuration
├── 🔧 requirements.txt       # Python dependencies
├── ⚙️  .env.example          # Environment configuration template
└── 📖 README.md              # This file
```

### Key Components

- **🤖 Gemini Integration**: Two modes available - Full Gemini or Hybrid (Gemini LLM + OpenAI embeddings)
- **📦 Single Startup**: One `start.bat` script handles everything
- **🎯 Clean Architecture**: Separated concerns with modular design
- **🐳 Docker Services**: Only Milvus and Redis, no unnecessary containers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review configuration settings  
3. Check service logs: `docker-compose -f docker-compose-milvus.yml logs`
4. Create an issue with system details

### Quick Troubleshooting

**Services not starting?**
```bash
# Check Docker
docker --version
docker-compose --version

# Restart services
docker-compose -f docker-compose-milvus.yml down
docker-compose -f docker-compose-milvus.yml up -d
```

**API key issues?**
1. Ensure `.env` file exists (copy from `.env.example`)
2. Add your `GOOGLE_API_KEY` from [Google AI Studio](https://ai.google.dev/)
3. Restart the application

---

**Happy querying with Gemini Flash 2.0! 🤖📚✨**