#!/bin/bash

# Streamlit RAG System Launcher
echo "🚀 Starting RAG System with Streamlit UI"
echo "========================================"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment is active: $VIRTUAL_ENV"
else
    echo "⚠️  Virtual environment not detected"
    echo "🔄 Attempting to activate venv..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found"
    echo "📝 Creating from template..."
    cp .env.example .env
    echo "❗ Please edit .env file with your OpenAI API key before proceeding"
    exit 1
fi

# Check if services are running
echo "🔍 Checking services..."

# Check Redis
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "✅ Redis is running"
    else
        echo "❌ Redis is not responding"
        echo "💡 Start with: docker-compose up -d redis"
    fi
else
    echo "⚠️  Redis CLI not found, cannot check Redis status"
fi

# Check if Docker is running (for Milvus)
if command -v docker &> /dev/null; then
    if docker info &> /dev/null; then
        echo "✅ Docker is running"
    else
        echo "❌ Docker is not running"
        echo "💡 Start Docker and run: docker-compose up -d"
    fi
else
    echo "⚠️  Docker not found"
fi

echo ""
echo "🎯 Starting Streamlit application..."
echo "🌐 Open your browser to: http://localhost:8501"
echo ""

# Start Streamlit with custom config
streamlit run streamlit_app.py \
    --server.port=8501 \
    --server.address=localhost \
    --server.headless=false \
    --server.fileWatcherType=auto \
    --theme.base=light