@echo off
REM Optimized RAG System Startup Script for Windows
REM This script handles all initialization and can start either CLI or Web interface

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   🤖 RAG System with Gemini Flash 2.0
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo 🔧 Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install/upgrade requirements
echo 📦 Installing/updating dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️  Warning: .env file not found!
    echo 📝 Please copy .env.example to .env and add your Google API key
    echo.
    copy .env.example .env >nul
    echo ✅ Created .env file from template
    echo ❗ Please edit .env and add your GOOGLE_API_KEY before continuing
    echo.
    pause
)

REM Check Docker services
echo 🐳 Checking Docker services...
docker-compose -f docker-compose-milvus.yml ps >nul 2>&1
if errorlevel 1 (
    echo 🚀 Starting Docker services...
    docker-compose -f docker-compose-milvus.yml up -d
    if errorlevel 1 (
        echo ❌ Failed to start Docker services
        echo 💡 Make sure Docker Desktop is running
        pause
        exit /b 1
    )
    echo ⏳ Waiting for services to be ready...
    timeout /t 10 /nobreak >nul
) else (
    echo ✅ Docker services already running
)

echo.
echo 🎯 Choose how to run the RAG system:
echo   1) 🌐 Web Interface (Streamlit)
echo   2) 💻 Command Line Interface
echo   3) 🔧 Just setup environment and exit
echo.

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo 🌐 Starting Streamlit web interface...
    echo 🔗 The app will open at: http://localhost:8501
    echo.
    streamlit run streamlit_app.py --server.port 8501 --server.headless true
) else if "%choice%"=="2" (
    echo.
    echo 💻 Starting CLI mode...
    echo 💡 Use --help to see available options
    echo.
    python main_gemini.py --interactive
) else if "%choice%"=="3" (
    echo.
    echo ✅ Environment setup complete!
    echo.
    echo 📖 Usage:
    echo   • Web: streamlit run streamlit_app.py
    echo   • CLI: python main_gemini.py --interactive
    echo   • Help: python main_gemini.py --help
    echo.
) else (
    echo ❌ Invalid choice. Exiting...
)

echo.
echo 👋 Thanks for using RAG System!
pause