# Logging configuration for RAG System
import logging
import sys
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str = "logs/rag_system.log"):
    """Setup logging configuration."""
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Define log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers
    loggers = {
        'pymilvus': logging.WARNING,
        'redis': logging.WARNING,
        'openai': logging.WARNING,
        'httpx': logging.WARNING,
        'urllib3': logging.WARNING
    }
    
    for logger_name, level in loggers.items():
        logging.getLogger(logger_name).setLevel(level)

if __name__ == "__main__":
    setup_logging()