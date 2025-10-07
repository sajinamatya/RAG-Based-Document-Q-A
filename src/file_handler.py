import os
import hashlib
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from llama_index.core import Document
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.core.node_parser import SentenceSplitter

from config.settings import settings


class FileUploadHandler:
    """Handles file uploads and document processing."""
    
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        
        # Initialize readers
        self.pdf_reader = PDFReader()
        self.docx_reader = DocxReader()
        
        # Initialize text splitter
        self.text_splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
    def _generate_file_hash(self, content: bytes) -> str:
        """Generate a hash for the file content to avoid duplicates."""
        return hashlib.md5(content).hexdigest()
    
    def _validate_file(self, file_path: Path, content: bytes) -> bool:
        """Validate file size and type."""
        # Check file size
        if len(content) > settings.max_file_size_mb * 1024 * 1024:
            raise ValueError(f"File size exceeds {settings.max_file_size_mb}MB limit")
        
        # Check file extension
        supported_extensions = {'.pdf', '.txt', '.docx'}
        if file_path.suffix.lower() not in supported_extensions:
            raise ValueError(f"Unsupported file type. Supported: {supported_extensions}")
        
        return True
    
    def save_uploaded_file(self, filename: str, content: bytes) -> Path:
        """Save uploaded file to disk."""
        file_path = Path(filename)
        self._validate_file(file_path, content)
        
        # Create unique filename with timestamp and hash
        file_hash = self._generate_file_hash(content)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{file_hash[:8]}_{file_path.name}"
        
        saved_path = self.upload_dir / unique_filename
        
        with open(saved_path, 'wb') as f:
            f.write(content)
        
        return saved_path
    
    def load_document(self, file_path: Path) -> List[Document]:
        """Load and parse document based on file type."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.pdf':
                documents = self.pdf_reader.load_data(file=file_path)
            elif file_extension == '.docx':
                documents = self.docx_reader.load_data(file=file_path)
            elif file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                documents = [Document(text=text)]
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Add metadata to documents
            for doc in documents:
                doc.metadata.update({
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'file_type': file_extension,
                    'upload_time': datetime.now().isoformat()
                })
            
            return documents
            
        except Exception as e:
            raise ValueError(f"Error processing file {file_path.name}: {str(e)}")
    
    def process_document(self, file_path: Path) -> List[Document]:
        """Load document and split into chunks."""
        documents = self.load_document(file_path)
        
        # Split documents into nodes/chunks
        nodes = []
        for document in documents:
            doc_nodes = self.text_splitter.get_nodes_from_documents([document])
            nodes.extend(doc_nodes)
        
        # Convert nodes back to documents for consistency
        processed_docs = []
        for node in nodes:
            doc = Document(
                text=node.text,
                metadata=node.metadata
            )
            processed_docs.append(doc)
        
        return processed_docs
    
    def get_uploaded_files(self) -> List[Path]:
        """Get list of all uploaded files."""
        return list(self.upload_dir.glob('*'))
    
    def delete_file(self, file_path: Path) -> bool:
        """Delete uploaded file."""
        try:
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception:
            return False