"""
Data ingestion service for PDF text extraction and chunking
"""

import uuid
import io
from io import BytesIO
from core.config import settings
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import PyPDF2 as pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    try:
        import pypdf
        PYPDF_AVAILABLE = True
    except ImportError:
        PYPDF_AVAILABLE = False    

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    RecursiveCharacterTextSplitter = None
    LANGCHAIN_AVAILABLE = False 

from utils.logger import get_logger
from services.vectordb_service import VectorDBService

logger = get_logger(__name__)


class DataIngestionService:
    def __init__(self):
        self.text_splitter = self._initialize_text_splitter()
        self.vectordb_service = VectorDBService()

    def _initialize_text_splitter(self):
        """Initialize the text splitter for chunking documents"""
        if RecursiveCharacterTextSplitter is None:
            logger.warning("LangChain not available, using simple text splitter")
            return None

        return RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def _simple_text_splitter(self, text: str) -> List[str]:
        """Simple text splitter fallback if Langchain is not available"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > settings.CHUNK_SIZE:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Keep overlap
                    overlap_words = current_chunk[-settings.CHUNK_OVERLAP // 10 :]
                    current_chunk = overlap_words + [word]
                    current_length = sum(len(w) for w in current_chunk) + len(
                        current_chunk
                    )
                else:
                    current_chunk = [word]
                    current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    async def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file content"""
        if  pypdf  is None:
            raise ImportError(
                "pypdf or PyPDF2 is required for PDF processing. Install with pip install PyPDF2"
            )

        try:
           # Handle both pypdf and PyPDF2 reader APIs safely
            reader_cls = getattr(pypdf, "PdfReader", None) or getattr(pypdf, "PdfFileReader", None)
            pdf_reader = reader_cls(io.BytesIO(file_content))
            text = ""

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                text += page_text + "\n"

            return text.strip()

        except Exception as e:
            logger.error((f"Error extracting text from PDF: {str(e)}"))
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    async def chunk_text(self, text: str) -> List[str]:
        """Split text info chunks"""

        try:
            if self.text_splitter:
                chunks = self.text_splitter.split_text(text)
            else:
                chunks = self._simple_text_splitter(text)

            # Filter out very small chunks
            filtered_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50] 

            logger.info(f"Created {len(filtered_chunks)} chunks from text")
            return filtered_chunks
        
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise ValueError(f"Failed to chunk text: {str(e)}")

    async def process_file(
        self,
        file_content: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
        file_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a file: extract text, chunk it, and store in Pinecone
        """

        try:
            # Generate file Id if not provided
            if not file_id:
                file_id = str(uuid.uuid4())

            logger.info(f"Processing file: {filename} (ID:{file_id})")

            # Extract text from PDF
            text = await self.extract_text_from_pdf(file_content)

            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")

            # Chunk the text
            chunks = await self.chunk_text(text)

            if not chunks:
                raise ValueError("No valid chunks created from the text")

            chunk_documents = []
            for i, chunk in enumerate(chunks):
                chunk_documents.append(
                    {
                        "id": f"{file_id}_chunk_{i}",
                        "text": chunk,
                        "metadata": {
                            "file_id": file_id,
                            "filename": filename,
                            "chunk_index": i,
                            "upload_date": datetime.now().isoformat(),
                            "custom_metadata": metadata if isinstance(metadata, (str, int, float, bool)) else str(metadata) if metadata else "",
                        },
                    }
                )

            await self.vectordb_service.store_embeddings(chunk_documents, namespace=file_id)

            logger.info(
                f"Successfully processed file {filename}: {len(chunk_documents)} chunks stored"
            )

            return {
                "file_id": file_id,
                "chunks_processed": len(chunk_documents),
                "total_chunks_created": len(chunks),
                "text_length": len(text),
            }

        except Exception as e:
            logger.error(f"Error processing file {filename}:{str(e)}")
            raise

    async def reprocess_file(self, file_id: str) -> Dict[str, Any]:
        """
        Reprocess an existing file (useful for updating chunk strategy or embeddings)
        """

        try:
            # This require storing original file content
            # For now raising an error suggesting to re-upload

            raise NotImplementedError(
                "File reprocessing not implemented. Please re-upload the file to update it."
            )

        except Exception as e:
            logger.error(f"Error reprocessing file {file_id}: {str(e)}")
            raise
