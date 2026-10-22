"""
File management endpoint routes for RAG application
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional, List
from pydantic import BaseModel
from services.data_ingestion_service import DataIngestionService
from utils.logger import get_logger
from services.vectordb_service import VectorDBService

logger = get_logger(__name__)
router = APIRouter()

class FileInfo(BaseModel):
    file_id: str
    filename: str
    upload_date: str
    status: str
    chunk_count: Optional[int] = None

class FileResponse(BaseModel):
    message: str
    file_id: str
    filename: str
    chunks_processed: Optional[int] = None


@router.post("/files/upload", response_model=FileResponse)
async def upload_file(
    file: UploadFile = File(...), metadata: Optional[str] = Form(None)
):
    """
    Upload and process a file for RAG
    """

    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Read file content
        content = await file.read()

        # Process file with data ingestion service
        ingestion_service = DataIngestionService()
        result = await ingestion_service.process_file(
            file_content=content,
            filename=file.filename,
            chunks_processed=result["chunks_processed"],
        )

        return FileResponse(
            message="File uploaded and processed successfully",
            file_id=result["file_id"],
            filename=file.filename,
            chunks_processed=result["chunks_processed"]
        )

    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/files", response_model=List[FileInfo])
async def list_files():
    """
    List all uploaded files
    """

    try:
        vectordb_service = VectorDBService()
        files = await vectordb_service.list_files()
        return files

    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")    
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/files/{file_id}", response_model=FileInfo)
async def get_file_info(file_id: str):
    """Get Information about a specific file"""
