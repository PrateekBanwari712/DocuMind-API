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

        # Initializing result as none
        result = None
        # Read file content
        content = await file.read()

        # Process file with data ingestion service
        ingestion_service = DataIngestionService()
        result = await ingestion_service.process_file(
            file_content=content,
            filename=file.filename,
            metadata=metadata,
        )

        return FileResponse(
            message="File uploaded and processed successfully",
            file_id=result["file_id"],
            filename=file.filename,
            chunks_processed=result["chunks_processed"],
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

    try:
        vectordb_service = VectorDBService()
        file_info = await vectordb_service.get_file_info(file_id)

        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")

        return file_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """
    Delete a file and its associated embeddings
    """

    try:
        vectordb_service = VectorDBService()
        result = await vectordb_service.delete_file(file_id)

        if not result:
            raise HTTPException(status_code=404, detail="File not found")

        return {"message": f"File {file_id} delete successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.put("/files/{file_id}")
async def update_file(
    file_id: str, file: UploadFile = File(...), metadata: Optional[str] = Form(None)
):
    """
    Update an existing file
    """

    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # First delete the existing file
        vectordb_service = VectorDBService()
        delete_result = await vectordb_service.delete_file(file_id)

        if not delete_result:
            raise HTTPException(status_code=404, detail="File not found")

        # Read new file content
        content = await file.read()

        # Process new file with the same file_id
        ingestion_service = DataIngestionService()
        result = await ingestion_service.process_file(
            file_content=content,
            filename=file.filename,
            metadata=metadata,
            file_id=file_id,
        )

        return FileResponse(
            message="File updated successfully",
            file_id=result["file_id"],
            filename=file.filename,
            chunks_processed=result["chunks_processed"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"internal server error: {str(e)}")
