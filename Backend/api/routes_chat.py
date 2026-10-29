"""
Chat endpoint routes for RAG application
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from utils.logger import get_logger
from langgraph_agent.agent import RAGAgent

logger = get_logger(__name__)
router = APIRouter()


class ChatMessage(BaseModel):
    message: str = Field(min_length=1)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[str] = Field(default_factory=list)
    confidence_score: Optional[float] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Main chat endpoint for RAG queries
    """
    try:
        query = message.message.strip()
        if not query:
            raise HTTPException(status_code=422, detail="Message cannot be empty")

        agent = RAGAgent()
        response = await agent.process_query(
            query=query, session_id=message.session_id or None
        )

        return {
            "answer": response.get("answer", "No response"),
            "session_id": response.get("session_id", ""),
            "sources": response.get("sources", []),
            "confidence_score": response.get("confidence_score", 0.0),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/chat/sessions/{session_id}")
async def get_chat_history(session_id: str):
    """
    Get chat history for a specific session
    """
    try:
        agent = RAGAgent()
        history = await agent.get_chat_history(session_id)
        return {"session_id": session_id, "history": history}

    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/chat/sessions/{session_id}")
async def clear_chat_session(session_id: str):
    """
    Clear chat history for a specific session
    """
    try:
        agent = RAGAgent()
        await agent.clear_session(session_id)
        return {"message": f"Session {session_id} cleared successfully"}

    except Exception as e:
        logger.error(f"Error clearing chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
