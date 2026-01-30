"""
Chat API Routes
Server-Sent Events endpoint for streaming conversational AI
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os

# Centralized logging
from ...logging_config import get_logger, log_execution

logger = get_logger(__name__)

from ...services.conversation_service import ConversationService
from ...services.lca_service import LungCancerAssistantService

router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize services - Neo4j and vector store enabled by default, configurable via env
use_neo4j = os.getenv("CHAT_USE_NEO4J", "true").lower() == "true"
use_vector_store = os.getenv("CHAT_USE_VECTOR_STORE", "true").lower() == "true"

lca_service = LungCancerAssistantService(
    use_neo4j=use_neo4j,
    use_vector_store=use_vector_store,
    enable_advanced_workflow=True,
    enable_provenance=True
)
conversation_service = ConversationService(lca_service)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat responses via Server-Sent Events
    
    Args:
        message: User message
        session_id: Session identifier for conversation history
        
    Returns:
        SSE stream with chat responses
    """
    try:
        async def event_generator():
            async for chunk in conversation_service.chat_stream(
                session_id=request.session_id,
                message=request.message
            ):
                yield chunk
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    
    except Exception as e:
        logger.error(f"Chat stream error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """
    Get conversation history for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        List of messages in the conversation
    """
    try:
        history = conversation_service.get_history(session_id)
        return {
            "session_id": session_id,
            "messages": history,
            "message_count": len(history)
        }
    except Exception as e:
        logger.error(f"Get history error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """
    Clear conversation history for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        Confirmation message
    """
    try:
        conversation_service.clear_session(session_id)
        return {
            "status": "success",
            "message": f"Session {session_id} cleared"
        }
    except Exception as e:
        logger.error(f"Clear session error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def chat_health():
    """Health check for chat service"""
    return {
        "status": "healthy",
        "service": "chat",
        "features": [
            "streaming_responses",
            "session_management",
            "patient_data_extraction",
            "workflow_integration"
        ]
    }
