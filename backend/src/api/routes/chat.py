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
    enable_advanced_workflow=False,  # Temporarily disabled for stability
    enable_provenance=False  # Temporarily disabled for stability
)

# Initialize conversation service with enhanced features disabled for stability
conversation_service = ConversationService(lca_service, enable_enhanced_features=False)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    use_enhanced_features: Optional[bool] = None  # Allow per-request control

class FollowUpRequest(BaseModel):
    session_id: str

class ThreadResetRequest(BaseModel):
    session_id: str


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat responses via Server-Sent Events with enhanced features
    
    Args:
        message: User message
        session_id: Session identifier for conversation history
        use_enhanced_features: Whether to use LangGraph enhanced features
        
    Returns:
        SSE stream with chat responses and potential follow-up suggestions
    """
    try:
        async def event_generator():
            async for chunk in conversation_service.chat_stream(
                session_id=request.session_id,
                message=request.message,
                use_enhanced_features=request.use_enhanced_features
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
            "workflow_integration",
            "enhanced_langgraph_features",
            "intelligent_followup_suggestions"
        ],
        "enhanced_features_available": conversation_service.enable_enhanced_features
    }


# =============================================================================
# Enhanced Conversation Endpoints
# =============================================================================

@router.get("/follow-up/{session_id}")
async def get_follow_up_suggestions(session_id: str):
    """
    Get intelligent follow-up suggestions for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        List of follow-up question suggestions
    """
    try:
        suggestions = conversation_service.get_follow_up_suggestions(session_id)
        return {
            "session_id": session_id,
            "suggestions": suggestions,
            "count": len(suggestions)
        }
    except Exception as e:
        logger.error(f"Follow-up suggestions error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-thread")
async def reset_conversation_thread(request: ThreadResetRequest):
    """
    Reset LangGraph conversation thread for enhanced features
    
    Args:
        session_id: Session identifier
        
    Returns:
        Reset confirmation
    """
    try:
        success = conversation_service.reset_conversation_thread(request.session_id)
        return {
            "status": "success" if success else "not_found",
            "session_id": request.session_id,
            "message": "Conversation thread reset successfully" if success else "Session not found"
        }
    except Exception as e:
        logger.error(f"Thread reset error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights/{session_id}")
async def get_conversation_insights(session_id: str):
    """
    Get analytics and insights about a conversation session
    
    Args:
        session_id: Session identifier
        
    Returns:
        Conversation insights and analytics
    """
    try:
        insights = await conversation_service.get_conversation_insights(session_id)
        return {
            "session_id": session_id,
            "insights": insights
        }
    except Exception as e:
        logger.error(f"Conversation insights error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))