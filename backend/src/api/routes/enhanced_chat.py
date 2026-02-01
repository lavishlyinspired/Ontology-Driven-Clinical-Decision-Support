"""
Enhanced Chat Endpoint with LangGraph Integration
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json

from ..services.enhanced_conversation_service import create_enhanced_conversation_service
from ..services.lca_service import LungCancerAssistantService
from ..db.neo4j_client import Neo4jGraphClient

router = APIRouter(prefix="/enhanced-chat", tags=["Enhanced Chat"])

# Initialize enhanced service (will be done in startup)
enhanced_service = None

@router.on_event("startup")
async def startup_enhanced_chat():
    """Initialize enhanced conversation service"""
    global enhanced_service
    
    # Get LCA service from dependency
    lca_service = LungCancerAssistantService()
    
    # Optional: Initialize Neo4j client
    graph_client = None
    try:
        graph_client = Neo4jGraphClient()
    except Exception:
        pass  # Neo4j optional
    
    enhanced_service = create_enhanced_conversation_service(
        lca_service=lca_service,
        graph_client=graph_client,
        enable_persistence=True
    )

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    session_id: Optional[str] = None

class HistoryRequest(BaseModel):
    thread_id: str

@router.post("/stream")
async def enhanced_chat_stream(request: ChatRequest):
    """
    Enhanced chat endpoint with LangGraph integration
    
    Features:
    - Intelligent follow-up questions
    - Conversation memory and state
    - Context-aware responses
    - Rich patient case analysis
    """
    if not enhanced_service:
        raise HTTPException(status_code=500, detail="Enhanced chat service not initialized")
    
    # Generate thread_id if not provided
    if not request.thread_id:
        import uuid
        request.thread_id = str(uuid.uuid4())
    
    async def generate():
        try:
            async for chunk in enhanced_service.chat_stream(
                message=request.message,
                thread_id=request.thread_id,
                session_id=request.session_id
            ):
                yield chunk
        except Exception as e:
            error_msg = json.dumps({
                "type": "error",
                "content": f"Chat error: {str(e)}"
            })
            yield f"data: {error_msg}\n\n"
    
    return StreamingResponse(generate(), media_type="text/plain")

@router.post("/history")
async def get_conversation_history(request: HistoryRequest):
    """Get conversation history for a thread"""
    if not enhanced_service:
        raise HTTPException(status_code=500, detail="Enhanced chat service not initialized")
    
    history = await enhanced_service.get_conversation_history(request.thread_id)
    return {"history": history, "thread_id": request.thread_id}

@router.post("/reset/{thread_id}")
async def reset_conversation(thread_id: str):
    """Reset/clear a conversation thread"""
    if not enhanced_service:
        raise HTTPException(status_code=500, detail="Enhanced chat service not initialized")
    
    success = await enhanced_service.reset_conversation(thread_id)
    return {"success": success, "thread_id": thread_id}

@router.get("/health")
async def health_check():
    """Health check for enhanced chat service"""
    return {
        "status": "healthy" if enhanced_service else "not_initialized",
        "features": [
            "LangGraph conversation flow",
            "Memory and state persistence",
            "Intelligent follow-up questions",
            "Context-aware responses",
            "Patient case analysis"
        ]
    }