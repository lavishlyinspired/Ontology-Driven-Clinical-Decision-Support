"""
WebSocket API for Real-Time Communication
Handles live updates, notifications, and bidirectional messaging
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

from src.services.websocket_service import websocket_service, WSChannels
from src.services.audit_service import audit_logger
from src.api.routes.auth import get_current_active_user

router = APIRouter(prefix="/ws", tags=["WebSocket"])


# ==================== WebSocket Connection Endpoint ====================

@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: Optional[str] = None,
    channels: Optional[str] = None
):
    """
    WebSocket connection endpoint for real-time updates
    
    Query Parameters:
    - user_id: User identifier (for authentication)
    - channels: Comma-separated list of channels to subscribe
    
    Available Channels:
    - system: System-wide announcements
    - patient_updates: Patient data changes
    - recommendations: Treatment recommendation updates
    - hitl_queue: HITL review queue updates
    - batch_jobs: Batch job progress updates
    - analytics: Analytics computation results
    
    Example URL:
    ws://localhost:8000/api/v1/ws/connect?user_id=user123&channels=patient_updates,recommendations
    """
    # Accept connection
    await websocket.accept()
    
    # Parse channels
    channel_list = []
    if channels:
        channel_list = [ch.strip() for ch in channels.split(",")]
    else:
        channel_list = [WSChannels.SYSTEM.value]  # Default to system channel
    
    try:
        # Connect to WebSocket service
        connection_id = await websocket_service.connect(
            websocket=websocket,
            user_id=user_id or "anonymous",
            channels=channel_list
        )
        
        # Send welcome message
        await websocket.send_json({
            "type": "connection_established",
            "connection_id": connection_id,
            "channels": channel_list,
            "timestamp": datetime.now().isoformat(),
            "message": "Connected to LCA WebSocket server"
        })
        
        # Log connection
        await audit_logger.log_event(
            action="WEBSOCKET_CONNECTED",
            user_id=user_id or "anonymous",
            resource_type="websocket",
            resource_id=connection_id,
            details={"channels": channel_list}
        )
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                msg_type = message.get("type")
                
                if msg_type == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif msg_type == "subscribe":
                    # Subscribe to additional channels
                    new_channels = message.get("channels", [])
                    await websocket_service.subscribe_to_channels(connection_id, new_channels)
                    await websocket.send_json({
                        "type": "subscribed",
                        "channels": new_channels,
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif msg_type == "unsubscribe":
                    # Unsubscribe from channels
                    remove_channels = message.get("channels", [])
                    await websocket_service.unsubscribe_from_channels(connection_id, remove_channels)
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "channels": remove_channels,
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif msg_type == "message":
                    # Broadcast message to channel
                    channel = message.get("channel")
                    content = message.get("content")
                    
                    if channel and content:
                        await websocket_service.broadcast_to_channel(
                            channel=channel,
                            message={
                                "type": "user_message",
                                "from_user": user_id or "anonymous",
                                "content": content,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                
                else:
                    # Unknown message type
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                        "timestamp": datetime.now().isoformat()
                    })
            
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        # Client disconnected
        await websocket_service.disconnect(connection_id)
        
        # Log disconnection
        await audit_logger.log_event(
            action="WEBSOCKET_DISCONNECTED",
            user_id=user_id or "anonymous",
            resource_type="websocket",
            resource_id=connection_id,
            details={"reason": "client_disconnect"}
        )
    
    except Exception as e:
        # Error occurred
        print(f"WebSocket error: {e}")
        
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass
        
        # Disconnect
        await websocket_service.disconnect(connection_id)
        
        # Log error
        await audit_logger.log_event(
            action="WEBSOCKET_ERROR",
            user_id=user_id or "anonymous",
            resource_type="websocket",
            resource_id=connection_id,
            details={"error": str(e)}
        )


# ==================== REST Endpoints for WebSocket Management ====================

@router.get("/channels")
async def list_channels(current_user: dict = Depends(get_current_active_user)):
    """
    List all available WebSocket channels
    """
    channels = [
        {
            "channel": WSChannels.SYSTEM.value,
            "description": "System-wide announcements and alerts"
        },
        {
            "channel": WSChannels.PATIENT_UPDATES.value,
            "description": "Patient data changes and updates"
        },
        {
            "channel": WSChannels.RECOMMENDATIONS.value,
            "description": "New treatment recommendations"
        },
        {
            "channel": WSChannels.HITL_QUEUE.value,
            "description": "HITL review queue updates"
        },
        {
            "channel": WSChannels.BATCH_JOBS.value,
            "description": "Batch job progress and completion"
        },
        {
            "channel": WSChannels.ANALYTICS.value,
            "description": "Analytics computation results"
        }
    ]
    
    return {"channels": channels}


@router.get("/connections")
async def list_active_connections(current_user: dict = Depends(get_current_active_user)):
    """
    List all active WebSocket connections (admin only)
    """
    # Check admin permission
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    connections = await websocket_service.get_active_connections()
    
    return {
        "total_connections": len(connections),
        "connections": connections
    }


@router.post("/broadcast")
async def broadcast_message(
    channel: str,
    message: Dict[str, Any],
    current_user: dict = Depends(get_current_active_user)
):
    """
    Broadcast a message to a WebSocket channel (admin only)
    
    Example:
    {
        "channel": "system",
        "message": {
            "type": "announcement",
            "title": "System Maintenance",
            "content": "Scheduled maintenance at 2 AM UTC",
            "severity": "info"
        }
    }
    """
    # Check admin permission
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # Add metadata
    message_with_metadata = {
        **message,
        "from_user": current_user["sub"],
        "timestamp": datetime.now().isoformat()
    }
    
    # Broadcast
    sent_count = await websocket_service.broadcast_to_channel(channel, message_with_metadata)
    
    # Log audit event
    await audit_logger.log_event(
        action="WEBSOCKET_BROADCAST",
        user_id=current_user["sub"],
        resource_type="websocket",
        resource_id=channel,
        details={
            "channel": channel,
            "message_type": message.get("type"),
            "recipients": sent_count
        }
    )
    
    return {
        "message": "Broadcast sent",
        "channel": channel,
        "recipients": sent_count
    }


@router.post("/send/{user_id}")
async def send_to_user(
    user_id: str,
    message: Dict[str, Any],
    current_user: dict = Depends(get_current_active_user)
):
    """
    Send a direct message to a specific user via WebSocket
    """
    # Add metadata
    message_with_metadata = {
        **message,
        "from_user": current_user["sub"],
        "timestamp": datetime.now().isoformat()
    }
    
    # Send to user
    sent = await websocket_service.send_to_user(user_id, message_with_metadata)
    
    if not sent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not connected"
        )
    
    # Log audit event
    await audit_logger.log_event(
        action="WEBSOCKET_DIRECT_MESSAGE",
        user_id=current_user["sub"],
        resource_type="websocket",
        resource_id=user_id,
        details={"message_type": message.get("type")}
    )
    
    return {
        "message": "Message sent",
        "recipient": user_id
    }


@router.get("/stats")
async def get_websocket_stats(current_user: dict = Depends(get_current_active_user)):
    """
    Get WebSocket connection statistics
    """
    # Check admin permission
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    stats = await websocket_service.get_connection_stats()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "statistics": stats
    }


@router.delete("/connections/{connection_id}")
async def disconnect_connection(
    connection_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Force disconnect a WebSocket connection (admin only)
    """
    # Check admin permission
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    success = await websocket_service.disconnect(connection_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Connection {connection_id} not found"
        )
    
    # Log audit event
    await audit_logger.log_event(
        action="WEBSOCKET_FORCE_DISCONNECT",
        user_id=current_user["sub"],
        resource_type="websocket",
        resource_id=connection_id,
        details={}
    )
    
    return {"message": f"Connection {connection_id} disconnected"}
