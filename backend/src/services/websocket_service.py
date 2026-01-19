"""
WebSocket Service

Provides real-time bidirectional communication for live updates.
Complements SSE (Server-Sent Events) with full duplex communication.
"""

from typing import Dict, Set, Optional, Any, List
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import json
import asyncio


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str  # 'subscribe', 'unsubscribe', 'update', 'notification', 'error'
    channel: Optional[str] = None
    data: Any = None
    timestamp: datetime = None
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now()
        super().__init__(**data)


class ConnectionInfo(BaseModel):
    """Information about a WebSocket connection."""
    connection_id: str
    user_id: Optional[str] = None
    connected_at: datetime
    subscriptions: Set[str] = set()
    metadata: Dict[str, Any] = {}


class WebSocketManager:
    """Manages WebSocket connections and channels."""
    
    def __init__(self):
        """Initialize WebSocket manager."""
        # Active connections: connection_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Connection metadata: connection_id -> ConnectionInfo
        self.connection_info: Dict[str, ConnectionInfo] = {}
        
        # Channel subscriptions: channel -> set of connection_ids
        self.channels: Dict[str, Set[str]] = {}
        
        # Connection counter
        self._connection_counter = 0
    
    async def connect(
        self,
        websocket: WebSocket,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Accept a new WebSocket connection.
        
        Returns:
            connection_id
        """
        await websocket.accept()
        
        # Generate connection ID
        self._connection_counter += 1
        connection_id = f"ws_{self._connection_counter:08d}"
        
        # Store connection
        self.active_connections[connection_id] = websocket
        
        # Store connection info
        self.connection_info[connection_id] = ConnectionInfo(
            connection_id=connection_id,
            user_id=user_id,
            connected_at=datetime.now(),
            subscriptions=set(),
            metadata=metadata or {}
        )
        
        print(f"🔌 WebSocket connected: {connection_id} (user: {user_id})")
        
        # Send welcome message
        await self._send_to_connection(
            connection_id,
            WebSocketMessage(
                type='connected',
                data={
                    'connection_id': connection_id,
                    'message': 'Connected to LCA WebSocket'
                }
            )
        )
        
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Disconnect a WebSocket connection."""
        if connection_id not in self.active_connections:
            return
        
        # Get connection info
        info = self.connection_info.get(connection_id)
        
        # Unsubscribe from all channels
        if info:
            for channel in info.subscriptions:
                self._unsubscribe_from_channel(connection_id, channel)
        
        # Remove connection
        del self.active_connections[connection_id]
        if connection_id in self.connection_info:
            del self.connection_info[connection_id]
        
        print(f"🔌 WebSocket disconnected: {connection_id}")
    
    async def subscribe(self, connection_id: str, channel: str):
        """Subscribe a connection to a channel."""
        if connection_id not in self.active_connections:
            return
        
        # Add to channel
        if channel not in self.channels:
            self.channels[channel] = set()
        
        self.channels[channel].add(connection_id)
        
        # Update connection info
        info = self.connection_info.get(connection_id)
        if info:
            info.subscriptions.add(channel)
        
        print(f"📢 {connection_id} subscribed to {channel}")
        
        # Send confirmation
        await self._send_to_connection(
            connection_id,
            WebSocketMessage(
                type='subscribed',
                channel=channel,
                data={'message': f'Subscribed to {channel}'}
            )
        )
    
    async def unsubscribe(self, connection_id: str, channel: str):
        """Unsubscribe a connection from a channel."""
        self._unsubscribe_from_channel(connection_id, channel)
        
        # Send confirmation
        await self._send_to_connection(
            connection_id,
            WebSocketMessage(
                type='unsubscribed',
                channel=channel,
                data={'message': f'Unsubscribed from {channel}'}
            )
        )
    
    def _unsubscribe_from_channel(self, connection_id: str, channel: str):
        """Internal: Unsubscribe from channel without sending message."""
        if channel in self.channels:
            self.channels[channel].discard(connection_id)
            
            # Remove empty channels
            if not self.channels[channel]:
                del self.channels[channel]
        
        # Update connection info
        info = self.connection_info.get(connection_id)
        if info:
            info.subscriptions.discard(channel)
    
    async def broadcast(self, channel: str, message: WebSocketMessage):
        """Broadcast a message to all subscribers of a channel."""
        if channel not in self.channels:
            return
        
        # Get all subscribers
        subscribers = self.channels[channel].copy()
        
        print(f"📡 Broadcasting to {channel}: {len(subscribers)} subscribers")
        
        # Send to all subscribers
        for connection_id in subscribers:
            await self._send_to_connection(connection_id, message)
    
    async def send_to_user(self, user_id: str, message: WebSocketMessage):
        """Send a message to all connections of a specific user."""
        # Find all connections for this user
        user_connections = [
            conn_id for conn_id, info in self.connection_info.items()
            if info.user_id == user_id
        ]
        
        for connection_id in user_connections:
            await self._send_to_connection(connection_id, message)
    
    async def send_to_connection(self, connection_id: str, message: WebSocketMessage):
        """Send a message to a specific connection."""
        await self._send_to_connection(connection_id, message)
    
    async def _send_to_connection(self, connection_id: str, message: WebSocketMessage):
        """Internal: Send message to connection."""
        if connection_id not in self.active_connections:
            return
        
        websocket = self.active_connections[connection_id]
        
        try:
            # Convert message to JSON
            message_dict = message.dict()
            message_dict['timestamp'] = message.timestamp.isoformat()
            
            await websocket.send_json(message_dict)
        except Exception as e:
            print(f"❌ Error sending to {connection_id}: {e}")
            # Connection might be dead, disconnect it
            self.disconnect(connection_id)
    
    async def handle_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle incoming message from client."""
        msg_type = message.get('type')
        
        if msg_type == 'subscribe':
            channel = message.get('channel')
            if channel:
                await self.subscribe(connection_id, channel)
        
        elif msg_type == 'unsubscribe':
            channel = message.get('channel')
            if channel:
                await self.unsubscribe(connection_id, channel)
        
        elif msg_type == 'ping':
            # Respond with pong
            await self._send_to_connection(
                connection_id,
                WebSocketMessage(type='pong', data={'timestamp': datetime.now().isoformat()})
            )
        
        else:
            print(f"⚠️ Unknown message type: {msg_type}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket statistics."""
        return {
            'total_connections': len(self.active_connections),
            'total_channels': len(self.channels),
            'connections_by_channel': {
                channel: len(subscribers)
                for channel, subscribers in self.channels.items()
            },
            'active_users': len(set(
                info.user_id for info in self.connection_info.values()
                if info.user_id
            ))
        }


# Global WebSocket manager instance
ws_manager = WebSocketManager()
websocket_service = ws_manager  # Alias for consistency with other services


# Channel names for different event types
class WSChannels:
    """WebSocket channel names."""
    # Analysis updates
    ANALYSIS = "analysis"  # All analysis updates
    ANALYSIS_PROGRESS = "analysis:{analysis_id}"  # Specific analysis
    
    # Patient updates
    PATIENT = "patient:{patient_id}"
    
    # Review queue updates
    REVIEW_QUEUE = "review_queue"
    REVIEW_CASE = "review:{case_id}"
    
    # System notifications
    NOTIFICATIONS = "notifications:{user_id}"
    SYSTEM = "system"
    
    @staticmethod
    def analysis_progress(analysis_id: str) -> str:
        """Get channel name for specific analysis."""
        return f"analysis:{analysis_id}"
    
    @staticmethod
    def patient_updates(patient_id: str) -> str:
        """Get channel name for specific patient."""
        return f"patient:{patient_id}"
    
    @staticmethod
    def review_case(case_id: str) -> str:
        """Get channel name for specific review case."""
        return f"review:{case_id}"
    
    @staticmethod
    def user_notifications(user_id: str) -> str:
        """Get channel name for user notifications."""
        return f"notifications:{user_id}"


# Helper functions for common notifications

async def notify_analysis_started(analysis_id: str, patient_id: str):
    """Notify that analysis has started."""
    await ws_manager.broadcast(
        WSChannels.ANALYSIS,
        WebSocketMessage(
            type='analysis_started',
            channel=WSChannels.ANALYSIS,
            data={
                'analysis_id': analysis_id,
                'patient_id': patient_id,
                'status': 'started'
            }
        )
    )


async def notify_analysis_progress(
    analysis_id: str,
    agent_name: str,
    progress: float,
    message: str
):
    """Notify about analysis progress."""
    channel = WSChannels.analysis_progress(analysis_id)
    
    await ws_manager.broadcast(
        channel,
        WebSocketMessage(
            type='analysis_progress',
            channel=channel,
            data={
                'analysis_id': analysis_id,
                'agent': agent_name,
                'progress': progress,
                'message': message
            }
        )
    )


async def notify_analysis_completed(
    analysis_id: str,
    patient_id: str,
    recommendation: Dict[str, Any]
):
    """Notify that analysis is complete."""
    await ws_manager.broadcast(
        WSChannels.ANALYSIS,
        WebSocketMessage(
            type='analysis_completed',
            channel=WSChannels.ANALYSIS,
            data={
                'analysis_id': analysis_id,
                'patient_id': patient_id,
                'status': 'completed',
                'recommendation': recommendation
            }
        )
    )


async def notify_review_case_created(case_id: str, priority: str):
    """Notify that a new review case was created."""
    await ws_manager.broadcast(
        WSChannels.REVIEW_QUEUE,
        WebSocketMessage(
            type='review_case_created',
            channel=WSChannels.REVIEW_QUEUE,
            data={
                'case_id': case_id,
                'priority': priority,
                'status': 'pending'
            }
        )
    )


async def notify_review_case_updated(
    case_id: str,
    status: str,
    reviewer_id: Optional[str] = None
):
    """Notify that a review case was updated."""
    channel = WSChannels.review_case(case_id)
    
    await ws_manager.broadcast(
        channel,
        WebSocketMessage(
            type='review_case_updated',
            channel=channel,
            data={
                'case_id': case_id,
                'status': status,
                'reviewer_id': reviewer_id
            }
        )
    )


async def notify_user(user_id: str, notification_type: str, data: Dict[str, Any]):
    """Send notification to a specific user."""
    channel = WSChannels.user_notifications(user_id)
    
    await ws_manager.broadcast(
        channel,
        WebSocketMessage(
            type=notification_type,
            channel=channel,
            data=data
        )
    )
