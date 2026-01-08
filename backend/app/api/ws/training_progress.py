"""
WebSocket endpoint for training progress streaming.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
from typing import Dict, Set

router = APIRouter()

# Connection manager for WebSocket clients
class ConnectionManager:
    """Manages WebSocket connections for training progress."""
    
    def __init__(self):
        # job_id -> set of connected websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        self.active_connections[job_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        """Remove a WebSocket connection."""
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
    
    async def broadcast(self, job_id: str, message: dict):
        """Broadcast a message to all clients watching a job."""
        if job_id in self.active_connections:
            disconnected = set()
            for websocket in self.active_connections[job_id]:
                try:
                    await websocket.send_json(message)
                except:
                    disconnected.add(websocket)
            
            # Clean up disconnected clients
            for ws in disconnected:
                self.active_connections[job_id].discard(ws)


manager = ConnectionManager()


@router.websocket("/training/{job_id}")
async def training_progress_websocket(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time training progress.
    
    Clients connect to receive:
    - progress: Current epoch progress
    - metric: Training/validation metrics
    - log: Log messages
    - completed: Training completed with results
    - error: Training failed
    """
    await manager.connect(websocket, job_id)
    
    try:
        # Send initial status
        await websocket.send_json({
            "event_type": "connected",
            "job_id": job_id,
            "message": f"Connected to training job {job_id}",
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages (could be used for cancellation)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,  # Heartbeat timeout
                )
                
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("type") == "cancel":
                    # Handle cancellation request
                    await websocket.send_json({
                        "event_type": "log",
                        "message": "Cancellation requested",
                    })
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)
    except Exception as e:
        manager.disconnect(websocket, job_id)
        raise


# Helper function for Celery workers to broadcast updates
async def send_training_update(job_id: str, update: dict):
    """
    Send a training update to all connected clients.
    
    Called from Celery tasks via Redis pub/sub or direct async call.
    """
    await manager.broadcast(job_id, update)
