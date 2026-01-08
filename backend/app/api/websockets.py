from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.api.v1.endpoints.training import _jobs
import asyncio
import json
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/training/{job_id}")
async def websocket_training_status(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        last_progress = -1.0
        last_status = ""
        
        while True:
            if job_id not in _jobs:
                await websocket.send_json({
                    "event_type": "error",
                    "error": "Job not found"
                })
                await websocket.close()
                return

            job = _jobs[job_id]
            
            # Send update if changed
            if job.progress != last_progress or job.status != last_status:
                payload = {
                    "event_type": "progress",
                    "job_id": job.job_id,
                    "status": job.status,
                    "progress": job.progress,
                    "epoch": job.current_epoch,
                    "total_epochs": job.total_epochs,
                    "train_loss": job.train_loss,
                    "val_loss": job.val_loss,
                    "error": job.error_message
                }
                await websocket.send_json(payload)
                last_progress = job.progress
                last_status = job.status
            
            if job.status in ["completed", "failed", "cancelled"]:
                # Send one final update to ensure 100% or error is caught
                payload = {
                    "event_type": "completed" if job.status == "completed" else "error",
                    "job_id": job.job_id,
                    "status": job.status,
                    "progress": job.progress,
                    "error": job.error_message
                }
                await websocket.send_json(payload)
                break
                
            await asyncio.sleep(0.5)
            
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass
