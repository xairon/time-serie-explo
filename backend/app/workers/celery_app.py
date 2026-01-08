"""
Celery application configuration.
"""

from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "junon_workers",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
    include=["app.workers.training_tasks"],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Paris",
    enable_utc=True,
    
    # Task settings
    task_track_started=True,
    task_time_limit=3600 * 4,  # 4 hours max per training
    task_soft_time_limit=3600 * 3.5,
    
    # Result settings
    result_expires=3600 * 24,  # Results expire after 24 hours
    
    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time for GPU workloads
    worker_concurrency=1,  # Single worker for GPU memory
)
