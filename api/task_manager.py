from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class TaskInfo:
    task_id: str
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    config: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    metrics_file: str | None = None
    thread: threading.Thread | None = field(default=None, repr=False)
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    created_at: float = field(default_factory=time.time)


class TaskManager:
    """Manages long-running background tasks (training, forecasting, etc.)."""

    MAX_AGE_SECONDS = 24 * 3600  # Clean up tasks older than 24h

    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = threading.Lock()

    def create(self, task_type: str, config: dict[str, Any] | None = None) -> TaskInfo:
        """Create a new task entry and return it."""
        task_id = uuid.uuid4().hex[:12]
        task = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            config=config or {},
        )
        with self._lock:
            self._tasks[task_id] = task
        self._cleanup_old()
        return task

    def get(self, task_id: str) -> TaskInfo | None:
        """Retrieve a task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        """Signal a task to stop. Returns True if the task was found."""
        task = self.get(task_id)
        if task is None:
            return False
        with task.lock:
            if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                task.stop_event.set()
                task.status = TaskStatus.CANCELLED
                logger.info("Task %s cancelled", task_id)
                return True
        return False

    def list_tasks(self, task_type: str | None = None) -> list[TaskInfo]:
        """List all tasks, optionally filtered by type."""
        with self._lock:
            tasks = list(self._tasks.values())
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def _cleanup_old(self) -> None:
        """Remove completed/failed tasks older than MAX_AGE_SECONDS."""
        now = time.time()
        terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
        with self._lock:
            to_remove = [
                tid
                for tid, t in self._tasks.items()
                if t.status in terminal and (now - t.created_at) > self.MAX_AGE_SECONDS
            ]
            for tid in to_remove:
                del self._tasks[tid]
        if to_remove:
            logger.debug("Cleaned up %d old tasks", len(to_remove))


task_manager = TaskManager()
