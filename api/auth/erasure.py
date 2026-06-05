import logging
import shutil
import uuid
from pathlib import Path

from api.config import settings
from api.task_manager import task_manager

logger = logging.getLogger(__name__)


def erase_user_artifacts(user_id: uuid.UUID) -> None:
    """Best-effort deletion of resources owned by a user.

    Covers in-memory tasks, prepared-dataset directories and MLflow runs tagged
    with the user's ``owner_id``. Failures are logged but never raised so account
    deletion always completes.
    """
    uid = str(user_id)

    # 1. In-memory tasks
    try:
        for t in task_manager.list_tasks():
            if getattr(t, "owner_id", None) == uid:
                task_manager.cancel(t.task_id)
    except Exception as e:
        logger.warning("task erase failed for %s: %s", uid, e)

    # 2. Dataset directories owned by the user
    try:
        from dashboard.utils.dataset_registry import DatasetRegistry

        reg = DatasetRegistry(Path(settings.data_dir) / "prepared")
        for ds in reg.scan_datasets(owner_id=uid):
            try:
                shutil.rmtree(ds.path, ignore_errors=True)
            except Exception:
                continue
    except Exception as e:
        logger.warning("dataset erase failed for %s: %s", uid, e)

    # 3. MLflow runs tagged owner_id == user
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()
        for exp in client.search_experiments():
            runs = client.search_runs(
                [exp.experiment_id], filter_string=f"tags.owner_id = '{uid}'"
            )
            for run in runs:
                try:
                    client.delete_run(run.info.run_id)
                except Exception:
                    continue
    except Exception as e:
        logger.warning("mlflow erase failed for %s: %s", uid, e)
