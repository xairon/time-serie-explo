from api.task_manager import TaskManager, TaskStatus


def test_create_task():
    tm = TaskManager()
    task_id = tm.create("training", {"model": "TFT"})
    assert len(task_id) == 12
    task = tm.get(task_id)
    assert task is not None
    assert task.status == TaskStatus.PENDING


def test_cancel_task():
    tm = TaskManager()
    task_id = tm.create("training", {})
    assert tm.cancel(task_id)
    task = tm.get(task_id)
    assert task.status == TaskStatus.CANCELLED
    assert task.stop_event.is_set()


def test_cancel_nonexistent():
    tm = TaskManager()
    assert not tm.cancel("nonexistent")


def test_list_tasks():
    tm = TaskManager()
    tm.create("training", {})
    tm.create("counterfactual", {})
    assert len(tm.list_tasks()) == 2
    assert len(tm.list_tasks(task_type="training")) == 1
