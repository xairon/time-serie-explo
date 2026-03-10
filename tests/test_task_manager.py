from api.task_manager import TaskManager, TaskStatus


def test_create_task():
    tm = TaskManager()
    task = tm.create("training", {"model": "TFT"})
    assert len(task.task_id) == 12
    fetched = tm.get(task.task_id)
    assert fetched is not None
    assert fetched.status == TaskStatus.PENDING


def test_cancel_task():
    tm = TaskManager()
    task = tm.create("training", {})
    assert tm.cancel(task.task_id)
    fetched = tm.get(task.task_id)
    assert fetched.status == TaskStatus.CANCELLED


def test_cancel_nonexistent():
    tm = TaskManager()
    assert not tm.cancel("nonexistent")


def test_list_tasks():
    tm = TaskManager()
    tm.create("training", {})
    tm.create("counterfactual", {})
    assert len(tm.list_tasks()) == 2
    assert len(tm.list_tasks(task_type="training")) == 1
