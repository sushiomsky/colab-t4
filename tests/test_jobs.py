import threading
import time

from colab_t4.jobs import JobManager


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def test_job_transitions_from_queued_to_running_to_succeeded():
    started = threading.Event()
    release = threading.Event()
    manager = JobManager()

    def work(context):
        started.set()
        context.progress("halfway", 50)
        release.wait(2)
        return "finished"

    job = manager.submit("example", work)
    assert job.to_dict()["status"] in {"queued", "running"}
    assert started.wait(2)
    assert manager.get(job.id).status == "running"
    assert manager.get(job.id).progress == {"message": "halfway", "percent": 50}

    release.set()
    wait_for(lambda: manager.get(job.id).status == "succeeded")
    data = manager.get(job.id).to_dict()
    assert data["result"] == "finished"
    assert data["error"] is None
    assert data["updated_at"] >= data["created_at"]


def test_job_captures_redacted_exception(monkeypatch):
    monkeypatch.setenv("COLAB_T4_API_KEY", "secret-value")
    manager = JobManager()

    def work(context):
        raise RuntimeError("failed with COLAB_T4_API_KEY=secret-value")

    job = manager.submit("failure", work)
    wait_for(lambda: manager.get(job.id).status == "failed")
    assert manager.get(job.id).error == "failed with COLAB_T4_API_KEY=[REDACTED]"


def test_cancel_sets_flag_and_prevents_queued_job_from_running():
    manager = JobManager()
    gate = threading.Event()
    first = manager.submit("first", lambda context: gate.wait(2))
    wait_for(lambda: manager.get(first.id).status == "running")
    second_started = threading.Event()
    second = manager.submit("second", lambda context: second_started.set())

    assert manager.cancel(second.id) is True
    assert manager.cancel(second.id) is False
    gate.set()
    wait_for(lambda: manager.get(second.id).status == "cancelled")
    assert not second_started.is_set()
    assert manager.get(second.id).cancelled


def test_cancelled_context_is_visible_to_running_target():
    manager = JobManager()
    started = threading.Event()
    observed = threading.Event()

    def work(context):
        started.set()
        while not context.cancelled:
            time.sleep(0.005)
        observed.set()

    job = manager.submit("cancellable", work)
    assert started.wait(2)
    assert manager.cancel(job.id) is True
    assert observed.wait(2)
    wait_for(lambda: manager.get(job.id).status == "cancelled")


def test_progress_result_and_error_are_redacted_recursively(monkeypatch):
    monkeypatch.setenv("COLAB_T4_API_KEY", "secret-value")
    manager = JobManager()

    def work(context):
        context.progress("token=secret-value", 25)
        return {"output": "secret-value", "items": ["safe", "secret-value"]}

    job = manager.submit("redaction", work)
    wait_for(lambda: manager.get(job.id).status == "succeeded")
    data = manager.get(job.id).to_dict()
    assert data["progress"]["message"] == "token=[REDACTED]"
    assert data["result"] == {"output": "[REDACTED]", "items": ["safe", "[REDACTED]"]}


def test_result_redacts_secret_bearing_dictionary_keys(monkeypatch):
    monkeypatch.setenv("COLAB_T4_API_KEY", "secret-value")
    manager = JobManager()

    job = manager.submit("keys", lambda context: {"secret-value": "safe"})
    wait_for(lambda: manager.get(job.id).status == "succeeded")

    assert manager.get(job.id).to_dict()["result"] == {"[REDACTED]": "safe"}


def test_result_redacts_unsupported_object_representation(monkeypatch):
    monkeypatch.setenv("COLAB_T4_API_KEY", "secret-value")
    manager = JobManager()

    class Unsupported:
        def __repr__(self):
            return "Unsupported(secret-value)"

    job = manager.submit("object", lambda context: Unsupported())
    wait_for(lambda: manager.get(job.id).status == "succeeded")

    assert manager.get(job.id).to_dict()["result"] == "Unsupported([REDACTED])"


def test_job_name_is_redacted(monkeypatch):
    monkeypatch.setenv("COLAB_T4_API_KEY", "secret-value")
    manager = JobManager()

    job = manager.submit("job-secret-value", lambda context: "done")
    wait_for(lambda: manager.get(job.id).status == "succeeded")

    assert job.name == "job-[REDACTED]"
    assert manager.get(job.id).to_dict()["name"] == "job-[REDACTED]"


def test_serialized_job_name_is_redacted_after_external_mutation(monkeypatch):
    monkeypatch.setenv("COLAB_T4_API_KEY", "secret-value")
    manager = JobManager()

    job = manager.submit("safe-name", lambda context: "done")
    wait_for(lambda: manager.get(job.id).status == "succeeded")
    job.name = "mutated-secret-value"

    assert job.to_dict()["name"] == "mutated-[REDACTED]"


def test_manager_keeps_only_latest_fifty_jobs():
    manager = JobManager()
    jobs = [manager.submit(str(index), lambda context: index) for index in range(51)]
    wait_for(lambda: manager.get(jobs[-1].id).status == "succeeded")
    assert manager.get(jobs[0].id) is None
    assert manager.get(jobs[-1].id) is not None


def test_manager_preserves_running_and_queued_jobs_when_terminal_history_is_full():
    manager = JobManager()
    release = threading.Event()
    running = manager.submit("running", lambda context: release.wait(2))
    wait_for(lambda: manager.get(running.id).status == "running")
    queued = manager.submit("queued", lambda context: "queued")
    queued_tail = [manager.submit(f"queued-{index}", lambda context: index) for index in range(50)]

    assert manager.get(running.id) is not None
    assert manager.get(queued.id) is not None
    assert manager.get(queued.id).status == "queued"

    release.set()
    wait_for(lambda: manager.get(queued_tail[-1].id).status == "succeeded")


def test_cancellation_after_target_return_does_not_replace_success_result():
    """A cancellation that races only with result commit is too late to cancel work."""
    manager = JobManager()
    started = threading.Event()
    allow_return = threading.Event()
    commit_waiting = threading.Event()
    allow_commit = threading.Event()

    class CommitGate:
        def __init__(self):
            self._lock = threading.RLock()
            self._blocked = False

        def __enter__(self):
            if not self._blocked and allow_return.is_set():
                self._blocked = True
                commit_waiting.set()
                assert allow_commit.wait(2)
            return self._lock.__enter__()

        def __exit__(self, *args):
            return self._lock.__exit__(*args)

    def work(context):
        started.set()
        assert allow_return.wait(2)
        return {"state": "completed"}

    job = manager.submit("commit-race", work)
    assert started.wait(2)
    job._lock = CommitGate()
    allow_return.set()
    assert commit_waiting.wait(2)
    assert manager.cancel(job.id) is True
    allow_commit.set()
    wait_for(lambda: manager.get(job.id).status in {"succeeded", "cancelled"})

    data = manager.get(job.id).to_dict()
    assert data["status"] == "succeeded"
    assert data["result"] == {"state": "completed"}
    assert manager.cancel(job.id) is False
