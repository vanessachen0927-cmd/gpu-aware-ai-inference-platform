import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.model import MockBackend, create_backend
from app.store import QueueFull
from app.worker import process_one


class MemoryStore:
    def __init__(self, max_depth=100):
        self.items = {}
        self.queue = []
        self.max_depth = max_depth
        self.counter = 0
        self.online = True

    def ping(self):
        return self.online

    def queue_depth(self):
        return len(self.queue)

    def submit(self, prompt, max_new_tokens):
        if len(self.queue) >= self.max_depth:
            raise QueueFull("Inference queue is at capacity")
        self.counter += 1
        job_id = f"test-{self.counter}"
        self.items[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "payload": {"prompt": prompt, "max_new_tokens": max_new_tokens},
        }
        self.queue.append(job_id)
        return job_id

    def get(self, job_id):
        return self.items.get(job_id)

    def update(self, job_id, **fields):
        self.items[job_id].update(fields)

    def next_job(self, timeout=2):
        return self.queue.pop(0) if self.queue else None


@pytest.fixture
def setup():
    store = MemoryStore()
    with TestClient(create_app(store)) as client:
        yield client, store


def test_health_and_metrics(setup):
    client, store = setup
    assert client.get("/health/live").json() == {"status": "alive"}
    assert client.get("/health/ready").status_code == 200
    assert "inference_queue_depth" in client.get("/metrics").text
    store.online = False
    assert client.get("/health/ready").status_code == 503


def test_submit_process_retrieve(setup):
    client, store = setup
    result = client.post("/v1/infer", json={"prompt": "Hello", "max_new_tokens": 5})
    assert result.status_code == 202
    job_id = result.json()["job_id"]
    assert client.get(f"/v1/jobs/{job_id}").json()["status"] == "queued"
    assert process_one(store, MockBackend(), store.next_job())
    completed = client.get(f"/v1/jobs/{job_id}").json()
    assert completed["status"] == "completed"
    assert completed["output"] == "[mock inference] Hello"
    assert completed["latency_seconds"] >= 0


def test_validation_and_unknown_job(setup):
    client, _ = setup
    assert client.post("/v1/infer", json={"prompt": ""}).status_code == 422
    assert client.post("/v1/infer", json={"prompt": "x", "max_new_tokens": 200}).status_code == 422
    assert client.get("/v1/jobs/missing").status_code == 404


def test_full_queue():
    store = MemoryStore(max_depth=1)
    with TestClient(create_app(store)) as client:
        assert client.post("/v1/infer", json={"prompt": "first"}).status_code == 202
        assert client.post("/v1/infer", json={"prompt": "second"}).status_code == 429


def test_worker_failure(setup):
    _, store = setup
    job_id = store.submit("hello", 2)

    class FailingBackend:
        def generate(self, prompt, max_new_tokens):
            raise RuntimeError("model unavailable")

    assert not process_one(store, FailingBackend(), store.next_job())
    assert store.get(job_id)["status"] == "failed"
    assert "RuntimeError" in store.get(job_id)["error"]


def test_model_backend_guard():
    assert create_backend("mock", "unused").generate("abc", 2) == "[mock inference] ab"
    with pytest.raises(ValueError):
        create_backend("unsupported", "unused")
