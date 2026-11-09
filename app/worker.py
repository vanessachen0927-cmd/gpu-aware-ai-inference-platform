import logging
import signal
import time

import redis

from app.config import Settings
from app.model import ModelBackend, create_backend
from app.store import RedisStore


logger = logging.getLogger(__name__)


def process_one(store: RedisStore, backend: ModelBackend, job_id: str) -> bool:
    record = store.get(job_id)
    if record is None or record["status"] != "queued":
        return False
    store.update(job_id, status="running")
    started = time.perf_counter()
    try:
        payload = record["payload"]
        output = backend.generate(payload["prompt"], payload["max_new_tokens"])
        store.update(
            job_id,
            status="completed",
            output=output,
            latency_seconds=round(time.perf_counter() - started, 6),
        )
        return True
    except Exception as exc:
        logger.exception("Inference failed for job %s", job_id)
        store.update(
            job_id,
            status="failed",
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
            latency_seconds=round(time.perf_counter() - started, 6),
        )
        return False


def run() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    store = RedisStore(
        redis.Redis.from_url(settings.redis_url),
        settings.max_queue_depth,
        settings.result_ttl_seconds,
    )
    backend = create_backend(settings.model_backend, settings.model_id)
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    logger.info("Worker started with backend=%s", settings.model_backend)
    while running:
        try:
            job_id = store.next_job(settings.worker_poll_seconds)
            if job_id:
                process_one(store, backend, job_id)
        except redis.RedisError:
            logger.exception("Redis error, retrying")
            time.sleep(2)


if __name__ == "__main__":
    run()
