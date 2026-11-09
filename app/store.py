import json
import uuid
from typing import Any

import redis


QUEUE_KEY = "inference:queue"
JOB_PREFIX = "inference:job:"


class QueueFull(Exception):
    pass


class RedisStore:
    def __init__(self, client: redis.Redis, max_depth: int = 100, ttl_seconds: int = 3600):
        self.client = client
        self.max_depth = max_depth
        self.ttl_seconds = ttl_seconds

    def ping(self) -> bool:
        return bool(self.client.ping())

    def queue_depth(self) -> int:
        return int(self.client.llen(QUEUE_KEY))

    def submit(self, prompt: str, max_new_tokens: int) -> str:
        job_id = str(uuid.uuid4())
        payload = {"prompt": prompt, "max_new_tokens": max_new_tokens}
        script = """
        if redis.call('LLEN', KEYS[1]) >= tonumber(ARGV[1]) then
            return 0
        end
        redis.call('SETEX', KEYS[2], tonumber(ARGV[2]), ARGV[3])
        redis.call('RPUSH', KEYS[1], ARGV[4])
        return 1
        """
        initial = json.dumps({"job_id": job_id, "status": "queued", "payload": payload})
        accepted = self.client.eval(
            script, 2, QUEUE_KEY, JOB_PREFIX + job_id,
            self.max_depth, self.ttl_seconds, initial, job_id
        )
        if not accepted:
            raise QueueFull("Inference queue is at capacity")
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        raw = self.client.get(JOB_PREFIX + job_id)
        return json.loads(raw) if raw else None

    def update(self, job_id: str, **fields: Any) -> None:
        key = JOB_PREFIX + job_id
        with self.client.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    if raw is None:
                        pipe.unwatch()
                        return
                    record = json.loads(raw)
                    record.update(fields)
                    pipe.multi()
                    pipe.setex(key, self.ttl_seconds, json.dumps(record))
                    pipe.execute()
                    return
                except redis.WatchError:
                    continue

    def next_job(self, timeout: int = 2) -> str | None:
        item = self.client.blpop(QUEUE_KEY, timeout=timeout)
        return item[1].decode() if item else None
