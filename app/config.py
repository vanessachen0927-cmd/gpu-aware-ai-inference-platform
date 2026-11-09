import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    max_queue_depth: int = int(os.getenv("MAX_QUEUE_DEPTH", "100"))
    result_ttl_seconds: int = int(os.getenv("RESULT_TTL_SECONDS", "3600"))
    model_backend: str = os.getenv("MODEL_BACKEND", "mock")
    model_id: str = os.getenv("MODEL_ID", "sshleifer/tiny-gpt2")
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "128"))
    worker_poll_seconds: int = int(os.getenv("WORKER_POLL_SECONDS", "2"))
