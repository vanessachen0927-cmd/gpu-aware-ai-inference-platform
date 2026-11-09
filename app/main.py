from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from app.config import Settings
from app.schemas import InferenceRequest, JobResult, Submission
from app.store import QueueFull, RedisStore


REQUESTS = Counter("inference_http_requests_total", "Inference submissions", ["outcome"])
QUEUE_DEPTH = Gauge("inference_queue_depth", "Queued inference requests")
REQUEST_LATENCY = Histogram("inference_submit_latency_seconds", "Submission processing latency")


def create_app(store: RedisStore | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if store is None:
            settings = Settings()
            app.state.store = RedisStore(
                redis.Redis.from_url(settings.redis_url),
                settings.max_queue_depth,
                settings.result_ttl_seconds,
            )
        else:
            app.state.store = store
        yield

    api = FastAPI(title="GPU-Aware AI Inference Platform", version="0.1.0", lifespan=lifespan)

    @api.get("/health/live")
    def live():
        return {"status": "alive"}

    @api.get("/health/ready")
    def ready():
        try:
            if api.state.store.ping():
                return {"status": "ready"}
        except redis.RedisError:
            pass
        raise HTTPException(status_code=503, detail="Redis is unavailable")

    @api.post("/v1/infer", response_model=Submission, status_code=202)
    def submit(payload: InferenceRequest):
        try:
            with REQUEST_LATENCY.time():
                job_id = api.state.store.submit(payload.prompt, payload.max_new_tokens)
            REQUESTS.labels(outcome="accepted").inc()
            return Submission(job_id=job_id, status="queued")
        except QueueFull as exc:
            REQUESTS.labels(outcome="rejected").inc()
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except redis.RedisError as exc:
            REQUESTS.labels(outcome="unavailable").inc()
            raise HTTPException(status_code=503, detail="Queue service unavailable") from exc

    @api.get("/v1/jobs/{job_id}", response_model=JobResult)
    def result(job_id: str):
        try:
            record = api.state.store.get(job_id)
        except redis.RedisError as exc:
            raise HTTPException(status_code=503, detail="Queue service unavailable") from exc
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found or expired")
        return JobResult(**record)

    @api.get("/metrics")
    def metrics():
        try:
            QUEUE_DEPTH.set(api.state.store.queue_depth())
        except redis.RedisError:
            QUEUE_DEPTH.set(-1)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return api


app = create_app()
