from typing import Literal
from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=32, ge=1, le=128)


class Submission(BaseModel):
    job_id: str
    status: Literal["queued"]


class JobResult(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    output: str | None = None
    error: str | None = None
    latency_seconds: float | None = None
