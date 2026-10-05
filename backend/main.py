import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Thread

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import database
from .config import get_mode_config
from .healthcheck import get_health_status, run_startup_self_test
from .job_queue import JobQueue
from .providers import log_provider_key_status

job_queue = JobQueue()


@asynccontextmanager
async def lifespan(_: FastAPI):
    log_provider_key_status()
    database.initialize()
    Thread(target=run_startup_self_test, name="startup-self-test", daemon=True).start()
    job_queue.start()
    try:
        yield
    finally:
        job_queue.stop()


app = FastAPI(title="PromptToVideo API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])
database.DATA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=database.DATA_DIR), name="media")


class JobRequest(BaseModel):
    mode: str
    fast_preview: bool = False
    prompt: str = Field(min_length=8, max_length=3000)
    duration: int
    language: str
    style: str
    voice: str

    @field_validator("mode")
    @classmethod
    def valid_mode(cls, value):
        if value not in ("video", "shorts"):
            raise ValueError("Mode must be video or shorts")
        return value

    @field_validator("duration")
    @classmethod
    def valid_duration(cls, value, info):
        mode = info.data.get("mode")
        if mode:
            config = get_mode_config(mode)
            min_duration = 180 if mode == "video" and info.data.get("fast_preview") else config.min_seconds
            if not min_duration <= value <= config.max_seconds:
                raise ValueError(f"Duration must be between {min_duration} and {config.max_seconds} seconds")
        return value

    @field_validator("language")
    @classmethod
    def valid_language(cls, value):
        if value not in ("Hindi", "English"):
            raise ValueError("Language must be Hindi or English")
        return value

    @field_validator("style")
    @classmethod
    def valid_style(cls, value):
        if value not in ("cinematic", "cartoon", "realistic", "storybook"):
            raise ValueError("Choose a supported visual style")
        return value

    @field_validator("voice")
    @classmethod
    def valid_voice(cls, value):
        if value not in ("male", "female"):
            raise ValueError("Voice must be male or female")
        return value


@app.get("/api/health")
def health():
    return get_health_status()


@app.get("/api/jobs")
def jobs():
    return database.list_jobs()


@app.post("/api/jobs", status_code=202)
def create_job(request: JobRequest):
    payload = request.model_dump()
    job_id = str(uuid.uuid4())
    database.create_job({"id": job_id, **payload})
    job_queue.notify()
    return database.get_job(job_id)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "running":
        raise HTTPException(status_code=409, detail="This job is already running")
    database.update_job(job_id, status="queued", stage="Queued for retry", stage_status="pending", error=None, warning=None)
    job_queue.notify()
    return database.get_job(job_id)


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str):
    job = database.get_job(job_id)
    if not job or not job.get("output_file"):
        raise HTTPException(status_code=404, detail="Video is not ready")
    output = Path(job["output_file"])
    if not output.exists():
        raise HTTPException(status_code=404, detail="Video file is missing")
    return FileResponse(output, media_type="video/mp4", filename=f"prompttovideo-{job_id[:8]}.mp4")