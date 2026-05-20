
import asyncio
import logging
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/openclaude", tags=["openclaude"])

# In-memory job store
_jobs: dict[str, dict] = {}


class OpenClaudeRequest(BaseModel):
    task: str


class OpenClaudeJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


async def _run_task(job_id: str, task: str):
    """Run openclaude in background and store result."""
    _jobs[job_id]["status"] = "running"
    task_escaped = task.replace("'", "'\\''")
    cmd = [
        "/usr/bin/docker", "exec", "-u", "gem", "-i", "deer-flow-sandbox",
        "bash", "-c",
        f"export PATH=/home/gem/.npm-global/bin:$PATH && "
        f"export HOME=/home/gem && "
        f"export CLAUDE_CODE_USE_OPENAI=1 && "
        f"export OPENAI_BASE_URL=https://opengateway.gitlawb.com/v1/xiaomi-mimo && "
        f"export OPENAI_API_KEY=gitlawb && "
        f"export OPENAI_MODEL=mimo-v2.5-pro && "
        f"cd /mnt/shared && "
        f"openclaude -p --dangerously-skip-permissions '{task_escaped}' < /dev/null"
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=900)
        output = stdout.decode("utf-8", errors="replace")
        _jobs[job_id]["status"] = "done" if proc.returncode == 0 else "failed"
        _jobs[job_id]["output"] = output
        _jobs[job_id]["success"] = proc.returncode == 0
    except asyncio.TimeoutError:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["output"] = "Timed out after 900 seconds"
        _jobs[job_id]["success"] = False
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["output"] = str(e)
        _jobs[job_id]["success"] = False


@router.post("")
async def start_openclaude(request: OpenClaudeRequest):
    """Start an openclaude task asynchronously. Returns a job_id to poll."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "output": "", "success": None}
    asyncio.create_task(_run_task(job_id, request.task))
    return {"job_id": job_id, "status": "queued", "message": f"Task started. Poll /openclaude/{job_id} for result."}


@router.get("/{job_id}")
async def get_openclaude_result(job_id: str):
    """Poll job status. Status: queued | running | done | failed."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "success": job.get("success"),
        "output": job.get("output", ""),
    }
