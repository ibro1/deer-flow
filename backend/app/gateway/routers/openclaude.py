import asyncio
import logging
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/openclaude", tags=["openclaude"])

# In-memory job store for async endpoint
_jobs: dict[str, dict] = {}


class OpenClaudeRequest(BaseModel):
    task: str


async def _execute(task: str) -> dict:
    """Core execution - run openclaude and return result."""
    task_escaped = task.replace("'", "'\\''")
    cmd = [
        "/usr/bin/docker", "exec", "-u", "gem", "-i", "deer-flow-sandbox",
        "bash", "-c",
        f"export PATH=/mnt/shared/.npm-global/bin:/home/gem/.npm-global/bin:$PATH && "
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
        return {"success": proc.returncode == 0, "output": output}
    except asyncio.TimeoutError:
        return {"success": False, "output": "Timed out after 900 seconds"}
    except Exception as e:
        return {"success": False, "output": str(e)}


@router.post("/run")
async def run_openclaude_sync(request: OpenClaudeRequest):
    """
    Synchronous endpoint — waits for openclaude to finish before returning.
    Use this for DeerFlow agent tasks. No polling needed.
    Returns: {"success": bool, "output": str}
    """
    return await _execute(request.task)


@router.post("")
async def start_openclaude_async(request: OpenClaudeRequest):
    """
    Async endpoint — returns job_id immediately.
    Use for long tasks where you want to poll separately.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "output": "", "success": None}

    async def _run():
        _jobs[job_id]["status"] = "running"
        result = await _execute(request.task)
        _jobs[job_id]["status"] = "done" if result["success"] else "failed"
        _jobs[job_id]["output"] = result["output"]
        _jobs[job_id]["success"] = result["success"]

    asyncio.create_task(_run())
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Task started. Poll /openclaude/{job_id} for result."
    }


@router.get("/{job_id}")
async def get_openclaude_result(job_id: str):
    """Poll async job status. Status: queued | running | done | failed."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "success": job.get("success"),
        "output": job.get("output", ""),
    }
