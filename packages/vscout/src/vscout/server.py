"""V-SCOUT API server.

Provides REST API for:
- Running valoscribe analysis on VODs
- Browsing match data (events, rounds, player states)
- Session management
"""

import subprocess
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vscout.data_loader import load_match, MatchData
from vscout.session_manager import SessionManager
from vscout.utils import setup_logger

logger = setup_logger("Server")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="V-SCOUT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_manager = SessionManager(str(DATA_DIR))


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------
class JobState:
    id: Optional[str] = None
    is_running: bool = False
    progress: float = 0.0
    status: str = "idle"
    session_id: Optional[str] = None


current_job = JobState()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    """Request to start VOD analysis."""
    vlr_url: Optional[str] = None
    youtube_url: Optional[str] = None
    local_video_path: Optional[str] = None
    start_time: Optional[float] = None
    duration: Optional[float] = None
    session_id: Optional[str] = None


class JobStatus(BaseModel):
    id: Optional[str]
    is_running: bool
    progress: float
    status: str
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Background analysis task
# ---------------------------------------------------------------------------
def run_valoscribe_task(job_id: str, req: AnalyzeRequest):
    """Run valoscribe in a subprocess."""
    global current_job
    current_job.id = job_id
    current_job.is_running = True
    current_job.status = "processing"
    current_job.progress = 0.0

    session_id = req.session_id or f"vs_{uuid.uuid4().hex[:12]}"
    current_job.session_id = session_id
    output_dir = DATA_DIR / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if req.vlr_url:
            # Use the shell script for full VLR pipeline
            cmd = [
                "bash", "packages/valoscribe/scripts/process_vlr_series.sh",
                req.vlr_url, str(output_dir),
            ]
        elif req.youtube_url or req.local_video_path:
            # Direct valoscribe orchestrate
            cmd = ["uv", "run", "valoscribe", "orchestrate", "process"]
            if req.youtube_url:
                cmd += ["--youtube-url", req.youtube_url]
            if req.local_video_path:
                cmd += ["--video-path", req.local_video_path]
            cmd += ["--output-dir", str(output_dir)]
            if req.start_time is not None:
                cmd += ["--start-time", str(req.start_time)]
        else:
            raise ValueError("Provide vlr_url, youtube_url, or local_video_path")

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

        if result.returncode == 0:
            current_job.status = "completed"
            current_job.progress = 1.0
        else:
            logger.error(f"valoscribe failed: {result.stderr[-500:]}")
            current_job.status = f"error: {result.stderr[-200:]}"

    except subprocess.TimeoutExpired:
        current_job.status = "error: timeout (2h)"
    except Exception as e:
        logger.error(f"Job failed: {e}")
        current_job.status = f"error: {str(e)}"
    finally:
        current_job.is_running = False


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health():
    return {"app": "V-SCOUT", "status": "running"}


@app.get("/api/status")
def get_status():
    return JobStatus(
        id=current_job.id,
        is_running=current_job.is_running,
        progress=current_job.progress,
        status=current_job.status,
        session_id=current_job.session_id,
    )


@app.post("/api/analyze")
def start_analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    if current_job.is_running:
        raise HTTPException(status_code=400, detail="A job is already running")
    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_valoscribe_task, job_id, req)
    return {"message": "Analysis started", "job_id": job_id}


@app.get("/api/sessions")
def list_sessions():
    """List all analysis sessions with their match info."""
    sessions = []
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        # Look for valoscribe output dirs (contain event_log.jsonl)
        event_logs = list(d.rglob("event_log.jsonl"))
        if not event_logs:
            continue
        sessions.append({
            "session_id": d.name,
            "maps": [
                {
                    "path": str(el.parent.relative_to(DATA_DIR)),
                    "name": el.parent.parent.name if el.parent.name == "output" else el.parent.name,
                }
                for el in event_logs
            ],
        })
    return {"sessions": sessions}


@app.get("/api/matches/{session_id}/{map_path:path}")
def get_match(session_id: str, map_path: str):
    """Get full match data for a specific map."""
    output_dir = DATA_DIR / session_id / map_path
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Match output not found")

    try:
        match = load_match(output_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "team1": match.team1,
        "team2": match.team2,
        "map_name": match.map_name,
        "final_score": match.final_score,
        "winner": match.winner,
        "total_events": len(match.events),
        "total_rounds": len(match.rounds),
        "rounds": [
            {
                "round_number": r.round_number,
                "start_timestamp": r.start_timestamp,
                "end_timestamp": r.end_timestamp,
                "duration": r.duration,
                "score": (r.score_team1, r.score_team2),
                "winner": r.winner,
                "kills": len(r.kills),
                "abilities": len(r.abilities),
                "ultimates": len(r.ultimates),
                "spike_events": len(r.spike_events),
            }
            for r in match.rounds
        ],
    }


@app.get("/api/matches/{session_id}/{map_path:path}/events")
def get_match_events(
    session_id: str,
    map_path: str,
    event_type: Optional[str] = None,
    round_number: Optional[int] = None,
):
    """Get events for a match, optionally filtered."""
    output_dir = DATA_DIR / session_id / map_path
    try:
        match = load_match(output_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    events = match.events
    if event_type:
        events = [e for e in events if e.type == event_type]
    if round_number is not None:
        rd = match.get_round(round_number)
        if rd:
            events = [
                e for e in events
                if rd.start_timestamp <= e.timestamp <= (rd.end_timestamp or float("inf"))
            ]
        else:
            events = []

    return {
        "count": len(events),
        "events": [
            {"type": e.type, "timestamp": e.timestamp, **e.data}
            for e in events
        ],
    }


@app.get("/api/matches/{session_id}/{map_path:path}/rounds/{round_number}")
def get_round_detail(session_id: str, map_path: str, round_number: int):
    """Get detailed data for a specific round."""
    output_dir = DATA_DIR / session_id / map_path
    try:
        match = load_match(output_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    rd = match.get_round(round_number)
    if not rd:
        raise HTTPException(status_code=404, detail=f"Round {round_number} not found")

    return {
        "round_number": rd.round_number,
        "start_timestamp": rd.start_timestamp,
        "end_timestamp": rd.end_timestamp,
        "duration": rd.duration,
        "score": (rd.score_team1, rd.score_team2),
        "winner": rd.winner,
        "kills": [
            {"type": e.type, "timestamp": e.timestamp, **e.data}
            for e in rd.kills
        ],
        "abilities": [
            {"type": e.type, "timestamp": e.timestamp, **e.data}
            for e in rd.abilities
        ],
        "ultimates": [
            {"type": e.type, "timestamp": e.timestamp, **e.data}
            for e in rd.ultimates
        ],
        "spike_events": [
            {"type": e.type, "timestamp": e.timestamp, **e.data}
            for e in rd.spike_events
        ],
        "deaths": [
            {"type": e.type, "timestamp": e.timestamp, **e.data}
            for e in rd.deaths
        ],
    }


@app.get("/api/matches/{session_id}/{map_path:path}/kills")
def get_kill_timeline(session_id: str, map_path: str):
    """Get kill timeline for a match."""
    output_dir = DATA_DIR / session_id / map_path
    try:
        match = load_match(output_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    kills = match.get_kill_timeline()
    return {
        "total_kills": len(kills),
        "kills": [
            {"timestamp": e.timestamp, **e.data}
            for e in kills
        ],
    }


def main():
    uvicorn.run("vscout.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
