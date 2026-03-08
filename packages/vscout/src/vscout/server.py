"""V-SCOUT API server.

Provides REST API for:
- Running valoscribe VLR analysis pipeline
- Browsing match data (events, rounds, player states)
- Session management
"""

import threading
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vscout.data_loader import load_match
from vscout.pipeline import PipelineState, run_vlr_pipeline
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


# ---------------------------------------------------------------------------
# Pipeline state (singleton)
# ---------------------------------------------------------------------------
current_pipeline: PipelineState | None = None


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    """Request to start VLR analysis."""
    vlr_url: str


class JobStatus(BaseModel):
    id: str | None
    is_running: bool
    progress: float
    status: str
    current_step: str
    total_maps: int
    current_map: int
    session_id: str | None = None
    steps_log: list[str] = []


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------
def _pipeline_thread(state: PipelineState, vlr_url: str, output_dir: Path) -> None:
    """Thin wrapper so that run_vlr_pipeline runs in a daemon thread."""
    try:
        run_vlr_pipeline(state, vlr_url, output_dir)
    except Exception as exc:
        logger.exception("Pipeline thread crashed: %s", exc)
        state.status = f"error: {exc}"
        state.error = str(exc)
        state.is_running = False


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"app": "V-SCOUT", "status": "running"}


@app.get("/api/status")
def get_status():
    if current_pipeline is None:
        return JobStatus(
            id=None,
            is_running=False,
            progress=0.0,
            status="idle",
            current_step="",
            total_maps=0,
            current_map=0,
        )
    return JobStatus(
        id=current_pipeline.job_id,
        is_running=current_pipeline.is_running,
        progress=current_pipeline.progress,
        status=current_pipeline.status,
        current_step=current_pipeline.current_step,
        total_maps=current_pipeline.total_maps,
        current_map=current_pipeline.current_map,
        session_id=current_pipeline.session_id,
        steps_log=list(current_pipeline.steps_log),
    )


@app.post("/api/stop")
def stop_analyze():
    if current_pipeline is None or not current_pipeline.is_running:
        raise HTTPException(status_code=400, detail="No job is running")
    current_pipeline.cancel_requested = True
    return {"message": "Stop requested"}


@app.post("/api/analyze")
def start_analyze(req: AnalyzeRequest):
    global current_pipeline

    if current_pipeline is not None and current_pipeline.is_running:
        raise HTTPException(status_code=400, detail="A job is already running")

    # Validate VLR URL
    url = req.vlr_url.strip()
    if not url or "vlr.gg" not in url:
        raise HTTPException(status_code=400, detail="有効なVLR.gg URLを入力してください")

    job_id = str(uuid.uuid4())
    state = PipelineState(job_id=job_id)

    current_pipeline = state

    thread = threading.Thread(
        target=_pipeline_thread,
        args=(state, url, DATA_DIR),
        daemon=True,
    )
    thread.start()

    return {"message": "Analysis started", "job_id": job_id}


@app.get("/api/sessions")
def list_sessions():
    """List all analysis sessions with their match info."""
    sessions = []
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        # Walk into series sub-directories
        event_logs = list(d.rglob("event_log.jsonl"))
        if not event_logs:
            continue
        sessions.append({
            "session_id": d.name,
            "maps": [
                {
                    "path": str(el.parent.relative_to(d)),
                    "name": el.parent.parent.name if el.parent.name == "output" else el.parent.name,
                }
                for el in event_logs
            ],
        })
    return {"sessions": sessions}


def _resolve_output_dir(session_id: str, map_path: str) -> Path:
    """Resolve the output directory for a session/map."""
    return DATA_DIR / session_id / map_path


@app.get("/api/matches/{session_id}/{map_path:path}/rounds/{round_number}")
def get_round_detail(session_id: str, map_path: str, round_number: int):
    """Get detailed data for a specific round."""
    output_dir = _resolve_output_dir(session_id, map_path)
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


@app.get("/api/matches/{session_id}/{map_path:path}/events")
def get_match_events(
    session_id: str,
    map_path: str,
    event_type: Optional[str] = None,
    round_number: Optional[int] = None,
):
    """Get events for a match, optionally filtered."""
    output_dir = _resolve_output_dir(session_id, map_path)
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


@app.get("/api/matches/{session_id}/{map_path:path}/kills")
def get_kill_timeline(session_id: str, map_path: str):
    """Get kill timeline for a match."""
    output_dir = _resolve_output_dir(session_id, map_path)
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


@app.get("/api/matches/{session_id}/{map_path:path}")
def get_match(session_id: str, map_path: str):
    """Get full match data for a specific map."""
    output_dir = _resolve_output_dir(session_id, map_path)
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


# ---------------------------------------------------------------------------
# Frontend static files (must be after all API routes)
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="frontend-assets")

    @app.get("/{path:path}")
    async def serve_frontend(request: Request, path: str):
        """Serve React SPA — all non-API routes return index.html."""
        file = FRONTEND_DIR / path
        if file.exists() and file.is_file():
            return FileResponse(file)
        return FileResponse(FRONTEND_DIR / "index.html")


def main():
    uvicorn.run("vscout.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
