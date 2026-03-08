"""VLR.gg match analysis pipeline.

Replaces the shell script ``process_vlr_series.sh`` with a pure-Python
implementation that exposes real-time progress via a shared
:class:`PipelineState` object the FastAPI server can poll.
"""

import json
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("vscout.pipeline")


# ---------------------------------------------------------------------------
# Shared state visible to the FastAPI polling endpoint
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """Mutable state object shared between the background worker and the API.

    The FastAPI endpoint reads this from the main thread while
    :func:`run_vlr_pipeline` mutates it from a background thread.
    All writes are simple attribute assignments on built-in types so they are
    effectively atomic on CPython (GIL).
    """

    job_id: str
    status: str = "idle"  # idle | scraping | downloading_map_N | processing_map_N | completed | error: ...
    is_running: bool = False
    progress: float = 0.0  # 0.0 – 1.0
    current_step: str = ""  # Human-readable description (Japanese)
    total_maps: int = 0
    current_map: int = 0
    session_id: str | None = None
    error: str | None = None
    cancel_requested: bool = False
    steps_log: list[str] = field(default_factory=list)  # Completed-step log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_step(state: PipelineState, message: str) -> None:
    """Append *message* to the steps log and update ``current_step``."""
    state.current_step = message
    state.steps_log.append(message)
    logger.info(message)


def _split_series_metadata(series_metadata: dict) -> list[dict]:
    """Convert series-level scraper output into per-map metadata dicts.

    Each returned dict matches the format expected by
    :class:`~valoscribe.orchestration.game_state_manager.GameStateManager`:

    .. code-block:: python

        {
            "teams": [{"name": ..., "starting_side": ...}, ...],
            "players": [{"name": ..., "team": ..., "agent": ...}, ...],
            "map": "Ascent",
            "map_number": 1,
            "vod_url": "https://...",
            "match_url": "https://...",
        }
    """
    result: list[dict] = []
    match_url = series_metadata.get("match_url", "")

    for map_data in series_metadata.get("maps", []):
        map_teams = map_data.get("teams", [])

        # Flatten players from nested team structure
        all_players: list[dict] = []
        for team in map_teams:
            team_name = team["name"]
            for player in team.get("players", []):
                all_players.append({
                    "name": player["name"],
                    "team": team_name,
                    "agent": player["agent"],
                })

        map_metadata = {
            "teams": [
                {
                    "name": team["name"],
                    "starting_side": team["starting_side"],
                }
                for team in map_teams
            ],
            "players": all_players,
            "map": map_data.get("map_name", "Unknown"),
            "map_number": map_data.get("map_number", 0),
            "vod_url": map_data.get("vod_url"),
            "match_url": match_url,
        }
        result.append(map_metadata)

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_vlr_pipeline(
    state: PipelineState,
    vlr_url: str,
    output_dir: Path,
) -> None:
    """Run the full VLR match-analysis pipeline.

    Designed to execute in a background thread.  Progress is communicated
    exclusively through *state* so that the FastAPI server can expose it to
    the frontend via a polling endpoint.

    Parameters
    ----------
    state:
        Shared mutable state object.
    vlr_url:
        VLR.gg match URL (e.g. ``https://www.vlr.gg/542272/...``).
    output_dir:
        Root directory for all outputs.  Sub-folders are created automatically
        per series / map.
    """
    # Late imports so the module can be imported without heavy deps at
    # import time (keeps the FastAPI startup fast).
    from valoscribe.scraper import scrape_match  # noqa: WPS433
    from valoscribe.video.youtube import download_youtube  # noqa: WPS433
    from valoscribe.orchestration.game_state_manager import GameStateManager  # noqa: WPS433
    from valoscribe.video.reader import VideoReader  # noqa: WPS433

    state.is_running = True
    state.progress = 0.0
    state.error = None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Temp directory for intermediate artefacts (downloaded VODs, etc.)
    temp_dir = output_dir / ".temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    vod_path: Path | None = None  # track for cleanup in finally blocks

    try:
        # ------------------------------------------------------------------
        # 1. Scrape VLR metadata
        # ------------------------------------------------------------------
        if state.cancel_requested:
            _abort(state)
            return

        state.status = "scraping"
        _log_step(state, "VLR.ggからメタデータを取得中...")

        try:
            series_metadata = scrape_match(vlr_url)
        except Exception as exc:
            _fail(state, f"VLRスクレイピング失敗: {exc}")
            return

        state.progress = 0.03

        # ------------------------------------------------------------------
        # 2. Extract series info & build folder structure
        # ------------------------------------------------------------------
        team_names = series_metadata.get("teams", ["Team1", "Team2"])
        _log_step(state, f"シリーズ情報を展開中: {team_names[0]} vs {team_names[1]}")

        # Derive a stable series folder name
        series_slug = (
            f"{team_names[0]}_vs_{team_names[1]}"
            .lower()
            .replace(" ", "_")
        )
        series_dir = output_dir / series_slug
        series_dir.mkdir(parents=True, exist_ok=True)

        # Persist raw series metadata
        series_meta_path = series_dir / "series_metadata.json"
        with open(series_meta_path, "w") as fh:
            json.dump(series_metadata, fh, indent=2, ensure_ascii=False)

        # ------------------------------------------------------------------
        # 3. Split metadata into per-map dicts
        # ------------------------------------------------------------------
        try:
            per_map_metadata = _split_series_metadata(series_metadata)
        except Exception as exc:
            _fail(state, f"メタデータ分割失敗: {exc}")
            return

        num_maps = len(per_map_metadata)
        state.total_maps = num_maps
        state.progress = 0.05
        _log_step(state, f"メタデータ分割完了 – {num_maps}マップ検出")

        if num_maps == 0:
            _fail(state, "マップが見つかりませんでした")
            return

        # Persist per-map metadata files
        metadata_dir = series_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        for mm in per_map_metadata:
            map_meta_path = metadata_dir / f"map{mm['map_number']}.json"
            with open(map_meta_path, "w") as fh:
                json.dump(mm, fh, indent=2, ensure_ascii=False)

        # ------------------------------------------------------------------
        # 4. Process each map
        # ------------------------------------------------------------------
        # Progress budget: 0.05 already used for scraping/splitting.
        # Remaining 0.95 is divided equally among maps.
        # Within each map: download = first half, process = second half.
        per_map_budget = 0.95 / num_maps

        for idx, map_meta in enumerate(per_map_metadata, start=1):
            # -- cancel check ------------------------------------------------
            if state.cancel_requested:
                _abort(state)
                return

            state.current_map = idx
            map_name: str = map_meta.get("map", "Unknown")
            vod_url: str | None = map_meta.get("vod_url")
            map_base_progress = 0.05 + (idx - 1) * per_map_budget

            # Human-readable folder: map1_haven
            map_folder_name = f"map{idx}_{map_name.lower()}"
            map_output_dir = series_dir / map_folder_name / "output"
            map_output_dir.mkdir(parents=True, exist_ok=True)

            # Copy metadata into map folder
            shutil.copy2(
                metadata_dir / f"map{map_meta['map_number']}.json",
                series_dir / map_folder_name / "metadata.json",
            )

            # ---- skip if no VOD URL ----------------------------------------
            if not vod_url:
                msg = f"マップ{idx} ({map_name}): VOD URLなし – スキップ"
                _log_step(state, msg)
                logger.warning(msg)
                state.progress = map_base_progress + per_map_budget
                continue

            # ---- 4a. Download VOD ------------------------------------------
            if state.cancel_requested:
                _abort(state)
                return

            state.status = f"downloading_map_{idx}"
            _log_step(state, f"マップ{idx}/{num_maps} ({map_name}) のVODをダウンロード中...")
            state.progress = map_base_progress

            vod_download_dir = temp_dir / "videos"
            vod_download_dir.mkdir(parents=True, exist_ok=True)

            try:
                dl_result = download_youtube(
                    vod_url,
                    out_dir=vod_download_dir,
                    prefer_height=1080,
                    prefer_fps=60,
                )
                vod_path = Path(dl_result.out_path)
            except Exception as exc:
                _fail(state, f"マップ{idx} VODダウンロード失敗: {exc}")
                return

            state.progress = map_base_progress + per_map_budget * 0.5
            _log_step(state, f"マップ{idx} ({map_name}) のVODダウンロード完了")

            # ---- 4b. Run orchestration -------------------------------------
            if state.cancel_requested:
                _cleanup_vod(vod_path)
                _abort(state)
                return

            state.status = f"processing_map_{idx}"
            _log_step(state, f"マップ{idx}/{num_maps} ({map_name}) を解析中...")

            try:
                manager = GameStateManager(
                    video_path=vod_path,
                    vlr_metadata=map_meta,
                    output_dir=map_output_dir,
                    fps=4.0,
                )

                with manager.output_writer:
                    video_reader = VideoReader(str(vod_path), fps_filter=4.0)
                    try:
                        for frame_info in video_reader:
                            if state.cancel_requested:
                                break
                            manager.process_frame(
                                frame_info.timestamp_sec,
                                frame_info.frame,
                            )
                    except StopIteration:
                        pass
                    manager._finalize_events()
            except Exception as exc:
                # Non-fatal per-map error: log but don't kill the whole pipeline
                msg = f"マップ{idx} ({map_name}) 解析失敗: {exc}"
                _log_step(state, msg)
                logger.exception(msg)
                # Continue to next map after cleanup
            finally:
                _cleanup_vod(vod_path)
                vod_path = None

            if state.cancel_requested:
                _abort(state)
                return

            state.progress = map_base_progress + per_map_budget
            _log_step(state, f"マップ{idx} ({map_name}) 解析完了")

        # ------------------------------------------------------------------
        # 5. Done
        # ------------------------------------------------------------------
        state.status = "completed"
        state.progress = 1.0
        state.session_id = series_dir.name
        _log_step(state, f"全{num_maps}マップの解析が完了しました")
        logger.info("Pipeline completed successfully: %s", series_dir)

    except Exception as exc:
        # Catch-all for truly unexpected errors
        _fail(state, f"予期しないエラー: {exc}")
        logger.exception("Unexpected error in pipeline")

    finally:
        state.is_running = False
        # Best-effort cleanup of temp directory
        _cleanup_vod(vod_path)
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fail(state: PipelineState, message: str) -> None:
    """Mark the pipeline as failed."""
    state.status = f"error: {message}"
    state.error = message
    _log_step(state, f"エラー: {message}")
    logger.error(message)


def _abort(state: PipelineState) -> None:
    """Mark the pipeline as cancelled."""
    state.status = "error: キャンセルされました"
    state.error = "キャンセルされました"
    _log_step(state, "パイプラインがキャンセルされました")
    logger.warning("Pipeline cancelled by user")


def _cleanup_vod(vod_path: Path | None) -> None:
    """Delete a downloaded VOD file if it exists."""
    if vod_path is None:
        return
    try:
        vod = Path(vod_path)
        if vod.exists():
            vod.unlink()
            logger.info("Cleaned up VOD: %s", vod)
    except Exception as exc:
        logger.warning("Failed to clean up VOD %s: %s", vod_path, exc)
