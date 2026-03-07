"""Data loader for valoscribe output files.

Reads event_log.jsonl and frame_states.csv produced by valoscribe
and provides structured access to match data.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class MatchEvent:
    """A single game event."""
    type: str
    timestamp: float
    data: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "MatchEvent":
        event_type = d.pop("type")
        timestamp = d.pop("timestamp")
        return cls(type=event_type, timestamp=timestamp, data=d)


@dataclass
class RoundData:
    """Aggregated data for a single round."""
    round_number: int
    start_timestamp: float
    end_timestamp: Optional[float] = None
    score_team1: int = 0
    score_team2: int = 0
    winner: Optional[str] = None
    kills: list[MatchEvent] = field(default_factory=list)
    abilities: list[MatchEvent] = field(default_factory=list)
    ultimates: list[MatchEvent] = field(default_factory=list)
    spike_events: list[MatchEvent] = field(default_factory=list)
    deaths: list[MatchEvent] = field(default_factory=list)
    revivals: list[MatchEvent] = field(default_factory=list)

    @property
    def all_events(self) -> list[MatchEvent]:
        events = self.kills + self.abilities + self.ultimates + self.spike_events + self.deaths + self.revivals
        return sorted(events, key=lambda e: e.timestamp)

    @property
    def duration(self) -> Optional[float]:
        if self.end_timestamp is not None:
            return self.end_timestamp - self.start_timestamp
        return None


@dataclass
class MatchData:
    """Complete match data loaded from valoscribe output."""
    team1: str = ""
    team2: str = ""
    map_name: str = ""
    final_score: tuple[int, int] = (0, 0)
    winner: Optional[str] = None
    events: list[MatchEvent] = field(default_factory=list)
    rounds: list[RoundData] = field(default_factory=list)
    frame_states: list[dict] = field(default_factory=list)

    def get_round(self, round_number: int) -> Optional[RoundData]:
        for r in self.rounds:
            if r.round_number == round_number:
                return r
        return None

    def get_events_by_type(self, event_type: str) -> list[MatchEvent]:
        return [e for e in self.events if e.type == event_type]

    def get_kill_timeline(self) -> list[MatchEvent]:
        return self.get_events_by_type("kill")

    def get_player_kills(self, player_name: str) -> list[MatchEvent]:
        return [
            e for e in self.events
            if e.type == "kill" and e.data.get("killer_name") == player_name
        ]


def load_event_log(path: Path) -> list[MatchEvent]:
    """Load events from a JSONL event log file."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                events.append(MatchEvent.from_dict(d))
    return events


def load_frame_states(path: Path, max_rows: Optional[int] = None) -> list[dict]:
    """Load frame states from CSV. Optionally limit rows for memory."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            rows.append(row)
    return rows


def build_match_data(events: list[MatchEvent], frame_states: Optional[list[dict]] = None) -> MatchData:
    """Build structured MatchData from flat event list."""
    match = MatchData(
        events=events,
        frame_states=frame_states or [],
    )

    # Extract match-level info
    for e in events:
        if e.type == "match_start":
            match.team1 = e.data.get("team1", "")
            match.team2 = e.data.get("team2", "")
            match.map_name = e.data.get("map_name", "")
        elif e.type == "match_end":
            match.final_score = (e.data.get("score_team1", 0), e.data.get("score_team2", 0))
            match.winner = e.data.get("winner")

    # Build rounds
    current_round: Optional[RoundData] = None
    for e in events:
        if e.type == "round_start":
            current_round = RoundData(
                round_number=e.data.get("round_number", 0),
                start_timestamp=e.timestamp,
                score_team1=e.data.get("score_team1", 0),
                score_team2=e.data.get("score_team2", 0),
            )
            match.rounds.append(current_round)
        elif e.type == "round_end" and current_round:
            current_round.end_timestamp = e.timestamp
            current_round.winner = e.data.get("winner")
            current_round = None
        elif current_round:
            if e.type == "kill":
                current_round.kills.append(e)
            elif e.type == "ability_used":
                current_round.abilities.append(e)
            elif e.type == "ultimate_used":
                current_round.ultimates.append(e)
            elif e.type == "spike_plant":
                current_round.spike_events.append(e)
            elif e.type == "death":
                current_round.deaths.append(e)
            elif e.type == "revival":
                current_round.revivals.append(e)

    return match


def load_match(output_dir: Path, load_frames: bool = False) -> MatchData:
    """Load a complete match from a valoscribe output directory.

    Args:
        output_dir: Directory containing event_log.jsonl and frame_states.csv
        load_frames: Whether to load frame_states.csv (can be large)

    Returns:
        MatchData with all events and optionally frame states
    """
    event_log_path = output_dir / "event_log.jsonl"
    if not event_log_path.exists():
        raise FileNotFoundError(f"Event log not found: {event_log_path}")

    events = load_event_log(event_log_path)

    frame_states = None
    if load_frames:
        frame_states_path = output_dir / "frame_states.csv"
        if frame_states_path.exists():
            frame_states = load_frame_states(frame_states_path)

    return build_match_data(events, frame_states)
