import axios from 'axios';

const api = axios.create({
  baseURL: '/',
});

// --- Types ---

export interface JobStatus {
  id: string | null;
  is_running: boolean;
  progress: number;
  status: string;
  session_id: string | null;
}

export interface MapInfo {
  path: string;
  name: string;
}

export interface Session {
  session_id: string;
  maps: MapInfo[];
}

export interface RoundSummary {
  round_number: number;
  start_timestamp: number;
  end_timestamp: number | null;
  duration: number | null;
  score: [number, number];
  winner: string | null;
  kills: number;
  abilities: number;
  ultimates: number;
  spike_events: number;
}

export interface MatchData {
  team1: string;
  team2: string;
  map_name: string;
  final_score: [number, number];
  winner: string | null;
  total_events: number;
  total_rounds: number;
  rounds: RoundSummary[];
}

export interface GameEvent {
  type: string;
  timestamp: number;
  [key: string]: unknown;
}

export interface RoundDetail {
  round_number: number;
  start_timestamp: number;
  end_timestamp: number | null;
  duration: number | null;
  score: [number, number];
  winner: string | null;
  kills: GameEvent[];
  abilities: GameEvent[];
  ultimates: GameEvent[];
  spike_events: GameEvent[];
  deaths: GameEvent[];
}

// --- API calls ---

export const getStatus = async () => {
  const res = await api.get<JobStatus>('/api/status');
  return res.data;
};

export const getSessions = async () => {
  const res = await api.get<{ sessions: Session[] }>('/api/sessions');
  return res.data;
};

export const getMatch = async (sessionId: string, mapPath: string) => {
  const res = await api.get<MatchData>(`/api/matches/${sessionId}/${mapPath}`);
  return res.data;
};

export const getMatchEvents = async (
  sessionId: string,
  mapPath: string,
  eventType?: string,
  roundNumber?: number,
) => {
  const params: Record<string, string> = {};
  if (eventType) params.event_type = eventType;
  if (roundNumber !== undefined) params.round_number = String(roundNumber);
  const res = await api.get<{ count: number; events: GameEvent[] }>(
    `/api/matches/${sessionId}/${mapPath}/events`,
    { params },
  );
  return res.data;
};

export const getRoundDetail = async (
  sessionId: string,
  mapPath: string,
  roundNumber: number,
) => {
  const res = await api.get<RoundDetail>(
    `/api/matches/${sessionId}/${mapPath}/rounds/${roundNumber}`,
  );
  return res.data;
};

export const getKillTimeline = async (sessionId: string, mapPath: string) => {
  const res = await api.get<{ total_kills: number; kills: GameEvent[] }>(
    `/api/matches/${sessionId}/${mapPath}/kills`,
  );
  return res.data;
};

export const startAnalyze = async (params: {
  vlr_url?: string;
  youtube_url?: string;
  local_video_path?: string;
  start_time?: number;
  duration?: number;
  session_id?: string;
}) => {
  const res = await api.post('/api/analyze', params);
  return res.data;
};

export const stopAnalyze = async () => {
  const res = await api.post('/api/stop');
  return res.data;
};
