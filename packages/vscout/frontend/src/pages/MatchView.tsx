import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getMatch, getRoundDetail } from '../api';
import type { MatchData, RoundDetail, GameEvent } from '../api';

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function EventBadge({ event }: { event: GameEvent }) {
  const colors: Record<string, string> = {
    kill: '#ff4655',
    death: '#666',
    ability_used: '#4fc3f7',
    ultimate_used: '#ffd740',
    spike_plant: '#66bb6a',
  };
  const color = colors[event.type] || '#888';

  let label = event.type;
  if (event.type === 'kill') {
    const killer = (event.killer_name as string) || (event.killer_agent as string) || '?';
    const victim = (event.victim_name as string) || (event.victim_agent as string) || '?';
    label = `${killer} → ${victim}`;
  } else if (event.type === 'ability_used') {
    label = `${event.player || event.agent} - ${event.ability_name || event.ability_slot}`;
  } else if (event.type === 'ultimate_used') {
    label = `${event.player || event.agent} ULT`;
  } else if (event.type === 'spike_plant') {
    label = `💣 Spike Plant`;
  }

  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '3px 10px',
        background: `${color}18`,
        border: `1px solid ${color}44`,
        borderRadius: 4,
        fontSize: '0.8rem',
        color,
      }}
    >
      <span style={{ opacity: 0.6, fontSize: '0.75rem' }}>{formatTime(event.timestamp)}</span>
      {label}
    </span>
  );
}

function RoundCard({
  roundNum,
  summary,
  isSelected,
  onClick,
}: {
  roundNum: number;
  summary: MatchData['rounds'][0];
  isSelected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem',
        padding: '0.6rem 1rem',
        background: isSelected ? '#ff465518' : '#1a1a1a',
        border: isSelected ? '1px solid #ff4655' : '1px solid #333',
        borderRadius: 6,
        cursor: 'pointer',
        color: '#eee',
        fontSize: '0.85rem',
        textAlign: 'left',
        width: '100%',
      }}
    >
      <span style={{ fontWeight: 700, minWidth: 24 }}>R{roundNum}</span>
      <span style={{ color: '#888' }}>
        {summary.score[0]}-{summary.score[1]}
      </span>
      <span style={{ color: '#ff4655', fontSize: '0.8rem' }}>
        {summary.kills > 0 && `${summary.kills}K`}
      </span>
      {summary.duration && (
        <span style={{ color: '#666', fontSize: '0.75rem', marginLeft: 'auto' }}>
          {formatTime(summary.duration)}
        </span>
      )}
      {summary.winner && (
        <span
          style={{
            fontSize: '0.7rem',
            padding: '1px 6px',
            borderRadius: 3,
            background: '#ffffff10',
            color: '#aaa',
          }}
        >
          {summary.winner}
        </span>
      )}
    </button>
  );
}

export function MatchView() {
  const params = useParams();
  const sessionId = params.sessionId || '';
  // Combine :mapPath and * segments to reconstruct full map path
  const mapPathBase = params.mapPath || '';
  const mapPathRest = params['*'] || '';
  const mapPath = mapPathRest ? `${mapPathBase}/${mapPathRest}` : mapPathBase;

  const [match, setMatch] = useState<MatchData | null>(null);
  const [selectedRound, setSelectedRound] = useState<number | null>(null);
  const [roundDetail, setRoundDetail] = useState<RoundDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!sessionId || !mapPath) return;
    setLoading(true);
    getMatch(sessionId, mapPath)
      .then((d) => {
        setMatch(d);
        if (d.rounds.length > 0) setSelectedRound(d.rounds[0].round_number);
      })
      .catch((e) => setError(e.response?.data?.detail || '読み込み失敗'))
      .finally(() => setLoading(false));
  }, [sessionId, mapPath]);

  useEffect(() => {
    if (!sessionId || !mapPath || selectedRound === null) return;
    getRoundDetail(sessionId, mapPath, selectedRound)
      .then(setRoundDetail)
      .catch(() => setRoundDetail(null));
  }, [sessionId, mapPath, selectedRound]);

  if (loading) return <p style={{ color: '#888' }}>読み込み中…</p>;
  if (error) return <p style={{ color: '#ff4655' }}>エラー: {error}</p>;
  if (!match) return null;

  return (
    <div>
      {/* Match header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '1.5rem',
          marginBottom: '2rem',
          padding: '1.25rem',
          background: '#1a1a1a',
          borderRadius: 8,
          border: '1px solid #333',
        }}
      >
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#eee' }}>
            {match.team1}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: '#ff4655' }}>
            {match.final_score[0]} - {match.final_score[1]}
          </div>
          <div style={{ fontSize: '0.8rem', color: '#888' }}>{match.map_name}</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#eee' }}>
            {match.team2}
          </div>
        </div>
        <div style={{ marginLeft: 'auto', textAlign: 'right', color: '#888', fontSize: '0.8rem' }}>
          <div>{match.total_rounds} ラウンド</div>
          <div>{match.total_events} イベント</div>
        </div>
      </div>

      {/* Two-column layout */}
      <div style={{ display: 'flex', gap: '1.5rem' }}>
        {/* Round list */}
        <div
          style={{
            width: 260,
            display: 'flex',
            flexDirection: 'column',
            gap: '0.35rem',
            flexShrink: 0,
          }}
        >
          <h3 style={{ color: '#aaa', fontSize: '0.85rem', margin: '0 0 0.5rem' }}>ラウンド</h3>
          {match.rounds.map((r) => (
            <RoundCard
              key={r.round_number}
              roundNum={r.round_number}
              summary={r}
              isSelected={selectedRound === r.round_number}
              onClick={() => setSelectedRound(r.round_number)}
            />
          ))}
        </div>

        {/* Round detail */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {roundDetail ? (
            <div>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '1rem',
                  marginBottom: '1.5rem',
                  padding: '1rem',
                  background: '#1a1a1a',
                  borderRadius: 8,
                  border: '1px solid #333',
                }}
              >
                <h3 style={{ color: '#eee', margin: 0 }}>Round {roundDetail.round_number}</h3>
                <span style={{ color: '#888', fontSize: '0.85rem' }}>
                  {roundDetail.score[0]}-{roundDetail.score[1]}
                </span>
                {roundDetail.winner && (
                  <span style={{ color: '#ff4655', fontSize: '0.85rem' }}>
                    Winner: {roundDetail.winner}
                  </span>
                )}
                {roundDetail.duration && (
                  <span style={{ color: '#666', fontSize: '0.85rem', marginLeft: 'auto' }}>
                    {formatTime(roundDetail.duration)}
                  </span>
                )}
              </div>

              {/* Kill events */}
              {roundDetail.kills.length > 0 && (
                <Section title="🗡️ キル" events={roundDetail.kills} />
              )}

              {/* Ability events */}
              {roundDetail.abilities.length > 0 && (
                <Section title="✨ アビリティ" events={roundDetail.abilities} />
              )}

              {/* Ultimate events */}
              {roundDetail.ultimates.length > 0 && (
                <Section title="⚡ アルティメット" events={roundDetail.ultimates} />
              )}

              {/* Spike events */}
              {roundDetail.spike_events.length > 0 && (
                <Section title="💣 スパイク" events={roundDetail.spike_events} />
              )}

              {roundDetail.kills.length === 0 &&
                roundDetail.abilities.length === 0 &&
                roundDetail.ultimates.length === 0 && (
                  <p style={{ color: '#666' }}>イベントなし</p>
                )}
            </div>
          ) : (
            <p style={{ color: '#666' }}>ラウンドを選択してください</p>
          )}
        </div>
      </div>
    </div>
  );
}

function Section({ title, events }: { title: string; events: GameEvent[] }) {
  return (
    <div style={{ marginBottom: '1.25rem' }}>
      <h4 style={{ color: '#aaa', fontSize: '0.85rem', margin: '0 0 0.5rem' }}>{title}</h4>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
        {events.map((e, i) => (
          <EventBadge key={i} event={e} />
        ))}
      </div>
    </div>
  );
}
