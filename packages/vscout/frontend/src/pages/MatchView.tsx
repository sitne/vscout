import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Layout } from '../components/Layout';
import { getMatch, getRoundDetail } from '../api';
import type { MatchData, RoundDetail, GameEvent } from '../api';
import { ArrowLeft, Crosshair, Zap, Sparkles, Bomb } from 'lucide-react';

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

const eventColors: Record<string, string> = {
  kill: 'var(--accent)',
  death: 'var(--text-secondary)',
  ability_used: '#4fc3f7',
  ultimate_used: 'var(--warning)',
  spike_plant: 'var(--success)',
};

function EventBadge({ event }: { event: GameEvent }) {
  const color = eventColors[event.type] || 'var(--text-secondary)';

  let label = event.type;
  if (event.type === 'kill') {
    const killer = (event.killer_name as string) || (event.killer_agent as string) || '?';
    const victim = (event.victim_name as string) || (event.victim_agent as string) || '?';
    label = `${killer} \u2192 ${victim}`;
  } else if (event.type === 'ability_used') {
    label = `${event.player || event.agent} - ${event.ability_name || event.ability_slot}`;
  } else if (event.type === 'ultimate_used') {
    label = `${event.player || event.agent} ULT`;
  } else if (event.type === 'spike_plant') {
    label = 'Spike Plant';
  }

  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '4px 10px',
        background: 'var(--bg-primary)',
        border: `1px solid ${color}44`,
        borderRadius: 4,
        fontSize: '0.8rem',
        color,
        transition: 'border-color 0.1s',
      }}
    >
      <span style={{ opacity: 0.5, fontSize: '0.75rem', fontFamily: 'monospace' }}>
        {formatTime(event.timestamp)}
      </span>
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
        background: isSelected ? 'var(--accent-glow)' : 'var(--bg-secondary)',
        border: isSelected ? '1px solid var(--accent)' : '1px solid var(--border)',
        borderRadius: 0,
        cursor: 'pointer',
        color: 'var(--text-primary)',
        fontSize: '0.85rem',
        textAlign: 'left',
        width: '100%',
        fontFamily: 'inherit',
        transition: 'all 0.1s',
        borderLeft: isSelected ? '3px solid var(--accent)' : '3px solid transparent',
      }}
    >
      <span style={{ fontWeight: 700, minWidth: 30, fontFamily: 'monospace' }}>R{roundNum}</span>
      <span style={{ color: 'var(--text-secondary)' }}>
        {summary.score[0]}-{summary.score[1]}
      </span>
      {summary.kills > 0 && (
        <span style={{ color: 'var(--accent)', fontSize: '0.8rem' }}>{summary.kills}K</span>
      )}
      {summary.duration != null && (
        <span
          style={{
            color: 'var(--text-secondary)',
            fontSize: '0.75rem',
            marginLeft: 'auto',
            fontFamily: 'monospace',
          }}
        >
          {formatTime(summary.duration)}
        </span>
      )}
      {summary.winner && (
        <span
          style={{
            fontSize: '0.7rem',
            padding: '1px 6px',
            borderRadius: 3,
            background: 'var(--bg-tertiary)',
            color: 'var(--text-secondary)',
          }}
        >
          {summary.winner}
        </span>
      )}
    </button>
  );
}

function Section({
  title,
  icon,
  events,
}: {
  title: string;
  icon: React.ReactNode;
  events: GameEvent[];
}) {
  return (
    <div style={{ marginBottom: '1.25rem' }}>
      <h4
        style={{
          color: 'var(--text-secondary)',
          fontSize: '0.85rem',
          margin: '0 0 0.5rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
        }}
      >
        {icon}
        {title}
      </h4>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
        {events.map((e, i) => (
          <EventBadge key={i} event={e} />
        ))}
      </div>
    </div>
  );
}

export function MatchView() {
  const params = useParams();
  const sessionId = params.sessionId || '';
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

  if (loading) {
    return (
      <Layout title="マッチ詳細">
        <div className="card" style={{ textAlign: 'center', padding: '2rem' }}>
          <div style={{ color: 'var(--text-secondary)' }}>読み込み中...</div>
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout title="マッチ詳細">
        <div className="card" style={{ textAlign: 'center', padding: '2rem' }}>
          <div style={{ color: 'var(--accent)', marginBottom: '1rem' }}>エラー: {error}</div>
          <Link to="/" className="btn" style={{ textDecoration: 'none' }}>
            戻る
          </Link>
        </div>
      </Layout>
    );
  }

  if (!match) return null;

  const backAction = (
    <Link
      to="/"
      className="btn"
      style={{
        textDecoration: 'none',
        fontSize: '0.85rem',
        textTransform: 'none',
        letterSpacing: 0,
      }}
    >
      <ArrowLeft size={16} /> 一覧に戻る
    </Link>
  );

  return (
    <Layout title="マッチ詳細" actions={backAction}>
      {/* Match header */}
      <div
        className="card"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '2rem',
          marginBottom: '2rem',
          padding: '1.5rem',
        }}
      >
        <div style={{ textAlign: 'center', flex: 1 }}>
          <div style={{ fontSize: '1.3rem', fontWeight: 700, textTransform: 'uppercase' }}>
            {match.team1}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div
            style={{
              fontSize: '2.5rem',
              fontWeight: 700,
              color: 'var(--accent)',
              lineHeight: 1,
            }}
          >
            {match.final_score[0]} - {match.final_score[1]}
          </div>
          <div
            style={{
              fontSize: '0.85rem',
              color: 'var(--text-secondary)',
              marginTop: '0.25rem',
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
            }}
          >
            {match.map_name}
          </div>
        </div>
        <div style={{ textAlign: 'center', flex: 1 }}>
          <div style={{ fontSize: '1.3rem', fontWeight: 700, textTransform: 'uppercase' }}>
            {match.team2}
          </div>
        </div>
      </div>

      {/* Stats bar */}
      <div
        style={{
          display: 'flex',
          gap: '2rem',
          marginBottom: '2rem',
          padding: '0.75rem 1rem',
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border)',
          borderRadius: 4,
          fontSize: '0.85rem',
          color: 'var(--text-secondary)',
        }}
      >
        <span>
          ラウンド:{' '}
          <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
            {match.total_rounds}
          </span>
        </span>
        <span>
          イベント:{' '}
          <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
            {match.total_events}
          </span>
        </span>
      </div>

      {/* Two-column layout */}
      <div style={{ display: 'flex', gap: '1.5rem' }}>
        {/* Round list */}
        <div
          style={{
            width: 280,
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
            flexShrink: 0,
          }}
        >
          <h3
            style={{
              color: 'var(--text-secondary)',
              fontSize: '0.85rem',
              margin: '0 0 0.75rem',
            }}
          >
            ラウンド
          </h3>
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
              {/* Round header */}
              <div
                className="card"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '1rem',
                  marginBottom: '1.5rem',
                }}
              >
                <h3 style={{ margin: 0, fontSize: '1.1rem' }}>
                  ROUND {roundDetail.round_number}
                </h3>
                <span
                  style={{
                    color: 'var(--text-secondary)',
                    fontSize: '0.85rem',
                    fontFamily: 'monospace',
                  }}
                >
                  {roundDetail.score[0]}-{roundDetail.score[1]}
                </span>
                {roundDetail.winner && (
                  <span
                    style={{
                      color: 'var(--accent)',
                      fontSize: '0.85rem',
                      fontWeight: 600,
                    }}
                  >
                    Winner: {roundDetail.winner}
                  </span>
                )}
                {roundDetail.duration != null && (
                  <span
                    style={{
                      color: 'var(--text-secondary)',
                      fontSize: '0.85rem',
                      marginLeft: 'auto',
                      fontFamily: 'monospace',
                    }}
                  >
                    {formatTime(roundDetail.duration)}
                  </span>
                )}
              </div>

              {roundDetail.kills.length > 0 && (
                <Section
                  title="キル"
                  icon={<Crosshair size={14} />}
                  events={roundDetail.kills}
                />
              )}
              {roundDetail.abilities.length > 0 && (
                <Section
                  title="アビリティ"
                  icon={<Sparkles size={14} />}
                  events={roundDetail.abilities}
                />
              )}
              {roundDetail.ultimates.length > 0 && (
                <Section
                  title="アルティメット"
                  icon={<Zap size={14} />}
                  events={roundDetail.ultimates}
                />
              )}
              {roundDetail.spike_events.length > 0 && (
                <Section
                  title="スパイク"
                  icon={<Bomb size={14} />}
                  events={roundDetail.spike_events}
                />
              )}

              {roundDetail.kills.length === 0 &&
                roundDetail.abilities.length === 0 &&
                roundDetail.ultimates.length === 0 &&
                roundDetail.spike_events.length === 0 && (
                  <div
                    style={{
                      textAlign: 'center',
                      padding: '2rem',
                      color: 'var(--text-secondary)',
                      background: 'var(--bg-secondary)',
                      border: '1px dashed var(--border)',
                      borderRadius: 4,
                    }}
                  >
                    このラウンドにイベントはありません
                  </div>
                )}
            </div>
          ) : (
            <div
              style={{
                textAlign: 'center',
                padding: '3rem',
                color: 'var(--text-secondary)',
              }}
            >
              ラウンドを選択してください
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
