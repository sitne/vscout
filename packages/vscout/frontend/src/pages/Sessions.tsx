import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getSessions } from '../api';
import type { Session } from '../api';

export function Sessions() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getSessions()
      .then((d) => setSessions(d.sessions))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p style={{ color: '#888' }}>読み込み中…</p>;

  if (sessions.length === 0) {
    return (
      <div>
        <h2 style={{ color: '#eee' }}>マッチ一覧</h2>
        <p style={{ color: '#888' }}>
          まだデータがありません。
          <Link to="/analyze" style={{ color: '#ff4655', marginLeft: 8 }}>
            新規分析を開始
          </Link>
        </p>
      </div>
    );
  }

  return (
    <div>
      <h2 style={{ color: '#eee', marginBottom: '1.5rem' }}>マッチ一覧</h2>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {sessions.map((s) => (
          <div
            key={s.session_id}
            style={{
              background: '#1a1a1a',
              border: '1px solid #333',
              borderRadius: 8,
              padding: '1.25rem',
            }}
          >
            <h3 style={{ color: '#eee', margin: '0 0 0.75rem', fontSize: '1.1rem' }}>
              {s.session_id}
            </h3>
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              {s.maps.map((m) => (
                <Link
                  key={m.path}
                  to={`/match/${s.session_id}/${m.path}`}
                  style={{
                    display: 'inline-block',
                    padding: '0.5rem 1rem',
                    background: '#252525',
                    border: '1px solid #444',
                    borderRadius: 6,
                    color: '#ccc',
                    textDecoration: 'none',
                    fontSize: '0.85rem',
                    transition: 'border-color 0.15s',
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.borderColor = '#ff4655')}
                  onMouseLeave={(e) => (e.currentTarget.style.borderColor = '#444')}
                >
                  🗺️ {m.name}
                </Link>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
