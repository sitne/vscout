import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Layout } from '../components/Layout';
import { getSessions } from '../api';
import type { Session } from '../api';
import { FolderOpen, Map } from 'lucide-react';

export function Sessions() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getSessions()
      .then((d) => setSessions(d.sessions))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <Layout title="マッチ一覧">
        <div className="card" style={{ maxWidth: 800, margin: '0 auto', textAlign: 'center', padding: '2rem' }}>
          <div style={{ color: 'var(--text-secondary)' }}>読み込み中...</div>
        </div>
      </Layout>
    );
  }

  if (sessions.length === 0) {
    return (
      <Layout title="マッチ一覧">
        <div
          style={{
            textAlign: 'center',
            padding: '4rem 2rem',
            color: 'var(--text-secondary)',
            background: 'var(--bg-secondary)',
            borderRadius: 4,
            border: '1px dashed var(--border)',
          }}
        >
          <div style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>
            まだマッチデータがありません
          </div>
          <div style={{ fontSize: '0.9rem', marginBottom: '1.5rem' }}>
            動画をアップロードして分析を開始してください
          </div>
          <Link to="/analyze" className="btn btn-primary" style={{ textDecoration: 'none' }}>
            新規分析を開始
          </Link>
        </div>
      </Layout>
    );
  }

  return (
    <Layout title="マッチ一覧">
      {/* サマリー */}
      <div
        style={{
          marginBottom: '2rem',
          padding: '1rem',
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border)',
          borderRadius: 4,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
          合計{' '}
          <span style={{ color: 'var(--accent)', fontWeight: 600 }}>
            {sessions.length}
          </span>{' '}
          セッション
        </div>
        <Link to="/analyze" className="btn btn-primary" style={{ textDecoration: 'none', fontSize: '0.85rem' }}>
          + 新規分析
        </Link>
      </div>

      {/* セッションリスト */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        {sessions.map((s) => (
          <div
            key={s.session_id}
            className="card"
            style={{
              borderLeft: '4px solid var(--accent)',
              display: 'grid',
              gap: '1rem',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
              }}
            >
              <FolderOpen size={20} style={{ color: 'var(--accent)', flexShrink: 0 }} />
              <h3
                style={{
                  margin: 0,
                  fontSize: '1rem',
                  fontWeight: 600,
                  fontFamily: 'monospace',
                  textTransform: 'none',
                  letterSpacing: 0,
                }}
              >
                {s.session_id}
              </h3>
              <span
                style={{
                  fontSize: '0.8rem',
                  color: 'var(--text-secondary)',
                  marginLeft: 'auto',
                }}
              >
                {s.maps.length} マップ
              </span>
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              {s.maps.map((m) => (
                <Link
                  key={m.path}
                  to={`/match/${s.session_id}/${m.path}`}
                  className="btn"
                  style={{
                    textDecoration: 'none',
                    fontSize: '0.85rem',
                    gap: '0.4rem',
                    textTransform: 'none',
                    letterSpacing: 0,
                  }}
                >
                  <Map size={14} />
                  {m.name}
                </Link>
              ))}
            </div>
          </div>
        ))}
      </div>
    </Layout>
  );
}
