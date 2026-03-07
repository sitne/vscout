import { useState } from 'react';
import { startAnalyze, getStatus } from '../api';
import type { JobStatus } from '../api';

export function Analyze() {
  const [vlrUrl, setVlrUrl] = useState('');
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState('');

  const handleSubmit = async () => {
    setError('');
    try {
      await startAnalyze({
        vlr_url: vlrUrl || undefined,
        youtube_url: youtubeUrl || undefined,
      });
      // Poll status
      const poll = setInterval(async () => {
        const s = await getStatus();
        setStatus(s);
        if (!s.is_running) clearInterval(poll);
      }, 2000);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    }
  };

  return (
    <div>
      <h2 style={{ color: '#eee', marginBottom: '1.5rem' }}>新規分析</h2>

      <div
        style={{
          background: '#1a1a1a',
          border: '1px solid #333',
          borderRadius: 8,
          padding: '1.5rem',
          maxWidth: 600,
        }}
      >
        <div style={{ marginBottom: '1rem' }}>
          <label style={{ color: '#aaa', fontSize: '0.85rem', display: 'block', marginBottom: 6 }}>
            VLR.gg URL
          </label>
          <input
            value={vlrUrl}
            onChange={(e) => setVlrUrl(e.target.value)}
            placeholder="https://www.vlr.gg/626529/..."
            style={{
              width: '100%',
              padding: '0.6rem',
              background: '#252525',
              border: '1px solid #444',
              borderRadius: 4,
              color: '#eee',
              fontSize: '0.9rem',
              boxSizing: 'border-box',
            }}
          />
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <label style={{ color: '#aaa', fontSize: '0.85rem', display: 'block', marginBottom: 6 }}>
            または YouTube URL
          </label>
          <input
            value={youtubeUrl}
            onChange={(e) => setYoutubeUrl(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=..."
            style={{
              width: '100%',
              padding: '0.6rem',
              background: '#252525',
              border: '1px solid #444',
              borderRadius: 4,
              color: '#eee',
              fontSize: '0.9rem',
              boxSizing: 'border-box',
            }}
          />
        </div>

        <button
          onClick={handleSubmit}
          disabled={status?.is_running}
          style={{
            padding: '0.6rem 1.5rem',
            background: status?.is_running ? '#333' : '#ff4655',
            color: '#fff',
            border: 'none',
            borderRadius: 4,
            cursor: status?.is_running ? 'not-allowed' : 'pointer',
            fontWeight: 600,
            fontSize: '0.9rem',
          }}
        >
          {status?.is_running ? '分析中…' : '分析開始'}
        </button>

        {error && <p style={{ color: '#ff4655', marginTop: '1rem' }}>{error}</p>}

        {status && (
          <div style={{ marginTop: '1.5rem' }}>
            <div
              style={{
                height: 6,
                background: '#333',
                borderRadius: 3,
                overflow: 'hidden',
                marginBottom: '0.5rem',
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: `${(status.progress * 100).toFixed(0)}%`,
                  background: '#ff4655',
                  transition: 'width 0.3s',
                }}
              />
            </div>
            <p style={{ color: '#888', fontSize: '0.8rem' }}>{status.status}</p>
          </div>
        )}
      </div>
    </div>
  );
}
