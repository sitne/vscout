import { useState } from 'react';
import type { KeyboardEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layout } from '../components/Layout';
import { startAnalyze } from '../api';
import { Play, AlertCircle } from 'lucide-react';

const parseTime = (t: string): number => {
  if (!t) return 0;
  t = t.trim();
  if (t.includes(':')) {
    const parts = t.split(':').map(Number).filter((n) => !isNaN(n));
    if (parts.length === 2) return parts[0] * 60 + parts[1];
    if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  return parseFloat(t) || 0;
};

export function Analyze() {
  const navigate = useNavigate();
  const [vlrUrl, setVlrUrl] = useState('');
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [startTime, setStartTime] = useState('');
  const [endTime, setEndTime] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const validate = (): boolean => {
    setError('');
    if (!vlrUrl && !youtubeUrl) {
      setError('VLR URLまたはYouTube URLを入力してください');
      return false;
    }
    if (startTime && endTime && parseTime(startTime) > parseTime(endTime)) {
      setError('開始時刻は終了時刻より前である必要があります');
      return false;
    }
    return true;
  };

  const handleStart = async () => {
    if (!validate()) return;
    try {
      setLoading(true);
      setError('');
      await startAnalyze({
        vlr_url: vlrUrl || undefined,
        youtube_url: youtubeUrl || undefined,
        start_time: startTime ? parseTime(startTime) : undefined,
        duration: endTime ? parseTime(endTime) - (startTime ? parseTime(startTime) : 0) : undefined,
      });
      navigate('/analysis');
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError('分析の開始に失敗しました: ' + msg);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') handleStart();
  };

  return (
    <Layout title="ビデオ分析">
      <div className="card" style={{ maxWidth: 600, margin: '0 auto', padding: '1.5rem' }}>
        <h3
          style={{
            marginBottom: '1.5rem',
            color: 'var(--text-secondary)',
            fontWeight: 600,
          }}
        >
          新規分析を開始
        </h3>

        {error && (
          <div
            style={{
              display: 'flex',
              gap: '0.75rem',
              padding: '0.75rem 1rem',
              background: 'rgba(255, 70, 85, 0.1)',
              border: '1px solid var(--accent)',
              borderRadius: 4,
              marginBottom: '1.5rem',
              alignItems: 'center',
            }}
          >
            <AlertCircle size={20} style={{ color: 'var(--accent)', flexShrink: 0 }} />
            <span style={{ color: 'var(--text-primary)', fontSize: '0.95rem' }}>
              {error}
            </span>
          </div>
        )}

        <div style={{ display: 'grid', gap: '1.5rem' }} onKeyDown={handleKeyDown}>
          {/* ビデオソース */}
          <div>
            <div
              style={{
                fontSize: '0.9rem',
                fontWeight: 600,
                marginBottom: '0.75rem',
                color: 'var(--text-secondary)',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
              }}
            >
              ビデオソース (必須)
            </div>
            <div style={{ display: 'grid', gap: '1rem' }}>
              <div>
                <label
                  style={{
                    display: 'block',
                    marginBottom: '0.5rem',
                    fontSize: '0.85rem',
                    fontWeight: 500,
                  }}
                >
                  VLR.gg URL
                </label>
                <input
                  type="text"
                  value={vlrUrl}
                  onChange={(e) => setVlrUrl(e.target.value)}
                  placeholder="https://www.vlr.gg/626529/..."
                  style={{ width: '100%' }}
                />
              </div>

              <div
                style={{
                  textAlign: 'center',
                  color: 'var(--text-secondary)',
                  fontSize: '0.85rem',
                }}
              >
                または
              </div>

              <div>
                <label
                  style={{
                    display: 'block',
                    marginBottom: '0.5rem',
                    fontSize: '0.85rem',
                    fontWeight: 500,
                  }}
                >
                  YouTube URL
                </label>
                <input
                  type="text"
                  value={youtubeUrl}
                  onChange={(e) => setYoutubeUrl(e.target.value)}
                  placeholder="https://www.youtube.com/watch?v=..."
                  style={{ width: '100%' }}
                />
              </div>
            </div>
          </div>

          {/* 分析範囲 */}
          <div>
            <div
              style={{
                fontSize: '0.9rem',
                fontWeight: 600,
                marginBottom: '0.75rem',
                color: 'var(--text-secondary)',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
              }}
            >
              分析範囲 (オプション)
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <label
                  style={{
                    display: 'block',
                    marginBottom: '0.5rem',
                    fontSize: '0.85rem',
                    fontWeight: 500,
                  }}
                >
                  開始時刻
                </label>
                <input
                  type="text"
                  value={startTime}
                  onChange={(e) => setStartTime(e.target.value)}
                  placeholder="mm:ss または秒数"
                  style={{ width: '100%' }}
                />
              </div>
              <div>
                <label
                  style={{
                    display: 'block',
                    marginBottom: '0.5rem',
                    fontSize: '0.85rem',
                    fontWeight: 500,
                  }}
                >
                  終了時刻
                </label>
                <input
                  type="text"
                  value={endTime}
                  onChange={(e) => setEndTime(e.target.value)}
                  placeholder="mm:ss または秒数"
                  style={{ width: '100%' }}
                />
              </div>
            </div>
            <div
              style={{
                fontSize: '0.8rem',
                color: 'var(--text-secondary)',
                marginTop: '0.5rem',
              }}
            >
              形式: 00:00 (分:秒) または 0 (秒数)
            </div>
          </div>

          {/* CTA */}
          <button
            className="btn btn-primary"
            style={{
              marginTop: '1rem',
              justifyContent: 'center',
              padding: '0.75rem 1.5rem',
              opacity: loading ? 0.6 : 1,
              cursor: loading ? 'not-allowed' : 'pointer',
            }}
            onClick={handleStart}
            disabled={loading}
          >
            <Play size={18} />
            {loading ? '分析中...' : 'ビデオを分析'}
          </button>
          <div
            style={{
              fontSize: '0.8rem',
              color: 'var(--text-secondary)',
              textAlign: 'center',
            }}
          >
            Ctrl+Enter でも実行可能
          </div>
        </div>
      </div>
    </Layout>
  );
}
