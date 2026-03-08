import { useState } from 'react';
import type { KeyboardEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layout } from '../components/Layout';
import { startAnalyze } from '../api';
import { Play, AlertCircle, ExternalLink } from 'lucide-react';

export function Analyze() {
  const navigate = useNavigate();
  const [vlrUrl, setVlrUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const validate = (): boolean => {
    setError('');
    const url = vlrUrl.trim();
    if (!url) {
      setError('VLR.gg URLを入力してください');
      return false;
    }
    if (!url.includes('vlr.gg')) {
      setError('有効なVLR.gg URLを入力してください (例: https://www.vlr.gg/626529/...)');
      return false;
    }
    return true;
  };

  const handleStart = async () => {
    if (!validate()) return;
    try {
      setLoading(true);
      setError('');
      await startAnalyze(vlrUrl.trim());
      navigate('/analysis');
    } catch (e: unknown) {
      if (e && typeof e === 'object' && 'response' in e) {
        const axiosErr = e as { response?: { data?: { detail?: string } } };
        setError(axiosErr.response?.data?.detail || '分析の開始に失敗しました');
      } else {
        const msg = e instanceof Error ? e.message : String(e);
        setError('分析の開始に失敗しました: ' + msg);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') handleStart();
  };

  return (
    <Layout title="VLR 分析">
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
          {/* VLR URL 入力 */}
          <div>
            <label
              style={{
                display: 'block',
                marginBottom: '0.5rem',
                fontSize: '0.9rem',
                fontWeight: 600,
                color: 'var(--text-secondary)',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
              }}
            >
              VLR.gg マッチ URL
            </label>
            <input
              type="text"
              value={vlrUrl}
              onChange={(e) => setVlrUrl(e.target.value)}
              placeholder="https://www.vlr.gg/626529/..."
              style={{ width: '100%' }}
              autoFocus
            />
            <div
              style={{
                fontSize: '0.8rem',
                color: 'var(--text-secondary)',
                marginTop: '0.5rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
              }}
            >
              <ExternalLink size={12} />
              VLR.ggのマッチページURLを貼り付けてください
            </div>
          </div>

          {/* 処理フロー説明 */}
          <div
            style={{
              background: 'var(--bg-primary)',
              border: '1px solid var(--border)',
              borderRadius: 4,
              padding: '1rem',
            }}
          >
            <div
              style={{
                fontSize: '0.85rem',
                fontWeight: 600,
                color: 'var(--text-secondary)',
                marginBottom: '0.75rem',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
              }}
            >
              処理フロー
            </div>
            <div style={{ display: 'grid', gap: '0.5rem', fontSize: '0.85rem' }}>
              {[
                { step: '1', text: 'VLR.ggからマッチメタデータをスクレイピング' },
                { step: '2', text: '各マップのYouTube VODを自動ダウンロード' },
                { step: '3', text: 'コンピュータビジョンでVODを解析' },
                { step: '4', text: 'イベントタイムラインを生成 (キル/スパイク/ULT)' },
              ].map((item) => (
                <div
                  key={item.step}
                  style={{
                    display: 'flex',
                    gap: '0.75rem',
                    alignItems: 'center',
                  }}
                >
                  <span
                    style={{
                      width: 22,
                      height: 22,
                      borderRadius: '50%',
                      background: 'var(--accent-glow)',
                      color: 'var(--accent)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.75rem',
                      fontWeight: 700,
                      flexShrink: 0,
                    }}
                  >
                    {item.step}
                  </span>
                  <span style={{ color: 'var(--text-primary)' }}>{item.text}</span>
                </div>
              ))}
            </div>
          </div>

          {/* CTA */}
          <button
            className="btn btn-primary"
            style={{
              marginTop: '0.5rem',
              justifyContent: 'center',
              padding: '0.75rem 1.5rem',
              opacity: loading ? 0.6 : 1,
              cursor: loading ? 'not-allowed' : 'pointer',
            }}
            onClick={handleStart}
            disabled={loading}
          >
            <Play size={18} />
            {loading ? '開始中...' : '分析を開始'}
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
