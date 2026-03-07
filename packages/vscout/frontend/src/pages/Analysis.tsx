import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layout } from '../components/Layout';
import { getStatus, stopAnalyze } from '../api';
import type { JobStatus } from '../api';
import { Square, CheckCircle, AlertCircle } from 'lucide-react';

export function Analysis() {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [showStop, setShowStop] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const poll = async () => {
      try {
        const s = await getStatus();
        setStatus(s);
        setShowStop(false);
      } catch (e) {
        console.error(e);
      }
    };
    poll();
    const interval = setInterval(poll, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleStop = async () => {
    try {
      await stopAnalyze();
      setShowStop(true);
    } catch (e) {
      console.error('Failed to stop:', e);
    }
  };

  if (!status) {
    return (
      <Layout title="分析ダッシュボード">
        <div className="card" style={{ maxWidth: 800, margin: '0 auto' }}>
          <div style={{ textAlign: 'center', padding: '2rem' }}>
            <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
              読み込み中...
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout title="分析ダッシュボード">
      <div className="card" style={{ maxWidth: 800, margin: '0 auto' }}>
        {/* ヘッダー */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '2rem',
            paddingBottom: '1.5rem',
            borderBottom: '1px solid var(--border)',
          }}
        >
          <div>
            <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem', fontWeight: 600 }}>
              {status.is_running ? '分析処理中...' : `ステータス: ${status.status}`}
            </h2>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              ジョブID: {status.id || 'N/A'}
            </div>
          </div>

          {status.is_running && (
            <button
              className="btn"
              onClick={handleStop}
              style={{ borderColor: 'var(--accent)', color: 'var(--accent)' }}
            >
              <Square size={16} fill="currentColor" /> 停止
            </button>
          )}
        </div>

        {/* 停止確認 */}
        {showStop && (
          <div
            style={{
              display: 'flex',
              gap: '0.75rem',
              padding: '0.75rem 1rem',
              background: 'rgba(255, 206, 0, 0.1)',
              border: '1px solid var(--warning)',
              borderRadius: 4,
              marginBottom: '1.5rem',
              alignItems: 'center',
            }}
          >
            <AlertCircle size={20} style={{ color: 'var(--warning)', flexShrink: 0 }} />
            <span style={{ color: 'var(--text-primary)', fontSize: '0.95rem' }}>
              停止要求を送信しました。処理が終了するまでお待ちください。
            </span>
          </div>
        )}

        {/* プログレス */}
        <div style={{ marginBottom: '2rem' }}>
          <div
            style={{
              background: 'var(--bg-tertiary)',
              height: 8,
              borderRadius: 4,
              overflow: 'hidden',
              marginBottom: '0.75rem',
            }}
          >
            <div
              style={{
                width: `${status.progress * 100}%`,
                background: 'var(--accent)',
                height: '100%',
                transition: 'width 0.2s ease',
                borderRadius: 4,
              }}
            />
          </div>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '0.9rem',
              color: 'var(--text-secondary)',
            }}
          >
            <span>進捗</span>
            <span>{(status.progress * 100).toFixed(1)}%</span>
          </div>
        </div>

        {/* ステータスログ */}
        <div
          style={{
            marginBottom: '2rem',
            background: 'var(--bg-primary)',
            border: '1px solid var(--border)',
            padding: '1rem',
            borderRadius: 4,
            fontFamily: 'monospace',
            fontSize: '0.85rem',
            minHeight: 60,
          }}
        >
          <div
            style={{
              color: status.is_running ? 'var(--accent)' : 'var(--success)',
              transition: 'color 0.1s',
            }}
          >
            <span style={{ opacity: 0.5 }}>{'> '}</span>
            {status.status}
          </div>
        </div>

        {/* 完了 */}
        {!status.is_running && status.progress >= 1.0 && (
          <div
            style={{
              padding: '1.5rem',
              background: 'rgba(0, 255, 159, 0.05)',
              border: '1px solid var(--success)',
              borderRadius: 4,
              textAlign: 'center',
            }}
          >
            <div
              style={{
                color: 'var(--success)',
                marginBottom: '1rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                fontSize: '1rem',
                fontWeight: 600,
              }}
            >
              <CheckCircle size={20} /> 分析が完了しました
            </div>
            <button className="btn btn-primary" onClick={() => navigate('/')} style={{ marginTop: '1rem' }}>
              結果を表示
            </button>
          </div>
        )}

        {/* エラー */}
        {!status.is_running && status.status.includes('error') && (
          <div
            style={{
              padding: '1.5rem',
              background: 'rgba(255, 70, 85, 0.05)',
              border: '1px solid var(--accent)',
              borderRadius: 4,
              textAlign: 'center',
            }}
          >
            <div
              style={{
                color: 'var(--accent)',
                marginBottom: '1rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                fontSize: '1rem',
                fontWeight: 600,
              }}
            >
              <AlertCircle size={20} /> エラーが発生しました
            </div>
            <div
              style={{
                fontSize: '0.9rem',
                color: 'var(--text-secondary)',
                marginTop: '0.75rem',
              }}
            >
              {status.status}
            </div>
            <button className="btn" onClick={() => navigate('/analyze')} style={{ marginTop: '1rem' }}>
              ホームに戻る
            </button>
          </div>
        )}

        {/* アイドル */}
        {!status.is_running && !status.status.includes('error') && status.progress < 1.0 && status.status === 'idle' && (
          <div
            style={{
              textAlign: 'center',
              padding: '2rem',
              color: 'var(--text-secondary)',
            }}
          >
            分析ジョブは実行されていません。
            <br />
            <button
              className="btn"
              onClick={() => navigate('/analyze')}
              style={{ marginTop: '1rem' }}
            >
              新規分析を開始
            </button>
          </div>
        )}
      </div>
    </Layout>
  );
}
