import { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layout } from '../components/Layout';
import { getStatus, stopAnalyze } from '../api';
import type { JobStatus } from '../api';
import { Square, CheckCircle, AlertCircle, Download, Cpu, Search, Play } from 'lucide-react';

const stepIcon = (status: string) => {
  if (status.startsWith('scraping')) return <Search size={16} />;
  if (status.startsWith('downloading')) return <Download size={16} />;
  if (status.startsWith('processing')) return <Cpu size={16} />;
  return <Play size={16} />;
};

export function Analysis() {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const navigate = useNavigate();
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const s = await getStatus();
        setStatus(s);
      } catch (e) {
        console.error(e);
      }
    };
    poll();
    const interval = setInterval(poll, 1500);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [status?.steps_log.length]);

  const handleStop = async () => {
    try {
      await stopAnalyze();
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

  const isCompleted = !status.is_running && status.progress >= 1.0;
  const isError = !status.is_running && status.status.includes('error');
  const isIdle = !status.is_running && !isCompleted && !isError && status.status === 'idle';

  return (
    <Layout title="分析ダッシュボード">
      <div style={{ maxWidth: 800, margin: '0 auto', display: 'grid', gap: '1.5rem' }}>
        {/* ステータスヘッダー */}
        <div className="card">
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '1.5rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              {status.is_running && stepIcon(status.status)}
              <div>
                <h2 style={{ fontSize: '1.25rem', marginBottom: '0.25rem', fontWeight: 600 }}>
                  {status.is_running
                    ? status.current_step || '処理中...'
                    : isCompleted
                      ? '分析完了'
                      : isError
                        ? 'エラー'
                        : '待機中'}
                </h2>
                {status.is_running && status.total_maps > 0 && (
                  <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                    マップ {status.current_map} / {status.total_maps}
                  </div>
                )}
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

          {/* プログレスバー */}
          <div style={{ marginBottom: '0.5rem' }}>
            <div
              style={{
                background: 'var(--bg-tertiary)',
                height: 8,
                borderRadius: 4,
                overflow: 'hidden',
                marginBottom: '0.5rem',
              }}
            >
              <div
                style={{
                  width: `${status.progress * 100}%`,
                  background: isError ? 'var(--accent)' : isCompleted ? 'var(--success)' : 'var(--accent)',
                  height: '100%',
                  transition: 'width 0.3s ease',
                  borderRadius: 4,
                }}
              />
            </div>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: '0.85rem',
                color: 'var(--text-secondary)',
              }}
            >
              <span>進捗</span>
              <span>{(status.progress * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* ステップログ */}
        {status.steps_log.length > 0 && (
          <div className="card" style={{ padding: 0 }}>
            <div
              style={{
                padding: '0.75rem 1rem',
                borderBottom: '1px solid var(--border)',
                fontSize: '0.85rem',
                fontWeight: 600,
                color: 'var(--text-secondary)',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
              }}
            >
              処理ログ
            </div>
            <div
              style={{
                maxHeight: 300,
                overflowY: 'auto',
                padding: '0.75rem 1rem',
                fontFamily: 'monospace',
                fontSize: '0.82rem',
                lineHeight: 1.8,
              }}
            >
              {status.steps_log.map((log, i) => (
                <div
                  key={i}
                  style={{
                    color: log.includes('エラー') || log.includes('失敗')
                      ? 'var(--accent)'
                      : log.includes('完了')
                        ? 'var(--success)'
                        : 'var(--text-primary)',
                    opacity: i === status.steps_log.length - 1 ? 1 : 0.7,
                  }}
                >
                  <span style={{ opacity: 0.4, marginRight: '0.5rem' }}>
                    [{String(i + 1).padStart(2, '0')}]
                  </span>
                  {log}
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>
        )}

        {/* 完了 */}
        {isCompleted && (
          <div
            className="card"
            style={{
              background: 'rgba(0, 255, 159, 0.05)',
              borderColor: 'var(--success)',
              textAlign: 'center',
              padding: '2rem',
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
            <button
              className="btn btn-primary"
              onClick={() => navigate('/')}
              style={{ marginTop: '0.5rem' }}
            >
              結果を表示
            </button>
          </div>
        )}

        {/* エラー */}
        {isError && (
          <div
            className="card"
            style={{
              background: 'rgba(255, 70, 85, 0.05)',
              borderColor: 'var(--accent)',
              textAlign: 'center',
              padding: '2rem',
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
                marginBottom: '1rem',
                fontFamily: 'monospace',
                wordBreak: 'break-all',
              }}
            >
              {status.status.replace('error: ', '')}
            </div>
            <button
              className="btn"
              onClick={() => navigate('/analyze')}
            >
              やり直す
            </button>
          </div>
        )}

        {/* アイドル */}
        {isIdle && (
          <div
            className="card"
            style={{
              textAlign: 'center',
              padding: '3rem 2rem',
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
