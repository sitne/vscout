import { NavLink } from 'react-router-dom';
import { Tv, Activity, BarChart3 } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

interface NavItem {
  icon: LucideIcon;
  label: string;
  path: string;
  description: string;
}

const navItems: NavItem[] = [
  { icon: Tv, label: 'ビデオ', path: '/analyze', description: '動画アップロード' },
  { icon: Activity, label: '分析', path: '/analysis', description: '処理状況' },
  { icon: BarChart3, label: 'マッチ', path: '/', description: '検出結果' },
];

export function Sidebar() {
  return (
    <div className="sidebar">
      {/* ロゴ */}
      <div className="header" style={{ justifyContent: 'center' }}>
        <h2 style={{ fontSize: '1.2rem', color: 'var(--accent)', margin: 0 }}>V-SCOUT</h2>
      </div>

      {/* ナビゲーション */}
      <nav style={{ flex: 1, padding: '1rem 0' }}>
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            style={({ isActive }) => ({
              display: 'flex',
              alignItems: 'center',
              padding: '0.75rem 1.5rem',
              color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
              background: isActive
                ? 'linear-gradient(90deg, var(--accent-glow) 0%, transparent 100%)'
                : 'transparent',
              borderLeft: isActive ? '3px solid var(--accent)' : '3px solid transparent',
              textDecoration: 'none',
              transition: 'all 0.15s',
              gap: '1rem',
              cursor: 'pointer',
            })}
          >
            <item.icon size={20} style={{ flexShrink: 0 }} />
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <span style={{ fontWeight: 500 }}>{item.label}</span>
              <span style={{ fontSize: '0.75rem', opacity: 0.7 }}>
                {item.description}
              </span>
            </div>
          </NavLink>
        ))}
      </nav>

      {/* ステータス */}
      <div style={{ padding: '1.5rem', borderTop: '1px solid var(--border)' }}>
        <div
          style={{
            fontSize: '0.8rem',
            color: 'var(--text-secondary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <div
            style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: 'var(--success)',
              animation: 'pulse 2s infinite',
            }}
          />
          <span>システム正常</span>
        </div>
      </div>


    </div>
  );
}
