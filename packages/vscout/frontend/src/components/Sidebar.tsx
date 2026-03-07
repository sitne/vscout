import { NavLink } from 'react-router-dom';

const links = [
  { to: '/', label: 'マッチ一覧' },
  { to: '/analyze', label: '新規分析' },
];

export function Sidebar() {
  return (
    <nav
      style={{
        width: 220,
        background: '#1a1a1a',
        borderRight: '1px solid #333',
        padding: '1.5rem 0',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.25rem',
      }}
    >
      <div
        style={{
          padding: '0 1.5rem 1.5rem',
          borderBottom: '1px solid #333',
          marginBottom: '0.5rem',
        }}
      >
        <h1 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#ff4655', margin: 0 }}>
          V-SCOUT
        </h1>
        <p style={{ fontSize: '0.75rem', color: '#888', margin: '0.25rem 0 0' }}>
          Valorant VOD Analyzer
        </p>
      </div>
      {links.map((link) => (
        <NavLink
          key={link.to}
          to={link.to}
          end={link.to === '/'}
          style={({ isActive }) => ({
            display: 'block',
            padding: '0.6rem 1.5rem',
            color: isActive ? '#ff4655' : '#ccc',
            background: isActive ? '#ff465510' : 'transparent',
            textDecoration: 'none',
            fontSize: '0.9rem',
            fontWeight: isActive ? 600 : 400,
            borderLeft: isActive ? '3px solid #ff4655' : '3px solid transparent',
          })}
        >
          {link.label}
        </NavLink>
      ))}
    </nav>
  );
}
