import type { ReactNode } from 'react';
import { Sidebar } from './Sidebar';

interface LayoutProps {
  children: ReactNode;
  title?: string;
  actions?: ReactNode;
}

export function Layout({ children, title, actions }: LayoutProps) {
  return (
    <div style={{ display: 'flex' }}>
      <Sidebar />
      <div className="main-content" style={{ flex: 1 }}>
        {(title || actions) && (
          <header
            style={{
              marginBottom: '2rem',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            {title && <h1>{title}</h1>}
            {actions && <div>{actions}</div>}
          </header>
        )}
        <main>{children}</main>
      </div>
    </div>
  );
}
