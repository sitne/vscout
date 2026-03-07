import type { ReactNode } from 'react';
import { Sidebar } from './Sidebar';

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: '#0f0f0f' }}>
      <Sidebar />
      <main style={{ flex: 1, padding: '2rem', overflow: 'auto' }}>
        {children}
      </main>
    </div>
  );
}
