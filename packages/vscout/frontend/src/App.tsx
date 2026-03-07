import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Sessions } from './pages/Sessions';
import { MatchView } from './pages/MatchView';
import { Analyze } from './pages/Analyze';

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Sessions />} />
          <Route path="/match/:sessionId/:mapPath/*" element={<MatchView />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
