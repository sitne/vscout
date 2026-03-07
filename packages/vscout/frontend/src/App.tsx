import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Sessions } from './pages/Sessions';
import { MatchView } from './pages/MatchView';
import { Analyze } from './pages/Analyze';
import { Analysis } from './pages/Analysis';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Sessions />} />
        <Route path="/match/:sessionId/:mapPath/*" element={<MatchView />} />
        <Route path="/analyze" element={<Analyze />} />
        <Route path="/analysis" element={<Analysis />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
