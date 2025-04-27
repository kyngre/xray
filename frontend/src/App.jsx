import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Landing from './pages/Landing';
import Analyze from './pages/Analyze';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/analyze" element={<Analyze />} />
      </Routes>
    </Router>
  );
}

export default App;
