import './components/FormStyle.css';
import React, { useState } from 'react';
import ScoreForm from './components/ScoreForm';
import PredictionResult from './components/PredictionResult';

function App() {
  const [result, setResult] = useState(null);

  return (
    <div style={{ textAlign: 'center' }}>
      <h1>🏏 IPL Score Predictor</h1>
      <ScoreForm setResult={setResult} />
      <PredictionResult result={result} />
    </div>
  );
}

export default App;
