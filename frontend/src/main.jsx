import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  async function predict(event) {
    event.preventDefault();
    if (!file) return;
    setLoading(true); setError(""); setResult(null);
    const started = performance.now();
    const body = new FormData(); body.append("file", file);
    try {
      const response = await fetch(`${API}/predict`, { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Prediction failed");
      setResult({ ...payload, clientLatency: performance.now() - started });
    } catch (reason) { setError(reason.message); } finally { setLoading(false); }
  }
  return <main>
    <header><span className="badge">FarmPulse AI</span><h1>Plant disease diagnosis</h1>
      <p>Upload a leaf image and get an EfficientNet transfer-learning prediction.</p></header>
    <form onSubmit={predict} className="card">
      <label className="drop"> <input type="file" accept="image/*" onChange={e => setFile(e.target.files[0])} />
        <strong>{file ? file.name : "Choose a leaf image"}</strong><small>JPG, PNG or WEBP</small>
      </label>
      <button disabled={!file || loading}>{loading ? "Analyzing..." : "Diagnose leaf"}</button>
    </form>
    {error && <p className="error">{error}</p>}
    {result && <section className="card result"><span className="badge">Prediction</span>
      <h2>{result.predicted_class.replaceAll("_", " ")}</h2>
      <div className="confidence">{(result.confidence * 100).toFixed(1)}% confidence</div>
      <p>Server: {result.latency_ms.toFixed(1)} ms · Client: {result.clientLatency.toFixed(1)} ms</p>
      <ul>{result.top_predictions.map(item => <li key={item.label}><span>{item.label.replaceAll("_", " ")}</span><b>{(item.confidence * 100).toFixed(1)}%</b></li>)}</ul>
    </section>}
  </main>;
}
createRoot(document.getElementById("root")).render(<App />);
