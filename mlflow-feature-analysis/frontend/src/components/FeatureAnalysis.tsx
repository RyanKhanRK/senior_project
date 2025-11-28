// frontend/src/components/FeatureAnalysis.tsx
import React, { useState, useEffect, useRef } from 'react';
import Plot from 'react-plotly.js';
import './FeatureAnalysis.css';

interface Run {
  run_id: string;
  experiment_id: string;
  status: string;
  start_time: string;
}

interface UploadData {
  filename: string;
  shape: [number, number];
  columns: string[];
  preview: Record<string, any>[];
}

interface ShapResults {
  shap_values: number[][];
  features: string[];
  feature_importance: Array<{ feature: string; importance: number }>;
  model_id: string;
  dataset_shape: [number, number];
  computed_at: string;
}

interface ProgressMessage {
  status: string;
  progress: number;
  error?: string;
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_URL = import.meta.env.VITE_API_WS_URL || 'ws://localhost:8000';

export const FeatureAnalysis: React.FC = () => {
  // State management
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>('');
  const [uploadedData, setUploadedData] = useState<UploadData | null>(null);
  const [shapResults, setShapResults] = useState<ShapResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressStatus, setProgressStatus] = useState('');
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Fetch runs on component mount
  useEffect(() => {
    fetchRuns();
  }, []);

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const fetchRuns = async () => {
    try {
      const response = await fetch(`${API_URL}/api/runs`);
      const data = await response.json();
      setRuns(data.runs || []);
      if (data.runs.length > 0) {
        setSelectedRun(data.runs[0].run_id);
      }
    } catch (err) {
      setError(`Failed to fetch runs: ${err}`);
      console.error('Error fetching runs:', err);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setError(null);
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const data: UploadData = await response.json();
      setUploadedData(data);
    } catch (err) {
      setError(`File upload failed: ${err}`);
      console.error('Upload error:', err);
    }
  };

  const handleComputeShap = async () => {
    if (!selectedRun || !uploadedData || !fileInputRef.current?.files?.[0]) {
      setError('Please select a run and upload data');
      return;
    }

    try {
      setError(null);
      setLoading(true);
      setProgress(0);

      const file = fileInputRef.current.files[0];
      const formData = new FormData();
      formData.append('run_id', selectedRun);
      formData.append('file', file);

      // Start computation
      const response = await fetch(`${API_URL}/api/shap/compute`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Computation failed: ${response.statusText}`);
      }

      const { computation_id } = await response.json();

      // Connect to WebSocket for real-time progress
      connectWebSocket(computation_id);
    } catch (err) {
      setError(`SHAP computation failed: ${err}`);
      setLoading(false);
      console.error('Computation error:', err);
    }
  };

  const connectWebSocket = (computationId: string) => {
    const wsUrl = `${WS_URL}/ws/shap/${computationId}`;
    
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      console.log('WebSocket connected');
    };

    wsRef.current.onmessage = (event) => {
      try {
        const message: ProgressMessage = JSON.parse(event.data);
        setProgress(message.progress);
        setProgressStatus(message.status);

        if (message.error) {
          setError(message.error);
          setLoading(false);
        }

        if (message.status === 'Complete') {
          fetchShapResults(computationId);
        }
      } catch (err) {
        console.error('WebSocket message error:', err);
      }
    };

    wsRef.current.onerror = (event) => {
      setError('WebSocket connection error');
      console.error('WebSocket error:', event);
      setLoading(false);
    };

    wsRef.current.onclose = () => {
      console.log('WebSocket closed');
    };
  };

  const fetchShapResults = async (computationId: string) => {
    try {
      const response = await fetch(`${API_URL}/api/shap/results/${computationId}`);
      const data: ShapResults = await response.json();
      setShapResults(data);
      setLoading(false);
    } catch (err) {
      setError(`Failed to fetch results: ${err}`);
      setLoading(false);
      console.error('Results error:', err);
    }
  };

  const downloadResults = async () => {
    if (!shapResults) return;

    try {
      const response = await fetch(`${API_URL}/api/shap/download/${selectedRun}`);
      const data = await response.json();
      
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `shap_results_${Date.now()}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(`Download failed: ${err}`);
      console.error('Download error:', err);
    }
  };

  return (
    <div className="feature-analysis-container">
      <div className="header">
        <h1>🔍 Feature Analysis Dashboard</h1>
        <p>Real-time SHAP explainability for MLflow models</p>
      </div>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}

      <div className="content">
        {/* Left Panel: Controls */}
        <div className="control-panel">
          <div className="section">
            <h3>1️⃣ Select MLflow Run</h3>
            <select
              value={selectedRun}
              onChange={(e) => setSelectedRun(e.target.value)}
              disabled={loading}
            >
              <option value="">-- Select Run --</option>
              {runs.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {run.run_id.substring(0, 8)}... ({run.status})
                </option>
              ))}
            </select>
            <button
              className="refresh-btn"
              onClick={fetchRuns}
              disabled={loading}
              title="Refresh runs list"
            >
              🔄 Refresh
            </button>
          </div>

          <div className="section">
            <h3>2️⃣ Upload Dataset</h3>
            <div
              className="upload-zone"
              onClick={() => fileInputRef.current?.click()}
              style={{
                borderStyle: uploadedData ? 'solid' : 'dashed',
                backgroundColor: uploadedData ? '#f0f8ff' : 'transparent',
              }}
            >
              {uploadedData ? (
                <div className="upload-success">
                  <span>✅ {uploadedData.filename}</span>
                  <small>
                    {uploadedData.shape[0]} rows × {uploadedData.shape[1]} columns
                  </small>
                </div>
              ) : (
                <div className="upload-placeholder">
                  <span>📁 Click or drag CSV file here</span>
                  <small>Supported: .csv (max 10MB)</small>
                </div>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
                disabled={loading}
              />
            </div>
          </div>

          <div className="section">
            <h3>3️⃣ Compute SHAP</h3>
            <button
              className="compute-btn"
              onClick={handleComputeShap}
              disabled={!selectedRun || !uploadedData || loading}
              style={{
                opacity: !selectedRun || !uploadedData || loading ? 0.5 : 1,
                cursor: !selectedRun || !uploadedData || loading ? 'not-allowed' : 'pointer',
              }}
            >
              {loading ? `⏳ Computing (${progress}%)` : '⚡ Compute SHAP'}
            </button>
          </div>

          {/* Progress Bar */}
          {loading && (
            <div className="progress-section">
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              <div className="progress-text">
                <span>{progress}%</span>
                <span>{progressStatus}</span>
              </div>
            </div>
          )}

          {/* Download Button */}
          {shapResults && (
            <div className="section">
              <button
                className="download-btn"
                onClick={downloadResults}
                title="Download SHAP results as JSON"
              >
                💾 Download Results
              </button>
            </div>
          )}
        </div>

        {/* Right Panel: Visualizations */}
        <div className="visualization-panel">
          {!shapResults ? (
            <div className="empty-state">
              <div className="empty-icon">📊</div>
              <h3>No Results Yet</h3>
              <p>Upload data and click "Compute SHAP" to see feature importance</p>
            </div>
          ) : (
            <>
              <div className="results-header">
                <h2>📈 Feature Importance (SHAP)</h2>
                <small>
                  Model: {shapResults.model_id.substring(0, 8)}... | Data:
                  {shapResults.dataset_shape[0]} samples × {shapResults.dataset_shape[1]} features
                </small>
              </div>

              {/* Bar Chart: Feature Importance */}
              <div className="chart-container">
                <Plot
                  data={[
                    {
                      x: shapResults.feature_importance.map((f) => f.importance),
                      y: shapResults.feature_importance.map((f) => f.feature),
                      type: 'bar',
                      orientation: 'h',
                      marker: { color: '#2E86AB' },
                    },
                  ]}
                  layout={{
                    title: 'Mean |SHAP| Feature Importance',
                    xaxis: { title: 'Mean Absolute SHAP Value' },
                    yaxis: { automargin: true },
                    height: 400,
                    margin: { l: 150 },
                  }}
                  config={{ responsive: true }}
                />
              </div>

              {/* Feature Details Table */}
              <div className="table-container">
                <h3>Feature Details</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Feature</th>
                      <th>Importance Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {shapResults.feature_importance.map((item, idx) => (
                      <tr key={idx}>
                        <td>#{idx + 1}</td>
                        <td>{item.feature}</td>
                        <td>
                          <div className="importance-bar">
                            <div
                              className="importance-fill"
                              style={{
                                width: `${(item.importance / Math.max(...shapResults.feature_importance.map((f) => f.importance))) * 100}%`,
                              }}
                            ></div>
                            <span className="importance-value">
                              {item.importance.toFixed(4)}
                            </span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Metadata */}
              <div className="metadata">
                <small>Computed: {new Date(shapResults.computed_at).toLocaleString()}</small>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default FeatureAnalysis;