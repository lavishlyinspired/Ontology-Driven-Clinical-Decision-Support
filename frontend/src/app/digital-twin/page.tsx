"use client";

import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function DigitalTwinPage() {
  const [patientId, setPatientId] = useState("DEMO_001");
  const [twinStatus, setTwinStatus] = useState<any>(null);
  const [predictions, setPredictions] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const initializeTwin = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/api/v1/digital-twin/initialize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: patientId,
          patient_data: {
            patient_id: patientId,
            age: 68,
            sex: "M",
            tnm_stage: "IIIA",
            histology_type: "Adenocarcinoma",
            performance_status: 1,
            cancer_type: "NSCLC",
            biomarker_profile: {
              egfr_mutation: "Exon 19 deletion",
              pdl1_tps: 45
            }
          }
        })
      });
      const data = await response.json();
      setTwinStatus(data);
    } catch (err) {
      setError("Failed to initialize twin. Make sure the API is running.");
      // Show demo data
      setTwinStatus({
        twin_id: `twin_${patientId}_demo`,
        state: "active",
        context_graph_nodes: 12,
        context_graph_edges: 18,
        created_at: new Date().toISOString()
      });
    }
    setLoading(false);
  };

  const getPredictions = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/digital-twin/${patientId}/predictions`);
      const data = await response.json();
      setPredictions(data);
    } catch (err) {
      // Show demo predictions
      setPredictions({
        pathways: [
          {
            description: "Standard EGFR TKI pathway with durable response",
            probability: 0.65,
            median_pfs_months: 18,
            recommended_treatments: ["Osimertinib", "Afatinib"]
          },
          {
            description: "EGFR TKI with acquired resistance requiring switch",
            probability: 0.25,
            median_pfs_months: 12,
            recommended_treatments: ["Sequential TKI", "Combination therapy"]
          },
          {
            description: "Early progression requiring chemotherapy backup",
            probability: 0.10,
            median_pfs_months: 6,
            recommended_treatments: ["Platinum doublet", "Docetaxel"]
          }
        ],
        confidence: 0.82,
        model_version: "2025.1"
      });
    }
    setLoading(false);
  };

  return (
    <div className="page-container">
      <header className="page-header">
        <h1>Digital Twin Engine</h1>
        <p className="muted">Living patient model with real-time predictions and alerts</p>
      </header>

      <section className="section">
        <h2>Twin Management</h2>
        <div className="card">
          <div className="form-group">
            <label>Patient ID</label>
            <input
              type="text"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              className="input"
            />
          </div>
          <div className="button-group">
            <button className="button" onClick={initializeTwin} disabled={loading}>
              {loading ? "Initializing..." : "Initialize Twin"}
            </button>
            <button className="button button-secondary" onClick={getPredictions} disabled={loading || !twinStatus}>
              Get Predictions
            </button>
          </div>
        </div>
      </section>

      {error && (
        <div className="card error-card">
          <p>{error}</p>
        </div>
      )}

      {twinStatus && (
        <section className="section">
          <h2>Twin Status</h2>
          <div className="card">
            <div className="status-grid">
              <div className="status-item">
                <span className="status-label">Twin ID</span>
                <span className="status-value">{twinStatus.twin_id}</span>
              </div>
              <div className="status-item">
                <span className="status-label">State</span>
                <span className={`status-badge status-${twinStatus.state}`}>{twinStatus.state}</span>
              </div>
              <div className="status-item">
                <span className="status-label">Context Graph</span>
                <span className="status-value">{twinStatus.context_graph_nodes} nodes, {twinStatus.context_graph_edges} edges</span>
              </div>
              <div className="status-item">
                <span className="status-label">Created</span>
                <span className="status-value">{new Date(twinStatus.created_at).toLocaleString()}</span>
              </div>
            </div>
          </div>
        </section>
      )}

      {predictions && (
        <section className="section">
          <h2>Trajectory Predictions</h2>
          <div className="predictions-container">
            {predictions.pathways?.map((pathway: any, index: number) => (
              <div key={index} className="card pathway-card">
                <div className="pathway-header">
                  <h3>Pathway {index + 1}</h3>
                  <span className="probability-badge">{(pathway.probability * 100).toFixed(0)}%</span>
                </div>
                <p className="pathway-description">{pathway.description}</p>
                <div className="pathway-stats">
                  <div className="stat">
                    <span className="stat-label">Median PFS</span>
                    <span className="stat-value">{pathway.median_pfs_months} months</span>
                  </div>
                </div>
                <div className="treatments-list">
                  <span className="treatments-label">Recommended:</span>
                  {pathway.recommended_treatments?.map((t: string, i: number) => (
                    <span key={i} className="treatment-badge">{t}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="card confidence-card">
            <span>Overall Confidence: </span>
            <strong>{(predictions.confidence * 100).toFixed(0)}%</strong>
            <span className="muted"> (Model: {predictions.model_version})</span>
          </div>
        </section>
      )}

      <section className="section">
        <h2>Twin Architecture</h2>
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))" }}>
          <div className="card">
            <h3>Context Graph Layer</h3>
            <p className="muted">Dynamic hypergraph maintaining patient state, relationships, and reasoning chains</p>
          </div>
          <div className="card">
            <h3>Agent Orchestration</h3>
            <p className="muted">Coordinates NSCLC, SCLC, Biomarker, and Comorbidity agents</p>
          </div>
          <div className="card">
            <h3>Temporal Analysis</h3>
            <p className="muted">Disease progression patterns and intervention windows</p>
          </div>
          <div className="card">
            <h3>Alert System</h3>
            <p className="muted">Real-time monitoring with severity-based notifications</p>
          </div>
        </div>
      </section>
    </div>
  );
}
