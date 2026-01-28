"use client";

import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const analyticsModules = [
  {
    id: "survival",
    title: "Survival Analysis",
    description: "Kaplan-Meier curves and Cox regression",
    icon: "S",
    endpoints: ["survival_kaplan_meier", "survival_cox_regression", "survival_compare_treatments"]
  },
  {
    id: "uncertainty",
    title: "Uncertainty Quantification",
    description: "Bayesian confidence estimation",
    icon: "U",
    endpoints: ["quantify_uncertainty"]
  },
  {
    id: "trials",
    title: "Clinical Trial Matching",
    description: "ClinicalTrials.gov API integration",
    icon: "T",
    endpoints: ["match_clinical_trials"]
  },
  {
    id: "counterfactual",
    title: "Counterfactual Analysis",
    description: "What-if scenario modeling",
    icon: "C",
    endpoints: ["analyze_counterfactuals"]
  },
  {
    id: "risk",
    title: "Risk Stratification",
    description: "Low/Intermediate/High risk grouping",
    icon: "R",
    endpoints: ["stratify_risk"]
  },
  {
    id: "temporal",
    title: "Temporal Analysis",
    description: "Disease progression patterns",
    icon: "P",
    endpoints: ["analyze_disease_progression", "identify_intervention_windows"]
  }
];

export default function AnalyticsPage() {
  const [selectedModule, setSelectedModule] = useState<string | null>(null);
  const [patientId, setPatientId] = useState("DEMO_001");
  const [treatment, setTreatment] = useState("Platinum-based chemotherapy");
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const runAnalysis = async (endpoint: string) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/analytics/${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: patientId,
          treatment: treatment,
          patient_data: {
            patient_id: patientId,
            tnm_stage: "IIIA",
            histology_type: "Adenocarcinoma",
            performance_status: 1
          }
        })
      });
      const data = await response.json();
      setResults(data);
    } catch (error) {
      setResults({ error: "Failed to run analysis. Make sure the API is running." });
    }
    setLoading(false);
  };

  return (
    <div className="page-container">
      <header className="page-header">
        <h1>Advanced Analytics Suite</h1>
        <p className="muted">Survival analysis, uncertainty quantification, clinical trials, and counterfactual reasoning</p>
      </header>

      <section className="section">
        <h2>Analytics Modules</h2>
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}>
          {analyticsModules.map((module) => (
            <article
              key={module.id}
              className={`card clickable ${selectedModule === module.id ? "selected" : ""}`}
              onClick={() => setSelectedModule(module.id)}
            >
              <div className="card-header">
                <span className="module-icon">{module.icon}</span>
                <h3>{module.title}</h3>
              </div>
              <p className="muted">{module.description}</p>
              <div className="endpoints-list">
                {module.endpoints.map((ep) => (
                  <span key={ep} className="endpoint-badge">{ep}</span>
                ))}
              </div>
            </article>
          ))}
        </div>
      </section>

      {selectedModule && (
        <section className="section">
          <h2>Run Analysis</h2>
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
            <div className="form-group">
              <label>Treatment</label>
              <input
                type="text"
                value={treatment}
                onChange={(e) => setTreatment(e.target.value)}
                className="input"
              />
            </div>
            <div className="button-group">
              {analyticsModules
                .find((m) => m.id === selectedModule)
                ?.endpoints.map((ep) => (
                  <button
                    key={ep}
                    className="button"
                    onClick={() => runAnalysis(ep)}
                    disabled={loading}
                  >
                    {loading ? "Running..." : `Run ${ep}`}
                  </button>
                ))}
            </div>
          </div>
        </section>
      )}

      {results && (
        <section className="section">
          <h2>Results</h2>
          <div className="card">
            <pre className="code-block">{JSON.stringify(results, null, 2)}</pre>
          </div>
        </section>
      )}
    </div>
  );
}
