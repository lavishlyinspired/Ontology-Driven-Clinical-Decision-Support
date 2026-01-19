"use client";

import { useState, useTransition } from "react";
import TreatmentRecommendations from "./TreatmentRecommendations";
import ArgumentPanel from "./ArgumentPanel";
import MDTSummary from "./MDTSummary";

type PatientInput = {
  name: string;
  sex: string;
  age: number;
  tnm_stage: string;
  histology_type: string;
  performance_status: number;
  laterality: string;
  diagnosis: string;
};

type AnalysisResult = {
  patient_id: string;
  mdt_summary: string;
  recommendations: Array<{
    treatment_type: string;
    rule_id: string;
    evidence_level: string;
    priority: number;
    contraindications?: string[];
  }>;
  arguments: Array<{
    treatment: string;
    arguments: string[];
  }>;
};

const defaultPatient: PatientInput = {
  name: "Jenny Sesen",
  sex: "F",
  age: 72,
  tnm_stage: "IIA",
  histology_type: "Carcinosarcoma",
  performance_status: 1,
  laterality: "Right",
  diagnosis: "Malignant Neoplasm of Lung"
};

const fallbackResult: AnalysisResult = {
  patient_id: "Jenny_Sesen_200312",
  mdt_summary:
    "Jenny Sesen (72F, Stage IIA Carcinosarcoma) is classified under R3 (radical radiotherapy). Treatment recommendations emphasize curative intent with radiotherapy planning and MDT discussion on surgical resection feasibility.",
  recommendations: [
    {
      treatment_type: "Radiotherapy",
      rule_id: "R3",
      evidence_level: "Grade B",
      priority: 82,
      contraindications: ["Prior chest radiotherapy", "Large tumor volume"]
    }
  ],
  arguments: [
    {
      treatment: "Radiotherapy",
      arguments: [
        "Claim: Radical radiotherapy offers a curative option for Stage I-IIIA disease.",
        "Evidence: NICE R3 + NDCT data; Dose escalation tolerated with PS 1.",
        "Opposing: Tumor proximity to cardiac structures raises toxicity concerns."
      ]
    }
  ]
};

export default function PatientForm() {
  const [patient, setPatient] = useState<PatientInput>(defaultPatient);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [isPending, startTransition] = useTransition();
  const [error, setError] = useState<string | null>(null);

  const handleChange = (field: keyof PatientInput, value: string | number) => {
    setPatient((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    startTransition(async () => {
      setError(null);
      try {
        const response = await fetch("/api/v2/patients/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(patient)
        });
        if (!response.ok) {
          throw new Error("Workflow API unavailable");
        }
        const result = await response.json();
        const parsed: AnalysisResult = {
          patient_id: result.patient_id,
          mdt_summary: result.mdt_summary,
          recommendations: result.recommendations.map((rec: any) => ({
            treatment_type: rec.treatment_type,
            rule_id: rec.rule_id,
            evidence_level: rec.evidence_level,
            priority: rec.priority,
            contraindications: rec.contraindications
          })),
          arguments: result.arguments || []
        };
        setAnalysis(parsed);
      } catch (err) {
        setAnalysis(fallbackResult);
        setError("Workflow API offline, showing static demonstration example.");
      }
    });
  };

  return (
    <section className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <h2 style={{ margin: 0 }}>Patient Simulation</h2>
        <p className="muted">Jenny Sesen readiness check</p>
      </div>
      <form onSubmit={handleSubmit} className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "1.25rem", marginTop: "1rem" }}>
        {["name", "sex", "tnm_stage", "histology_type", "laterality"].map((field) => (
          <label key={field} style={{ display: "flex", flexDirection: "column" }}>
            <span style={{ fontSize: "0.85rem", marginBottom: "0.25rem" }}>{field.replace(/_/g, " ")}</span>
            <input
              value={(patient as any)[field]}
              onChange={(event) => handleChange(field as keyof PatientInput, event.target.value)}
              className="card"
              style={{ padding: "0.6rem", borderRadius: "12px", border: "1px solid var(--border)", background: "rgba(255,255,255,0.02)" }}
            />
          </label>
        ))}
        {[
          { label: "Age", field: "age", type: "number" },
          { label: "Performance Status", field: "performance_status", type: "number" }
        ].map(({ label, field, type }) => (
          <label key={field} style={{ display: "flex", flexDirection: "column" }}>
            <span style={{ fontSize: "0.85rem", marginBottom: "0.25rem" }}>{label}</span>
            <input
              type={type}
              value={(patient as any)[field]}
              onChange={(event) => handleChange(field as keyof PatientInput, Number(event.target.value))}
              className="card"
              style={{ padding: "0.6rem", borderRadius: "12px", border: "1px solid var(--border)", background: "rgba(255,255,255,0.02)" }}
            />
          </label>
        ))}
      </form>
      <button
        type="button"
        onClick={handleSubmit}
        disabled={isPending}
        style={{
          marginTop: "1rem",
          borderRadius: "999px",
          padding: "0.65rem 1.75rem",
          border: "none",
          fontWeight: 600,
          background: "linear-gradient(135deg, var(--accent), var(--accent-strong))",
          color: "#030711",
          cursor: isPending ? "progress" : "pointer"
        }}
      >
        {isPending ? "Analyzing…" : "Run 11-Agent Analysis"}
      </button>
      {error && <p className="muted" style={{ marginTop: "0.75rem" }}>{error}</p>}
      {analysis && (
        <div className="section">
          <TreatmentRecommendations recommendations={analysis.recommendations} />
          <ArgumentPanel entries={analysis.arguments} />
          <MDTSummary summary={analysis.mdt_summary} />
        </div>
      )}
    </section>
  );
}
