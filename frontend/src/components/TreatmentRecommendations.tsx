type Recommendation = {
  treatment_type: string;
  rule_id: string;
  evidence_level: string;
  priority: number;
  contraindications?: string[];
};

export default function TreatmentRecommendations({
  recommendations
}: {
  recommendations: Recommendation[];
}) {
  return (
    <div className="section card">
      <h3>Recommendations</h3>
      <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
        {recommendations.map((rec) => (
          <article key={rec.rule_id} className="card" style={{ padding: "1rem" }}>
            <p className="muted" style={{ margin: 0 }}>
              {rec.rule_id} · {rec.evidence_level}
            </p>
            <h4 style={{ margin: "0.25rem 0 0.5rem" }}>{rec.treatment_type}</h4>
            <p style={{ margin: 0 }}>Priority: {rec.priority}</p>
            {rec.contraindications && rec.contraindications.length > 0 && (
              <p style={{ margin: "0.35rem 0 0", fontSize: "0.85rem" }}>
                Contraindications: {rec.contraindications.join(", ")}
              </p>
            )}
          </article>
        ))}
      </div>
    </div>
  );
}
