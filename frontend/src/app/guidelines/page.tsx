"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

const fallbackGuidelines = [
  { rule_id: "R1", name: "Chemotherapy Advanced NSCLC", evidence_level: "Grade A" },
  { rule_id: "R2", name: "Surgery Early NSCLC", evidence_level: "Grade A" },
  { rule_id: "R3", name: "Radiotherapy Inoperable NSCLC", evidence_level: "Grade B" },
  { rule_id: "R7", name: "Immunotherapy Advanced NSCLC", evidence_level: "Grade A" }
];

export default function GuidelinesPage() {
  const [guidelines, setGuidelines] = useState<typeof fallbackGuidelines>(fallbackGuidelines);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchGuidelines = async () => {
      try {
        const response = await fetch("/api/v1/guidelines");
        if (!response.ok) throw new Error("Guidelines fetched failed");
        const data = await response.json();
        setGuidelines(data.slice(0, 6));
      } catch (error) {
        setGuidelines(fallbackGuidelines);
      } finally {
        setLoading(false);
      }
    };
    fetchGuidelines();
  }, []);

  return (
    <main>
      <section className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ margin: 0 }}>Guideline Explorer</h2>
          <Link href="/" style={{ color: "var(--accent)", fontWeight: 600 }}>
            Back to overview →
          </Link>
        </div>
        <p>Track NICE CG121 rules through to Immunotherapy, Targeted therapy, and MDT-ready arguments.</p>
      </section>

      <section className="section">
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))" }}>
          {(loading ? fallbackGuidelines : guidelines).map((rule) => (
            <article key={rule.rule_id} className="card">
              <p className="muted">{rule.rule_id}</p>
              <h3>{rule.name}</h3>
              <p style={{ marginBottom: "0", fontSize: "0.9rem" }}>Evidence: {rule.evidence_level}</p>
              <p style={{ marginTop: "0.5rem", fontSize: "0.85rem" }}>
                Click to see outcomes, argumentation, and Neo4j persistence hooks.
              </p>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
