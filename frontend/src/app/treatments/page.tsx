"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

const treatmentTypes = [
  "Radiotherapy",
  "Chemotherapy",
  "PalliativeCare",
  "Immunotherapy",
  "TargetedTherapy"
];

const fallbackStats = [
  { treatment: "Radiotherapy", patient_count: 128, avg_survival_days: 540, median_survival_days: 510, stages_treated: ["IB", "IIA", "IIIA"] },
  { treatment: "Chemotherapy", patient_count: 220, avg_survival_days: 360, median_survival_days: 330, stages_treated: ["IIIA", "IV"] },
  { treatment: "Immunotherapy", patient_count: 64, avg_survival_days: 470, median_survival_days: 450, stages_treated: ["IIIB", "IV"] },
  { treatment: "TargetedTherapy", patient_count: 42, avg_survival_days: 630, median_survival_days: 610, stages_treated: ["IIIB", "IV"] },
  { treatment: "PalliativeCare", patient_count: 54, avg_survival_days: 210, median_survival_days: 190, stages_treated: ["IV"] }
];

type TreatmentStat = {
  treatment: string;
  patient_count: number;
  avg_survival_days?: number;
  median_survival_days?: number;
  stages_treated?: string[];
};

export default function TreatmentsPage() {
  const [stats, setStats] = useState<TreatmentStat[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const responses = await Promise.all(
          treatmentTypes.map((type) =>
            fetch(`/api/v1/treatments/${type}/statistics`).then((res) => res.json())
          )
        );
        setStats(responses as TreatmentStat[]);
      } catch (error) {
        setStats(fallbackStats);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  return (
    <main>
      <section className="card">
        <h2>Treatment Intelligence</h2>
        <p>Orbit the NHS-grade NICE rules while understanding outcomes, patient spreads, and survival expectations.</p>
        <Link href="/" style={{ color: "var(--accent)", fontWeight: 600 }}>
          Return to dashboard →
        </Link>
      </section>

      <section className="section grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))" }}>
        {(loading ? fallbackStats : stats).map((stat) => (
          <article key={stat.treatment} className="card" style={{ padding: "1.5rem" }}>
            <p className="muted" style={{ marginBottom: "0.25rem" }}>
              {stat.treatment}
            </p>
            <h3 style={{ margin: "0" }}>{stat.patient_count.toLocaleString()} patients</h3>
            <p style={{ margin: "0.4rem 0" }}>Avg survival: {stat.avg_survival_days ?? "—"} days</p>
            <p style={{ margin: 0 }}>Median survival: {stat.median_survival_days ?? "—"} days</p>
            <p style={{ marginTop: "0.75rem", fontSize: "0.85rem" }}>
              Stage coverage: {stat.stages_treated?.join(", ") ?? "—"}
            </p>
          </article>
        ))}
      </section>
    </main>
  );
}
