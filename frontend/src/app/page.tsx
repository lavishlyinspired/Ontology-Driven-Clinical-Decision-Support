import Link from "next/link";

const features = [
  {
    title: "11-Agent Integrated Workflow",
    body: "Core Processing (4), Specialized Clinical (5), and Orchestration (2) agents work in harmony: from data ingestion through SNOMED/LUCADA mapping, cancer-specific analysis (NSCLC/SCLC), biomarker interpretation, to explanation and persistence." 
  },
  {
    title: "Neo4j Auditability",
    body: "Read-only agents harvest observations, while a single Persistence layer writes decisions and creates an immutable audit record for regulators."
  },
  {
    title: "Clinician-Ready Output",
    body: "Automated MDT summaries, ranking of NICE rules, and treatment arguments keep the multidisciplinary team grounded in evidence."
  }
];

const highlights = [
  {
    label: "Patient-centric",
    value: "Figure 2 data model"
  },
  {
    label: "Guideline coverage",
    value: "R1-R10 (Chemotherapy → Targeted therapy)"
  },
  {
    label: "Latency",
    value: "< 2s per patient (typical cache)"
  }
];

const gradientBar = [
  "MDT orchestration",
  "SNOMED mapping",
  "Neo4j persistence",
  "LLM-explanation"
];

export default function HomePage() {
  return (
    <main>
      <section className="card glow-border">
        <div>
          <p className="muted">Lung Cancer Assistant</p>
          <h1 style={{ fontSize: "2.8rem", marginBottom: "0.5rem" }}>
            Guideline-driven decision support that feels intentional and modern.
          </h1>
          <p>Unified dashboard for oncology teams bridging ontology, vector search, and LangGraph coordination.</p>
          <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
            {highlights.map((item) => (
              <div key={item.label} className="card" style={{ background: "rgba(255,255,255,0.04)" }}>
                <p className="muted" style={{ margin: 0 }}>{item.label}</p>
                <h3 style={{ margin: "0.25rem 0 0" }}>{item.value}</h3>
              </div>
            ))}
          </div>
        </div>
        <div style={{ marginTop: "2rem", display: "flex", flexWrap: "wrap", gap: "0.75rem" }}>
          {gradientBar.map((text) => (
            <span
              key={text}
              style={{
                padding: "0.5rem 1rem",
                borderRadius: "999px",
                border: "1px solid rgba(255, 255, 255, 0.2)",
                fontSize: "0.85rem"
              }}
            >
              {text}
            </span>
          ))}
        </div>
        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", marginTop: "1.5rem" }}>
          <Link href="/chat" className="glow-border card" style={{ flex: 1, textAlign: "center" }}>
            <strong>💬 Chat Assistant</strong>
            <p style={{ margin: "0.4rem 0 0" }}>Conversational analysis</p>
          </Link>
          <Link href="/patients" className="glow-border card" style={{ flex: 1, textAlign: "center" }}>
            <strong>Run a practice patient</strong>
            <p style={{ margin: "0.4rem 0 0" }}>Jenny Sesen + modern cohorts</p>
          </Link>
          <Link href="/guidelines" className="glow-border card" style={{ flex: 1, textAlign: "center" }}>
            <strong>Inspect NICE rules</strong>
            <p style={{ margin: "0.4rem 0 0" }}>R1-R10 mapping & outcomes</p>
          </Link>
        </div>
      </section>

      <section className="section">
        <h2>Experience the Narrative</h2>
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))" }}>
          {features.map((feature) => (
            <article key={feature.title} className="card">
              <h3>{feature.title}</h3>
              <p>{feature.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section">
        <h2>Design Direction</h2>
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))" }}>
          <article className="card">
            <h3>Expressive Typography</h3>
            <p>Space Grotesk headline energy and purposeful spacing to signal tension-free expertise.</p>
          </article>
          <article className="card">
            <h3>Color & Light</h3>
            <p>Gradient surfaces, warm accent amber, and emerald glows keep focus on critical insights.</p>
          </article>
          <article className="card">
            <h3>Motion</h3>
            <p>Buttons and cards animate subtle hover glimmers for feel of a living clinical dashboard.</p>
          </article>
        </div>
      </section>
    </main>
  );
}
