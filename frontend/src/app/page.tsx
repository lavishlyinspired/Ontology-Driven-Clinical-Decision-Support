import Link from "next/link";
import { Activity, Shield, Database, MessageSquare, Users, FileText } from "lucide-react";

const features = [
  {
    title: "11-Agent Integrated Workflow",
    body: "Core Processing (4), Specialized Clinical (5), and Orchestration (2) agents work in harmony: from data ingestion through SNOMED/LUCADA mapping, cancer-specific analysis (NSCLC/SCLC), biomarker interpretation, to explanation and persistence.",
    icon: Users
  },
  {
    title: "Neo4j Auditability",
    body: "Read-only agents harvest observations, while a single Persistence layer writes decisions and creates an immutable audit record for regulators.",
    icon: Database
  },
  {
    title: "Clinician-Ready Output",
    body: "Automated MDT summaries, ranking of NICE rules, and treatment arguments keep the multidisciplinary team grounded in evidence.",
    icon: FileText
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

const actions = [
  {
    title: "💬 Chat Assistant",
    description: "Conversational analysis",
    href: "/chat"
  },
  {
    title: "🔬 Run a practice patient",
    description: "Jenny Sesen + modern cohorts",
    href: "/patients"
  },
  {
    title: "📋 Inspect NICE rules",
    description: "R1-R10 mapping & outcomes",
    href: "/guidelines"
  }
];

export default function HomePage() {
  return (
    <main className="max-w-7xl mx-auto">
      {/* Hero Section */}
      <section className="mb-12">
        <div className="mb-8">
          <p className="text-blue-600 font-medium mb-2">Lung Cancer Assistant</p>
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            Guideline-driven decision support that feels intentional and modern.
          </h1>
          <p className="text-lg text-gray-600 mb-8">
            Unified dashboard for oncology teams bridging ontology, vector search, and LangGraph coordination.
          </p>
          
          {/* Highlights Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {highlights.map((item) => (
              <div key={item.label} className="card bg-gray-50">
                <p className="text-sm text-gray-500 mb-1">{item.label}</p>
                <h3 className="text-xl font-semibold text-gray-900">{item.value}</h3>
              </div>
            ))}
          </div>

          {/* Tags */}
          <div className="flex flex-wrap gap-2 mb-8">
            {["MDT orchestration", "SNOMED mapping", "Neo4j persistence", "LLM-explanation"].map((text) => (
              <span
                key={text}
                className="px-4 py-2 bg-blue-50 text-blue-700 text-sm font-medium rounded-full border border-blue-200"
              >
                {text}
              </span>
            ))}
          </div>

          {/* Action Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {actions.map((action) => (
              <Link key={action.href} href={action.href} className="card hover:border-blue-300 hover:shadow-lg transition-all">
                <div className="text-center">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">{action.title}</h3>
                  <p className="text-sm text-gray-600">{action.description}</p>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="section">
        <h2 className="text-3xl font-bold text-gray-900 mb-6">Experience the Narrative</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {features.map((feature) => {
            const Icon = feature.icon;
            return (
              <article key={feature.title} className="card">
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                  <Icon className="w-6 h-6 text-blue-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">{feature.title}</h3>
                <p className="text-gray-600">{feature.body}</p>
              </article>
            );
          })}
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
