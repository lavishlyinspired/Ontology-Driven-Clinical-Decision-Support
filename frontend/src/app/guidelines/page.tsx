"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ChevronRight, FileText, AlertCircle, CheckCircle, X } from "lucide-react";

interface Guideline {
  rule_id: string;
  name: string;
  evidence_level: string;
  description?: string;
  conditions?: string[];
  treatments?: string[];
}

const fallbackGuidelines: Guideline[] = [
  {
    rule_id: "R1",
    name: "Chemotherapy Advanced NSCLC",
    evidence_level: "Grade A",
    description: "Platinum-based chemotherapy for advanced NSCLC without driver mutations",
    conditions: ["Stage IIIB-IV NSCLC", "No actionable mutations", "PS 0-2"],
    treatments: ["Carboplatin + Pemetrexed", "Cisplatin + Gemcitabine"]
  },
  {
    rule_id: "R2",
    name: "Surgery Early NSCLC",
    evidence_level: "Grade A",
    description: "Surgical resection for early-stage NSCLC with curative intent",
    conditions: ["Stage I-II NSCLC", "Medically operable", "Adequate pulmonary function"],
    treatments: ["Lobectomy", "Segmentectomy", "Wedge resection"]
  },
  {
    rule_id: "R3",
    name: "Radiotherapy Inoperable NSCLC",
    evidence_level: "Grade B",
    description: "Definitive radiotherapy for medically inoperable early-stage NSCLC",
    conditions: ["Stage I-II NSCLC", "Medically inoperable", "PS 0-2"],
    treatments: ["SBRT/SABR", "Conventional RT"]
  },
  {
    rule_id: "R4",
    name: "EGFR TKI First-Line",
    evidence_level: "Grade A",
    description: "EGFR tyrosine kinase inhibitors for EGFR-mutant NSCLC",
    conditions: ["EGFR mutation positive", "Ex19del or L858R", "Any stage"],
    treatments: ["Osimertinib", "Gefitinib", "Erlotinib"]
  },
  {
    rule_id: "R5",
    name: "ALK Inhibitor First-Line",
    evidence_level: "Grade A",
    description: "ALK inhibitors for ALK-rearranged NSCLC",
    conditions: ["ALK rearrangement positive", "Any stage"],
    treatments: ["Alectinib", "Brigatinib", "Lorlatinib"]
  },
  {
    rule_id: "R6",
    name: "Concurrent Chemoradiotherapy",
    evidence_level: "Grade A",
    description: "Concurrent platinum-based chemoradiotherapy for unresectable Stage III",
    conditions: ["Stage III NSCLC", "Unresectable", "PS 0-1"],
    treatments: ["Cisplatin + Etoposide + RT", "Carboplatin + Paclitaxel + RT"]
  },
  {
    rule_id: "R7",
    name: "Immunotherapy Advanced NSCLC",
    evidence_level: "Grade A",
    description: "Immune checkpoint inhibitors for advanced NSCLC",
    conditions: ["Stage IV NSCLC", "PD-L1 expression", "No driver mutations"],
    treatments: ["Pembrolizumab", "Nivolumab", "Atezolizumab"]
  },
  {
    rule_id: "R8",
    name: "Durvalumab Consolidation",
    evidence_level: "Grade A",
    description: "Durvalumab maintenance after chemoradiotherapy for Stage III",
    conditions: ["Stage III NSCLC", "Completed cCRT", "No progression"],
    treatments: ["Durvalumab for 12 months"]
  },
  {
    rule_id: "R9",
    name: "SCLC Limited Stage",
    evidence_level: "Grade A",
    description: "Concurrent chemoradiotherapy for limited-stage SCLC",
    conditions: ["Limited-stage SCLC", "PS 0-2"],
    treatments: ["Cisplatin + Etoposide + Thoracic RT", "PCI if response"]
  },
  {
    rule_id: "R10",
    name: "SCLC Extensive Stage",
    evidence_level: "Grade A",
    description: "Chemo-immunotherapy for extensive-stage SCLC",
    conditions: ["Extensive-stage SCLC", "PS 0-2"],
    treatments: ["Carboplatin + Etoposide + Atezolizumab/Durvalumab"]
  }
];

export default function GuidelinesPage() {
  const [guidelines, setGuidelines] = useState<Guideline[]>(fallbackGuidelines);
  const [loading, setLoading] = useState(true);
  const [selectedGuideline, setSelectedGuideline] = useState<Guideline | null>(null);

  useEffect(() => {
    const fetchGuidelines = async () => {
      try {
        const response = await fetch("http://localhost:8000/api/v1/guidelines");
        if (!response.ok) throw new Error("Guidelines fetch failed");
        const data = await response.json();
        // Merge fetched data with fallback details
        const merged = data.map((g: any) => {
          const fallback = fallbackGuidelines.find(f => f.rule_id === g.rule_id);
          return { ...fallback, ...g };
        });
        setGuidelines(merged.length > 0 ? merged : fallbackGuidelines);
      } catch (error) {
        console.warn('Using fallback guidelines:', error);
        setGuidelines(fallbackGuidelines);
      } finally {
        setLoading(false);
      }
    };
    fetchGuidelines();
  }, []);

  const getEvidenceColor = (level: string) => {
    if (level.includes('A')) return '#10b981';  // Green
    if (level.includes('B')) return '#f59e0b';  // Amber
    if (level.includes('C')) return '#ef4444';  // Red
    return '#6b7280';
  };

  return (
    <main>
      <section className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h1 style={{ margin: 0, fontSize: '2rem' }}>Guideline Explorer</h1>
            <p className="muted" style={{ margin: '0.5rem 0 0' }}>
              NICE CG121 and NCCN guidelines for lung cancer treatment decisions
            </p>
          </div>
          <Link href="/chat" style={{
            color: "#60a5fa",
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            textDecoration: 'none'
          }}>
            Try in Chat <ChevronRight size={18} />
          </Link>
        </div>
      </section>

      <section className="section">
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: '1.5rem' }}>
          {(loading ? fallbackGuidelines : guidelines).map((rule) => (
            <article
              key={rule.rule_id}
              className="card glow-border"
              onClick={() => setSelectedGuideline(rule)}
              style={{
                cursor: 'pointer',
                transition: 'transform 0.2s, box-shadow 0.2s',
                position: 'relative'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-4px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                <span style={{
                  background: 'rgba(96, 165, 250, 0.2)',
                  padding: '0.25rem 0.75rem',
                  borderRadius: '999px',
                  fontSize: '0.8rem',
                  fontWeight: '600',
                  color: '#60a5fa'
                }}>
                  {rule.rule_id}
                </span>
                <span style={{
                  background: `${getEvidenceColor(rule.evidence_level)}20`,
                  color: getEvidenceColor(rule.evidence_level),
                  padding: '0.25rem 0.75rem',
                  borderRadius: '999px',
                  fontSize: '0.75rem',
                  fontWeight: '600'
                }}>
                  {rule.evidence_level}
                </span>
              </div>
              <h3 style={{ margin: '0 0 0.5rem', fontSize: '1.1rem' }}>{rule.name}</h3>
              <p className="muted" style={{ margin: 0, fontSize: '0.85rem', lineHeight: '1.5' }}>
                {rule.description || 'Click to see full guideline details, conditions, and treatment options.'}
              </p>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                marginTop: '1rem',
                color: '#60a5fa',
                fontSize: '0.85rem'
              }}>
                <FileText size={14} />
                <span>View details</span>
                <ChevronRight size={14} />
              </div>
            </article>
          ))}
        </div>
      </section>

      {/* Guideline Detail Modal */}
      {selectedGuideline && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.8)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '2rem'
          }}
          onClick={() => setSelectedGuideline(null)}
        >
          <div
            className="card"
            style={{
              maxWidth: '600px',
              width: '100%',
              maxHeight: '80vh',
              overflow: 'auto',
              position: 'relative'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setSelectedGuideline(null)}
              style={{
                position: 'absolute',
                top: '1rem',
                right: '1rem',
                background: 'transparent',
                border: 'none',
                color: '#fff',
                cursor: 'pointer',
                padding: '0.5rem'
              }}
            >
              <X size={24} />
            </button>

            <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem' }}>
              <span style={{
                background: 'rgba(96, 165, 250, 0.2)',
                padding: '0.25rem 0.75rem',
                borderRadius: '999px',
                fontSize: '0.85rem',
                fontWeight: '600',
                color: '#60a5fa'
              }}>
                {selectedGuideline.rule_id}
              </span>
              <span style={{
                background: `${getEvidenceColor(selectedGuideline.evidence_level)}20`,
                color: getEvidenceColor(selectedGuideline.evidence_level),
                padding: '0.25rem 0.75rem',
                borderRadius: '999px',
                fontSize: '0.8rem',
                fontWeight: '600'
              }}>
                {selectedGuideline.evidence_level}
              </span>
            </div>

            <h2 style={{ margin: '0 0 1rem', fontSize: '1.5rem' }}>{selectedGuideline.name}</h2>

            <p style={{ color: '#94a3b8', lineHeight: '1.6', marginBottom: '1.5rem' }}>
              {selectedGuideline.description}
            </p>

            {selectedGuideline.conditions && selectedGuideline.conditions.length > 0 && (
              <div style={{ marginBottom: '1.5rem' }}>
                <h4 style={{ margin: '0 0 0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <AlertCircle size={18} style={{ color: '#f59e0b' }} />
                  Conditions
                </h4>
                <ul style={{ margin: 0, paddingLeft: '1.5rem', color: '#94a3b8' }}>
                  {selectedGuideline.conditions.map((c, i) => (
                    <li key={i} style={{ marginBottom: '0.25rem' }}>{c}</li>
                  ))}
                </ul>
              </div>
            )}

            {selectedGuideline.treatments && selectedGuideline.treatments.length > 0 && (
              <div style={{ marginBottom: '1.5rem' }}>
                <h4 style={{ margin: '0 0 0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <CheckCircle size={18} style={{ color: '#10b981' }} />
                  Treatment Options
                </h4>
                <ul style={{ margin: 0, paddingLeft: '1.5rem', color: '#94a3b8' }}>
                  {selectedGuideline.treatments.map((t, i) => (
                    <li key={i} style={{ marginBottom: '0.25rem' }}>{t}</li>
                  ))}
                </ul>
              </div>
            )}

            <div style={{
              display: 'flex',
              gap: '1rem',
              marginTop: '2rem',
              paddingTop: '1rem',
              borderTop: '1px solid rgba(255,255,255,0.1)'
            }}>
              <Link
                href={`/chat?q=Apply ${selectedGuideline.rule_id} ${selectedGuideline.name}`}
                style={{
                  flex: 1,
                  padding: '0.75rem',
                  background: '#60a5fa',
                  color: '#000',
                  textAlign: 'center',
                  borderRadius: '0.5rem',
                  textDecoration: 'none',
                  fontWeight: '600'
                }}
              >
                Try in Chat
              </Link>
              <button
                onClick={() => setSelectedGuideline(null)}
                style={{
                  flex: 1,
                  padding: '0.75rem',
                  background: 'transparent',
                  border: '1px solid rgba(255,255,255,0.2)',
                  color: '#fff',
                  borderRadius: '0.5rem',
                  cursor: 'pointer'
                }}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
