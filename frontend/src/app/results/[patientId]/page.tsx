'use client'

import { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import Link from 'next/link'
import { ArrowLeft, Download, FileText, AlertCircle, CheckCircle, TrendingUp } from 'lucide-react'

interface Recommendation {
  treatment: string
  rule_id: string
  evidence_level: string
  confidence: number
  treatment_intent: string
  rationale: string[]
  contraindications: string[]
  expected_outcomes?: {
    response_rate: number
    median_pfs_months: number
    median_os_months?: number
  }
}

interface PatientResults {
  patient_id: string
  patient_data: any
  primary_recommendation: Recommendation
  alternative_recommendations: Recommendation[]
  mdt_summary: string
  analysis_metadata: {
    processing_time_ms: number
    agents_executed: string[]
    timestamp: string
  }
}

export default function ResultsPage() {
  const params = useParams()
  const patientId = params.patientId as string
  
  const [results, setResults] = useState<PatientResults | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchResults()
  }, [patientId])

  const fetchResults = async () => {
    try {
      // Try to get results from API
      const response = await fetch(`/api/v2/patients/${patientId}/analysis`)
      
      if (response.ok) {
        const data = await response.json()
        setResults(data)
      } else {
        // Mock data for demonstration
        setResults({
          patient_id: patientId,
          patient_data: {
            name: 'John D.',
            age: 65,
            sex: 'M',
            tnm_stage: 'IIIA',
            histology_type: 'Adenocarcinoma',
            performance_status: 1
          },
          primary_recommendation: {
            treatment: 'Osimertinib (EGFR TKI)',
            rule_id: 'Biomarker',
            evidence_level: 'A',
            confidence: 0.92,
            treatment_intent: 'Curative',
            rationale: [
              'EGFR Exon 19 deletion detected',
              'Stage IIIA adenocarcinoma',
              'Performance status 1 (good functional status)',
              'No contraindications identified'
            ],
            contraindications: [],
            expected_outcomes: {
              response_rate: 0.75,
              median_pfs_months: 18.9,
              median_os_months: 38.6
            }
          },
          alternative_recommendations: [
            {
              treatment: 'Chemotherapy + Pembrolizumab',
              rule_id: 'R7',
              evidence_level: 'A',
              confidence: 0.78,
              treatment_intent: 'Curative',
              rationale: ['Stage IIIA with good performance status', 'Alternative to targeted therapy'],
              contraindications: []
            },
            {
              treatment: 'Platinum-based Chemotherapy',
              rule_id: 'R1',
              evidence_level: 'A',
              confidence: 0.65,
              treatment_intent: 'Curative',
              rationale: ['Standard first-line option', 'No molecular target identified'],
              contraindications: []
            }
          ],
          mdt_summary: 'Patient presents with Stage IIIA adenocarcinoma with EGFR Exon 19 deletion. Osimertinib is recommended as first-line therapy based on molecular profiling showing actionable EGFR mutation. Expected response rate 75% with median PFS of 18.9 months.',
          analysis_metadata: {
            processing_time_ms: 1624,
            agents_executed: ['IngestionAgent', 'SemanticMappingAgent', 'ClassificationAgent', 'BiomarkerAgent', 'ExplanationAgent', 'PersistenceAgent'],
            timestamp: new Date().toISOString()
          }
        })
      }
      setLoading(false)
    } catch (err) {
      setError('Failed to load results')
      setLoading(false)
    }
  }

  const exportPDF = () => {
    alert('PDF export functionality would be implemented here')
  }

  const exportMDT = () => {
    alert('MDT summary export functionality would be implemented here')
  }

  if (loading) {
    return (
      <main className="section">
        <div style={{ textAlign: 'center', padding: '4rem 0' }}>
          <div className="animate-spin" style={{ margin: '0 auto', width: '48px', height: '48px', border: '4px solid rgba(96, 165, 250, 0.2)', borderTopColor: '#60a5fa', borderRadius: '50%' }} />
          <p style={{ marginTop: '1rem' }}>Loading results...</p>
        </div>
      </main>
    )
  }

  if (error || !results) {
    return (
      <main className="section">
        <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
          <AlertCircle size={48} style={{ margin: '0 auto', color: '#ef4444' }} />
          <h2 style={{ marginTop: '1rem' }}>Error Loading Results</h2>
          <p className="muted" style={{ marginTop: '0.5rem' }}>{error || 'Results not found'}</p>
          <Link href="/patients/analyze" className="glow-border card" style={{ display: 'inline-block', marginTop: '1.5rem', padding: '1rem 2rem', textDecoration: 'none' }}>
            Start New Analysis
          </Link>
        </div>
      </main>
    )
  }

  const { patient_data, primary_recommendation, alternative_recommendations, mdt_summary } = results

  return (
    <main>
      {/* Header */}
      <section className="card glow-border">
        <div>
          <Link href="/dashboard" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', color: '#60a5fa', textDecoration: 'none' }}>
            <ArrowLeft size={20} />
            Back to Dashboard
          </Link>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <h1 style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>
                Treatment Recommendations
              </h1>
              <p className="muted">
                Patient: {patient_data.name} | {patient_data.sex}, {patient_data.age}y | 
                Stage {patient_data.tnm_stage} {patient_data.histology_type}
              </p>
            </div>
            <div style={{ display: 'flex', gap: '1rem' }}>
              <button
                onClick={exportMDT}
                className="card"
                style={{
                  padding: '0.75rem 1.5rem',
                  border: '1px solid rgba(255,255,255,0.2)',
                  background: 'transparent',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
              >
                <FileText size={18} />
                MDT Summary
              </button>
              <button
                onClick={exportPDF}
                className="glow-border card"
                style={{
                  padding: '0.75rem 1.5rem',
                  background: '#60a5fa',
                  color: '#000',
                  cursor: 'pointer',
                  fontWeight: 'bold',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
              >
                <Download size={18} />
                Export PDF
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Primary Recommendation */}
      <section className="section">
        <div className="card glow-border" style={{ padding: '2rem', background: 'linear-gradient(135deg, rgba(96, 165, 250, 0.1), rgba(16, 185, 129, 0.1))' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <CheckCircle size={24} style={{ color: '#10b981' }} />
                <h2 style={{ margin: 0 }}>PRIMARY RECOMMENDATION</h2>
              </div>
              <p className="muted" style={{ margin: 0 }}>Evidence Level: {primary_recommendation.evidence_level} | Source: {primary_recommendation.rule_id}</p>
            </div>
            <div>
              <div style={{ textAlign: 'right' }}>
                <p className="muted" style={{ margin: 0, fontSize: '0.85rem' }}>Confidence</p>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.25rem' }}>
                  <div style={{ 
                    width: '100px', 
                    height: '8px', 
                    background: 'rgba(255,255,255,0.1)', 
                    borderRadius: '999px',
                    overflow: 'hidden'
                  }}>
                    <div style={{ 
                      width: `${primary_recommendation.confidence * 100}%`, 
                      height: '100%', 
                      background: '#10b981',
                      borderRadius: '999px'
                    }} />
                  </div>
                  <span style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
                    {Math.round(primary_recommendation.confidence * 100)}%
                  </span>
                </div>
              </div>
            </div>
          </div>

          <h1 style={{ fontSize: '2.5rem', marginBottom: '1.5rem', color: '#60a5fa' }}>
            {primary_recommendation.treatment}
          </h1>

          <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>
            <div>
              <h3 style={{ marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <CheckCircle size={20} style={{ color: '#10b981' }} />
                Supporting Evidence
              </h3>
              <ul style={{ margin: 0, paddingLeft: '1.5rem' }}>
                {primary_recommendation.rationale.map((item, i) => (
                  <li key={i} style={{ marginBottom: '0.5rem' }}>{item}</li>
                ))}
              </ul>
            </div>

            {primary_recommendation.expected_outcomes && (
              <div>
                <h3 style={{ marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <TrendingUp size={20} style={{ color: '#10b981' }} />
                  Expected Outcomes
                </h3>
                <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <div className="card" style={{ padding: '1rem', background: 'rgba(255,255,255,0.05)' }}>
                    <p className="muted" style={{ margin: 0, fontSize: '0.85rem' }}>Response Rate</p>
                    <p style={{ margin: '0.25rem 0 0', fontSize: '1.5rem', fontWeight: 'bold' }}>
                      {Math.round(primary_recommendation.expected_outcomes.response_rate * 100)}%
                    </p>
                  </div>
                  <div className="card" style={{ padding: '1rem', background: 'rgba(255,255,255,0.05)' }}>
                    <p className="muted" style={{ margin: 0, fontSize: '0.85rem' }}>Median PFS</p>
                    <p style={{ margin: '0.25rem 0 0', fontSize: '1.5rem', fontWeight: 'bold' }}>
                      {primary_recommendation.expected_outcomes.median_pfs_months} mo
                    </p>
                  </div>
                  {primary_recommendation.expected_outcomes.median_os_months && (
                    <div className="card" style={{ padding: '1rem', background: 'rgba(255,255,255,0.05)', gridColumn: 'span 2' }}>
                      <p className="muted" style={{ margin: 0, fontSize: '0.85rem' }}>Median OS</p>
                      <p style={{ margin: '0.25rem 0 0', fontSize: '1.5rem', fontWeight: 'bold' }}>
                        {primary_recommendation.expected_outcomes.median_os_months} mo
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          <div className="card" style={{ padding: '1.5rem', background: 'rgba(96, 165, 250, 0.1)', border: '1px solid rgba(96, 165, 250, 0.3)' }}>
            <h4 style={{ margin: '0 0 0.5rem 0' }}>Clinical Considerations</h4>
            <p style={{ margin: 0, lineHeight: '1.6' }}>{mdt_summary}</p>
          </div>
        </div>
      </section>

      {/* Alternative Options */}
      <section className="section">
        <h2 style={{ marginBottom: '1.5rem' }}>Alternative Treatment Options</h2>
        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '1.5rem' }}>
          {alternative_recommendations.map((rec, idx) => (
            <div key={idx} className="card" style={{ padding: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                <div>
                  <h3 style={{ margin: 0 }}>{rec.treatment}</h3>
                  <p className="muted" style={{ margin: '0.25rem 0 0', fontSize: '0.85rem' }}>
                    Evidence: Grade {rec.evidence_level} ({rec.rule_id})
                  </p>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <p className="muted" style={{ margin: 0, fontSize: '0.75rem' }}>Confidence</p>
                  <p style={{ margin: '0.25rem 0 0', fontSize: '1.25rem', fontWeight: 'bold' }}>
                    {Math.round(rec.confidence * 100)}%
                  </p>
                </div>
              </div>

              <ul style={{ margin: '1rem 0 0', paddingLeft: '1.5rem', fontSize: '0.9rem' }}>
                {rec.rationale.map((item, i) => (
                  <li key={i} style={{ marginBottom: '0.5rem' }}>{item}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      {/* Analysis Metadata */}
      <section className="section">
        <div className="card" style={{ padding: '1.5rem', background: 'rgba(255,255,255,0.02)' }}>
          <h3 style={{ marginBottom: '1rem' }}>Analysis Metadata</h3>
          <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            <div>
              <p className="muted" style={{ margin: 0, fontSize: '0.85rem' }}>Processing Time</p>
              <p style={{ margin: '0.25rem 0 0', fontWeight: 'bold' }}>
                {results.analysis_metadata.processing_time_ms}ms
              </p>
            </div>
            <div>
              <p className="muted" style={{ margin: 0, fontSize: '0.85rem' }}>Timestamp</p>
              <p style={{ margin: '0.25rem 0 0', fontWeight: 'bold' }}>
                {new Date(results.analysis_metadata.timestamp).toLocaleString()}
              </p>
            </div>
            <div>
              <p className="muted" style={{ margin: 0, fontSize: '0.85rem' }}>Agents Executed</p>
              <p style={{ margin: '0.25rem 0 0', fontWeight: 'bold' }}>
                {results.analysis_metadata.agents_executed.length}
              </p>
            </div>
          </div>
          <details style={{ marginTop: '1rem' }}>
            <summary style={{ cursor: 'pointer', fontWeight: '500' }}>Agent Execution Chain</summary>
            <div style={{ marginTop: '0.75rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {results.analysis_metadata.agents_executed.map((agent, idx) => (
                <span key={idx} style={{
                  padding: '0.5rem 1rem',
                  background: 'rgba(96, 165, 250, 0.1)',
                  border: '1px solid rgba(96, 165, 250, 0.3)',
                  borderRadius: '999px',
                  fontSize: '0.85rem'
                }}>
                  {idx + 1}. {agent}
                </span>
              ))}
            </div>
          </details>
        </div>
      </section>
    </main>
  )
}
