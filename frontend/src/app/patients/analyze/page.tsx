'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { ArrowLeft, Play, Save, Trash2, Download } from 'lucide-react'
import Link from 'next/link'

interface PatientFormData {
  patient_id?: string
  name: string
  sex: 'M' | 'F' | 'U'
  age: number
  tnm_stage: string
  histology_type: string
  performance_status: number
  laterality: string
  fev1_percent?: number
  comorbidities: string[]
  biomarkers?: {
    egfr_mutation?: string
    alk_status?: string
    pd_l1_score?: number
    kras_mutation?: string
  }
}

export default function AnalyzePage() {
  const router = useRouter()
  const [formData, setFormData] = useState<PatientFormData>({
    name: '',
    sex: 'M',
    age: 65,
    tnm_stage: 'IIIA',
    histology_type: 'Adenocarcinoma',
    performance_status: 1,
    laterality: 'Right',
    comorbidities: [],
    biomarkers: {}
  })
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [showBiomarkers, setShowBiomarkers] = useState(false)

  const tnmStages = ['IA', 'IB', 'IIA', 'IIB', 'IIIA', 'IIIB', 'IIIC', 'IV']
  const histologyTypes = [
    'Adenocarcinoma',
    'Squamous Cell Carcinoma',
    'Large Cell Carcinoma',
    'Small Cell Lung Cancer',
    'Adenosquamous Carcinoma',
    'Carcinoid Tumor'
  ]
  
  const egfrOptions = ['Negative', 'Exon 19 deletion', 'L858R', 'Exon 20 insertion', 'T790M']
  const alkOptions = ['Negative', 'Positive']
  const krasOptions = ['Wild-type', 'G12C', 'G12V', 'G12D', 'G12A']

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {}

    if (!formData.name.trim()) {
      newErrors.name = 'Patient name is required'
    }
    if (formData.age < 18 || formData.age > 120) {
      newErrors.age = 'Age must be between 18 and 120'
    }
    if (formData.fev1_percent && (formData.fev1_percent < 0 || formData.fev1_percent > 150)) {
      newErrors.fev1_percent = 'FEV1 must be between 0 and 150%'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleAnalyze = async () => {
    if (!validateForm()) {
      return
    }

    setIsAnalyzing(true)
    try {
      const response = await fetch('/api/v2/patients/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })

      const result = await response.json()
      
      if (response.ok) {
        // Navigate to results page
        router.push(`/results/${result.patient_id || 'latest'}`)
      } else {
        setErrors({ submit: result.detail || 'Analysis failed' })
      }
    } catch (error) {
      setErrors({ submit: 'Failed to connect to API' })
    } finally {
      setIsAnalyzing(false)
    }
  }

  const loadSampleData = () => {
    setFormData({
      name: 'Jenny Sesen (Sample)',
      sex: 'F',
      age: 72,
      tnm_stage: 'IIA',
      histology_type: 'Adenocarcinoma',
      performance_status: 1,
      laterality: 'Right',
      fev1_percent: 75,
      comorbidities: ['Hypertension'],
      biomarkers: {
        egfr_mutation: 'Exon 19 deletion',
        alk_status: 'Negative',
        pd_l1_score: 55,
        kras_mutation: 'Wild-type'
      }
    })
    setShowBiomarkers(true)
  }

  const addComorbidity = (comorbidity: string) => {
    if (comorbidity && !formData.comorbidities.includes(comorbidity)) {
      setFormData({
        ...formData,
        comorbidities: [...formData.comorbidities, comorbidity]
      })
    }
  }

  const removeComorbidity = (index: number) => {
    setFormData({
      ...formData,
      comorbidities: formData.comorbidities.filter((_, i) => i !== index)
    })
  }

  return (
    <main>
      {/* Header */}
      <section className="card glow-border">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <Link href="/dashboard" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', color: '#60a5fa', textDecoration: 'none' }}>
              <ArrowLeft size={20} />
              Back to Dashboard
            </Link>
            <h1 style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>
              Patient Analysis
            </h1>
            <p className="muted">Enter patient clinical data for treatment recommendations</p>
          </div>
          <div style={{ display: 'flex', gap: '1rem' }}>
            <button
              onClick={loadSampleData}
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
              <Download size={18} />
              Load Sample Data
            </button>
          </div>
        </div>
      </section>

      {/* Form */}
      <section className="section">
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          
          {/* Demographics Section */}
          <div className="card" style={{ marginBottom: '2rem' }}>
            <h2 style={{ marginBottom: '1.5rem' }}>Demographics</h2>
            <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                  Patient Name *
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="Enter patient name"
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    background: 'rgba(255,255,255,0.05)',
                    border: errors.name ? '1px solid #ef4444' : '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '0.5rem',
                    color: '#fff'
                  }}
                />
                {errors.name && <p style={{ color: '#ef4444', fontSize: '0.85rem', marginTop: '0.25rem' }}>{errors.name}</p>}
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                  Sex *
                </label>
                <select
                  value={formData.sex}
                  onChange={(e) => setFormData({ ...formData, sex: e.target.value as 'M' | 'F' | 'U' })}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '0.5rem',
                    color: '#fff'
                  }}
                >
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                  <option value="U">Unknown</option>
                </select>
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                  Age at Diagnosis *
                </label>
                <input
                  type="number"
                  value={formData.age}
                  onChange={(e) => setFormData({ ...formData, age: parseInt(e.target.value) })}
                  min={18}
                  max={120}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    background: 'rgba(255,255,255,0.05)',
                    border: errors.age ? '1px solid #ef4444' : '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '0.5rem',
                    color: '#fff'
                  }}
                />
                {errors.age && <p style={{ color: '#ef4444', fontSize: '0.85rem', marginTop: '0.25rem' }}>{errors.age}</p>}
              </div>
            </div>
          </div>

          {/* Clinical Data Section */}
          <div className="card" style={{ marginBottom: '2rem' }}>
            <h2 style={{ marginBottom: '1.5rem' }}>Clinical Data</h2>
            <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                  TNM Stage *
                </label>
                <select
                  value={formData.tnm_stage}
                  onChange={(e) => setFormData({ ...formData, tnm_stage: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '0.5rem',
                    color: '#fff'
                  }}
                >
                  {tnmStages.map(stage => (
                    <option key={stage} value={stage}>{stage}</option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                  Histology Type *
                </label>
                <select
                  value={formData.histology_type}
                  onChange={(e) => setFormData({ ...formData, histology_type: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '0.5rem',
                    color: '#fff'
                  }}
                >
                  {histologyTypes.map(type => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                  Performance Status (0-4) *
                </label>
                <input
                  type="number"
                  value={formData.performance_status}
                  onChange={(e) => setFormData({ ...formData, performance_status: parseInt(e.target.value) })}
                  min={0}
                  max={4}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '0.5rem',
                    color: '#fff'
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                  Laterality
                </label>
                <select
                  value={formData.laterality}
                  onChange={(e) => setFormData({ ...formData, laterality: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '0.5rem',
                    color: '#fff'
                  }}
                >
                  <option value="Right">Right</option>
                  <option value="Left">Left</option>
                  <option value="Bilateral">Bilateral</option>
                  <option value="Unknown">Unknown</option>
                </select>
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                  FEV1 (%)
                </label>
                <input
                  type="number"
                  value={formData.fev1_percent || ''}
                  onChange={(e) => setFormData({ ...formData, fev1_percent: e.target.value ? parseFloat(e.target.value) : undefined })}
                  placeholder="Optional"
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '0.5rem',
                    color: '#fff'
                  }}
                />
              </div>
            </div>
          </div>

          {/* Biomarkers Section */}
          <div className="card" style={{ marginBottom: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h2 style={{ margin: 0 }}>Biomarkers</h2>
              <button
                onClick={() => setShowBiomarkers(!showBiomarkers)}
                className="card"
                style={{
                  padding: '0.5rem 1rem',
                  border: '1px solid rgba(255,255,255,0.2)',
                  background: 'transparent',
                  cursor: 'pointer',
                  fontSize: '0.9rem'
                }}
              >
                {showBiomarkers ? 'Hide' : 'Show'} Biomarkers
              </button>
            </div>

            {showBiomarkers && (
              <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem' }}>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                    EGFR Mutation
                  </label>
                  <select
                    value={formData.biomarkers?.egfr_mutation || ''}
                    onChange={(e) => setFormData({ 
                      ...formData, 
                      biomarkers: { ...formData.biomarkers, egfr_mutation: e.target.value }
                    })}
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      background: 'rgba(255,255,255,0.05)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '0.5rem',
                      color: '#fff'
                    }}
                  >
                    <option value="">Not tested</option>
                    {egfrOptions.map(opt => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                    ALK Status
                  </label>
                  <select
                    value={formData.biomarkers?.alk_status || ''}
                    onChange={(e) => setFormData({ 
                      ...formData, 
                      biomarkers: { ...formData.biomarkers, alk_status: e.target.value }
                    })}
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      background: 'rgba(255,255,255,0.05)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '0.5rem',
                      color: '#fff'
                    }}
                  >
                    <option value="">Not tested</option>
                    {alkOptions.map(opt => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                    PD-L1 Score (%)
                  </label>
                  <input
                    type="number"
                    value={formData.biomarkers?.pd_l1_score || ''}
                    onChange={(e) => setFormData({ 
                      ...formData, 
                      biomarkers: { ...formData.biomarkers, pd_l1_score: e.target.value ? parseFloat(e.target.value) : undefined }
                    })}
                    placeholder="0-100"
                    min={0}
                    max={100}
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      background: 'rgba(255,255,255,0.05)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '0.5rem',
                      color: '#fff'
                    }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                    KRAS Mutation
                  </label>
                  <select
                    value={formData.biomarkers?.kras_mutation || ''}
                    onChange={(e) => setFormData({ 
                      ...formData, 
                      biomarkers: { ...formData.biomarkers, kras_mutation: e.target.value }
                    })}
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      background: 'rgba(255,255,255,0.05)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '0.5rem',
                      color: '#fff'
                    }}
                  >
                    <option value="">Not tested</option>
                    {krasOptions.map(opt => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end' }}>
            <button
              onClick={() => setFormData({
                name: '',
                sex: 'M',
                age: 65,
                tnm_stage: 'IIIA',
                histology_type: 'Adenocarcinoma',
                performance_status: 1,
                laterality: 'Right',
                comorbidities: [],
                biomarkers: {}
              })}
              className="card"
              style={{
                padding: '1rem 2rem',
                border: '1px solid rgba(255,255,255,0.2)',
                background: 'transparent',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}
            >
              <Trash2 size={18} />
              Clear Form
            </button>

            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className="glow-border card"
              style={{
                padding: '1rem 2rem',
                background: '#60a5fa',
                color: '#000',
                cursor: isAnalyzing ? 'not-allowed' : 'pointer',
                fontWeight: 'bold',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                opacity: isAnalyzing ? 0.6 : 1
              }}
            >
              <Play size={18} />
              {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
            </button>
          </div>

          {errors.submit && (
            <div style={{ 
              marginTop: '1rem', 
              padding: '1rem', 
              background: 'rgba(239, 68, 68, 0.1)', 
              border: '1px solid #ef4444',
              borderRadius: '0.5rem',
              color: '#ef4444'
            }}>
              {errors.submit}
            </div>
          )}
        </div>
      </section>
    </main>
  )
}
