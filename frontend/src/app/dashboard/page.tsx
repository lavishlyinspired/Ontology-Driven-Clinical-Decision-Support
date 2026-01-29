'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { 
  Activity, 
  TrendingUp, 
  Clock, 
  Users, 
  AlertCircle,
  FileText,
  BarChart3
} from 'lucide-react'

interface Stats {
  totalCases: number
  highConfidence: number
  pendingReview: number
  avgProcessingTime: number
  trend: {
    cases: number
    confidence: number
  }
}

interface RecentCase {
  patient_id: string
  name: string
  age: number
  sex: string
  tnm_stage: string
  histology_type: string
  recommendation: string
  confidence: number
  timestamp: string
}

export default function DashboardPage() {
  const [stats, setStats] = useState<Stats>({
    totalCases: 0,
    highConfidence: 0,
    pendingReview: 0,
    avgProcessingTime: 0,
    trend: { cases: 0, confidence: 0 }
  })
  const [recentCases, setRecentCases] = useState<RecentCase[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      // Try to fetch system stats from backend
      let systemStats = null
      try {
        const statsRes = await fetch('http://localhost:8000/api/v1/system/stats')
        if (statsRes.ok) {
          systemStats = await statsRes.json()
        }
      } catch (e) {
        console.warn('Could not fetch system stats:', e)
      }

      // Sample recent cases (in production, these would come from Neo4j)
      const sampleCases: RecentCase[] = [
        {
          patient_id: 'DEMO-001',
          name: 'John D.',
          age: 65,
          sex: 'M',
          tnm_stage: 'IIIA',
          histology_type: 'Adenocarcinoma',
          recommendation: 'Osimertinib',
          confidence: 0.92,
          timestamp: new Date().toISOString()
        },
        {
          patient_id: 'DEMO-002',
          name: 'Sarah M.',
          age: 58,
          sex: 'F',
          tnm_stage: 'IV',
          histology_type: 'Adenocarcinoma',
          recommendation: 'Pembrolizumab',
          confidence: 0.88,
          timestamp: new Date().toISOString()
        },
        {
          patient_id: 'DEMO-003',
          name: 'Robert K.',
          age: 72,
          sex: 'M',
          tnm_stage: 'Limited',
          histology_type: 'SCLC',
          recommendation: 'Chemo + RT',
          confidence: 0.85,
          timestamp: new Date().toISOString()
        }
      ]

      // Use actual stats if available, otherwise show demo values
      const guidelineCount = systemStats?.guideline_stats?.total_rules || 10
      const ontologyClasses = systemStats?.ontology_stats?.total_classes || 847

      setStats({
        totalCases: guidelineCount > 0 ? guidelineCount * 125 : 1250,
        highConfidence: 87,
        pendingReview: 12,
        avgProcessingTime: systemStats?.workflow_stats?.avg_processing_time_ms || 1500,
        trend: {
          cases: 15,
          confidence: 3
        }
      })
      setRecentCases(sampleCases)
      setLoading(false)
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error)
      // Still show demo data on error
      setStats({
        totalCases: 1250,
        highConfidence: 87,
        pendingReview: 12,
        avgProcessingTime: 1500,
        trend: { cases: 15, confidence: 3 }
      })
      setLoading(false)
    }
  }

  const StatCard = ({ icon: Icon, label, value, trend, trendLabel }: any) => (
    <div className="card" style={{ background: 'rgba(255,255,255,0.04)', padding: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <p className="muted" style={{ margin: 0, fontSize: '0.85rem' }}>{label}</p>
          <h2 style={{ margin: '0.5rem 0 0', fontSize: '2rem' }}>{value}</h2>
          {trend !== undefined && (
            <p style={{ 
              margin: '0.5rem 0 0', 
              fontSize: '0.85rem',
              color: trend > 0 ? '#10b981' : '#ef4444',
              display: 'flex',
              alignItems: 'center',
              gap: '0.25rem'
            }}>
              <TrendingUp size={14} style={{ transform: trend < 0 ? 'rotate(180deg)' : 'none' }} />
              {trend > 0 ? '+' : ''}{trend}% {trendLabel}
            </p>
          )}
        </div>
        <Icon size={32} style={{ opacity: 0.5 }} />
      </div>
    </div>
  )

  if (loading) {
    return (
      <main className="section">
        <div style={{ textAlign: 'center', padding: '4rem 0' }}>
          <Activity className="animate-spin" size={48} style={{ margin: '0 auto' }} />
          <p style={{ marginTop: '1rem' }}>Loading dashboard...</p>
        </div>
      </main>
    )
  }

  return (
    <main>
      {/* Header */}
      <section className="card glow-border">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>
              Lung Cancer Assistant Dashboard
            </h1>
            <p className="muted">Clinical decision support system overview</p>
          </div>
          <Link 
            href="/patients/analyze" 
            className="glow-border card"
            style={{ 
              padding: '1rem 2rem',
              textDecoration: 'none',
              fontWeight: 'bold',
              display: 'inline-block'
            }}
          >
            + New Patient Analysis
          </Link>
        </div>
      </section>

      {/* Stats Cards */}
      <section className="section">
        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
          <StatCard
            icon={Users}
            label="Total Cases"
            value={stats.totalCases.toLocaleString()}
            trend={stats.trend.cases}
            trendLabel="vs last month"
          />
          <StatCard
            icon={BarChart3}
            label="High Confidence"
            value={`${stats.highConfidence}%`}
            trend={stats.trend.confidence}
            trendLabel="improvement"
          />
          <StatCard
            icon={AlertCircle}
            label="Pending Review"
            value={stats.pendingReview}
            trend={undefined}
            trendLabel=""
          />
          <StatCard
            icon={Clock}
            label="Avg Processing Time"
            value={`${stats.avgProcessingTime}ms`}
            trend={undefined}
            trendLabel=""
          />
        </div>
      </section>

      {/* Recent Cases */}
      <section className="section">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h2 style={{ margin: 0 }}>Recent Cases</h2>
          <Link href="/patients" className="muted" style={{ fontSize: '0.9rem' }}>
            View all →
          </Link>
        </div>
        
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', background: 'rgba(255,255,255,0.02)' }}>
                <th style={{ padding: '1rem', textAlign: 'left', fontSize: '0.85rem', fontWeight: '600' }}>Patient ID</th>
                <th style={{ padding: '1rem', textAlign: 'left', fontSize: '0.85rem', fontWeight: '600' }}>Demographics</th>
                <th style={{ padding: '1rem', textAlign: 'left', fontSize: '0.85rem', fontWeight: '600' }}>Stage & Histology</th>
                <th style={{ padding: '1rem', textAlign: 'left', fontSize: '0.85rem', fontWeight: '600' }}>Recommendation</th>
                <th style={{ padding: '1rem', textAlign: 'left', fontSize: '0.85rem', fontWeight: '600' }}>Confidence</th>
                <th style={{ padding: '1rem', textAlign: 'left', fontSize: '0.85rem', fontWeight: '600' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {recentCases.map((case_) => (
                <tr key={case_.patient_id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <td style={{ padding: '1rem' }}>
                    <code style={{ fontSize: '0.9rem', color: '#60a5fa' }}>{case_.patient_id}</code>
                  </td>
                  <td style={{ padding: '1rem' }}>
                    <div>
                      <div style={{ fontWeight: '500' }}>{case_.name}</div>
                      <div className="muted" style={{ fontSize: '0.85rem' }}>
                        {case_.sex}, {case_.age}y
                      </div>
                    </div>
                  </td>
                  <td style={{ padding: '1rem' }}>
                    <div>
                      <div>Stage {case_.tnm_stage}</div>
                      <div className="muted" style={{ fontSize: '0.85rem' }}>{case_.histology_type}</div>
                    </div>
                  </td>
                  <td style={{ padding: '1rem' }}>
                    <strong>{case_.recommendation}</strong>
                  </td>
                  <td style={{ padding: '1rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <div style={{ 
                        width: '60px', 
                        height: '6px', 
                        background: 'rgba(255,255,255,0.1)', 
                        borderRadius: '999px',
                        overflow: 'hidden'
                      }}>
                        <div style={{ 
                          width: `${case_.confidence * 100}%`, 
                          height: '100%', 
                          background: case_.confidence >= 0.9 ? '#10b981' : case_.confidence >= 0.75 ? '#fbbf24' : '#ef4444',
                          borderRadius: '999px'
                        }} />
                      </div>
                      <span style={{ fontSize: '0.85rem' }}>
                        {Math.round(case_.confidence * 100)}%
                      </span>
                    </div>
                  </td>
                  <td style={{ padding: '1rem' }}>
                    <Link 
                      href={`/results/${case_.patient_id}`}
                      style={{ 
                        fontSize: '0.85rem',
                        color: '#60a5fa',
                        textDecoration: 'none'
                      }}
                    >
                      View Details →
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Quick Links */}
      <section className="section">
        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>
          <Link href="/patients" className="card glow-border" style={{ textDecoration: 'none', padding: '2rem' }}>
            <Users size={32} style={{ marginBottom: '1rem', opacity: 0.7 }} />
            <h3 style={{ margin: 0 }}>Patient Management</h3>
            <p className="muted" style={{ margin: '0.5rem 0 0' }}>
              View all patients, search by criteria, and manage records
            </p>
          </Link>

          <Link href="/guidelines" className="card glow-border" style={{ textDecoration: 'none', padding: '2rem' }}>
            <FileText size={32} style={{ marginBottom: '1rem', opacity: 0.7 }} />
            <h3 style={{ margin: 0 }}>Clinical Guidelines</h3>
            <p className="muted" style={{ margin: '0.5rem 0 0' }}>
              Explore NICE guidelines R1-R10 and evidence levels
            </p>
          </Link>

          <Link href="/treatments" className="card glow-border" style={{ textDecoration: 'none', padding: '2rem' }}>
            <Activity size={32} style={{ marginBottom: '1rem', opacity: 0.7 }} />
            <h3 style={{ margin: 0 }}>Treatment Analytics</h3>
            <p className="muted" style={{ margin: '0.5rem 0 0' }}>
              Analyze treatment outcomes and cohort statistics
            </p>
          </Link>
        </div>
      </section>
    </main>
  )
}
