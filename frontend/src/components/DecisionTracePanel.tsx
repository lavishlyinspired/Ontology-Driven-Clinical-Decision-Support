'use client'

import { useState, useEffect } from 'react'
import { ChevronLeft, Loader2, AlertCircle, CheckCircle, Clock, ArrowUp, ArrowDown } from 'lucide-react'
import type { GraphNode, TreatmentDecision, SimilarDecision, CausalChain } from '@/lib/api'
import { getSimilarDecisions, getCausalChain } from '@/lib/api'

// Decision type colors
const DECISION_TYPE_COLORS: Record<string, string> = {
  treatment_recommendation: 'bg-green-100 text-green-800',
  biomarker_test: 'bg-blue-100 text-blue-800',
  staging: 'bg-purple-100 text-purple-800',
  surgery_recommendation: 'bg-red-100 text-red-800',
  chemotherapy_recommendation: 'bg-orange-100 text-orange-800',
  immunotherapy_recommendation: 'bg-teal-100 text-teal-800',
  targeted_therapy: 'bg-cyan-100 text-cyan-800',
  radiation_recommendation: 'bg-yellow-100 text-yellow-800',
}

// Category colors
const CATEGORY_COLORS: Record<string, string> = {
  NSCLC: 'bg-blue-50 text-blue-700 border-blue-200',
  SCLC: 'bg-purple-50 text-purple-700 border-purple-200',
  immunotherapy: 'bg-teal-50 text-teal-700 border-teal-200',
  targeted_therapy: 'bg-cyan-50 text-cyan-700 border-cyan-200',
  chemotherapy: 'bg-orange-50 text-orange-700 border-orange-200',
  radiation: 'bg-yellow-50 text-yellow-700 border-yellow-200',
  surgery: 'bg-red-50 text-red-700 border-red-200',
}

interface DecisionTracePanelProps {
  decision: TreatmentDecision | null
  onDecisionSelect: (decision: TreatmentDecision | null) => void
  graphDecisions?: GraphNode[]
  className?: string
}

export function DecisionTracePanel({
  decision,
  onDecisionSelect,
  graphDecisions = [],
  className = '',
}: DecisionTracePanelProps) {
  const [similarDecisions, setSimilarDecisions] = useState<SimilarDecision[]>([])
  const [causalChain, setCausalChain] = useState<CausalChain | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!decision) {
      setSimilarDecisions([])
      setCausalChain(null)
      return
    }

    const fetchData = async () => {
      setLoading(true)
      try {
        const [similar, chain] = await Promise.all([
          getSimilarDecisions(decision.id, 5, 'hybrid'),
          getCausalChain(decision.id, 2),
        ])
        setSimilarDecisions(similar)
        setCausalChain(chain)
      } catch (error) {
        console.error('Failed to fetch decision data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [decision])

  // Convert GraphNode to TreatmentDecision
  const convertToDecision = (node: GraphNode): TreatmentDecision => ({
    id: node.id,
    decision_type: (node.properties.decision_type as string) || 'treatment_recommendation',
    category: (node.properties.category as string) || (node.properties.histology_type as string) || 'NSCLC',
    status: (node.properties.status as string) || 'recommended',
    reasoning: (node.properties.reasoning as string) || '',
    reasoning_summary: (node.properties.reasoning_summary as string) || (node.properties.treatment as string) || '',
    treatment: (node.properties.treatment as string) || 'Unknown treatment',
    confidence_score: (node.properties.confidence_score as number) || (node.properties.confidence as number),
    confidence: (node.properties.confidence as number) || (node.properties.confidence_score as number),
    risk_factors: (node.properties.risk_factors as string[]) || [],
    guidelines_applied: node.properties.guideline_reference ? [node.properties.guideline_reference as string] : [],
    timestamp: (node.properties.decision_timestamp as string) || (node.properties.created_at as string),
    decision_timestamp: (node.properties.decision_timestamp as string) || (node.properties.created_at as string),
  })

  // Show decisions list when no decision is selected
  if (!decision) {
    return (
      <div className={`p-4 ${className}`}>
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-gray-800">Decisions in Graph</h3>
          <p className="text-sm text-gray-500">
            {graphDecisions.length > 0
              ? 'Click a decision node in the graph or select from below to view details.'
              : 'Use the chat to search for patients or decisions. Decision nodes will appear here.'}
          </p>

          {graphDecisions.length > 0 && (
            <span className="inline-block px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded">
              {graphDecisions.length} decision{graphDecisions.length !== 1 ? 's' : ''} in graph
            </span>
          )}

          {graphDecisions.length > 0 ? (
            <div className="space-y-2">
              {graphDecisions.map((node) => {
                const dec = convertToDecision(node)
                return (
                  <DecisionCard
                    key={node.id}
                    decision={dec}
                    onClick={() => onDecisionSelect(dec)}
                  />
                )
              })}
            </div>
          ) : (
            <p className="text-center text-gray-400 py-4">No decisions in graph yet.</p>
          )}
        </div>
      </div>
    )
  }

  const typeColor = DECISION_TYPE_COLORS[decision.decision_type] || 'bg-gray-100 text-gray-800'
  const categoryColor = CATEGORY_COLORS[decision.category] || 'bg-gray-50 text-gray-700 border-gray-200'

  return (
    <div className={`p-4 overflow-auto ${className}`}>
      <div className="space-y-4">
        {/* Back button */}
        <button
          onClick={() => onDecisionSelect(null)}
          className="flex items-center gap-1 text-sm text-gray-600 hover:text-gray-900 transition-colors"
        >
          <ChevronLeft className="w-4 h-4" />
          Back to list
        </button>

        {/* Decision Header */}
        <div>
          <div className="flex flex-wrap gap-2 mb-2">
            <span className={`px-2 py-1 text-xs font-medium rounded ${typeColor}`}>
              {decision.decision_type.replace(/_/g, ' ')}
            </span>
            <span className={`px-2 py-1 text-xs font-medium rounded border ${categoryColor}`}>
              {decision.category}
            </span>
            <span
              className={`px-2 py-1 text-xs font-medium rounded ${
                decision.status === 'recommended'
                  ? 'bg-green-100 text-green-800'
                  : decision.status === 'declined'
                    ? 'bg-red-100 text-red-800'
                    : 'bg-yellow-100 text-yellow-800'
              }`}
            >
              {decision.status}
            </span>
          </div>
          <p className="text-xs text-gray-500">
            {decision.timestamp
              ? new Date(decision.timestamp).toLocaleString()
              : decision.decision_timestamp
                ? new Date(decision.decision_timestamp).toLocaleString()
                : 'Unknown date'}
          </p>
          <p className="text-xs text-gray-400 mt-1">ID: {decision.id.slice(0, 8)}...</p>
        </div>

        <hr className="border-gray-200" />

        {/* Treatment */}
        {decision.treatment && (
          <div>
            <h4 className="text-sm font-semibold text-gray-800 mb-1">Treatment</h4>
            <p className="text-sm text-gray-700 bg-blue-50 p-2 rounded">{decision.treatment}</p>
          </div>
        )}

        {/* Reasoning */}
        <div>
          <h4 className="text-sm font-semibold text-gray-800 mb-1">Reasoning</h4>
          <div className="bg-gray-50 p-3 rounded text-sm text-gray-700 whitespace-pre-wrap">
            {decision.reasoning || decision.reasoning_summary || 'No reasoning provided.'}
          </div>
        </div>

        {/* Confidence */}
        <div className="flex gap-4">
          <div>
            <p className="text-xs text-gray-500 mb-1">Confidence Score</p>
            <p className="text-lg font-semibold text-gray-900">
              {decision.confidence_score !== undefined
                ? `${(decision.confidence_score * 100).toFixed(0)}%`
                : decision.confidence !== undefined
                  ? `${(decision.confidence * 100).toFixed(0)}%`
                  : 'N/A'}
            </p>
          </div>
        </div>

        {/* Risk Factors */}
        {decision.risk_factors && decision.risk_factors.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold text-gray-800 mb-2">Risk Factors</h4>
            <div className="flex flex-wrap gap-1">
              {decision.risk_factors.map((factor, idx) => (
                <span
                  key={idx}
                  className="px-2 py-1 text-xs bg-orange-100 text-orange-800 rounded"
                >
                  {String(factor).replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          </div>
        )}

        <hr className="border-gray-200" />

        {/* Causal Chain */}
        <div>
          <h4 className="text-sm font-semibold text-gray-800 mb-2">Causal Chain</h4>
          {loading ? (
            <div className="flex justify-center py-4">
              <Loader2 className="w-5 h-5 animate-spin text-gray-400" />
            </div>
          ) : causalChain ? (
            <div className="space-y-3">
              {/* Causes */}
              {causalChain.causes && causalChain.causes.length > 0 && (
                <div>
                  <p className="text-xs text-gray-500 mb-1 flex items-center gap-1">
                    <ArrowUp className="w-3 h-3" />
                    Caused by ({causalChain.causes.length})
                  </p>
                  {causalChain.causes.map((cause) => (
                    <CausalDecisionCard
                      key={cause.id}
                      decision={cause}
                      onClick={() => onDecisionSelect(cause)}
                      direction="cause"
                    />
                  ))}
                </div>
              )}

              {/* Effects */}
              {causalChain.effects && causalChain.effects.length > 0 && (
                <div>
                  <p className="text-xs text-gray-500 mb-1 flex items-center gap-1">
                    <ArrowDown className="w-3 h-3" />
                    Led to ({causalChain.effects.length})
                  </p>
                  {causalChain.effects.map((effect) => (
                    <CausalDecisionCard
                      key={effect.id}
                      decision={effect}
                      onClick={() => onDecisionSelect(effect)}
                      direction="effect"
                    />
                  ))}
                </div>
              )}

              {(!causalChain.causes || causalChain.causes.length === 0) &&
                (!causalChain.effects || causalChain.effects.length === 0) && (
                  <p className="text-sm text-gray-500">No causal relationships found.</p>
                )}
            </div>
          ) : (
            <p className="text-sm text-gray-500">No causal chain data.</p>
          )}
        </div>

        <hr className="border-gray-200" />

        {/* Similar Decisions */}
        <div>
          <h4 className="text-sm font-semibold text-gray-800 mb-2">Similar Decisions</h4>
          {loading ? (
            <div className="flex justify-center py-4">
              <Loader2 className="w-5 h-5 animate-spin text-gray-400" />
            </div>
          ) : similarDecisions.length > 0 ? (
            <div className="space-y-2">
              {similarDecisions.map((similar) => (
                <SimilarDecisionCard
                  key={similar.decision.id}
                  similarDecision={similar}
                  onClick={() => onDecisionSelect(similar.decision)}
                />
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-500">No similar decisions found.</p>
          )}
        </div>
      </div>
    </div>
  )
}

// Decision card for list view
function DecisionCard({
  decision,
  onClick,
}: {
  decision: TreatmentDecision
  onClick: () => void
}) {
  const typeColor = DECISION_TYPE_COLORS[decision.decision_type] || 'bg-gray-100 text-gray-800'
  const categoryColor = CATEGORY_COLORS[decision.category] || 'bg-gray-50 text-gray-700 border-gray-200'

  // Determine confidence display
  const confidence = decision.confidence_score || decision.confidence || 0
  const confidencePercent = confidence <= 1 ? Math.round(confidence * 100) : Math.round(confidence)

  // Determine border color based on status/rank
  const borderColor = decision.status === 'recommended' ? 'border-green-500' :
                      decision.status === 'alternative' ? 'border-blue-400' :
                      'border-purple-500'

  return (
    <div
      onClick={onClick}
      className={`bg-gray-50 p-3 rounded-lg cursor-pointer hover:bg-gray-100 transition-colors border-l-4 ${borderColor}`}
    >
      {/* Treatment name prominently displayed */}
      {decision.treatment && (
        <p className="text-sm font-semibold text-gray-800 mb-2 line-clamp-2">
          {decision.treatment}
        </p>
      )}

      <div className="flex flex-wrap justify-between items-start gap-1 mb-2">
        <div className="flex flex-wrap gap-1">
          <span className={`px-1.5 py-0.5 text-xs font-medium rounded ${typeColor}`}>
            {decision.decision_type.replace(/_/g, ' ')}
          </span>
          <span className={`px-1.5 py-0.5 text-xs font-medium rounded border ${categoryColor}`}>
            {decision.category}
          </span>
          {decision.status && (
            <span className={`px-1.5 py-0.5 text-xs font-medium rounded ${
              decision.status === 'recommended' ? 'bg-green-100 text-green-700' :
              decision.status === 'alternative' ? 'bg-blue-100 text-blue-700' :
              'bg-gray-100 text-gray-700'
            }`}>
              {decision.status}
            </span>
          )}
        </div>
        {confidencePercent > 0 && (
          <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${
            confidencePercent >= 80 ? 'bg-green-100 text-green-700' :
            confidencePercent >= 60 ? 'bg-yellow-100 text-yellow-700' :
            'bg-gray-100 text-gray-600'
          }`}>
            {confidencePercent}%
          </span>
        )}
      </div>

      {/* Reasoning preview */}
      {(decision.reasoning || decision.reasoning_summary) && (
        <p className="text-xs text-gray-500 line-clamp-2 mb-2">
          {(decision.reasoning_summary || decision.reasoning || '').slice(0, 150)}
          {(decision.reasoning_summary || decision.reasoning || '').length > 150 ? '...' : ''}
        </p>
      )}

      <div className="flex justify-between items-center">
        <div className="flex gap-2">
          {decision.guidelines_applied && decision.guidelines_applied.length > 0 && (
            <span className="text-xs bg-teal-50 text-teal-700 px-1.5 py-0.5 rounded">
              {decision.guidelines_applied[0]}
            </span>
          )}
        </div>
        {decision.risk_factors && decision.risk_factors.length > 0 && (
          <span className="text-xs bg-orange-100 text-orange-700 px-1.5 py-0.5 rounded">
            {decision.risk_factors.length} risk factors
          </span>
        )}
      </div>
    </div>
  )
}

// Causal decision card
function CausalDecisionCard({
  decision,
  onClick,
  direction,
}: {
  decision: TreatmentDecision
  onClick: () => void
  direction: 'cause' | 'effect'
}) {
  const typeColor = DECISION_TYPE_COLORS[decision.decision_type] || 'bg-gray-100 text-gray-800'

  return (
    <div
      onClick={onClick}
      className="bg-gray-50 p-2 rounded cursor-pointer hover:bg-gray-100 transition-colors mb-1"
    >
      <div className="flex items-center gap-2">
        <span className={direction === 'cause' ? 'text-blue-500' : 'text-green-500'}>
          {direction === 'cause' ? '↑' : '↓'}
        </span>
        <span className={`px-1.5 py-0.5 text-xs font-medium rounded ${typeColor}`}>
          {decision.decision_type.replace(/_/g, ' ')}
        </span>
        <span className="text-xs text-gray-500 truncate flex-1">{decision.category}</span>
      </div>
    </div>
  )
}

// Similar decision card
function SimilarDecisionCard({
  similarDecision,
  onClick,
}: {
  similarDecision: SimilarDecision
  onClick: () => void
}) {
  const { decision, similarity_score, similarity_type } = similarDecision
  const typeColor = DECISION_TYPE_COLORS[decision.decision_type] || 'bg-gray-100 text-gray-800'

  return (
    <div
      onClick={onClick}
      className="bg-gray-50 p-3 rounded-lg cursor-pointer hover:bg-gray-100 transition-colors"
    >
      <div className="flex justify-between items-start mb-1">
        <span className={`px-1.5 py-0.5 text-xs font-medium rounded ${typeColor}`}>
          {decision.decision_type.replace(/_/g, ' ')}
        </span>
        <div className="flex gap-1 items-center">
          <span className="text-xs bg-gray-200 px-1.5 py-0.5 rounded">{similarity_type}</span>
          <span className="text-xs font-bold text-blue-600">
            {(similarity_score * 100).toFixed(0)}%
          </span>
        </div>
      </div>
      <p className="text-sm text-gray-600 line-clamp-2">
        {decision.reasoning?.slice(0, 150) || 'No reasoning'}...
      </p>
      <p className="text-xs text-gray-400 mt-1">
        {decision.timestamp
          ? new Date(decision.timestamp).toLocaleDateString()
          : decision.decision_timestamp
            ? new Date(decision.decision_timestamp).toLocaleDateString()
            : ''}
      </p>
    </div>
  )
}
