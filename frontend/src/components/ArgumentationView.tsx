'use client'

import React, { useState, useMemo } from 'react'

interface Argument {
  id: string
  type: 'support' | 'attack' | 'neutral'
  source: string
  claim: string
  evidence?: string
  strength: 'strong' | 'moderate' | 'weak'
  references?: string[]
}

interface ArgumentationChain {
  conclusion: string
  treatment: string
  confidence: number
  arguments: Argument[]
  counterArguments?: Argument[]
  patientFactors?: string[]
}

interface ArgumentationViewProps {
  chain: ArgumentationChain | null
  onArgumentClick?: (arg: Argument) => void
  onReferenceClick?: (reference: string) => void
  className?: string
  compact?: boolean
}

export function ArgumentationView({
  chain,
  onArgumentClick,
  onReferenceClick,
  className = '',
  compact = false
}: ArgumentationViewProps) {
  const [expandedArgs, setExpandedArgs] = useState<Set<string>>(new Set())
  const [showCounterArgs, setShowCounterArgs] = useState(true)

  // Calculate overall argument strength
  const overallStrength = useMemo(() => {
    if (!chain?.arguments) return 0

    let score = 0
    chain.arguments.forEach(arg => {
      const strengthScore = arg.strength === 'strong' ? 3 : arg.strength === 'moderate' ? 2 : 1
      score += arg.type === 'support' ? strengthScore : arg.type === 'attack' ? -strengthScore : 0
    })

    if (chain.counterArguments) {
      chain.counterArguments.forEach(arg => {
        const strengthScore = arg.strength === 'strong' ? 3 : arg.strength === 'moderate' ? 2 : 1
        score -= strengthScore
      })
    }

    return Math.max(0, Math.min(100, 50 + score * 10))
  }, [chain])

  const toggleArgument = (argId: string) => {
    setExpandedArgs(prev => {
      const next = new Set(prev)
      if (next.has(argId)) {
        next.delete(argId)
      } else {
        next.add(argId)
      }
      return next
    })
  }

  const getTypeIcon = (type: Argument['type']) => {
    switch (type) {
      case 'support':
        return (
          <svg className="w-4 h-4 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        )
      case 'attack':
        return (
          <svg className="w-4 h-4 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        )
      default:
        return (
          <svg className="w-4 h-4 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )
    }
  }

  const getStrengthBadge = (strength: Argument['strength']) => {
    const colors = {
      strong: 'bg-green-500/20 text-green-400 border-green-500/30',
      moderate: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
      weak: 'bg-zinc-500/20 text-zinc-400 border-zinc-500/30'
    }
    return (
      <span className={`px-1.5 py-0.5 text-[10px] rounded border ${colors[strength]}`}>
        {strength}
      </span>
    )
  }

  const renderArgument = (arg: Argument, isCounter = false) => {
    const isExpanded = expandedArgs.has(arg.id)

    return (
      <div
        key={arg.id}
        className={`border rounded-lg transition-colors ${
          isCounter
            ? 'border-red-500/30 bg-red-500/5'
            : arg.type === 'support'
            ? 'border-green-500/30 bg-green-500/5'
            : arg.type === 'attack'
            ? 'border-red-500/30 bg-red-500/5'
            : 'border-zinc-700 bg-zinc-800/50'
        }`}
      >
        <div
          className="p-3 cursor-pointer hover:bg-white/5 transition-colors"
          onClick={() => {
            toggleArgument(arg.id)
            onArgumentClick?.(arg)
          }}
        >
          <div className="flex items-start gap-3">
            <div className="mt-0.5">{getTypeIcon(arg.type)}</div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-xs text-zinc-500">{arg.source}</span>
                {getStrengthBadge(arg.strength)}
              </div>
              <p className="text-sm text-zinc-200 mt-1">{arg.claim}</p>
            </div>
            <svg
              className={`w-4 h-4 text-zinc-500 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>

        {isExpanded && (
          <div className="px-3 pb-3 border-t border-zinc-700/50">
            {arg.evidence && (
              <div className="mt-3">
                <span className="text-xs text-zinc-500">Evidence</span>
                <p className="text-sm text-zinc-300 mt-1">{arg.evidence}</p>
              </div>
            )}
            {arg.references && arg.references.length > 0 && (
              <div className="mt-3">
                <span className="text-xs text-zinc-500">References</span>
                <div className="flex flex-wrap gap-2 mt-1">
                  {arg.references.map((ref, i) => (
                    <button
                      key={i}
                      onClick={(e) => {
                        e.stopPropagation()
                        onReferenceClick?.(ref)
                      }}
                      className="text-xs text-violet-400 hover:text-violet-300 hover:underline"
                    >
                      {ref}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  if (!chain) {
    return (
      <div className={`p-4 text-center text-zinc-500 ${className}`}>
        <svg className="w-8 h-8 mx-auto mb-2 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <p className="text-sm">No argumentation data available</p>
      </div>
    )
  }

  return (
    <div className={`bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-zinc-800 bg-gradient-to-r from-violet-600/20 to-purple-600/20">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-violet-600 to-purple-600 flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <div className="flex-1">
            <h3 className="text-sm font-semibold text-zinc-100">Treatment Decision Analysis</h3>
            <p className="text-xs text-zinc-400">{chain.treatment}</p>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold text-zinc-100">{chain.confidence}%</div>
            <div className="text-xs text-zinc-500">confidence</div>
          </div>
        </div>

        {/* Confidence bar */}
        <div className="mt-3">
          <div className="h-2 bg-zinc-700 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                chain.confidence >= 80
                  ? 'bg-green-500'
                  : chain.confidence >= 60
                  ? 'bg-yellow-500'
                  : 'bg-red-500'
              }`}
              style={{ width: `${chain.confidence}%` }}
            />
          </div>
        </div>
      </div>

      {/* Conclusion */}
      <div className="p-4 border-b border-zinc-800">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-full bg-violet-500/20 flex items-center justify-center flex-shrink-0">
            <svg className="w-4 h-4 text-violet-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <div>
            <span className="text-xs text-zinc-500">Conclusion</span>
            <p className="text-sm text-zinc-200 mt-1">{chain.conclusion}</p>
          </div>
        </div>
      </div>

      {/* Patient Factors */}
      {chain.patientFactors && chain.patientFactors.length > 0 && (
        <div className="p-4 border-b border-zinc-800">
          <span className="text-xs text-zinc-500">Patient Factors Considered</span>
          <div className="flex flex-wrap gap-2 mt-2">
            {chain.patientFactors.map((factor, i) => (
              <span
                key={i}
                className="px-2 py-1 bg-zinc-800 text-zinc-300 text-xs rounded-md"
              >
                {factor}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Supporting Arguments */}
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs text-zinc-500 uppercase tracking-wider">
            Supporting Arguments ({chain.arguments.filter(a => a.type === 'support').length})
          </span>
        </div>
        <div className="space-y-2">
          {chain.arguments
            .filter(a => a.type === 'support')
            .map(arg => renderArgument(arg))}
        </div>
      </div>

      {/* Counter Arguments */}
      {chain.counterArguments && chain.counterArguments.length > 0 && (
        <div className="p-4 border-t border-zinc-800">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs text-zinc-500 uppercase tracking-wider">
              Considerations/Risks ({chain.counterArguments.length})
            </span>
            <button
              onClick={() => setShowCounterArgs(!showCounterArgs)}
              className="text-xs text-zinc-400 hover:text-zinc-200"
            >
              {showCounterArgs ? 'Hide' : 'Show'}
            </button>
          </div>
          {showCounterArgs && (
            <div className="space-y-2">
              {chain.counterArguments.map(arg => renderArgument(arg, true))}
            </div>
          )}
        </div>
      )}

      {/* Argument Strength Summary */}
      {!compact && (
        <div className="p-4 border-t border-zinc-800 bg-zinc-800/30">
          <div className="flex items-center justify-between">
            <span className="text-xs text-zinc-500">Overall Argument Strength</span>
            <div className="flex items-center gap-2">
              <div className="w-20 h-2 bg-zinc-700 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${
                    overallStrength >= 70
                      ? 'bg-green-500'
                      : overallStrength >= 40
                      ? 'bg-yellow-500'
                      : 'bg-red-500'
                  }`}
                  style={{ width: `${overallStrength}%` }}
                />
              </div>
              <span className="text-xs text-zinc-400">{overallStrength}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Demo data for testing
export const DEMO_ARGUMENTATION_CHAIN: ArgumentationChain = {
  conclusion: 'Osimertinib is the recommended first-line treatment for this patient with EGFR-mutated metastatic NSCLC.',
  treatment: 'Osimertinib 80mg daily',
  confidence: 92,
  patientFactors: ['Stage IV NSCLC', 'EGFR L858R+', 'No brain metastases', 'ECOG PS 1', 'PD-L1: 45%'],
  arguments: [
    {
      id: 'arg1',
      type: 'support',
      source: 'NCCN Guidelines v2024',
      claim: 'Osimertinib is Category 1 recommended as preferred first-line therapy for EGFR exon 19 deletion or L858R mutations.',
      evidence: 'Based on FLAURA trial showing superior OS vs first-generation TKIs.',
      strength: 'strong',
      references: ['FLAURA Trial', 'NCCN NSCLC v1.2024']
    },
    {
      id: 'arg2',
      type: 'support',
      source: 'FLAURA Trial',
      claim: 'Median overall survival of 38.6 months vs 31.8 months with comparator TKIs (HR 0.80).',
      evidence: 'Phase III randomized trial with 556 patients.',
      strength: 'strong',
      references: ['N Engl J Med 2020;382:41-50']
    },
    {
      id: 'arg3',
      type: 'support',
      source: 'CNS Efficacy Data',
      claim: 'Superior CNS penetration provides prophylaxis against brain metastases.',
      evidence: 'FLAURA CNS subanalysis showed 52% vs 33% CNS ORR.',
      strength: 'moderate',
      references: ['J Clin Oncol 2020']
    }
  ],
  counterArguments: [
    {
      id: 'counter1',
      type: 'attack',
      source: 'Resistance Consideration',
      claim: 'Using osimertinib first-line may limit options at resistance as T790M-mediated escape is eliminated.',
      strength: 'moderate',
      references: ['Resistance Mechanisms Review']
    },
    {
      id: 'counter2',
      type: 'attack',
      source: 'Cardiotoxicity Risk',
      claim: 'Monitor for QTc prolongation, especially with concurrent medications.',
      strength: 'weak',
      references: ['Osimertinib Prescribing Information']
    }
  ]
}

export default ArgumentationView
