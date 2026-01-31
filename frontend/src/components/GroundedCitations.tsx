'use client'

import React, { useMemo, useState } from 'react'

// Citation types
type CitationType = 'guideline' | 'trial' | 'publication' | 'ontology' | 'patient' | 'drug'

interface Citation {
  type: CitationType
  id: string
  label: string
  fullText: string
  url?: string
  metadata?: Record<string, unknown>
}

interface CitationSource {
  id: string
  name: string
  type: CitationType
  description?: string
  url?: string
  version?: string
  lastUpdated?: string
}

interface GroundedCitationsProps {
  text: string
  onCitationClick?: (citation: Citation) => void
  onCitationHover?: (citation: Citation | null) => void
  className?: string
  showTooltips?: boolean
  renderAs?: 'inline' | 'footnotes'
}

// Known citation sources
const CITATION_SOURCES: Record<string, CitationSource> = {
  'NCCN': {
    id: 'nccn',
    name: 'NCCN Guidelines',
    type: 'guideline',
    description: 'National Comprehensive Cancer Network Clinical Practice Guidelines',
    url: 'https://www.nccn.org/guidelines',
    version: '2024.1'
  },
  'ESMO': {
    id: 'esmo',
    name: 'ESMO Guidelines',
    type: 'guideline',
    description: 'European Society for Medical Oncology Clinical Practice Guidelines',
    url: 'https://www.esmo.org/guidelines'
  },
  'ASCO': {
    id: 'asco',
    name: 'ASCO Guidelines',
    type: 'guideline',
    description: 'American Society of Clinical Oncology Clinical Practice Guidelines',
    url: 'https://www.asco.org/guidelines'
  },
  'FLAURA': {
    id: 'flaura',
    name: 'FLAURA Trial',
    type: 'trial',
    description: 'Phase III trial of osimertinib vs erlotinib/gefitinib in EGFR-mutated NSCLC',
    url: 'https://clinicaltrials.gov/study/NCT02296125'
  },
  'KEYNOTE-024': {
    id: 'keynote024',
    name: 'KEYNOTE-024 Trial',
    type: 'trial',
    description: 'Phase III trial of pembrolizumab vs chemotherapy in PD-L1 high NSCLC',
    url: 'https://clinicaltrials.gov/study/NCT02142738'
  },
  'ALEX': {
    id: 'alex',
    name: 'ALEX Trial',
    type: 'trial',
    description: 'Phase III trial of alectinib vs crizotinib in ALK-positive NSCLC',
    url: 'https://clinicaltrials.gov/study/NCT02075840'
  },
  'SNOMED': {
    id: 'snomed',
    name: 'SNOMED-CT',
    type: 'ontology',
    description: 'Systematized Nomenclature of Medicine - Clinical Terms',
    url: 'https://www.snomed.org'
  },
  'LUCADA': {
    id: 'lucada',
    name: 'LUCADA Ontology',
    type: 'ontology',
    description: 'Lung Cancer Data Ontology',
  },
  'FDA': {
    id: 'fda',
    name: 'FDA Label',
    type: 'drug',
    description: 'FDA Drug Prescribing Information',
    url: 'https://www.accessdata.fda.gov/scripts/cder/daf/'
  }
}

// Parse citation pattern: [[Type:ID]] or [[Type:ID|Label]]
const CITATION_PATTERN = /\[\[([A-Za-z]+):([^\]|]+)(?:\|([^\]]+))?\]\]/g

export function GroundedCitations({
  text,
  onCitationClick,
  onCitationHover,
  className = '',
  showTooltips = true,
  renderAs = 'inline'
}: GroundedCitationsProps) {
  const [hoveredCitation, setHoveredCitation] = useState<string | null>(null)
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 })

  // Parse and extract citations
  const { segments, citations } = useMemo(() => {
    const foundCitations: Citation[] = []
    const textSegments: Array<{ type: 'text' | 'citation'; content: string; citation?: Citation }> = []

    let lastIndex = 0
    let match: RegExpExecArray | null

    // Reset regex
    CITATION_PATTERN.lastIndex = 0

    while ((match = CITATION_PATTERN.exec(text)) !== null) {
      // Add text before citation
      if (match.index > lastIndex) {
        textSegments.push({
          type: 'text',
          content: text.slice(lastIndex, match.index)
        })
      }

      const [fullMatch, type, id, customLabel] = match
      const sourceKey = id.toUpperCase()
      const source = CITATION_SOURCES[sourceKey]

      const citation: Citation = {
        type: (type.toLowerCase() as CitationType) || 'guideline',
        id: id,
        label: customLabel || source?.name || id,
        fullText: fullMatch,
        url: source?.url,
        metadata: source ? { ...source } : undefined
      }

      foundCitations.push(citation)
      textSegments.push({
        type: 'citation',
        content: citation.label,
        citation
      })

      lastIndex = match.index + fullMatch.length
    }

    // Add remaining text
    if (lastIndex < text.length) {
      textSegments.push({
        type: 'text',
        content: text.slice(lastIndex)
      })
    }

    return { segments: textSegments, citations: foundCitations }
  }, [text])

  const handleCitationClick = (citation: Citation, e: React.MouseEvent) => {
    e.preventDefault()
    onCitationClick?.(citation)

    // If URL exists and no custom handler, open in new tab
    if (!onCitationClick && citation.url) {
      window.open(citation.url, '_blank', 'noopener,noreferrer')
    }
  }

  const handleCitationHover = (citation: Citation | null, e?: React.MouseEvent) => {
    if (citation && e) {
      const rect = (e.target as HTMLElement).getBoundingClientRect()
      setTooltipPosition({
        x: rect.left + rect.width / 2,
        y: rect.top - 8
      })
    }
    setHoveredCitation(citation?.id || null)
    onCitationHover?.(citation)
  }

  const getCitationColor = (type: CitationType) => {
    const colors: Record<CitationType, string> = {
      guideline: 'bg-violet-500/20 text-violet-400 border-violet-500/30 hover:bg-violet-500/30',
      trial: 'bg-blue-500/20 text-blue-400 border-blue-500/30 hover:bg-blue-500/30',
      publication: 'bg-green-500/20 text-green-400 border-green-500/30 hover:bg-green-500/30',
      ontology: 'bg-amber-500/20 text-amber-400 border-amber-500/30 hover:bg-amber-500/30',
      patient: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30 hover:bg-cyan-500/30',
      drug: 'bg-rose-500/20 text-rose-400 border-rose-500/30 hover:bg-rose-500/30'
    }
    return colors[type] || colors.guideline
  }

  const getCitationIcon = (type: CitationType) => {
    switch (type) {
      case 'guideline':
        return (
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        )
      case 'trial':
        return (
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
          </svg>
        )
      case 'publication':
        return (
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
          </svg>
        )
      case 'ontology':
        return (
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
        )
      case 'drug':
        return (
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
          </svg>
        )
      default:
        return (
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
          </svg>
        )
    }
  }

  // Render inline citations
  if (renderAs === 'inline') {
    return (
      <span className={`relative ${className}`}>
        {segments.map((segment, index) => {
          if (segment.type === 'text') {
            return <span key={index}>{segment.content}</span>
          }

          const citation = segment.citation!
          const isHovered = hoveredCitation === citation.id

          return (
            <span key={index} className="relative inline-block">
              <button
                onClick={(e) => handleCitationClick(citation, e)}
                onMouseEnter={(e) => handleCitationHover(citation, e)}
                onMouseLeave={() => handleCitationHover(null)}
                className={`inline-flex items-center gap-1 px-1.5 py-0.5 mx-0.5 text-xs font-medium rounded border transition-colors cursor-pointer ${getCitationColor(citation.type)}`}
              >
                {getCitationIcon(citation.type)}
                <span>{citation.label}</span>
                {citation.url && (
                  <svg className="w-2.5 h-2.5 opacity-60" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                )}
              </button>

              {/* Tooltip */}
              {showTooltips && isHovered && citation.metadata && (
                <div
                  className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-zinc-800 border border-zinc-700 rounded-lg shadow-xl"
                  style={{ pointerEvents: 'none' }}
                >
                  <div className="flex items-start gap-2">
                    <div className={`p-1.5 rounded ${getCitationColor(citation.type)}`}>
                      {getCitationIcon(citation.type)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-zinc-100">
                        {(citation.metadata as unknown as CitationSource)?.name || citation.label}
                      </div>
                      {(citation.metadata as unknown as CitationSource)?.description && (
                        <div className="text-xs text-zinc-400 mt-1">
                          {(citation.metadata as unknown as CitationSource).description}
                        </div>
                      )}
                      {(citation.metadata as unknown as CitationSource)?.version && (
                        <div className="text-xs text-zinc-500 mt-1">
                          Version: {(citation.metadata as unknown as CitationSource).version}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-full">
                    <div className="border-8 border-transparent border-t-zinc-800" />
                  </div>
                </div>
              )}
            </span>
          )
        })}
      </span>
    )
  }

  // Render with footnotes
  return (
    <div className={className}>
      <div className="mb-4">
        {segments.map((segment, index) => {
          if (segment.type === 'text') {
            return <span key={index}>{segment.content}</span>
          }

          const citation = segment.citation!
          const citationIndex = citations.findIndex(c => c.id === citation.id) + 1

          return (
            <span key={index}>
              <sup
                onClick={(e) => handleCitationClick(citation, e as any)}
                className={`cursor-pointer px-1 py-0.5 text-[10px] font-medium rounded ${getCitationColor(citation.type)}`}
              >
                [{citationIndex}]
              </sup>
            </span>
          )
        })}
      </div>

      {citations.length > 0 && (
        <div className="border-t border-zinc-700 pt-4">
          <h4 className="text-xs font-medium text-zinc-400 uppercase tracking-wider mb-2">References</h4>
          <ol className="space-y-2">
            {citations.map((citation, index) => (
              <li key={citation.id} className="flex items-start gap-2 text-sm">
                <span className="text-zinc-500 font-medium">[{index + 1}]</span>
                <div>
                  <button
                    onClick={(e) => handleCitationClick(citation, e)}
                    className="text-zinc-300 hover:text-white transition-colors"
                  >
                    {citation.label}
                  </button>
                  {citation.metadata && (citation.metadata as unknown as CitationSource)?.description && (
                    <span className="text-zinc-500 ml-2">
                      - {(citation.metadata as unknown as CitationSource).description}
                    </span>
                  )}
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  )
}

// Utility function to add citations to text
export function addCitation(text: string, type: CitationType, id: string, label?: string): string {
  const citation = label ? `[[${type}:${id}|${label}]]` : `[[${type}:${id}]]`
  return `${text} ${citation}`
}

// Parse citations from text
export function parseCitations(text: string): Citation[] {
  const citations: Citation[] = []
  let match: RegExpExecArray | null

  CITATION_PATTERN.lastIndex = 0

  while ((match = CITATION_PATTERN.exec(text)) !== null) {
    const [fullMatch, type, id, customLabel] = match
    const sourceKey = id.toUpperCase()
    const source = CITATION_SOURCES[sourceKey]

    citations.push({
      type: (type.toLowerCase() as CitationType) || 'guideline',
      id: id,
      label: customLabel || source?.name || id,
      fullText: fullMatch,
      url: source?.url,
      metadata: source ? { ...source } : undefined
    })
  }

  return citations
}

export default GroundedCitations
