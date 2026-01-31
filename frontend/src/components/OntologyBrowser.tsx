'use client'

import React, { useState, useEffect, useCallback } from 'react'

interface OntologyTerm {
  id: string
  label: string
  snomedCode?: string
  definition?: string
  synonyms?: string[]
  parents?: string[]
  children?: string[]
  domain: string
}

interface OntologyDomain {
  id: string
  name: string
  description: string
  termCount: number
}

interface OntologyBrowserProps {
  onTermSelect?: (term: OntologyTerm) => void
  onTermApply?: (term: OntologyTerm) => void
  selectedDomain?: string
  className?: string
  compact?: boolean
}

export function OntologyBrowser({
  onTermSelect,
  onTermApply,
  selectedDomain,
  className = '',
  compact = false
}: OntologyBrowserProps) {
  const [domains, setDomains] = useState<OntologyDomain[]>([])
  const [currentDomain, setCurrentDomain] = useState<string>(selectedDomain || 'all')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<OntologyTerm[]>([])
  const [selectedTerm, setSelectedTerm] = useState<OntologyTerm | null>(null)
  const [hierarchy, setHierarchy] = useState<OntologyTerm[]>([])
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set())
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch domains on mount
  useEffect(() => {
    fetchDomains()
  }, [])

  const fetchDomains = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/api/v1/ontology/domains`)
      if (response.ok) {
        const data = await response.json()
        setDomains(data.domains || [])
      } else {
        throw new Error('API not available')
      }
    } catch (err) {
      console.error('Failed to fetch domains:', err)
      // Use fallback domains
      setDomains([
        { id: 'Histology', name: 'Histology', description: 'Tumor histology types', termCount: 10 },
        { id: 'Biomarker', name: 'Biomarker', description: 'Molecular biomarkers', termCount: 15 },
        { id: 'Procedure', name: 'Procedure', description: 'Medical procedures', termCount: 12 },
        { id: 'PerformanceStatus', name: 'PS', description: 'Performance status', termCount: 5 },
        { id: 'Comorbidity', name: 'Comorbidity', description: 'Patient comorbidities', termCount: 8 }
      ])
    }
  }

  const searchOntology = useCallback(async (query: string) => {
    if (query.length < 2) {
      setSearchResults([])
      return
    }

    setLoading(true)
    setError(null)

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const params = new URLSearchParams({ term: query })
      if (currentDomain !== 'all') {
        params.append('domain', currentDomain)
      }

      const response = await fetch(`${apiUrl}/api/v1/ontology/search?${params}`)
      if (response.ok) {
        const data = await response.json()
        // Transform API response to match expected format
        const results = (data.matches || []).map((match: any) => ({
          id: match.snomed_code || match.term,
          label: match.display || match.term,
          snomedCode: match.snomed_code,
          domain: match.type?.toLowerCase() || 'general',
          definition: `${match.type}: ${match.display}`
        }))
        setSearchResults(results)
      } else {
        throw new Error('Search failed')
      }
    } catch (err) {
      console.error('Search error:', err)
      // Use mock results for demo
      setSearchResults(getMockResults(query))
    } finally {
      setLoading(false)
    }
  }, [currentDomain])

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      if (searchQuery) {
        searchOntology(searchQuery)
      }
    }, 300)

    return () => clearTimeout(timer)
  }, [searchQuery, searchOntology])

  const fetchHierarchy = async (termId: string) => {
    try {
      const response = await fetch(`/api/v1/ontology/hierarchy/${termId}`)
      if (response.ok) {
        const data = await response.json()
        setHierarchy(data.hierarchy || [])
      }
    } catch (err) {
      console.error('Failed to fetch hierarchy:', err)
    }
  }

  const handleTermClick = (term: OntologyTerm) => {
    setSelectedTerm(term)
    onTermSelect?.(term)
    fetchHierarchy(term.id)
  }

  const handleTermApply = () => {
    if (selectedTerm) {
      onTermApply?.(selectedTerm)
    }
  }

  const toggleNode = (nodeId: string) => {
    setExpandedNodes(prev => {
      const next = new Set(prev)
      if (next.has(nodeId)) {
        next.delete(nodeId)
      } else {
        next.add(nodeId)
      }
      return next
    })
  }

  const getMockResults = (query: string): OntologyTerm[] => {
    const mockTerms: OntologyTerm[] = [
      { id: 'egfr_mut', label: 'EGFR Mutation', snomedCode: '405824009', domain: 'biomarker', definition: 'Mutation in EGFR gene' },
      { id: 'nsclc', label: 'Non-Small Cell Lung Cancer', snomedCode: '254637007', domain: 'diagnosis', definition: 'NSCLC - most common type of lung cancer' },
      { id: 'pembrolizumab', label: 'Pembrolizumab', snomedCode: '716125002', domain: 'treatment', definition: 'PD-1 inhibitor immunotherapy' },
      { id: 'osimertinib', label: 'Osimertinib', snomedCode: '716097008', domain: 'treatment', definition: '3rd generation EGFR TKI' },
      { id: 'alk_fusion', label: 'ALK Rearrangement', snomedCode: '405825005', domain: 'biomarker', definition: 'ALK gene fusion' },
      { id: 'pdl1', label: 'PD-L1 Expression', snomedCode: '414916001', domain: 'biomarker', definition: 'Programmed death-ligand 1 expression level' },
      { id: 'stage_4', label: 'Stage IV', snomedCode: '258219007', domain: 'staging', definition: 'Metastatic disease' },
      { id: 'lobectomy', label: 'Lobectomy', snomedCode: '173171007', domain: 'procedure', definition: 'Surgical removal of lung lobe' }
    ]

    const q = query.toLowerCase()
    return mockTerms.filter(t =>
      t.label.toLowerCase().includes(q) ||
      t.snomedCode?.includes(q) ||
      t.definition?.toLowerCase().includes(q)
    )
  }

  const getDomainColor = (domain: string) => {
    const colors: Record<string, string> = {
      diagnosis: 'bg-blue-500/20 text-blue-400',
      treatment: 'bg-purple-500/20 text-purple-400',
      biomarker: 'bg-green-500/20 text-green-400',
      staging: 'bg-yellow-500/20 text-yellow-400',
      procedure: 'bg-orange-500/20 text-orange-400'
    }
    return colors[domain] || 'bg-zinc-500/20 text-zinc-400'
  }

  return (
    <div className={`flex flex-col bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden ${className}`}>
      {/* Header */}
      <div className="p-3 border-b border-zinc-800">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-purple-600 flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-zinc-100">Ontology Browser</h3>
            <p className="text-xs text-zinc-500">LUCADA + SNOMED-CT</p>
          </div>
        </div>

        {/* Search Input */}
        <div className="relative">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search terms, SNOMED codes..."
            className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-3 py-2 pl-9 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-violet-500"
          />
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          {loading && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <div className="w-4 h-4 border-2 border-violet-500 border-t-transparent rounded-full animate-spin" />
            </div>
          )}
        </div>
      </div>

      {/* Domain Filter */}
      {!compact && (
        <div className="p-2 border-b border-zinc-800 flex gap-1 overflow-x-auto">
          <button
            onClick={() => setCurrentDomain('all')}
            className={`px-2 py-1 rounded text-xs whitespace-nowrap transition-colors ${
              currentDomain === 'all'
                ? 'bg-violet-600 text-white'
                : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
            }`}
          >
            All
          </button>
          {domains.map(domain => (
            <button
              key={domain.id}
              onClick={() => setCurrentDomain(domain.id)}
              className={`px-2 py-1 rounded text-xs whitespace-nowrap transition-colors ${
                currentDomain === domain.id
                  ? 'bg-violet-600 text-white'
                  : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
              }`}
            >
              {domain.name}
            </button>
          ))}
        </div>
      )}

      {/* Results / Hierarchy */}
      <div className="flex-1 overflow-y-auto min-h-0" style={{ maxHeight: compact ? '200px' : '300px' }}>
        {searchResults.length > 0 ? (
          <div className="p-2 space-y-1">
            {searchResults.map(term => (
              <div
                key={term.id}
                onClick={() => handleTermClick(term)}
                className={`p-2 rounded cursor-pointer transition-colors ${
                  selectedTerm?.id === term.id
                    ? 'bg-violet-600/20 border border-violet-500'
                    : 'bg-zinc-800/50 border border-transparent hover:bg-zinc-800'
                }`}
              >
                <div className="flex items-start gap-2">
                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${getDomainColor(term.domain)}`}>
                    {term.domain}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-zinc-100 font-medium truncate">{term.label}</div>
                    {term.snomedCode && (
                      <div className="text-xs text-zinc-500">SNOMED: {term.snomedCode}</div>
                    )}
                    {term.definition && !compact && (
                      <div className="text-xs text-zinc-400 mt-1 line-clamp-2">{term.definition}</div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : searchQuery.length >= 2 && !loading ? (
          <div className="p-4 text-center text-zinc-500 text-sm">
            No results found
          </div>
        ) : (
          <div className="p-4 text-center text-zinc-500 text-sm">
            {compact ? 'Type to search' : 'Search for ontology terms or browse by domain'}
          </div>
        )}
      </div>

      {/* Selected Term Details */}
      {selectedTerm && !compact && (
        <div className="p-3 border-t border-zinc-800 bg-zinc-800/50">
          <div className="flex items-start justify-between mb-2">
            <div>
              <div className="text-sm font-medium text-zinc-100">{selectedTerm.label}</div>
              <div className="text-xs text-zinc-500">
                {selectedTerm.snomedCode && `SNOMED: ${selectedTerm.snomedCode}`}
              </div>
            </div>
            <button
              onClick={handleTermApply}
              className="px-3 py-1.5 bg-violet-600 hover:bg-violet-500 text-white text-xs font-medium rounded transition-colors"
            >
              Apply
            </button>
          </div>
          {selectedTerm.definition && (
            <p className="text-xs text-zinc-400">{selectedTerm.definition}</p>
          )}
          {selectedTerm.synonyms && selectedTerm.synonyms.length > 0 && (
            <div className="mt-2">
              <span className="text-xs text-zinc-500">Synonyms: </span>
              <span className="text-xs text-zinc-400">{selectedTerm.synonyms.join(', ')}</span>
            </div>
          )}
        </div>
      )}

      {/* Quick Actions */}
      {compact && selectedTerm && (
        <div className="p-2 border-t border-zinc-800">
          <button
            onClick={handleTermApply}
            className="w-full px-3 py-1.5 bg-violet-600 hover:bg-violet-500 text-white text-xs font-medium rounded transition-colors"
          >
            Apply: {selectedTerm.label}
          </button>
        </div>
      )}
    </div>
  )
}

export default OntologyBrowser
