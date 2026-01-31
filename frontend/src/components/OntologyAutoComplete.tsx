'use client'

import React, { useState, useEffect, useCallback, useRef } from 'react'

interface OntologyTerm {
  id: string
  label: string
  snomedCode?: string
  domain: string
  definition?: string
}

interface OntologyAutoCompleteProps {
  value?: string
  onChange?: (value: string) => void
  onTermSelect?: (term: OntologyTerm) => void
  placeholder?: string
  domain?: string
  className?: string
  inputClassName?: string
  disabled?: boolean
  showSnomedCode?: boolean
  showDomain?: boolean
  maxSuggestions?: number
  debounceMs?: number
}

export function OntologyAutoComplete({
  value: controlledValue,
  onChange,
  onTermSelect,
  placeholder = 'Search ontology terms...',
  domain,
  className = '',
  inputClassName = '',
  disabled = false,
  showSnomedCode = true,
  showDomain = true,
  maxSuggestions = 8,
  debounceMs = 200
}: OntologyAutoCompleteProps) {
  const [inputValue, setInputValue] = useState(controlledValue || '')
  const [suggestions, setSuggestions] = useState<OntologyTerm[]>([])
  const [isOpen, setIsOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [highlightedIndex, setHighlightedIndex] = useState(-1)
  const [isFocused, setIsFocused] = useState(false)

  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLUListElement>(null)
  const debounceRef = useRef<NodeJS.Timeout>()

  // Sync with controlled value
  useEffect(() => {
    if (controlledValue !== undefined) {
      setInputValue(controlledValue)
    }
  }, [controlledValue])

  // Mock ontology data - in production, this would be an API call
  const mockOntologySearch = useCallback(async (query: string): Promise<OntologyTerm[]> => {
    const allTerms: OntologyTerm[] = [
      // Diagnoses
      { id: 'nsclc', label: 'Non-Small Cell Lung Cancer', snomedCode: '254637007', domain: 'diagnosis', definition: 'Primary lung malignancy excluding small cell' },
      { id: 'sclc', label: 'Small Cell Lung Cancer', snomedCode: '254632001', domain: 'diagnosis', definition: 'Aggressive neuroendocrine lung cancer' },
      { id: 'adenocarcinoma', label: 'Adenocarcinoma', snomedCode: '35917007', domain: 'diagnosis', definition: 'Cancer forming glandular structures' },
      { id: 'squamous', label: 'Squamous Cell Carcinoma', snomedCode: '59367005', domain: 'diagnosis', definition: 'Cancer of squamous epithelial cells' },
      { id: 'large_cell', label: 'Large Cell Carcinoma', snomedCode: '22049006', domain: 'diagnosis', definition: 'Undifferentiated NSCLC' },

      // Biomarkers
      { id: 'egfr', label: 'EGFR Mutation', snomedCode: '405824009', domain: 'biomarker', definition: 'Epidermal growth factor receptor mutation' },
      { id: 'egfr_l858r', label: 'EGFR L858R', snomedCode: '405824009', domain: 'biomarker', definition: 'Exon 21 L858R point mutation' },
      { id: 'egfr_ex19del', label: 'EGFR Exon 19 Deletion', snomedCode: '405824009', domain: 'biomarker', definition: 'In-frame deletion in exon 19' },
      { id: 'egfr_t790m', label: 'EGFR T790M', snomedCode: '405824009', domain: 'biomarker', definition: 'Resistance mutation' },
      { id: 'alk', label: 'ALK Rearrangement', snomedCode: '405825005', domain: 'biomarker', definition: 'Anaplastic lymphoma kinase fusion' },
      { id: 'ros1', label: 'ROS1 Rearrangement', snomedCode: '405826006', domain: 'biomarker', definition: 'ROS1 gene fusion' },
      { id: 'kras', label: 'KRAS Mutation', snomedCode: '405827002', domain: 'biomarker', definition: 'KRAS oncogene mutation' },
      { id: 'kras_g12c', label: 'KRAS G12C', snomedCode: '405827002', domain: 'biomarker', definition: 'Glycine to cysteine at codon 12' },
      { id: 'pdl1', label: 'PD-L1 Expression', snomedCode: '414916001', domain: 'biomarker', definition: 'Programmed death-ligand 1 level' },
      { id: 'braf', label: 'BRAF Mutation', snomedCode: '405828007', domain: 'biomarker', definition: 'BRAF V600E mutation' },
      { id: 'met', label: 'MET Amplification', snomedCode: '405829004', domain: 'biomarker', definition: 'MET gene copy number gain' },
      { id: 'ret', label: 'RET Fusion', snomedCode: '405830009', domain: 'biomarker', definition: 'RET gene rearrangement' },
      { id: 'ntrk', label: 'NTRK Fusion', snomedCode: '405831008', domain: 'biomarker', definition: 'Neurotrophic tyrosine receptor kinase fusion' },
      { id: 'her2', label: 'HER2 Mutation', snomedCode: '405832001', domain: 'biomarker', definition: 'ERBB2 gene alteration' },

      // Treatments
      { id: 'osimertinib', label: 'Osimertinib', snomedCode: '716097008', domain: 'treatment', definition: '3rd-gen EGFR TKI' },
      { id: 'pembrolizumab', label: 'Pembrolizumab', snomedCode: '716125002', domain: 'treatment', definition: 'Anti-PD-1 immunotherapy' },
      { id: 'nivolumab', label: 'Nivolumab', snomedCode: '716124003', domain: 'treatment', definition: 'Anti-PD-1 immunotherapy' },
      { id: 'atezolizumab', label: 'Atezolizumab', snomedCode: '716126001', domain: 'treatment', definition: 'Anti-PD-L1 immunotherapy' },
      { id: 'durvalumab', label: 'Durvalumab', snomedCode: '716127005', domain: 'treatment', definition: 'Anti-PD-L1 immunotherapy' },
      { id: 'alectinib', label: 'Alectinib', snomedCode: '716098003', domain: 'treatment', definition: 'ALK inhibitor' },
      { id: 'brigatinib', label: 'Brigatinib', snomedCode: '716099006', domain: 'treatment', definition: 'ALK inhibitor' },
      { id: 'lorlatinib', label: 'Lorlatinib', snomedCode: '716100003', domain: 'treatment', definition: 'ALK/ROS1 inhibitor' },
      { id: 'crizotinib', label: 'Crizotinib', snomedCode: '716101004', domain: 'treatment', definition: 'ALK/ROS1/MET inhibitor' },
      { id: 'sotorasib', label: 'Sotorasib', snomedCode: '716102006', domain: 'treatment', definition: 'KRAS G12C inhibitor' },
      { id: 'carboplatin', label: 'Carboplatin', snomedCode: '386903001', domain: 'treatment', definition: 'Platinum chemotherapy' },
      { id: 'pemetrexed', label: 'Pemetrexed', snomedCode: '409093004', domain: 'treatment', definition: 'Antifolate chemotherapy' },
      { id: 'docetaxel', label: 'Docetaxel', snomedCode: '386917002', domain: 'treatment', definition: 'Taxane chemotherapy' },

      // Staging
      { id: 'stage_1a', label: 'Stage IA', snomedCode: '258215001', domain: 'staging', definition: 'T1a-c N0 M0' },
      { id: 'stage_1b', label: 'Stage IB', snomedCode: '258216000', domain: 'staging', definition: 'T2a N0 M0' },
      { id: 'stage_2a', label: 'Stage IIA', snomedCode: '258217009', domain: 'staging', definition: 'T2b N0 M0' },
      { id: 'stage_2b', label: 'Stage IIB', snomedCode: '258218004', domain: 'staging', definition: 'T1-2 N1 M0 or T3 N0 M0' },
      { id: 'stage_3a', label: 'Stage IIIA', snomedCode: '258219007', domain: 'staging', definition: 'T1-2 N2 M0 or T3-4 N1 M0' },
      { id: 'stage_3b', label: 'Stage IIIB', snomedCode: '258220001', domain: 'staging', definition: 'T1-2 N3 M0 or T3-4 N2 M0' },
      { id: 'stage_3c', label: 'Stage IIIC', snomedCode: '258221002', domain: 'staging', definition: 'T3-4 N3 M0' },
      { id: 'stage_4a', label: 'Stage IVA', snomedCode: '258222009', domain: 'staging', definition: 'M1a-b' },
      { id: 'stage_4b', label: 'Stage IVB', snomedCode: '258223004', domain: 'staging', definition: 'M1c' },

      // Procedures
      { id: 'lobectomy', label: 'Lobectomy', snomedCode: '173171007', domain: 'procedure', definition: 'Surgical lobe removal' },
      { id: 'pneumonectomy', label: 'Pneumonectomy', snomedCode: '173170008', domain: 'procedure', definition: 'Complete lung removal' },
      { id: 'wedge_resection', label: 'Wedge Resection', snomedCode: '173172000', domain: 'procedure', definition: 'Limited lung tissue removal' },
      { id: 'sbrt', label: 'SBRT', snomedCode: '441783008', domain: 'procedure', definition: 'Stereotactic body radiation therapy' },
      { id: 'bronchoscopy', label: 'Bronchoscopy', snomedCode: '10847001', domain: 'procedure', definition: 'Airway visualization procedure' },
      { id: 'pet_ct', label: 'PET-CT', snomedCode: '443952007', domain: 'procedure', definition: 'Positron emission tomography with CT' },
      { id: 'brain_mri', label: 'Brain MRI', snomedCode: '241601008', domain: 'procedure', definition: 'Magnetic resonance imaging of brain' }
    ]

    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 100))

    const q = query.toLowerCase()
    let results = allTerms.filter(term =>
      term.label.toLowerCase().includes(q) ||
      term.snomedCode?.includes(q) ||
      term.definition?.toLowerCase().includes(q)
    )

    // Filter by domain if specified
    if (domain) {
      results = results.filter(term => term.domain === domain)
    }

    return results.slice(0, maxSuggestions)
  }, [domain, maxSuggestions])

  // Search for suggestions
  const searchSuggestions = useCallback(async (query: string) => {
    if (query.length < 2) {
      setSuggestions([])
      setIsOpen(false)
      return
    }

    setLoading(true)

    try {
      // Try API first, fall back to mock data
      const response = await fetch(`/api/v1/ontology/search?term=${encodeURIComponent(query)}${domain ? `&domain=${domain}` : ''}`)
      if (response.ok) {
        const data = await response.json()
        setSuggestions(data.results?.slice(0, maxSuggestions) || [])
      } else {
        // Fall back to mock data
        const results = await mockOntologySearch(query)
        setSuggestions(results)
      }
    } catch {
      // Fall back to mock data on error
      const results = await mockOntologySearch(query)
      setSuggestions(results)
    } finally {
      setLoading(false)
      setIsOpen(true)
      setHighlightedIndex(-1)
    }
  }, [domain, maxSuggestions, mockOntologySearch])

  // Handle input change with debounce
  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value
    setInputValue(newValue)
    onChange?.(newValue)

    // Debounce search
    if (debounceRef.current) {
      clearTimeout(debounceRef.current)
    }

    debounceRef.current = setTimeout(() => {
      searchSuggestions(newValue)
    }, debounceMs)
  }, [onChange, searchSuggestions, debounceMs])

  // Handle term selection
  const handleSelectTerm = useCallback((term: OntologyTerm) => {
    setInputValue(term.label)
    onChange?.(term.label)
    onTermSelect?.(term)
    setIsOpen(false)
    setSuggestions([])
    setHighlightedIndex(-1)
  }, [onChange, onTermSelect])

  // Handle keyboard navigation
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (!isOpen || suggestions.length === 0) {
      if (e.key === 'ArrowDown' && inputValue.length >= 2) {
        searchSuggestions(inputValue)
      }
      return
    }

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault()
        setHighlightedIndex(prev =>
          prev < suggestions.length - 1 ? prev + 1 : 0
        )
        break
      case 'ArrowUp':
        e.preventDefault()
        setHighlightedIndex(prev =>
          prev > 0 ? prev - 1 : suggestions.length - 1
        )
        break
      case 'Enter':
        e.preventDefault()
        if (highlightedIndex >= 0 && highlightedIndex < suggestions.length) {
          handleSelectTerm(suggestions[highlightedIndex])
        }
        break
      case 'Escape':
        setIsOpen(false)
        setHighlightedIndex(-1)
        break
      case 'Tab':
        setIsOpen(false)
        break
    }
  }, [isOpen, suggestions, highlightedIndex, inputValue, handleSelectTerm, searchSuggestions])

  // Scroll highlighted item into view
  useEffect(() => {
    if (highlightedIndex >= 0 && listRef.current) {
      const item = listRef.current.children[highlightedIndex] as HTMLElement
      if (item) {
        item.scrollIntoView({ block: 'nearest' })
      }
    }
  }, [highlightedIndex])

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (inputRef.current && !inputRef.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current)
      }
    }
  }, [])

  const getDomainColor = (d: string) => {
    const colors: Record<string, string> = {
      diagnosis: 'bg-blue-500/20 text-blue-400',
      treatment: 'bg-purple-500/20 text-purple-400',
      biomarker: 'bg-green-500/20 text-green-400',
      staging: 'bg-yellow-500/20 text-yellow-400',
      procedure: 'bg-orange-500/20 text-orange-400'
    }
    return colors[d] || 'bg-zinc-500/20 text-zinc-400'
  }

  return (
    <div className={`relative ${className}`}>
      <div className="relative">
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onFocus={() => {
            setIsFocused(true)
            if (inputValue.length >= 2 && suggestions.length > 0) {
              setIsOpen(true)
            }
          }}
          onBlur={() => setIsFocused(false)}
          placeholder={placeholder}
          disabled={disabled}
          className={`w-full bg-zinc-800 border border-zinc-700 rounded-md px-3 py-2 pl-9 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 disabled:opacity-50 disabled:cursor-not-allowed ${inputClassName}`}
          autoComplete="off"
          role="combobox"
          aria-expanded={isOpen}
          aria-haspopup="listbox"
          aria-autocomplete="list"
        />

        {/* Search icon */}
        <svg
          className={`absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 transition-colors ${
            isFocused ? 'text-violet-400' : 'text-zinc-500'
          }`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
          />
        </svg>

        {/* Loading spinner */}
        {loading && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2">
            <div className="w-4 h-4 border-2 border-violet-500 border-t-transparent rounded-full animate-spin" />
          </div>
        )}
      </div>

      {/* Suggestions dropdown */}
      {isOpen && suggestions.length > 0 && (
        <ul
          ref={listRef}
          className="absolute z-50 w-full mt-1 bg-zinc-800 border border-zinc-700 rounded-md shadow-lg max-h-64 overflow-y-auto"
          role="listbox"
        >
          {suggestions.map((term, index) => (
            <li
              key={term.id}
              onClick={() => handleSelectTerm(term)}
              onMouseEnter={() => setHighlightedIndex(index)}
              className={`px-3 py-2 cursor-pointer transition-colors ${
                index === highlightedIndex
                  ? 'bg-violet-600/30 border-l-2 border-violet-500'
                  : 'hover:bg-zinc-700/50 border-l-2 border-transparent'
              }`}
              role="option"
              aria-selected={index === highlightedIndex}
            >
              <div className="flex items-start gap-2">
                {showDomain && (
                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium shrink-0 ${getDomainColor(term.domain)}`}>
                    {term.domain}
                  </span>
                )}
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-zinc-100 font-medium truncate">
                    {term.label}
                  </div>
                  <div className="flex items-center gap-2 text-xs text-zinc-500">
                    {showSnomedCode && term.snomedCode && (
                      <span>SNOMED: {term.snomedCode}</span>
                    )}
                    {term.definition && (
                      <span className="truncate">{term.definition}</span>
                    )}
                  </div>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}

      {/* No results message */}
      {isOpen && !loading && suggestions.length === 0 && inputValue.length >= 2 && (
        <div className="absolute z-50 w-full mt-1 bg-zinc-800 border border-zinc-700 rounded-md shadow-lg p-3 text-center text-zinc-500 text-sm">
          No matching terms found
        </div>
      )}
    </div>
  )
}

export default OntologyAutoComplete
