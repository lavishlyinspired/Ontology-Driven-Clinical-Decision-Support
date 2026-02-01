'use client'

/**
 * Medications Management Page
 * Provides drug formulary search, interaction checking, and therapeutic alternatives
 */

import { useState } from 'react'
import { searchMedications, checkDrugInteractions, getTherapeuticAlternatives, getMedicationDetails } from '@/lib/api'
import type { Medication, DrugInteraction } from '@/lib/api'

export default function MedicationsPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<Medication[]>([])
  const [selectedMedication, setSelectedMedication] = useState<any>(null)
  const [drugList, setDrugList] = useState<string[]>([])
  const [interactions, setInteractions] = useState<DrugInteraction[]>([])
  const [alternatives, setAlternatives] = useState<Medication[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'search' | 'interactions' | 'alternatives'>('search')

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    setIsLoading(true)
    try {
      const results = await searchMedications(searchQuery)
      setSearchResults(results)
    } catch (error) {
      console.error('Search failed:', error)
      setSearchResults([])
    } finally {
      setIsLoading(false)
    }
  }

  const handleMedicationSelect = async (medication: Medication) => {
    setSelectedMedication(medication)
    if (medication.rxcui) {
      try {
        const details = await getMedicationDetails(medication.rxcui)
        setSelectedMedication({ ...medication, ...details })
      } catch (error) {
        console.error('Failed to fetch details:', error)
      }
    }
  }

  const handleCheckInteractions = async () => {
    if (drugList.length < 2) {
      alert('Please add at least 2 medications to check interactions')
      return
    }
    setIsLoading(true)
    try {
      const result = await checkDrugInteractions(drugList)
      setInteractions(result.interactions || [])
      setActiveTab('interactions')
    } catch (error) {
      console.error('Interaction check failed:', error)
      setInteractions([])
    } finally {
      setIsLoading(false)
    }
  }

  const handleFindAlternatives = async (rxcui: string) => {
    setIsLoading(true)
    try {
      const alts = await getTherapeuticAlternatives(rxcui)
      setAlternatives(alts)
      setActiveTab('alternatives')
    } catch (error) {
      console.error('Failed to fetch alternatives:', error)
      setAlternatives([])
    } finally {
      setIsLoading(false)
    }
  }

  const addToDrugList = (drugName: string) => {
    if (!drugList.includes(drugName)) {
      setDrugList([...drugList, drugName])
    }
  }

  const removeFromDrugList = (drugName: string) => {
    setDrugList(drugList.filter(d => d !== drugName))
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">💊 Medication Management</h1>
          <p className="mt-2 text-gray-600">
            Search drug formulary, check interactions, and find therapeutic alternatives
          </p>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200 mb-6">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('search')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'search'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Drug Search
            </button>
            <button
              onClick={() => setActiveTab('interactions')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'interactions'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Interaction Checker ({drugList.length})
            </button>
            <button
              onClick={() => setActiveTab('alternatives')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'alternatives'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Therapeutic Alternatives
            </button>
          </nav>
        </div>

        {/* Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Content */}
          <div className="lg:col-span-2">
            {activeTab === 'search' && (
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4">Search Drug Formulary</h2>
                <div className="flex gap-2 mb-6">
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                    placeholder="Enter drug name (e.g., Osimertinib, Pembrolizumab)"
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <button
                    onClick={handleSearch}
                    disabled={isLoading}
                    className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                  >
                    {isLoading ? 'Searching...' : 'Search'}
                  </button>
                </div>

                {searchResults.length > 0 && (
                  <div className="space-y-2">
                    {searchResults.map((med, index) => (
                      <div
                        key={index}
                        className="border rounded-lg p-4 hover:bg-gray-50 cursor-pointer"
                        onClick={() => handleMedicationSelect(med)}
                      >
                        <div className="flex justify-between items-start">
                          <div>
                            <h3 className="font-semibold">{med.drug_name || med.name}</h3>
                            {med.rxcui && <p className="text-sm text-gray-500">RxCUI: {med.rxcui}</p>}
                            {med.drug_class && (
                              <span className="inline-block mt-2 px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded">
                                {med.drug_class}
                              </span>
                            )}
                          </div>
                          <div className="flex gap-2">
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                addToDrugList(med.drug_name || med.name || '')
                              }}
                              className="px-3 py-1 text-sm bg-green-600 text-white rounded hover:bg-green-700"
                            >
                              + Add to List
                            </button>
                            {med.rxcui && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  handleFindAlternatives(med.rxcui!)
                                }}
                                className="px-3 py-1 text-sm bg-purple-600 text-white rounded hover:bg-purple-700"
                              >
                                Find Alternatives
                              </button>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'interactions' && (
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4">Drug Interactions</h2>
                {interactions.length > 0 ? (
                  <div className="space-y-3">
                    {interactions.map((interaction, index) => (
                      <div
                        key={index}
                        className={`p-4 rounded border ${
                          interaction.severity === 'SEVERE'
                            ? 'bg-red-50 border-red-300'
                            : interaction.severity === 'MODERATE'
                            ? 'bg-yellow-50 border-yellow-300'
                            : 'bg-blue-50 border-blue-300'
                        }`}
                      >
                        <div className="flex items-start gap-2">
                          <span className="text-xl">
                            {interaction.severity === 'SEVERE' ? '🔴' : interaction.severity === 'MODERATE' ? '🟡' : '🔵'}
                          </span>
                          <div className="flex-1">
                            <h3 className="font-semibold">
                              {interaction.drug1} + {interaction.drug2}
                            </h3>
                            <p className="text-sm mt-1">{interaction.clinical_effect}</p>
                            {interaction.recommendation && (
                              <p className="text-sm mt-2 font-medium">
                                <strong>Recommendation:</strong> {interaction.recommendation}
                              </p>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500">No interactions found. Add medications to the list and click "Check Interactions".</p>
                )}
              </div>
            )}

            {activeTab === 'alternatives' && (
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4">Therapeutic Alternatives</h2>
                {alternatives.length > 0 ? (
                  <div className="space-y-2">
                    {alternatives.map((alt, index) => (
                      <div key={index} className="border rounded-lg p-4 hover:bg-gray-50">
                        <h3 className="font-semibold">{alt.drug_name || alt.name}</h3>
                        {alt.drug_class && (
                          <span className="inline-block mt-2 px-2 py-1 text-xs bg-green-100 text-green-700 rounded">
                            {alt.drug_class}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500">Select a medication and click "Find Alternatives" to see therapeutic alternatives.</p>
                )}
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow p-6 sticky top-4">
              <h3 className="font-semibold mb-4">Medication List</h3>
              {drugList.length > 0 ? (
                <>
                  <div className="space-y-2 mb-4">
                    {drugList.map((drug, index) => (
                      <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                        <span className="text-sm">{drug}</span>
                        <button
                          onClick={() => removeFromDrugList(drug)}
                          className="text-red-600 hover:text-red-800 text-sm"
                        >
                          Remove
                        </button>
                      </div>
                    ))}
                  </div>
                  <button
                    onClick={handleCheckInteractions}
                    disabled={drugList.length < 2 || isLoading}
                    className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                  >
                    Check Interactions
                  </button>
                </>
              ) : (
                <p className="text-sm text-gray-500">
                  Add medications from search results to check for drug interactions
                </p>
              )}

              {selectedMedication && (
                <div className="mt-6 pt-6 border-t">
                  <h3 className="font-semibold mb-3">Selected Medication</h3>
                  <div className="space-y-2 text-sm">
                    <p><strong>Name:</strong> {selectedMedication.drug_name || selectedMedication.name}</p>
                    {selectedMedication.rxcui && <p><strong>RxCUI:</strong> {selectedMedication.rxcui}</p>}
                    {selectedMedication.drug_class && <p><strong>Class:</strong> {selectedMedication.drug_class}</p>}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
