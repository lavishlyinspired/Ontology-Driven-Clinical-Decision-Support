'use client'

/**
 * Laboratory Management Page
 * Provides LOINC code browsing, lab interpretation, and lab panel viewing
 */

import { useState } from 'react'
import { searchLoincCodes, interpretLabResult, getLabPanel, batchInterpretLabs } from '@/lib/api'
import type { LabResult, LabInterpretation } from '@/lib/api'

export default function LaboratoryPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [selectedPanel, setSelectedPanel] = useState<string>('')
  const [panelTests, setPanelTests] = useState<LabResult[]>([])
  const [labValue, setLabValue] = useState('')
  const [labUnits, setLabUnits] = useState('')
  const [selectedTest, setSelectedTest] = useState<any>(null)
  const [interpretation, setInterpretation] = useState<LabInterpretation | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'search' | 'interpret' | 'panels'>('search')

  const labPanels = [
    'baseline_staging',
    'molecular_testing',
    'immunotherapy_workup',
    'chemotherapy_monitoring',
    'targeted_therapy_monitoring'
  ]

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    setIsLoading(true)
    try {
      const results = await searchLoincCodes(searchQuery)
      setSearchResults(results)
    } catch (error) {
      console.error('Search failed:', error)
      setSearchResults([])
    } finally {
      setIsLoading(false)
    }
  }

  const handleInterpret = async () => {
    if (!selectedTest || !labValue) {
      alert('Please select a test and enter a value')
      return
    }
    setIsLoading(true)
    try {
      const result = await interpretLabResult(
        selectedTest.loinc_code,
        parseFloat(labValue),
        labUnits || selectedTest.units || ''
      )
      setInterpretation(result)
    } catch (error) {
      console.error('Interpretation failed:', error)
      setInterpretation(null)
    } finally {
      setIsLoading(false)
    }
  }

  const handleLoadPanel = async (panelType: string) => {
    setIsLoading(true)
    setSelectedPanel(panelType)
    try {
      const tests = await getLabPanel(panelType)
      setPanelTests(tests)
      setActiveTab('panels')
    } catch (error) {
      console.error('Failed to load panel:', error)
      setPanelTests([])
    } finally {
      setIsLoading(false)
    }
  }

  const getSeverityColor = (severity?: string) => {
    if (severity === 'grade4' || severity === 'critical') return 'text-red-700 bg-red-100'
    if (severity === 'grade3') return 'text-red-600 bg-red-50'
    if (severity === 'grade2') return 'text-yellow-600 bg-yellow-50'
    if (severity === 'grade1') return 'text-yellow-500 bg-yellow-50'
    return 'text-green-700 bg-green-100'
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">🧪 Laboratory Management</h1>
          <p className="mt-2 text-gray-600">
            Browse LOINC codes, interpret lab values, and view predefined lab panels
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
              LOINC Browser
            </button>
            <button
              onClick={() => setActiveTab('interpret')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'interpret'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Lab Interpreter
            </button>
            <button
              onClick={() => setActiveTab('panels')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'panels'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Lab Panels
            </button>
          </nav>
        </div>

        {/* Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Content */}
          <div className="lg:col-span-2">
            {activeTab === 'search' && (
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4">Search LOINC Codes</h2>
                <div className="flex gap-2 mb-6">
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                    placeholder="Enter test name (e.g., CBC, Creatinine, ALT)"
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
                    {searchResults.map((test, index) => (
                      <div
                        key={index}
                        className="border rounded-lg p-4 hover:bg-gray-50 cursor-pointer"
                        onClick={() => {
                          setSelectedTest(test)
                          setActiveTab('interpret')
                        }}
                      >
                        <div className="flex justify-between items-start">
                          <div>
                            <h3 className="font-semibold">{test.loinc_name || test.name}</h3>
                            <p className="text-sm text-gray-500">LOINC: {test.loinc_code}</p>
                            {test.component && (
                              <p className="text-sm text-gray-600 mt-1">
                                Component: {test.component}
                              </p>
                            )}
                          </div>
                          {test.category && (
                            <span className="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded">
                              {test.category}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'interpret' && (
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4">Interpret Lab Value</h2>

                {selectedTest ? (
                  <div className="space-y-4">
                    <div className="bg-blue-50 p-4 rounded">
                      <p className="font-semibold">{selectedTest.loinc_name || selectedTest.name}</p>
                      <p className="text-sm text-gray-600">LOINC: {selectedTest.loinc_code}</p>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Lab Value
                        </label>
                        <input
                          type="number"
                          step="any"
                          value={labValue}
                          onChange={(e) => setLabValue(e.target.value)}
                          placeholder="Enter value"
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Units
                        </label>
                        <input
                          type="text"
                          value={labUnits}
                          onChange={(e) => setLabUnits(e.target.value)}
                          placeholder="e.g., mg/dL, U/L"
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>
                    </div>

                    <button
                      onClick={handleInterpret}
                      disabled={isLoading || !labValue}
                      className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                    >
                      {isLoading ? 'Interpreting...' : 'Interpret Result'}
                    </button>

                    {interpretation && (
                      <div className="mt-6 p-4 border rounded-lg">
                        <h3 className="font-semibold mb-3">Interpretation</h3>
                        <div className="space-y-2">
                          <div>
                            <strong>Value:</strong> {interpretation.value} {interpretation.units}
                          </div>
                          <div>
                            <strong>Reference Range:</strong> {interpretation.reference_range || 'N/A'}
                          </div>
                          <div>
                            <strong>Interpretation:</strong>{' '}
                            <span className={`px-2 py-1 text-xs font-semibold rounded ${getSeverityColor(interpretation.severity)}`}>
                              {interpretation.interpretation}
                            </span>
                          </div>
                          {interpretation.clinical_significance && (
                            <div>
                              <strong>Clinical Significance:</strong>
                              <p className="mt-1 text-gray-700">{interpretation.clinical_significance}</p>
                            </div>
                          )}
                          {interpretation.recommendation && (
                            <div>
                              <strong>Recommendation:</strong>
                              <p className="mt-1 text-gray-700">{interpretation.recommendation}</p>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-gray-500">Search for a test from the LOINC Browser to interpret values</p>
                )}
              </div>
            )}

            {activeTab === 'panels' && (
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4">Lab Panels</h2>

                {panelTests.length > 0 ? (
                  <div>
                    <h3 className="font-semibold mb-3">
                      {selectedPanel.replace(/_/g, ' ').toUpperCase()}
                    </h3>
                    <div className="space-y-2">
                      {panelTests.map((test, index) => (
                        <div key={index} className="border rounded p-3 hover:bg-gray-50">
                          <div className="font-medium">{test.loinc_name || test.test_name}</div>
                          <div className="text-sm text-gray-500">LOINC: {test.loinc_code}</div>
                          {test.reference_range && (
                            <div className="text-sm text-gray-600 mt-1">
                              Reference: {test.reference_range}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <p className="text-gray-500">Select a lab panel from the sidebar to view included tests</p>
                )}
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow p-6 sticky top-4">
              <h3 className="font-semibold mb-4">Predefined Lab Panels</h3>
              <div className="space-y-2">
                {labPanels.map((panel) => (
                  <button
                    key={panel}
                    onClick={() => handleLoadPanel(panel)}
                    className={`w-full text-left px-4 py-2 rounded-lg transition-colors ${
                      selectedPanel === panel
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    {panel.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </button>
                ))}
              </div>

              {selectedTest && (
                <div className="mt-6 pt-6 border-t">
                  <h3 className="font-semibold mb-3">Selected Test</h3>
                  <div className="space-y-2 text-sm">
                    <p><strong>Name:</strong> {selectedTest.loinc_name || selectedTest.name}</p>
                    <p><strong>LOINC:</strong> {selectedTest.loinc_code}</p>
                    {selectedTest.component && <p><strong>Component:</strong> {selectedTest.component}</p>}
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
