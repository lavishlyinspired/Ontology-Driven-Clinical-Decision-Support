'use client'

import { useState, useEffect } from 'react'
import { Network, ChevronLeft, ChevronRight, X } from 'lucide-react'
import { SigmaGraph } from './SigmaGraph'
import { useApp } from '@/contexts/AppContext'

export default function ContextGraphPanel() {
  const { graphData, isGraphPanelOpen, setIsGraphPanelOpen } = useApp()
  const [isCollapsed, setIsCollapsed] = useState(false)

  // Auto-show when graph data is available
  useEffect(() => {
    if (graphData && graphData.nodes.length > 0) {
      setIsCollapsed(false)
    }
  }, [graphData])

  const handleToggle = () => {
    const newCollapsed = !isCollapsed
    setIsCollapsed(newCollapsed)
    setIsGraphPanelOpen(!newCollapsed)
  }

  if (!isGraphPanelOpen) return null

  return (
    <>
      <div className={`context-graph-panel ${isCollapsed ? 'context-graph-panel-collapsed' : ''}`}>
        {/* Panel Header */}
        <div className="panel-header">
          <div className="flex items-center gap-2">
            <Network className="w-5 h-5 text-violet-400" />
            {!isCollapsed && <h2 className="panel-title">Context Graph</h2>}
          </div>
          <button
            onClick={handleToggle}
            className="panel-toggle-btn"
            title={isCollapsed ? "Expand graph panel" : "Collapse graph panel"}
          >
            {isCollapsed ? (
              <ChevronRight className="w-4 h-4" />
            ) : (
              <ChevronLeft className="w-4 h-4" />
            )}
          </button>
        </div>

        {/* Panel Content */}
        {!isCollapsed && (
          <div className="panel-content">
            {graphData && graphData.nodes.length > 0 ? (
              <div className="graph-container">
                <SigmaGraph
                  data={{
                    nodes: graphData.nodes.map(n => {
                      // Extract meaningful label from properties
                      const props = n.properties || {}
                      const nodeType = n.labels[0] || ''
                      
                      // Special handling for Inference nodes
                      let label: string
                      if (nodeType.includes('Inference') || nodeType === 'CancerClassification' || nodeType === 'TherapyInference') {
                        label = 
                          props.cancer_subtype as string || // e.g., "NSCLC_Adenocarcinoma"
                          props.therapy_class as string ||   // e.g., "EGFR_TKI"
                          props.inference_type as string ||
                          `Inference: ${props.rule || 'Unknown'}`
                      } else if (nodeType === 'Patient' || nodeType === 'Diagnosis') {
                        // Include histology for Patient/Diagnosis nodes
                        const histology = props.histology_type as string
                        const patientName = props.name as string
                        label = histology 
                          ? (patientName ? `${patientName} (${histology})` : histology)
                          : (patientName || n.id.split(':').pop() || n.id)
                      } else {
                        // Standard label extraction
                        label = 
                          props.name as string ||
                          props.treatment as string ||
                          props.drug_name as string ||
                          props.label as string ||
                          props.title as string ||
                          props.guideline_name as string ||
                          props.test_name as string ||
                          props.loinc_name as string ||
                          props.concept_name as string ||
                          props.description as string ||
                          props.code as string ||
                          (typeof props.id === 'string' ? props.id : null) ||
                          n.id.split(':').pop() ||
                          n.id
                      }
                      
                      return {
                        id: n.id,
                        label: String(label).substring(0, 50), // Limit length
                        type: n.labels[0],
                        properties: n.properties,
                        color: undefined,
                        size: undefined
                      }
                    }),
                    edges: graphData.relationships.map(r => ({
                      id: r.id,
                      source: r.startNodeId,
                      target: r.endNodeId,
                      label: r.type
                    }))
                  }}
                  height="100%"
                  showLabels={true}
                  enableZoom={true}
                  enablePan={true}
                  layout="force"
                />
                {/* Stats */}
                <div className="graph-stats">
                  <p className="text-xs text-zinc-400">
                    {graphData.nodes.length} nodes | {graphData.relationships.length} relationships
                  </p>
                </div>
              </div>
            ) : (
              <div className="empty-state">
                <Network className="w-12 h-12 text-zinc-600 mb-4" />
                <p className="text-sm text-zinc-500 text-center">
                  No graph data available.<br />
                  Start a conversation to see the context graph.
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Overlay for mobile */}
      {!isCollapsed && (
        <div
          className="panel-overlay md:hidden"
          onClick={() => setIsCollapsed(true)}
        />
      )}
    </>
  )
}