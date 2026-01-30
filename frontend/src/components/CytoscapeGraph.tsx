'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import cytoscape, { Core, NodeSingular, EdgeSingular } from 'cytoscape'
import { X, Loader2, ZoomIn, ZoomOut, Maximize2, Minimize2 } from 'lucide-react'
import type { GraphData, GraphNode, GraphRelationship } from '@/lib/api'
import { expandNode, getRelationshipsBetween, mergeGraphData } from '@/lib/api'

// Node color mapping by label
const NODE_COLORS: Record<string, string> = {
  Patient: '#4299E1',
  TreatmentDecision: '#9F7AEA',
  Biomarker: '#48BB78',
  Guideline: '#38B2AC',
  Comorbidity: '#ED8936',
  ClinicalTrial: '#F56565',
  SNOMED: '#63B3ED',
  Inference: '#B794F4',
}

// Relationship colors
const REL_COLORS: Record<string, string> = {
  ABOUT: '#A0AEC0',
  BASED_ON: '#48BB78',
  APPLIED_GUIDELINE: '#38B2AC',
  CAUSED: '#E53E3E',
  INFLUENCED: '#D69E2E',
  FOLLOWED_PRECEDENT: '#9F7AEA',
  HAS_BIOMARKER: '#48BB78',
  HAS_COMORBIDITY: '#ED8936',
  CONSIDERS: '#F6AD55',
}

interface SelectedElement {
  type: 'node' | 'edge'
  data: GraphNode | GraphRelationship
}

interface CytoscapeGraphProps {
  graphData: GraphData | null
  onNodeClick?: (nodeId: string, labels: string[]) => void
  onDecisionNodeClick?: (node: GraphNode) => void
  onGraphDataChange?: (graphData: GraphData) => void
  height?: string
  showLegend?: boolean
  className?: string
}

export function CytoscapeGraph({
  graphData,
  onNodeClick,
  onDecisionNodeClick,
  onGraphDataChange,
  height = '100%',
  showLegend = true,
  className = '',
}: CytoscapeGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const cyRef = useRef<Core | null>(null)
  const [selectedElement, setSelectedElement] = useState<SelectedElement | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isExpanding, setIsExpanding] = useState(false)
  const [expandedNodeIds, setExpandedNodeIds] = useState<Set<string>>(new Set())
  const [internalGraphData, setInternalGraphData] = useState<GraphData | null>(graphData)

  // Update internal graph data when prop changes
  useEffect(() => {
    if (graphData) {
      setInternalGraphData(graphData)
      setExpandedNodeIds(new Set())
    }
  }, [graphData])

  // Build caption for node
  const buildNodeCaption = useCallback((node: GraphNode): string => {
    const primaryLabel = node.labels[0] || 'Unknown'
    let caption = ''

    if (primaryLabel === 'Patient') {
      const age = node.properties.age_at_diagnosis || node.properties.age || ''
      const sex = node.properties.sex || ''
      const stage = node.properties.tnm_stage || ''
      const name = node.properties.name || node.properties.patient_id || ''
      caption = name ? `${name}` : `Patient ${age}${sex}`
      if (stage) caption += ` (${stage})`
    } else if (primaryLabel === 'TreatmentDecision') {
      const treatment = node.properties.treatment as string || ''
      const confidence = node.properties.confidence_score as number
      caption = treatment.length > 30 ? treatment.substring(0, 27) + '...' : treatment
      if (confidence) caption += ` (${Math.round(confidence * 100)}%)`
    } else if (primaryLabel === 'Biomarker') {
      const markerType = node.properties.marker_type as string || ''
      const value = node.properties.value as string || node.properties.status as string || ''
      caption = markerType
      if (value) caption += `: ${value}`
    } else if (primaryLabel === 'Guideline') {
      caption = (node.properties.name as string) || (node.properties.guideline_id as string) || 'Guideline'
    } else if (primaryLabel === 'Comorbidity') {
      caption = (node.properties.name as string) || (node.properties.condition as string) || 'Comorbidity'
    } else {
      caption = (node.properties.name as string) ||
        (node.properties.patient_id as string) ||
        (node.properties.treatment as string) ||
        primaryLabel
    }

    return caption || primaryLabel
  }, [])

  // Initialize Cytoscape
  useEffect(() => {
    if (!containerRef.current || !internalGraphData || internalGraphData.nodes.length === 0) return

    setIsLoading(true)

    // Convert graph data to Cytoscape format
    const elements = {
      nodes: internalGraphData.nodes.map(node => {
        const primaryLabel = node.labels[0] || 'Unknown'
        const caption = buildNodeCaption(node)
        const isExpanded = expandedNodeIds.has(node.id)
        
        return {
          data: {
            id: node.id,
            label: caption,
            primaryLabel,
            labels: node.labels,
            properties: node.properties,
            color: isExpanded ? '#38A169' : NODE_COLORS[primaryLabel] || '#718096',
            nodeData: node,
          }
        }
      }),
      edges: internalGraphData.relationships.map(rel => ({
        data: {
          id: rel.id,
          source: rel.startNodeId,
          target: rel.endNodeId,
          label: rel.type.replace(/_/g, ' '),
          type: rel.type,
          color: REL_COLORS[rel.type] || '#A0AEC0',
          relData: rel,
        }
      }))
    }

    // Destroy existing instance
    if (cyRef.current) {
      cyRef.current.destroy()
    }

    // Create new Cytoscape instance
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            'label': 'data(label)',
            'color': '#1F2937',
            'text-valign': 'bottom',
            'text-halign': 'center',
            'text-margin-y': 5,
            'font-size': '11px',
            'font-weight': 'bold',
            'text-wrap': 'wrap',
            'text-max-width': '120px',
            'width': 40,
            'height': 40,
            'border-width': 3,
            'border-color': '#ffffff',
            'overlay-padding': '6px',
            'text-background-color': '#ffffff',
            'text-background-opacity': 0.9,
            'text-background-padding': '4px',
            'text-background-shape': 'roundrectangle',
          }
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 4,
            'border-color': '#DC2626',
            'overlay-opacity': 0.1,
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': 'data(color)',
            'target-arrow-color': 'data(color)',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(label)',
            'font-size': '9px',
            'text-rotation': 'autorotate',
            'text-margin-y': -10,
            'color': '#64748B',
            'text-background-color': '#ffffff',
            'text-background-opacity': 0.8,
            'text-background-padding': '2px',
            'arrow-scale': 1.2,
          }
        },
        {
          selector: 'edge:selected',
          style: {
            'width': 3,
            'line-color': '#DC2626',
            'target-arrow-color': '#DC2626',
          }
        }
      ],
      layout: {
        name: 'cose', // Physics-based layout similar to Neo4j
        animate: true,
        animationDuration: 500,
        animationEasing: 'ease-out',
        nodeRepulsion: 8000,
        idealEdgeLength: 120,
        edgeElasticity: 100,
        nestingFactor: 1.2,
        gravity: 0.25,
        numIter: 1000,
        initialTemp: 200,
        coolingFactor: 0.95,
        minTemp: 1.0,
      },
      wheelSensitivity: 0.2,
      minZoom: 0.2,
      maxZoom: 4,
    })

    cyRef.current = cy

    // Event handlers
    cy.on('tap', 'node', (evt) => {
      const node = evt.target
      const nodeData: GraphNode = node.data('nodeData')
      
      if (nodeData) {
        setSelectedElement({ type: 'node', data: nodeData })
        onNodeClick?.(node.id(), nodeData.labels)
        
        if (nodeData.labels.includes('TreatmentDecision') && onDecisionNodeClick) {
          onDecisionNodeClick(nodeData)
        }
      }
    })

    cy.on('tap', 'edge', (evt) => {
      const edge = evt.target
      const edgeData: GraphRelationship = edge.data('relData')
      
      if (edgeData) {
        setSelectedElement({ type: 'edge', data: edgeData })
      }
    })

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        setSelectedElement(null)
        cy.$(':selected').unselect()
      }
    })

    // Double-click to expand
    cy.on('dbltap', 'node', async (evt) => {
      const nodeId = evt.target.id()
      
      if (!internalGraphData || isExpanding || expandedNodeIds.has(nodeId)) return
      
      setIsExpanding(true)

      try {
        const expandedData = await expandNode(nodeId)

        if (expandedData.nodes.length === 0) {
          setIsExpanding(false)
          return
        }

        const updatedGraphData = mergeGraphData(internalGraphData, expandedData)
        const allNodeIds = updatedGraphData.nodes.map((n) => n.id)
        const additionalRels = await getRelationshipsBetween(allNodeIds)

        const existingRelIds = new Set(updatedGraphData.relationships.map((r) => r.id))
        const newRels = additionalRels.filter((r) => !existingRelIds.has(r.id))

        const finalGraphData: GraphData = {
          nodes: updatedGraphData.nodes,
          relationships: [...updatedGraphData.relationships, ...newRels],
        }

        setInternalGraphData(finalGraphData)
        setExpandedNodeIds((prev) => new Set([...prev, nodeId]))
        onGraphDataChange?.(finalGraphData)
      } catch (error) {
        console.error('Error expanding node:', error)
      } finally {
        setIsExpanding(false)
      }
    })

    setIsLoading(false)

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy()
      }
    }
  }, [internalGraphData, expandedNodeIds, isExpanding, buildNodeCaption, onNodeClick, onDecisionNodeClick, onGraphDataChange])

  const handleZoomIn = useCallback(() => {
    if (cyRef.current) {
      cyRef.current.zoom(cyRef.current.zoom() * 1.3)
      cyRef.current.center()
    }
  }, [])

  const handleZoomOut = useCallback(() => {
    if (cyRef.current) {
      cyRef.current.zoom(cyRef.current.zoom() / 1.3)
      cyRef.current.center()
    }
  }, [])

  const handleResetView = useCallback(() => {
    if (cyRef.current) {
      cyRef.current.fit(undefined, 50)
      cyRef.current.zoom(1)
    }
  }, [])

  const handleClosePanel = useCallback(() => {
    setSelectedElement(null)
    if (cyRef.current) {
      cyRef.current.$(':selected').unselect()
    }
  }, [])

  if (!internalGraphData || internalGraphData.nodes.length === 0) {
    return (
      <div
        className={`flex flex-col items-center justify-center gap-4 p-8 text-center ${className}`}
        style={{ height }}
      >
        <p className="text-gray-500">No graph data to display.</p>
        <p className="text-sm text-gray-400">
          Use the chat to search for patients or decisions to visualize the context graph.
        </p>
      </div>
    )
  }

  return (
    <div className={`relative ${className}`} style={{ height }}>
      {/* Legend */}
      {showLegend && (
        <div className="absolute top-2 left-2 z-10 bg-white/90 backdrop-blur-sm rounded-lg p-2 shadow-sm border border-gray-200 flex flex-wrap gap-1 max-w-[200px]">
          {Object.entries(NODE_COLORS).slice(0, 6).map(([label, color]) => (
            <span
              key={label}
              className="px-2 py-0.5 text-xs font-medium text-white rounded"
              style={{ backgroundColor: color }}
            >
              {label}
            </span>
          ))}
        </div>
      )}

      {/* Properties Panel */}
      {selectedElement && (
        <div className="absolute top-2 right-2 z-10 bg-white rounded-xl shadow-xl border border-gray-200 max-w-[320px] max-h-[500px] overflow-hidden">
          <div
            className="p-3 border-b border-gray-100"
            style={{
              background: selectedElement.type === 'node'
                ? `linear-gradient(135deg, ${NODE_COLORS[(selectedElement.data as GraphNode).labels[0]] || '#718096'}20, ${NODE_COLORS[(selectedElement.data as GraphNode).labels[0]] || '#718096'}05)`
                : 'linear-gradient(135deg, #A0AEC020, #A0AEC005)'
            }}
          >
            <div className="flex justify-between items-start">
              <div>
                {selectedElement.type === 'node' && (
                  <>
                    <div className="flex gap-1 flex-wrap mb-2">
                      {(selectedElement.data as GraphNode).labels.map((label) => (
                        <span
                          key={label}
                          className="px-2 py-1 text-xs font-semibold text-white rounded-full shadow-sm"
                          style={{ backgroundColor: NODE_COLORS[label] || '#718096' }}
                        >
                          {label}
                        </span>
                      ))}
                    </div>
                    <h3 className="text-sm font-bold text-gray-800">
                      {(selectedElement.data as GraphNode).properties.name as string ||
                       (selectedElement.data as GraphNode).properties.treatment as string ||
                       (selectedElement.data as GraphNode).properties.patient_id as string ||
                       'Node Details'}
                    </h3>
                  </>
                )}
                {selectedElement.type === 'edge' && (
                  <>
                    <span className="px-2 py-1 text-xs font-semibold bg-gray-600 text-white rounded-full">
                      Relationship
                    </span>
                    <h3 className="text-sm font-bold text-gray-800 mt-2">
                      {(selectedElement.data as GraphRelationship).type.replace(/_/g, ' ')}
                    </h3>
                  </>
                )}
              </div>
              <button
                onClick={handleClosePanel}
                className="p-1.5 hover:bg-white/50 rounded-full transition-colors"
              >
                <X className="w-4 h-4 text-gray-500" />
              </button>
            </div>
          </div>

          <div className="p-3 max-h-[380px] overflow-y-auto">
            {selectedElement.type === 'node' && (
              <div className="space-y-3">
                {Object.entries((selectedElement.data as GraphNode).properties).map(([key, value]) => {
                  if (key === 'id' || key.startsWith('_')) return null

                  const displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
                  let displayValue = ''

                  if (typeof value === 'boolean') {
                    displayValue = value ? 'Yes' : 'No'
                  } else if (typeof value === 'number') {
                    if (key.includes('confidence') || key.includes('score')) {
                      displayValue = `${Math.round(value * 100)}%`
                    } else {
                      displayValue = String(value)
                    }
                  } else if (typeof value === 'object' && value !== null) {
                    displayValue = JSON.stringify(value, null, 2)
                  } else {
                    displayValue = String(value ?? '-')
                  }

                  return (
                    <div key={key} className="bg-gray-50 p-2 rounded-lg border border-gray-100">
                      <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide block mb-0.5">
                        {displayKey}
                      </span>
                      <span className="text-sm font-medium text-gray-700 break-words">
                        {displayValue.length > 200 ? displayValue.slice(0, 200) + '...' : displayValue}
                      </span>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="absolute bottom-2 left-2 z-10 bg-white/80 backdrop-blur-sm rounded-lg px-2 py-1 shadow-sm border border-gray-200 opacity-80">
        <p className="text-xs text-gray-500">
          Drag nodes | Scroll to zoom | Click to inspect | Double-click to expand
        </p>
      </div>

      {/* Loading indicator */}
      {(isLoading || isExpanding) && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-white/70">
          <div className="bg-white rounded-lg p-3 shadow-md border border-gray-200 flex items-center gap-2">
            <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
            <span className="text-sm text-gray-700">
              {isExpanding ? 'Expanding node…' : 'Loading graph…'}
            </span>
          </div>
        </div>
      )}

      {/* Graph Container */}
      <div 
        ref={containerRef} 
        className="h-full w-full bg-gradient-to-br from-slate-50 to-gray-100"
      />

      {/* Zoom controls */}
      <div className="absolute bottom-12 right-2 flex flex-col gap-1 z-10">
        <button
          onClick={handleZoomIn}
          className="w-8 h-8 bg-white rounded-lg shadow border border-gray-200 flex items-center justify-center hover:bg-gray-50"
          title="Zoom in"
        >
          <ZoomIn className="w-4 h-4 text-gray-600" />
        </button>
        <button
          onClick={handleZoomOut}
          className="w-8 h-8 bg-white rounded-lg shadow border border-gray-200 flex items-center justify-center hover:bg-gray-50"
          title="Zoom out"
        >
          <ZoomOut className="w-4 h-4 text-gray-600" />
        </button>
        <button
          onClick={handleResetView}
          className="w-8 h-8 bg-white rounded-lg shadow border border-gray-200 flex items-center justify-center hover:bg-gray-50"
          title="Reset view"
        >
          <Maximize2 className="w-4 h-4 text-gray-600" />
        </button>
      </div>
    </div>
  )
}

