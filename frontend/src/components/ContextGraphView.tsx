'use client'

import { useEffect, useCallback, useMemo, useState, useRef } from 'react'
import { X, Loader2, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react'
import type { GraphData, GraphNode, GraphRelationship } from '@/lib/api'
import { expandNode, getRelationshipsBetween, mergeGraphData } from '@/lib/api'
import { CytoscapeGraph } from './CytoscapeGraph'

// NVL types
interface NvlNode {
  id: string
  caption?: string
  color?: string
  size?: number
  selected?: boolean
}

interface NvlRelationship {
  id: string
  from: string
  to: string
  caption?: string
  color?: string
  selected?: boolean
}

// Node color mapping by label - GitNexus-inspired vibrant clinical palette
const NODE_COLORS: Record<string, string> = {
  Patient: '#3b82f6',           // Blue (vibrant) - central entity
  TreatmentDecision: '#a855f7', // Purple (prominent) - key decisions
  Biomarker: '#10b981',         // Emerald (vivid) - biological markers
  Guideline: '#f59e0b',         // Amber (stands out) - clinical guidelines
  Comorbidity: '#f97316',       // Orange (bright) - conditions
  ClinicalTrial: '#f43f5e',     // Rose (execution indicator) - trials
  SNOMED: '#06b6d4',            // Cyan (distinct) - ontology codes
  Inference: '#8b5cf6',         // Violet (analytic) - derived knowledge
  Treatment: '#14b8a6',         // Teal - therapies
  Diagnosis: '#eab308',         // Yellow - diagnoses
  Procedure: '#ec4899',         // Pink - medical procedures
  Medication: '#22c55e',        // Green - drugs/medications
}

// Node size by label
const NODE_SIZES: Record<string, number> = {
  Patient: 30,
  TreatmentDecision: 28,
  Biomarker: 22,
  Guideline: 24,
  Comorbidity: 20,
  ClinicalTrial: 22,
  SNOMED: 18,
  Inference: 24,
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
  type: 'node' | 'relationship'
  data: GraphNode | GraphRelationship
}

interface ContextGraphViewProps {
  graphData: GraphData | null
  onNodeClick?: (nodeId: string, labels: string[]) => void
  onGraphDataChange?: (graphData: GraphData) => void
  onDecisionNodesChange?: (decisions: GraphNode[]) => void
  onDecisionNodeClick?: (node: GraphNode) => void
  selectedNodeId?: string
  height?: string
  showLegend?: boolean
  className?: string
}

export function ContextGraphView({
  graphData,
  onNodeClick,
  onGraphDataChange,
  onDecisionNodesChange,
  onDecisionNodeClick,
  selectedNodeId,
  height = '100%',
  showLegend = true,
  className = '',
}: ContextGraphViewProps) {
  const [selectedElement, setSelectedElement] = useState<SelectedElement | null>(null)
  const [internalSelectedNodeId, setInternalSelectedNodeId] = useState<string | null>(selectedNodeId || null)
  const [internalSelectedRelId, setInternalSelectedRelId] = useState<string | null>(null)
  const [isExpanding, setIsExpanding] = useState(false)
  const [expandedNodeIds, setExpandedNodeIds] = useState<Set<string>>(new Set())
  const [internalGraphData, setInternalGraphData] = useState<GraphData | null>(graphData)
  const previousDecisionNodesRef = useRef<string>('')

  // Update internal graph data when prop changes
  useEffect(() => {
    if (graphData) {
      console.log('[ContextGraphView] Received graph data:', graphData.nodes.length, 'nodes,', graphData.relationships.length, 'rels')
      setInternalGraphData(graphData)
      setExpandedNodeIds(new Set())
    }
  }, [graphData])

  // Memoize decision nodes to prevent unnecessary recalculations
  const decisionNodes = useMemo(() => {
    if (!internalGraphData) return []
    
    const decisions = internalGraphData.nodes.filter((node) => {
      const hasDecisionLabel =
        node.labels.includes('TreatmentDecision') ||
        node.labels.includes('Decision') ||
        node.labels.includes('Inference')

      const hasDecisionProperties =
        node.properties?.decision_type === 'treatment_recommendation' ||
        (node.properties?.treatment !== undefined && node.id.includes('decision'))

      return hasDecisionLabel || hasDecisionProperties
    })

    console.log('[ContextGraphView] Found', decisions.length, 'decision nodes in graph data')
    return decisions
  }, [internalGraphData])

  // Notify parent when decision nodes change (only when actually different)
  useEffect(() => {
    if (!onDecisionNodesChange) return
    
    // Create a stable identifier for the current decision nodes
    const currentKey = decisionNodes.map(d => d.id).sort().join(',')
    
    // Only notify if the decision nodes have actually changed
    if (currentKey && currentKey !== previousDecisionNodesRef.current) {
      console.log('[ContextGraphView] Notifying parent of', decisionNodes.length, 'decisions')
      previousDecisionNodesRef.current = currentKey
      onDecisionNodesChange(decisionNodes)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [decisionNodes]) // onDecisionNodesChange deliberately excluded to prevent infinite loop

  // Transform graph data to NVL format
  const nvlData = useMemo(() => {
    if (!internalGraphData) return { nodes: [], relationships: [] }

    console.log('[ContextGraphView] Transforming', internalGraphData.nodes.length, 'nodes to NVL format')

    const nodes: NvlNode[] = internalGraphData.nodes.map((node) => {
      const primaryLabel = node.labels[0] || 'Unknown'
      const isExpanded = expandedNodeIds.has(node.id)
      const isSelected = internalSelectedNodeId === node.id

      // Build a meaningful caption based on node type
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
        const decisionType = (node.properties.decision_type as string || '').replace(/_/g, ' ')
        const confidence = node.properties.confidence_score as number
        const evidenceLevel = node.properties.evidence_level as string || ''

        if (treatment) {
          const shortTreatment = treatment.length > 30 ? treatment.substring(0, 27) + '...' : treatment
          caption = shortTreatment
        } else if (decisionType) {
          caption = decisionType
        }

        if (evidenceLevel) {
          caption += ` [${evidenceLevel}]`
        } else if (confidence) {
          caption += ` (${Math.round(confidence * 100)}%)`
        }
      } else if (primaryLabel === 'Biomarker') {
        const markerType = node.properties.marker_type as string || ''
        const value = node.properties.value as string || node.properties.status as string || ''
        caption = markerType
        if (value) caption += `: ${value}`
      } else if (primaryLabel === 'Guideline') {
        caption = (node.properties.name as string) || (node.properties.guideline_id as string) || 'Guideline'
        const source = node.properties.source as string
        if (source) caption += ` (${source})`
      } else if (primaryLabel === 'Comorbidity') {
        caption = (node.properties.name as string) || (node.properties.condition as string) || 'Comorbidity'
      } else if (primaryLabel === 'ClinicalTrial') {
        caption = (node.properties.trial_id as string) || (node.properties.name as string) || 'Trial'
      } else if (primaryLabel === 'Inference') {
        caption = (node.properties.inference_type as string) || (node.properties.reasoning as string)?.substring(0, 30) || 'Inference'
      } else {
        caption =
          (node.properties.name as string) ||
          (node.properties.patient_id as string) ||
          (node.properties.decision_type as string) ||
          (node.properties.marker_type as string) ||
          primaryLabel
      }

      // Ensure we always have a caption
      if (!caption) caption = primaryLabel

      return {
        id: node.id,
        caption,
        color: isSelected
          ? '#E53E3E'
          : isExpanded
            ? '#38A169'
            : NODE_COLORS[primaryLabel] || '#718096',
        size: isSelected
          ? (NODE_SIZES[primaryLabel] || 20) * 1.3
          : NODE_SIZES[primaryLabel] || 20,
        selected: isSelected,
      }
    })

    const relationships: NvlRelationship[] = internalGraphData.relationships.map((rel) => {
      const isSelected = internalSelectedRelId === rel.id
      return {
        id: rel.id,
        from: rel.startNodeId,
        to: rel.endNodeId,
        caption: rel.type,
        color: isSelected ? '#E53E3E' : REL_COLORS[rel.type] || '#A0AEC0',
        selected: isSelected,
      }
    })

    console.log('[ContextGraphView] Created', nodes.length, 'NVL nodes')
    return { nodes, relationships }
  }, [internalGraphData, internalSelectedNodeId, internalSelectedRelId, expandedNodeIds])

  const handleNodeClick = useCallback(
    (node: NvlNode) => {
      if (internalGraphData) {
        const originalNode = internalGraphData.nodes.find((n) => n.id === node.id)
        if (originalNode) {
          setSelectedElement({ type: 'node', data: originalNode })
          setInternalSelectedNodeId(node.id)
          setInternalSelectedRelId(null)
          onNodeClick?.(node.id, originalNode.labels)

          if (originalNode.labels.includes('TreatmentDecision') && onDecisionNodeClick) {
            onDecisionNodeClick(originalNode)
          }
        }
      }
    },
    [internalGraphData, onNodeClick, onDecisionNodeClick]
  )

  const handleNodeDoubleClick = useCallback(
    async (node: NvlNode) => {
      if (!internalGraphData || isExpanding || expandedNodeIds.has(node.id)) return

      setIsExpanding(true)

      try {
        const expandedData = await expandNode(node.id)

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
        setExpandedNodeIds((prev) => new Set([...prev, node.id]))
        onGraphDataChange?.(finalGraphData)
      } catch (error) {
        console.error('Error expanding node:', error)
      } finally {
        setIsExpanding(false)
      }
    },
    [internalGraphData, isExpanding, expandedNodeIds, onGraphDataChange]
  )

  const handleRelationshipClick = useCallback(
    (rel: NvlRelationship) => {
      if (internalGraphData) {
        const originalRel = internalGraphData.relationships.find((r) => r.id === rel.id)
        if (originalRel) {
          setSelectedElement({ type: 'relationship', data: originalRel })
          setInternalSelectedRelId(rel.id)
          setInternalSelectedNodeId(null)
        }
      }
    },
    [internalGraphData]
  )

  const handleCanvasClick = useCallback(() => {
    setSelectedElement(null)
    setInternalSelectedNodeId(null)
    setInternalSelectedRelId(null)
  }, [])

  const handleClosePanel = useCallback(() => {
    setSelectedElement(null)
    setInternalSelectedNodeId(null)
    setInternalSelectedRelId(null)
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
      {/* Use Cytoscape for better performance and Neo4j-like experience */}
      <CytoscapeGraph
        graphData={internalGraphData}
        onNodeClick={onNodeClick}
        onDecisionNodeClick={onDecisionNodeClick}
        onGraphDataChange={onGraphDataChange}
        height={height}
        showLegend={showLegend}
        className={className}
      />
    </div>
  )
}

// Simple force-directed graph positions calculator
function calculateNodePositions(
  nodes: NvlNode[],
  relationships: NvlRelationship[],
  width: number,
  height: number
): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>()

  if (nodes.length === 0) return positions

  const centerX = width / 2
  const centerY = height / 2
  const radius = Math.min(width, height) * 0.3

  // Initialize in circle
  nodes.forEach((node, i) => {
    const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2
    positions.set(node.id, {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
    })
  })

  // Simple force simulation - fewer iterations for performance
  const iterations = Math.min(25, Math.max(10, 40 - nodes.length))

  for (let iter = 0; iter < iterations; iter++) {
    // Repulsion - increased force for better spacing
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const pos1 = positions.get(nodes[i].id)!
        const pos2 = positions.get(nodes[j].id)!
        const dx = pos2.x - pos1.x
        const dy = pos2.y - pos1.y
        const distSq = dx * dx + dy * dy

        if (distSq > 80000) continue // Skip if far (increased from 40000)

        const dist = Math.max(Math.sqrt(distSq), 1)
        const force = 8000 / distSq // Increased from 2500 for better spacing
        const fx = (dx / dist) * force
        const fy = (dy / dist) * force
        pos1.x -= fx
        pos1.y -= fy
        pos2.x += fx
        pos2.y += fy
      }
    }

    // Attraction along edges
    relationships.forEach((rel) => {
      const pos1 = positions.get(rel.from)
      const pos2 = positions.get(rel.to)
      if (!pos1 || !pos2) return
      const dx = pos2.x - pos1.x
      const dy = pos2.y - pos1.y
      const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1)
      const force = (dist - 120) * 0.025 // Increased target distance from 80 to 120
      const fx = (dx / dist) * force
      const fy = (dy / dist) * force
      pos1.x += fx
      pos1.y += fy
      pos2.x -= fx
      pos2.y -= fy
    })

    // Center gravity
    nodes.forEach((node) => {
      const pos = positions.get(node.id)!
      pos.x += (centerX - pos.x) * 0.02
      pos.y += (centerY - pos.y) * 0.02
    })
  }

  // Bound to viewport
  const padding = 70
  nodes.forEach((node) => {
    const pos = positions.get(node.id)!
    pos.x = Math.max(padding, Math.min(width - padding, pos.x))
    pos.y = Math.max(padding, Math.min(height - padding, pos.y))
  })

  return positions
}

// SVG-based graph component
function SvgGraph({
  nodes,
  relationships,
  onNodeClick,
  onNodeDoubleClick,
  onRelationshipClick,
  onCanvasClick,
}: {
  nodes: NvlNode[]
  relationships: NvlRelationship[]
  onNodeClick: (node: NvlNode) => void
  onNodeDoubleClick: (node: NvlNode) => void
  onRelationshipClick: (rel: NvlRelationship) => void
  onCanvasClick: () => void
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
  const [positions, setPositions] = useState<Map<string, { x: number; y: number }>>(new Map())
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [draggedNodeId, setDraggedNodeId] = useState<string | null>(null)
  const [dragNodeStart, setDragNodeStart] = useState({ x: 0, y: 0 })

  // Measure container
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const updateDimensions = () => {
      const rect = container.getBoundingClientRect()
      if (rect.width > 0 && rect.height > 0) {
        setDimensions({ width: rect.width, height: rect.height })
      }
    }

    updateDimensions()

    const resizeObserver = new ResizeObserver(updateDimensions)
    resizeObserver.observe(container)

    return () => resizeObserver.disconnect()
  }, [])

  // Calculate positions
  useEffect(() => {
    if (nodes.length > 0 && dimensions.width > 100 && dimensions.height > 100) {
      console.log('[SvgGraph] Calculating positions for', nodes.length, 'nodes in', dimensions.width, 'x', dimensions.height)
      const newPositions = calculateNodePositions(nodes, relationships, dimensions.width, dimensions.height)
      console.log('[SvgGraph] Calculated', newPositions.size, 'positions')
      setPositions(newPositions)
    }
  }, [nodes, relationships, dimensions])

  // Wheel zoom
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault()
      const delta = e.deltaY > 0 ? 0.9 : 1.1
      setZoom((z) => Math.max(0.2, Math.min(4, z * delta)))
    }

    container.addEventListener('wheel', handleWheel, { passive: false })
    return () => container.removeEventListener('wheel', handleWheel)
  }, [])

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!draggedNodeId) {
      setIsDragging(true)
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y })
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (draggedNodeId) {
      // Dragging a node
      const newX = (e.clientX - pan.x - dragNodeStart.x) / zoom
      const newY = (e.clientY - pan.y - dragNodeStart.y) / zoom
      setPositions(prev => {
        const newPositions = new Map(prev)
        newPositions.set(draggedNodeId, { x: newX, y: newY })
        return newPositions
      })
    } else if (isDragging) {
      // Panning the canvas
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y })
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
    setDraggedNodeId(null)
  }

  const handleNodeMouseDown = (e: React.MouseEvent, nodeId: string) => {
    e.stopPropagation()
    const pos = positions.get(nodeId)
    if (pos) {
      setDraggedNodeId(nodeId)
      setDragNodeStart({ 
        x: (e.clientX - pan.x) / zoom - pos.x, 
        y: (e.clientY - pan.y) / zoom - pos.y 
      })
    }
  }

  console.log('[SvgGraph] Rendering', nodes.length, 'nodes,', positions.size, 'positions, dims:', dimensions)

  if (nodes.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500">
        <p>No graph data</p>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className="h-full w-full overflow-hidden bg-gradient-to-br from-slate-50 to-gray-100 relative"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onClick={(e) => {
        if (e.target === e.currentTarget) onCanvasClick()
      }}
    >
      <svg
        width={dimensions.width}
        height={dimensions.height}
        style={{ cursor: isDragging ? 'grabbing' : 'grab', display: 'block' }}
        onClick={(e) => {
          if (e.target === e.currentTarget || (e.target as SVGElement).tagName === 'rect') {
            onCanvasClick()
          }
        }}
      >
        {/* Background */}
        <rect width={dimensions.width} height={dimensions.height} fill="transparent" />

        {/* Definitions */}
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#94A3B8" />
          </marker>
          <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="1" dy="1" stdDeviation="2" floodOpacity="0.15" />
          </filter>
        </defs>

        <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
          {/* Edges */}
          {relationships.map((rel) => {
            const fromPos = positions.get(rel.from)
            const toPos = positions.get(rel.to)
            if (!fromPos || !toPos) return null

            const dx = toPos.x - fromPos.x
            const dy = toPos.y - fromPos.y
            const dist = Math.sqrt(dx * dx + dy * dy)
            if (dist < 1) return null

            const r = 20
            const startX = fromPos.x + (dx / dist) * r
            const startY = fromPos.y + (dy / dist) * r
            const endX = toPos.x - (dx / dist) * (r + 8)
            const endY = toPos.y - (dy / dist) * (r + 8)
            const midX = (startX + endX) / 2
            const midY = (startY + endY) / 2

            return (
              <g key={rel.id} onClick={(e) => { e.stopPropagation(); onRelationshipClick(rel) }} style={{ cursor: 'pointer' }}>
                <line
                  x1={startX} y1={startY} x2={endX} y2={endY}
                  stroke={rel.color || '#94A3B8'}
                  strokeWidth={2}
                  markerEnd="url(#arrow)"
                />
                {rel.caption && (
                  <>
                    <rect
                      x={midX - rel.caption.length * 3}
                      y={midY - 8}
                      width={rel.caption.length * 6}
                      height={14}
                      fill="white"
                      fillOpacity={0.9}
                      rx={3}
                    />
                    <text x={midX} y={midY + 2} textAnchor="middle" fontSize={9} fill="#64748B" fontWeight={500}>
                      {rel.caption}
                    </text>
                  </>
                )}
              </g>
            )
          })}

          {/* Nodes */}
          {nodes.map((node) => {
            const pos = positions.get(node.id)
            if (!pos) {
              console.log('[SvgGraph] No position for node:', node.id)
              return null
            }

            const r = (node.size || 20) * 0.8
            const caption = node.caption || node.id.slice(0, 8)
            const shortCaption = caption.length > 20 ? caption.slice(0, 17) + '...' : caption

            return (
              <g
                key={node.id}
                transform={`translate(${pos.x}, ${pos.y})`}
                onMouseDown={(e) => handleNodeMouseDown(e, node.id)}
                onClick={(e) => { e.stopPropagation(); onNodeClick(node) }}
                onDoubleClick={(e) => { e.stopPropagation(); onNodeDoubleClick(node) }}
                style={{ cursor: draggedNodeId === node.id ? 'grabbing' : 'grab' }}
              >
                {/* Glow */}
                <circle r={r + 4} fill={node.color || '#718096'} opacity={0.2} />
                {/* Main */}
                <circle r={r} fill={node.color || '#718096'} stroke="white" strokeWidth={2.5} filter="url(#shadow)" />
                {/* Selection ring */}
                {node.selected && <circle r={r + 6} fill="none" stroke="#DC2626" strokeWidth={3} strokeDasharray="5 3" />}
                {/* Label bg */}
                <rect
                  x={-shortCaption.length * 3.2}
                  y={r + 4}
                  width={shortCaption.length * 6.4}
                  height={16}
                  fill="white"
                  fillOpacity={0.95}
                  rx={4}
                />
                {/* Label */}
                <text y={r + 16} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1F2937">
                  {shortCaption}
                </text>
              </g>
            )
          })}
        </g>
      </svg>

      {/* Zoom controls */}
      <div className="absolute bottom-12 right-2 flex flex-col gap-1 z-10">
        <button
          onClick={() => setZoom(z => Math.min(4, z * 1.3))}
          className="w-8 h-8 bg-white rounded-lg shadow border border-gray-200 flex items-center justify-center hover:bg-gray-50"
          title="Zoom in"
        >
          <ZoomIn className="w-4 h-4 text-gray-600" />
        </button>
        <button
          onClick={() => setZoom(z => Math.max(0.2, z / 1.3))}
          className="w-8 h-8 bg-white rounded-lg shadow border border-gray-200 flex items-center justify-center hover:bg-gray-50"
          title="Zoom out"
        >
          <ZoomOut className="w-4 h-4 text-gray-600" />
        </button>
        <button
          onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }) }}
          className="w-8 h-8 bg-white rounded-lg shadow border border-gray-200 flex items-center justify-center hover:bg-gray-50"
          title="Reset view"
        >
          <Maximize2 className="w-4 h-4 text-gray-600" />
        </button>
      </div>
    </div>
  )
}

// Compact inline graph for chat messages
interface InlineGraphProps {
  graphData: GraphData
  height?: string
  onNodeClick?: (nodeId: string, labels: string[]) => void
  className?: string
}

export function InlineGraph({
  graphData,
  height = '200px',
  onNodeClick,
  className = '',
}: InlineGraphProps) {
  return (
    <div
      className={`rounded-lg border border-gray-200 overflow-hidden my-2 ${className}`}
      style={{ height }}
    >
      <ContextGraphView
        graphData={graphData}
        onNodeClick={onNodeClick}
        showLegend={false}
        height={height}
      />
    </div>
  )
}
