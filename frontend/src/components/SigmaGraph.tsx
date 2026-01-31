'use client'

import React, { useEffect, useRef, useState, useCallback } from 'react'

// Graph data types
interface GraphNode {
  id: string
  label: string
  x?: number
  y?: number
  size?: number
  color?: string
  type?: string
  properties?: Record<string, unknown>
}

interface GraphEdge {
  id: string
  source: string
  target: string
  label?: string
  color?: string
  size?: number
}

interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

interface SigmaGraphProps {
  data: GraphData | null
  onNodeClick?: (nodeId: string, nodeData: GraphNode) => void
  onEdgeClick?: (edgeId: string, edgeData: GraphEdge) => void
  onNodeHover?: (nodeId: string | null) => void
  height?: string | number
  width?: string | number
  className?: string
  nodeColorMap?: Record<string, string>
  showLabels?: boolean
  enableZoom?: boolean
  enablePan?: boolean
  layout?: 'force' | 'circular' | 'random' | 'hierarchical'
}

// GitNexus-inspired vibrant color scheme for clinical nodes
const DEFAULT_NODE_COLORS: Record<string, string> = {
  Patient: '#3b82f6',           // Blue (vibrant)
  Diagnosis: '#eab308',         // Yellow
  Treatment: '#14b8a6',         // Teal
  TreatmentDecision: '#a855f7', // Purple (prominent)
  Biomarker: '#10b981',         // Emerald
  Guideline: '#f59e0b',         // Amber (stands out)
  ClinicalTrial: '#f43f5e',     // Rose
  Comorbidity: '#f97316',       // Orange
  SNOMED: '#06b6d4',            // Cyan
  Inference: '#8b5cf6',         // Violet
  Procedure: '#ec4899',         // Pink
  Medication: '#22c55e',        // Green
  default: '#64748b'            // Slate (muted)
}

export function SigmaGraph({
  data,
  onNodeClick,
  onEdgeClick,
  onNodeHover,
  height = '100%',
  width = '100%',
  className = '',
  nodeColorMap = DEFAULT_NODE_COLORS,
  showLabels = true,
  enableZoom = true,
  enablePan = true,
  layout = 'force'
}: SigmaGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [showNodeDetails, setShowNodeDetails] = useState(false)
  const [showExplorer, setShowExplorer] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedTypes, setSelectedTypes] = useState<Set<string>>(new Set())
  const [minConnections, setMinConnections] = useState(0)
  const [currentLayout, setCurrentLayout] = useState<'force' | 'circular' | 'random' | 'hierarchical'>(layout)
  const [showStats, setShowStats] = useState(false)
  const [selectedRelationships, setSelectedRelationships] = useState<Set<string>>(new Set())
  const [focusDepth, setFocusDepth] = useState(0) // 0 = show all
  const [focusNodeId, setFocusNodeId] = useState<string | null>(null)

  // Graph state
  const [processedNodes, setProcessedNodes] = useState<Map<string, GraphNode>>(new Map())
  const [processedEdges, setProcessedEdges] = useState<GraphEdge[]>([])

  // Camera/viewport state
  const [camera, setCamera] = useState({ x: 0, y: 0, zoom: 1 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })

  // Process graph data and apply layout
  useEffect(() => {
    if (!data || data.nodes.length === 0) {
      setProcessedNodes(new Map())
      setProcessedEdges([])
      setIsLoading(false)
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      // Process nodes with positions and colors
      const nodeMap = new Map<string, GraphNode>()
      const positions = calculateLayout(data.nodes, data.edges, currentLayout)

      data.nodes.forEach((node, index) => {
        const nodeType = node.type || 'default'
        const color = nodeColorMap[nodeType] || nodeColorMap.default || '#A0AEC0'
        const pos = positions[index] || { x: 0, y: 0 }

        nodeMap.set(node.id, {
          ...node,
          x: pos.x,
          y: pos.y,
          color,
          size: node.size || getNodeSize(node),
          label: node.label || node.id
        })
      })

      // Process edges
      const edges = data.edges.map(edge => ({
        ...edge,
        color: edge.color || '#6B7280',
        size: edge.size || 1
      }))

      setProcessedNodes(nodeMap)
      setProcessedEdges(edges)
      setIsLoading(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process graph')
      setIsLoading(false)
    }
  }, [data, currentLayout, nodeColorMap])

  // Calculate node size based on type/importance
  const getNodeSize = (node: GraphNode): number => {
    const typeSize: Record<string, number> = {
      Patient: 12,
      TreatmentDecision: 10,
      Treatment: 8,
      Biomarker: 7,
      Guideline: 7,
      default: 6
    }
    return typeSize[node.type || 'default'] || 6
  }

  // Calculate layout positions
  const calculateLayout = (
    nodes: GraphNode[],
    edges: GraphEdge[],
    layoutType: string
  ): Array<{ x: number; y: number }> => {
    const count = nodes.length
    if (count === 0) return []

    switch (layoutType) {
      case 'circular': {
        return nodes.map((_, i) => {
          const angle = (2 * Math.PI * i) / count
          const radius = Math.min(200, 50 + count * 5)
          return {
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius
          }
        })
      }

      case 'force': {
        // Simple force-directed layout
        const positions = nodes.map((_, i) => ({
          x: (Math.random() - 0.5) * 400,
          y: (Math.random() - 0.5) * 400
        }))

        // Build adjacency for force calculation
        const adjacency = new Map<string, Set<string>>()
        nodes.forEach(n => adjacency.set(n.id, new Set()))
        edges.forEach(e => {
          adjacency.get(e.source)?.add(e.target)
          adjacency.get(e.target)?.add(e.source)
        })

        // Iterate force simulation
        const iterations = 50
        const k = Math.sqrt((400 * 400) / count) // Optimal distance

        for (let iter = 0; iter < iterations; iter++) {
          const cooling = 1 - iter / iterations

          // Calculate repulsive forces
          const forces = nodes.map(() => ({ x: 0, y: 0 }))

          for (let i = 0; i < count; i++) {
            for (let j = i + 1; j < count; j++) {
              const dx = positions[i].x - positions[j].x
              const dy = positions[i].y - positions[j].y
              const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy))
              const force = (k * k) / dist

              forces[i].x += (dx / dist) * force
              forces[i].y += (dy / dist) * force
              forces[j].x -= (dx / dist) * force
              forces[j].y -= (dy / dist) * force
            }
          }

          // Calculate attractive forces along edges
          edges.forEach(edge => {
            const sourceIdx = nodes.findIndex(n => n.id === edge.source)
            const targetIdx = nodes.findIndex(n => n.id === edge.target)
            if (sourceIdx === -1 || targetIdx === -1) return

            const dx = positions[targetIdx].x - positions[sourceIdx].x
            const dy = positions[targetIdx].y - positions[sourceIdx].y
            const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy))
            const force = (dist * dist) / k

            forces[sourceIdx].x += (dx / dist) * force * 0.5
            forces[sourceIdx].y += (dy / dist) * force * 0.5
            forces[targetIdx].x -= (dx / dist) * force * 0.5
            forces[targetIdx].y -= (dy / dist) * force * 0.5
          })

          // Apply forces with cooling
          for (let i = 0; i < count; i++) {
            const magnitude = Math.sqrt(forces[i].x ** 2 + forces[i].y ** 2)
            if (magnitude > 0) {
              const maxDisp = 10 * cooling
              const disp = Math.min(magnitude, maxDisp)
              positions[i].x += (forces[i].x / magnitude) * disp
              positions[i].y += (forces[i].y / magnitude) * disp
            }
          }
        }

        return positions
      }

      case 'hierarchical': {
        // Simple hierarchical layout - group by type and arrange in levels
        const typeGroups = new Map<string, GraphNode[]>()
        nodes.forEach(node => {
          const type = node.type || 'default'
          if (!typeGroups.has(type)) {
            typeGroups.set(type, [])
          }
          typeGroups.get(type)!.push(node)
        })

        const positions: Array<{ x: number; y: number }> = []
        const types = Array.from(typeGroups.keys())
        const levelHeight = 150
        const nodeSpacing = 100

        types.forEach((type, levelIndex) => {
          const nodesInLevel = typeGroups.get(type)!
          const levelWidth = nodesInLevel.length * nodeSpacing
          const startX = -levelWidth / 2

          nodesInLevel.forEach((_, nodeIndex) => {
            positions.push({
              x: startX + nodeIndex * nodeSpacing,
              y: levelIndex * levelHeight - (types.length * levelHeight) / 2
            })
          })
        })

        return positions
      }

      case 'random':
      default:
        return nodes.map(() => ({
          x: (Math.random() - 0.5) * 400,
          y: (Math.random() - 0.5) * 400
        }))
    }
  }

  // Render the graph
  const render = useCallback(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx || processedNodes.size === 0) return

    const { width: w, height: h } = canvas.getBoundingClientRect()
    canvas.width = w * window.devicePixelRatio
    canvas.height = h * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    // Clear canvas
    ctx.fillStyle = '#0d0d12'
    ctx.fillRect(0, 0, w, h)

    // Transform for camera
    ctx.save()
    ctx.translate(w / 2 + camera.x, h / 2 + camera.y)
    ctx.scale(camera.zoom, camera.zoom)

    // Draw edges
    ctx.lineWidth = 1 / camera.zoom
    processedEdges.forEach(edge => {
      const source = processedNodes.get(edge.source)
      const target = processedNodes.get(edge.target)
      if (!source || !target) return

      ctx.beginPath()
      ctx.strokeStyle = edge.color || '#4B5563'
      ctx.globalAlpha = hoveredNode
        ? (edge.source === hoveredNode || edge.target === hoveredNode ? 1 : 0.2)
        : 0.6
      ctx.moveTo(source.x || 0, source.y || 0)
      ctx.lineTo(target.x || 0, target.y || 0)
      ctx.stroke()
    })

    ctx.globalAlpha = 1

    // Draw nodes
    processedNodes.forEach((node, nodeId) => {
      const x = node.x || 0
      const y = node.y || 0
      const size = (node.size || 6) / camera.zoom
      const isHovered = hoveredNode === nodeId
      const isSelected = selectedNode === nodeId

      // Node circle
      ctx.beginPath()
      ctx.arc(x, y, size * (isHovered ? 1.3 : 1), 0, Math.PI * 2)

      // Fill
      ctx.fillStyle = node.color || '#A0AEC0'
      ctx.globalAlpha = hoveredNode && !isHovered ? 0.4 : 1
      ctx.fill()

      // Border for selected/hovered
      if (isSelected || isHovered) {
        ctx.strokeStyle = '#fff'
        ctx.lineWidth = 2 / camera.zoom
        ctx.stroke()
      }

      // Label
      if (showLabels && (isHovered || camera.zoom > 0.8)) {
        ctx.globalAlpha = 1
        ctx.fillStyle = '#fff'
        ctx.font = `${10 / camera.zoom}px sans-serif`
        ctx.textAlign = 'center'
        ctx.textBaseline = 'top'
        ctx.fillText(node.label || node.id, x, y + size + 4 / camera.zoom)
      }
    })

    ctx.restore()

    // Draw legend
    drawLegend(ctx, w, h)
  }, [processedNodes, processedEdges, camera, hoveredNode, selectedNode, showLabels])

  // Draw legend
  const drawLegend = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
    const types = Array.from(new Set(Array.from(processedNodes.values()).map(n => n.type || 'default')))
    if (types.length === 0) return

    const padding = 10
    const itemHeight = 20
    const legendHeight = types.length * itemHeight + padding * 2
    const legendWidth = 120

    ctx.fillStyle = 'rgba(24, 24, 27, 0.9)'
    ctx.fillRect(w - legendWidth - padding, padding, legendWidth, legendHeight)

    types.forEach((type, i) => {
      const y = padding + padding + i * itemHeight
      const color = nodeColorMap[type] || nodeColorMap.default || '#A0AEC0'

      ctx.beginPath()
      ctx.arc(w - legendWidth, y + 6, 5, 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.fill()

      ctx.fillStyle = '#e4e4e7'
      ctx.font = '11px sans-serif'
      ctx.textAlign = 'left'
      ctx.textBaseline = 'middle'
      ctx.fillText(type, w - legendWidth + 12, y + 6)
    })
  }

  // Re-render on state changes
  useEffect(() => {
    render()
  }, [render])

  // Handle mouse events
  const getNodeAtPosition = useCallback((clientX: number, clientY: number): string | null => {
    const canvas = canvasRef.current
    if (!canvas) return null

    const rect = canvas.getBoundingClientRect()
    const x = (clientX - rect.left - rect.width / 2 - camera.x) / camera.zoom
    const y = (clientY - rect.top - rect.height / 2 - camera.y) / camera.zoom

    for (const [nodeId, node] of processedNodes) {
      const dx = (node.x || 0) - x
      const dy = (node.y || 0) - y
      const dist = Math.sqrt(dx * dx + dy * dy)
      const size = (node.size || 6) / camera.zoom

      if (dist <= size * 1.5) {
        return nodeId
      }
    }

    return null
  }, [processedNodes, camera])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging && enablePan) {
      const dx = e.clientX - dragStart.x
      const dy = e.clientY - dragStart.y
      setCamera(prev => ({
        ...prev,
        x: prev.x + dx,
        y: prev.y + dy
      }))
      setDragStart({ x: e.clientX, y: e.clientY })
    } else {
      const nodeId = getNodeAtPosition(e.clientX, e.clientY)
      setHoveredNode(nodeId)
      onNodeHover?.(nodeId)
    }
  }, [isDragging, enablePan, dragStart, getNodeAtPosition, onNodeHover])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const nodeId = getNodeAtPosition(e.clientX, e.clientY)
    if (nodeId) {
      setSelectedNode(nodeId)
      setFocusNodeId(nodeId)
      setShowNodeDetails(true)
      const node = processedNodes.get(nodeId)
      if (node) {
        onNodeClick?.(nodeId, node)
      }
    } else {
      setIsDragging(true)
      setDragStart({ x: e.clientX, y: e.clientY })
    }
  }, [getNodeAtPosition, processedNodes, onNodeClick])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (!enableZoom) return
    e.preventDefault()

    const delta = e.deltaY > 0 ? 0.9 : 1.1
    setCamera(prev => ({
      ...prev,
      zoom: Math.max(0.1, Math.min(3, prev.zoom * delta))
    }))
  }, [enableZoom])

  // Reset view
  const resetView = useCallback(() => {
    setCamera({ x: 0, y: 0, zoom: 1 })
  }, [])

  // Get nodes within focus depth from a node
  const getNodesWithinDepth = useCallback((startNodeId: string, depth: number): Set<string> => {
    if (depth === 0) return new Set(Array.from(processedNodes.keys()))

    const visited = new Set<string>([startNodeId])
    let currentLevel = new Set([startNodeId])

    for (let i = 0; i < depth; i++) {
      const nextLevel = new Set<string>()
      currentLevel.forEach(nodeId => {
        processedEdges.forEach(edge => {
          if (edge.source === nodeId && !visited.has(edge.target)) {
            nextLevel.add(edge.target)
            visited.add(edge.target)
          }
          if (edge.target === nodeId && !visited.has(edge.source)) {
            nextLevel.add(edge.source)
            visited.add(edge.source)
          }
        })
      })
      currentLevel = nextLevel
    }

    return visited
  }, [processedNodes, processedEdges])

  // Filter nodes based on search and filters
  const filteredNodes = React.useMemo(() => {
    let nodes = Array.from(processedNodes.entries())

    // Focus depth filter (if a node is selected)
    if (focusNodeId && focusDepth > 0) {
      const nodesInRange = getNodesWithinDepth(focusNodeId, focusDepth)
      nodes = nodes.filter(([nodeId, _]) => nodesInRange.has(nodeId))
    }

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      nodes = nodes.filter(([_, node]) =>
        node.label?.toLowerCase().includes(query) ||
        node.id.toLowerCase().includes(query) ||
        node.type?.toLowerCase().includes(query)
      )
    }

    // Type filter
    if (selectedTypes.size > 0) {
      nodes = nodes.filter(([_, node]) => selectedTypes.has(node.type || 'default'))
    }

    // Connection filter
    if (minConnections > 0) {
      nodes = nodes.filter(([nodeId, _]) => {
        const connections = processedEdges.filter(
          e => e.source === nodeId || e.target === nodeId
        ).length
        return connections >= minConnections
      })
    }

    // Relationship filter (only show nodes connected by selected relationship types)
    if (selectedRelationships.size > 0) {
      const nodeIdsWithSelectedRels = new Set<string>()
      processedEdges.forEach(edge => {
        if (selectedRelationships.has(edge.label || 'unlabeled')) {
          nodeIdsWithSelectedRels.add(edge.source)
          nodeIdsWithSelectedRels.add(edge.target)
        }
      })
      nodes = nodes.filter(([nodeId, _]) => nodeIdsWithSelectedRels.has(nodeId))
    }

    return nodes
  }, [processedNodes, searchQuery, selectedTypes, minConnections, processedEdges, selectedRelationships, focusNodeId, focusDepth, getNodesWithinDepth])

  // Get unique node types
  const nodeTypes = React.useMemo(() => {
    const types = new Set<string>()
    processedNodes.forEach(node => {
      types.add(node.type || 'default')
    })
    return Array.from(types).sort()
  }, [processedNodes])

  // Get unique relationship types
  const relationshipTypes = React.useMemo(() => {
    const types = new Set<string>()
    processedEdges.forEach(edge => {
      types.add(edge.label || 'unlabeled')
    })
    return Array.from(types).sort()
  }, [processedEdges])

  // Toggle type filter
  const toggleTypeFilter = (type: string) => {
    setSelectedTypes(prev => {
      const newSet = new Set(prev)
      if (newSet.has(type)) {
        newSet.delete(type)
      } else {
        newSet.add(type)
      }
      return newSet
    })
  }

  // Toggle relationship filter
  const toggleRelationshipFilter = (type: string) => {
    setSelectedRelationships(prev => {
      const newSet = new Set(prev)
      if (newSet.has(type)) {
        newSet.delete(type)
      } else {
        newSet.add(type)
      }
      return newSet
    })
  }

  // Clear all filters
  const clearFilters = () => {
    setSearchQuery('')
    setSelectedTypes(new Set())
    setMinConnections(0)
    setSelectedRelationships(new Set())
    setFocusDepth(0)
    setFocusNodeId(null)
  }

  // Export graph as PNG
  const exportAsPNG = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const link = document.createElement('a')
    link.download = `graph-${Date.now()}.png`
    link.href = canvas.toDataURL('image/png')
    link.click()
  }, [])

  // Export graph data as JSON
  const exportAsJSON = useCallback(() => {
    if (!data) return

    const jsonStr = JSON.stringify(data, null, 2)
    const blob = new Blob([jsonStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.download = `graph-data-${Date.now()}.json`
    link.href = url
    link.click()
    URL.revokeObjectURL(url)
  }, [data])

  // Calculate graph statistics
  const graphStats = React.useMemo(() => {
    if (processedNodes.size === 0) return null

    const degrees = new Map<string, number>()
    processedNodes.forEach((_, nodeId) => {
      const degree = processedEdges.filter(
        e => e.source === nodeId || e.target === nodeId
      ).length
      degrees.set(nodeId, degree)
    })

    const degreeValues = Array.from(degrees.values())
    const avgDegree = degreeValues.reduce((a, b) => a + b, 0) / degreeValues.length
    const maxDegree = Math.max(...degreeValues)
    const minDegree = Math.min(...degreeValues)

    return {
      nodeCount: processedNodes.size,
      edgeCount: processedEdges.length,
      avgDegree: avgDegree.toFixed(2),
      maxDegree,
      minDegree,
      density: ((2 * processedEdges.length) / (processedNodes.size * (processedNodes.size - 1))).toFixed(4)
    }
  }, [processedNodes, processedEdges])

  if (error) {
    return (
      <div className={`flex items-center justify-center bg-zinc-900 text-red-400 ${className}`} style={{ height, width }}>
        <div className="text-center p-4">
          <svg className="w-8 h-8 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <p className="text-sm">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className={`relative bg-zinc-900 overflow-hidden ${className}`}
      style={{ height, width }}
    >
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-zinc-900/80 z-10">
          <div className="flex flex-col items-center gap-2">
            <div className="w-8 h-8 border-2 border-violet-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-zinc-400">Loading graph...</span>
          </div>
        </div>
      )}

      {/* Explorer Sidebar */}
      {showExplorer && (
        <div className="absolute top-0 left-0 h-full w-72 bg-zinc-900/95 backdrop-blur-sm border-r border-zinc-700 shadow-2xl overflow-hidden flex flex-col z-20">
          {/* Header */}
          <div className="bg-zinc-800 border-b border-zinc-700 p-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
              </svg>
              Explorer
            </h3>
            <button
              onClick={() => setShowExplorer(false)}
              className="p-1 hover:bg-zinc-700 rounded transition-colors text-zinc-400 hover:text-zinc-200"
              title="Hide explorer"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Search */}
          <div className="p-3 border-b border-zinc-700">
            <div className="relative">
              <svg className="absolute left-2.5 top-2.5 w-4 h-4 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search nodes..."
                className="w-full pl-8 pr-3 py-2 bg-zinc-800 border border-zinc-700 rounded text-sm text-zinc-200 placeholder-zinc-500 focus:outline-none focus:ring-1 focus:ring-violet-500"
              />
            </div>
          </div>

          {/* Filters */}
          <div className="border-b border-zinc-700">
            <details className="group" open>
              <summary className="p-3 cursor-pointer hover:bg-zinc-800/50 transition-colors flex items-center justify-between text-xs font-medium text-zinc-400 uppercase tracking-wider">
                <span>Filters</span>
                <svg className="w-4 h-4 group-open:rotate-180 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </summary>
              <div className="px-3 pb-3 space-y-3">
                {/* Type filters */}
                <div>
                  <label className="text-xs text-zinc-500 mb-1.5 block">Node Types</label>
                  <div className="space-y-1 max-h-32 overflow-y-auto">
                    {nodeTypes.map(type => {
                      const count = Array.from(processedNodes.values()).filter(n => (n.type || 'default') === type).length
                      return (
                        <label key={type} className="flex items-center gap-2 p-1.5 rounded hover:bg-zinc-800/50 cursor-pointer group">
                          <input
                            type="checkbox"
                            checked={selectedTypes.has(type)}
                            onChange={() => toggleTypeFilter(type)}
                            className="w-3.5 h-3.5 rounded border-zinc-600 bg-zinc-800 text-violet-500 focus:ring-violet-500 focus:ring-offset-0"
                          />
                          <div
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: nodeColorMap[type] || nodeColorMap.default }}
                          />
                          <span className="text-xs text-zinc-300 flex-1">{type}</span>
                          <span className="text-xs text-zinc-500">{count}</span>
                        </label>
                      )
                    })}
                  </div>
                </div>

                {/* Connection filter */}
                <div>
                  <label className="text-xs text-zinc-500 mb-1.5 block">Min Connections: {minConnections}</label>
                  <input
                    type="range"
                    min="0"
                    max="20"
                    value={minConnections}
                    onChange={(e) => setMinConnections(parseInt(e.target.value))}
                    className="w-full accent-violet-500"
                  />
                </div>

                {/* Relationship filters */}
                <div>
                  <label className="text-xs text-zinc-500 mb-1.5 block">Relationship Types</label>
                  <div className="space-y-1 max-h-32 overflow-y-auto">
                    {relationshipTypes.map(type => {
                      const count = processedEdges.filter(e => (e.label || 'unlabeled') === type).length
                      return (
                        <label key={type} className="flex items-center gap-2 p-1.5 rounded hover:bg-zinc-800/50 cursor-pointer group">
                          <input
                            type="checkbox"
                            checked={selectedRelationships.has(type)}
                            onChange={() => toggleRelationshipFilter(type)}
                            className="w-3.5 h-3.5 rounded border-zinc-600 bg-zinc-800 text-violet-500 focus:ring-violet-500 focus:ring-offset-0"
                          />
                          <span className="text-xs text-zinc-300 flex-1 truncate">{type}</span>
                          <span className="text-xs text-zinc-500">{count}</span>
                        </label>
                      )
                    })}
                  </div>
                </div>

                {/* Focus depth */}
                <div>
                  <label className="text-xs text-zinc-500 mb-1.5 block">
                    Focus Depth: {focusDepth === 0 ? 'All' : focusDepth}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="5"
                    value={focusDepth}
                    onChange={(e) => setFocusDepth(parseInt(e.target.value))}
                    className="w-full accent-violet-500"
                    disabled={!selectedNode}
                  />
                  <p className="text-xs text-zinc-600 mt-1">
                    {selectedNode ? (focusDepth === 0 ? 'Showing all nodes' : `Showing nodes within ${focusDepth} ${focusDepth === 1 ? 'hop' : 'hops'}`) : 'Select a node to enable'}
                  </p>
                </div>

                {/* Clear filters */}
                {(searchQuery || selectedTypes.size > 0 || minConnections > 0 || selectedRelationships.size > 0 || focusDepth > 0) && (
                  <button
                    onClick={clearFilters}
                    className="w-full py-1.5 px-3 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 rounded text-xs text-zinc-300 transition-colors"
                  >
                    Clear Filters
                  </button>
                )}
              </div>
            </details>
          </div>

          {/* Node List */}
          <div className="flex-1 overflow-y-auto p-3 space-y-1">
            <div className="text-xs text-zinc-500 mb-2">
              {filteredNodes.length} of {processedNodes.size} nodes
            </div>
            {filteredNodes.map(([nodeId, node]) => {
              const connections = processedEdges.filter(
                e => e.source === nodeId || e.target === nodeId
              ).length
              const isSelected = selectedNode === nodeId

              return (
                <button
                  key={nodeId}
                  onClick={() => {
                    setSelectedNode(nodeId)
                    setShowNodeDetails(true)
                    // Center camera on node
                    if (node.x !== undefined && node.y !== undefined) {
                      setCamera(prev => ({
                        ...prev,
                        x: -node.x! * prev.zoom,
                        y: -node.y! * prev.zoom
                      }))
                    }
                  }}
                  onMouseEnter={() => setHoveredNode(nodeId)}
                  onMouseLeave={() => setHoveredNode(null)}
                  className={`w-full text-left p-2 rounded border transition-all group ${
                    isSelected
                      ? 'bg-violet-900/30 border-violet-500'
                      : 'bg-zinc-800/30 border-zinc-700/50 hover:bg-zinc-700/50 hover:border-zinc-600'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full flex-shrink-0"
                      style={{ backgroundColor: node.color }}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-medium text-zinc-200 truncate group-hover:text-zinc-100">
                        {node.label}
                      </div>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span className="text-xs text-zinc-500">{node.type}</span>
                        <span className="text-xs text-zinc-600">•</span>
                        <span className="text-xs text-zinc-500">{connections} connections</span>
                      </div>
                    </div>
                  </div>
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Toggle Explorer Button (when hidden) */}
      {!showExplorer && (
        <button
          onClick={() => setShowExplorer(true)}
          className="absolute top-4 left-4 p-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-zinc-300 transition-colors shadow-lg z-20"
          title="Show explorer"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
          </svg>
        </button>
      )}

      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-grab active:cursor-grabbing"
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      />

      {/* Top Toolbar */}
      <div className="absolute top-4 right-4 flex items-center gap-2 z-10">
        {/* Layout Selector */}
        <select
          value={currentLayout}
          onChange={(e) => setCurrentLayout(e.target.value as any)}
          className="px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-xs text-zinc-300 focus:outline-none focus:ring-1 focus:ring-violet-500"
          title="Change layout"
        >
          <option value="force">Force-Directed</option>
          <option value="circular">Circular</option>
          <option value="hierarchical">Hierarchical</option>
          <option value="random">Random</option>
        </select>

        {/* Export Dropdown */}
        <div className="relative group">
          <button className="px-3 py-2 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 rounded-lg text-xs text-zinc-300 transition-colors flex items-center gap-1">
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Export
          </button>
          <div className="absolute right-0 mt-1 w-40 bg-zinc-800 border border-zinc-700 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all">
            <button
              onClick={exportAsPNG}
              className="w-full px-3 py-2 text-left text-xs text-zinc-300 hover:bg-zinc-700 rounded-t-lg transition-colors flex items-center gap-2"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              Export as PNG
            </button>
            <button
              onClick={exportAsJSON}
              className="w-full px-3 py-2 text-left text-xs text-zinc-300 hover:bg-zinc-700 rounded-b-lg transition-colors flex items-center gap-2"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Export as JSON
            </button>
          </div>
        </div>

        {/* Stats Toggle */}
        <button
          onClick={() => setShowStats(!showStats)}
          className={`px-3 py-2 border rounded-lg text-xs transition-colors flex items-center gap-1 ${
            showStats
              ? 'bg-violet-600 border-violet-500 text-white'
              : 'bg-zinc-800 hover:bg-zinc-700 border-zinc-700 text-zinc-300'
          }`}
          title="Toggle statistics"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Stats
        </button>
      </div>

      {/* Statistics Panel */}
      {showStats && graphStats && (
        <div className="absolute top-20 right-4 w-64 bg-zinc-900/95 backdrop-blur-sm border border-zinc-700 rounded-lg shadow-xl p-4 z-10">
          <h3 className="text-sm font-semibold text-zinc-100 mb-3 flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Graph Statistics
          </h3>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-xs text-zinc-400">Nodes</span>
              <span className="text-sm font-medium text-zinc-200">{graphStats.nodeCount}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-zinc-400">Edges</span>
              <span className="text-sm font-medium text-zinc-200">{graphStats.edgeCount}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-zinc-400">Avg Degree</span>
              <span className="text-sm font-medium text-zinc-200">{graphStats.avgDegree}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-zinc-400">Max Degree</span>
              <span className="text-sm font-medium text-zinc-200">{graphStats.maxDegree}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-zinc-400">Min Degree</span>
              <span className="text-sm font-medium text-zinc-200">{graphStats.minDegree}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-zinc-400">Density</span>
              <span className="text-sm font-medium text-zinc-200">{graphStats.density}</span>
            </div>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="absolute bottom-4 right-4 flex flex-col gap-2">
        <button
          onClick={() => setCamera(prev => ({ ...prev, zoom: Math.min(3, prev.zoom * 1.2) }))}
          className="p-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-zinc-300 transition-colors"
          title="Zoom in"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
          </svg>
        </button>
        <button
          onClick={() => setCamera(prev => ({ ...prev, zoom: Math.max(0.1, prev.zoom * 0.8) }))}
          className="p-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-zinc-300 transition-colors"
          title="Zoom out"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
          </svg>
        </button>
        <button
          onClick={resetView}
          className="p-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-zinc-300 transition-colors"
          title="Reset view"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
        </button>
      </div>

      {/* Stats */}
      <div className="absolute bottom-4 left-4 text-xs text-zinc-500">
        {processedNodes.size} nodes | {processedEdges.length} edges | {Math.round(camera.zoom * 100)}%
      </div>

      {/* Node Details Panel */}
      {showNodeDetails && selectedNode && processedNodes.get(selectedNode) && (
        <div className="absolute top-0 right-0 h-full w-80 bg-zinc-900/95 backdrop-blur-sm border-l border-zinc-700 shadow-2xl overflow-y-auto">
          <div className="sticky top-0 bg-zinc-800 border-b border-zinc-700 p-4 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-zinc-100">Node Details</h3>
            <button
              onClick={() => {
                setShowNodeDetails(false)
                setSelectedNode(null)
              }}
              className="p-1 hover:bg-zinc-700 rounded transition-colors text-zinc-400 hover:text-zinc-200"
              title="Close"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="p-4 space-y-4">
            {(() => {
              const node = processedNodes.get(selectedNode)!
              return (
                <>
                  {/* Node visual indicator */}
                  <div className="flex items-center gap-3 pb-3 border-b border-zinc-700">
                    <div
                      className="w-10 h-10 rounded-full flex items-center justify-center text-white font-medium"
                      style={{ backgroundColor: node.color }}
                    >
                      {node.label?.charAt(0).toUpperCase()}
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4 className="text-sm font-medium text-zinc-100 truncate">{node.label}</h4>
                      <p className="text-xs text-zinc-500">{node.type || 'Node'}</p>
                    </div>
                  </div>

                  {/* Node ID */}
                  <div>
                    <label className="text-xs font-medium text-zinc-400 uppercase tracking-wider">ID</label>
                    <p className="text-sm text-zinc-200 mt-1 font-mono break-all">{node.id}</p>
                  </div>

                  {/* Node Type */}
                  {node.type && (
                    <div>
                      <label className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Type</label>
                      <div className="mt-1">
                        <span className="inline-block px-2 py-1 text-xs rounded-full bg-zinc-800 text-zinc-200 border border-zinc-700">
                          {node.type}
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Properties */}
                  {node.properties && Object.keys(node.properties).length > 0 && (
                    <div>
                      <label className="text-xs font-medium text-zinc-400 uppercase tracking-wider mb-2 block">Properties</label>
                      <div className="space-y-2 bg-zinc-800/50 rounded-lg p-3 border border-zinc-700">
                        {Object.entries(node.properties).map(([key, value]) => (
                          <div key={key} className="flex flex-col gap-1">
                            <span className="text-xs text-zinc-500 font-medium">{key}</span>
                            <span className="text-sm text-zinc-200 break-all">
                              {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Connected nodes */}
                  {(() => {
                    const connectedEdges = processedEdges.filter(
                      e => e.source === selectedNode || e.target === selectedNode
                    )
                    const incomingCount = connectedEdges.filter(e => e.target === selectedNode).length
                    const outgoingCount = connectedEdges.filter(e => e.source === selectedNode).length

                    return (
                      <div>
                        <label className="text-xs font-medium text-zinc-400 uppercase tracking-wider mb-2 block">Connections</label>
                        <div className="grid grid-cols-2 gap-2">
                          <div className="bg-zinc-800/50 rounded-lg p-3 border border-zinc-700">
                            <div className="text-xs text-zinc-500 mb-1">Incoming</div>
                            <div className="text-lg font-semibold text-blue-400">{incomingCount}</div>
                          </div>
                          <div className="bg-zinc-800/50 rounded-lg p-3 border border-zinc-700">
                            <div className="text-xs text-zinc-500 mb-1">Outgoing</div>
                            <div className="text-lg font-semibold text-violet-400">{outgoingCount}</div>
                          </div>
                        </div>

                        {connectedEdges.length > 0 && (
                          <div className="mt-3 space-y-1 max-h-40 overflow-y-auto">
                            {connectedEdges.map(edge => {
                              const isIncoming = edge.target === selectedNode
                              const connectedNodeId = isIncoming ? edge.source : edge.target
                              const connectedNode = processedNodes.get(connectedNodeId)

                              return (
                                <button
                                  key={edge.id}
                                  onClick={() => {
                                    setSelectedNode(connectedNodeId)
                                  }}
                                  className="w-full text-left p-2 rounded bg-zinc-800/30 hover:bg-zinc-700/50 border border-zinc-700/50 transition-colors group"
                                >
                                  <div className="flex items-center gap-2">
                                    <span className="text-xs text-zinc-500">
                                      {isIncoming ? '←' : '→'}
                                    </span>
                                    <div className="flex-1 min-w-0">
                                      <div className="text-xs text-zinc-300 truncate group-hover:text-zinc-100">
                                        {connectedNode?.label || connectedNodeId}
                                      </div>
                                      {edge.label && (
                                        <div className="text-xs text-zinc-500 truncate">{edge.label}</div>
                                      )}
                                    </div>
                                  </div>
                                </button>
                              )
                            })}
                          </div>
                        )}
                      </div>
                    )
                  })()}
                </>
              )
            })()}
          </div>
        </div>
      )}
    </div>
  )
}

export default SigmaGraph
