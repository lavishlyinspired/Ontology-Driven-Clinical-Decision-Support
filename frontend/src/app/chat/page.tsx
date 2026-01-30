'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import ChatInterface from '@/components/ChatInterface'
import { ContextGraphView } from '@/components/ContextGraphView'
import { DecisionTracePanel } from '@/components/DecisionTracePanel'
import { Network, ChevronLeft, ChevronRight, Maximize2, Minimize2, GripVertical, X } from 'lucide-react'
import type { GraphData, GraphNode, TreatmentDecision } from '@/lib/api'

export default function ChatPage() {
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [selectedDecision, setSelectedDecision] = useState<TreatmentDecision | null>(null)
  const [graphDecisions, setGraphDecisions] = useState<GraphNode[]>([])
  const [isGraphPanelCollapsed, setIsGraphPanelCollapsed] = useState(false)
  const [activeTab, setActiveTab] = useState<'graph' | 'decisions'>('graph')
  const [isPopout, setIsPopout] = useState(false)
  const [panelWidth, setPanelWidth] = useState(40) // percentage
  const [isResizing, setIsResizing] = useState(false)

  const containerRef = useRef<HTMLDivElement>(null)
  const previousDecisionsKeyRef = useRef<string>('')

  // Handle graph data updates from chat
  const handleGraphDataChange = useCallback((data: GraphData) => {
    console.log('[ChatPage] Received graph data:', data.nodes.length, 'nodes,', data.relationships.length, 'relationships')

    // Deduplicate nodes by id
    const uniqueNodes = new Map<string, GraphNode>()
    data.nodes.forEach(node => {
      if (!uniqueNodes.has(node.id)) {
        uniqueNodes.set(node.id, node)
      }
    })

    // Deduplicate relationships by id
    const uniqueRels = new Map<string, any>()
    data.relationships.forEach(rel => {
      if (!uniqueRels.has(rel.id)) {
        uniqueRels.set(rel.id, rel)
      }
    })

    setGraphData({
      nodes: Array.from(uniqueNodes.values()),
      relationships: Array.from(uniqueRels.values())
    })
  }, [])

  // Handle decision node changes
  const handleDecisionNodesChange = useCallback((decisions: GraphNode[]) => {
    // Create a stable key for the current decisions
    const currentKey = decisions.map(d => d.id).sort().join(',')
    
    // Only process if decisions actually changed
    if (currentKey === previousDecisionsKeyRef.current) {
      return
    }
    
    console.log('[ChatPage] Received decision nodes:', decisions.length, 'decisions')
    previousDecisionsKeyRef.current = currentKey

    // Deduplicate decisions by treatment name to avoid showing duplicate recommendations
    const uniqueDecisions = new Map<string, GraphNode>()
    decisions.forEach(d => {
      const treatment = d.properties?.treatment as string || d.id
      const existingDecision = uniqueDecisions.get(treatment)

      // Keep the one with 'recommended' status or higher confidence
      if (!existingDecision) {
        uniqueDecisions.set(treatment, d)
      } else {
        const existingStatus = existingDecision.properties?.status as string
        const newStatus = d.properties?.status as string
        if (newStatus === 'recommended' && existingStatus !== 'recommended') {
          uniqueDecisions.set(treatment, d)
        }
      }
    })

    // Sort by status (recommended first) then by confidence
    const sortedDecisions = Array.from(uniqueDecisions.values()).sort((a, b) => {
      const aStatus = a.properties?.status as string
      const bStatus = b.properties?.status as string
      if (aStatus === 'recommended' && bStatus !== 'recommended') return -1
      if (bStatus === 'recommended' && aStatus !== 'recommended') return 1

      const aConf = (a.properties?.confidence_score as number) || 0
      const bConf = (b.properties?.confidence_score as number) || 0
      return bConf - aConf
    })

    setGraphDecisions(sortedDecisions)
  }, [])

  // Handle decision selection from graph
  const handleDecisionNodeClick = useCallback((node: GraphNode) => {
    const decision: TreatmentDecision = {
      id: node.id,
      decision_type: (node.properties.decision_type as string) || 'treatment_recommendation',
      category: (node.properties.category as string) || 'NSCLC',
      status: (node.properties.status as string) || 'recommended',
      reasoning: (node.properties.reasoning as string) || '',
      reasoning_summary: (node.properties.reasoning_summary as string) || '',
      treatment: (node.properties.treatment as string) || 'Unknown',
      confidence_score: (node.properties.confidence_score as number) || 0,
      confidence: (node.properties.confidence as number) || (node.properties.confidence_score as number) || 0,
      risk_factors: (node.properties.risk_factors as string[]) || [],
      guidelines_applied: node.properties.guideline_reference ? [node.properties.guideline_reference as string] : [],
      timestamp: (node.properties.decision_timestamp as string) || '',
    }
    setSelectedDecision(decision)
    setActiveTab('decisions')
  }, [])

  // Handle node click in graph
  const handleNodeClick = useCallback((nodeId: string, labels: string[]) => {
    console.log('Node clicked:', nodeId, labels)
  }, [])

  // Handle resize
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setIsResizing(true)
  }, [])

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing || !containerRef.current) return

      const containerRect = containerRef.current.getBoundingClientRect()
      const newWidth = ((containerRect.right - e.clientX) / containerRect.width) * 100

      // Clamp between 20% and 60%
      setPanelWidth(Math.min(60, Math.max(20, newWidth)))
    }

    const handleMouseUp = () => {
      setIsResizing(false)
    }

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isResizing])

  // Popout window content
  const GraphPanelContent = () => (
    <>
      {/* Panel Header */}
      <div className="flex items-center justify-between p-3 border-b border-gray-200 bg-gradient-to-r from-purple-600 to-blue-600">
        <div className="flex items-center gap-2">
          <Network className="w-5 h-5 text-white" />
          <h2 className="text-sm font-semibold text-white">Context Graph</h2>
        </div>
        <div className="flex items-center gap-2">
          {/* Tab Buttons */}
          <div className="flex bg-white/20 rounded-lg p-0.5">
            <button
              onClick={() => setActiveTab('graph')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                activeTab === 'graph'
                  ? 'bg-white text-purple-600 shadow-sm'
                  : 'text-white hover:bg-white/10'
              }`}
            >
              Graph
            </button>
            <button
              onClick={() => setActiveTab('decisions')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                activeTab === 'decisions'
                  ? 'bg-white text-purple-600 shadow-sm'
                  : 'text-white hover:bg-white/10'
              }`}
            >
              Decisions {graphDecisions.length > 0 && (
                <span className="ml-1 px-1.5 py-0.5 bg-white/30 rounded-full text-[10px]">
                  {graphDecisions.length}
                </span>
              )}
            </button>
          </div>

          {/* Popout/Minimize Button */}
          <button
            onClick={() => setIsPopout(!isPopout)}
            className="p-1.5 text-white hover:bg-white/20 rounded-md transition-colors"
            title={isPopout ? "Dock panel" : "Pop out panel"}
          >
            {isPopout ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>

          {/* Collapse Button */}
          {!isPopout && (
            <button
              onClick={() => {
                console.log('[ChatPage] Collapsing graph panel')
                setIsGraphPanelCollapsed(true)
              }}
              className="p-1.5 text-white hover:bg-white/20 rounded-md transition-colors"
              title="Collapse panel"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          )}

          {/* Close button for popout */}
          {isPopout && (
            <button
              onClick={() => setIsPopout(false)}
              className="p-1.5 text-white hover:bg-white/20 rounded-md transition-colors"
              title="Close"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Panel Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'graph' ? (
          <ContextGraphView
            graphData={graphData}
            onNodeClick={handleNodeClick}
            onGraphDataChange={handleGraphDataChange}
            onDecisionNodesChange={handleDecisionNodesChange}
            onDecisionNodeClick={handleDecisionNodeClick}
            height="100%"
            showLegend={true}
          />
        ) : (
          <DecisionTracePanel
            decision={selectedDecision}
            onDecisionSelect={setSelectedDecision}
            graphDecisions={graphDecisions}
            className="h-full overflow-auto"
          />
        )}
      </div>

      {/* Panel Footer - Graph Stats */}
      {activeTab === 'graph' && graphData && (
        <div className="p-2 border-t border-gray-200 bg-gray-50">
          <div className="flex justify-between items-center">
            <p className="text-xs text-gray-500">
              {graphData.nodes.length} nodes | {graphData.relationships.length} relationships
            </p>
            <div className="flex gap-1">
              {['Patient', 'TreatmentDecision', 'Biomarker', 'Guideline'].map(label => {
                const count = graphData.nodes.filter(n => n.labels.includes(label)).length
                if (count === 0) return null
                return (
                  <span
                    key={label}
                    className="px-1.5 py-0.5 text-[10px] rounded bg-gray-200 text-gray-600"
                  >
                    {label.replace('TreatmentDecision', 'Decision')}: {count}
                  </span>
                )
              })}
            </div>
          </div>
        </div>
      )}
    </>
  )

  return (
    <div ref={containerRef} className="flex h-screen bg-gray-100 relative overflow-hidden">
      {/* Chat Panel */}
      <div
        className="flex flex-col transition-all duration-300 ease-in-out min-w-0"
        style={{
          width: isGraphPanelCollapsed || isPopout ? '100%' : `${100 - panelWidth}%`,
          flexShrink: 0
        }}
      >
        <ChatInterface
          onGraphDataChange={handleGraphDataChange}
          onDecisionNodesChange={handleDecisionNodesChange}
        />
      </div>

      {/* Graph Panel Toggle Button (when collapsed) */}
      {isGraphPanelCollapsed && !isPopout && (
        <button
          onClick={() => {
            console.log('[ChatPage] Opening graph panel')
            setIsGraphPanelCollapsed(false)
          }}
          className="absolute right-0 top-1/2 transform -translate-y-1/2 z-30 bg-gradient-to-r from-purple-600 to-blue-600 text-white p-3 rounded-l-xl shadow-lg hover:from-purple-700 hover:to-blue-700 transition-all group"
          title="Show Context Graph"
        >
          <ChevronLeft className="w-5 h-5 group-hover:animate-pulse" />
          <span className="absolute right-12 top-1/2 -translate-y-1/2 bg-gray-900 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
            Open Graph Panel
          </span>
        </button>
      )}

      {/* Resizer Handle */}
      {!isGraphPanelCollapsed && !isPopout && (
        <div
          className={`w-1 bg-gray-200 hover:bg-blue-400 cursor-col-resize flex items-center justify-center transition-colors ${
            isResizing ? 'bg-blue-500' : ''
          }`}
          onMouseDown={handleMouseDown}
        >
          <div className="w-4 h-8 flex items-center justify-center rounded bg-gray-300 hover:bg-blue-400 transition-colors">
            <GripVertical className="w-3 h-3 text-gray-500" />
          </div>
        </div>
      )}

      {/* Graph Panel - Inline */}
      {!isGraphPanelCollapsed && !isPopout && (
        <div
          className="flex flex-col border-l border-gray-200 bg-white shadow-lg"
          style={{ width: `${panelWidth}%` }}
        >
          <GraphPanelContent />
        </div>
      )}

      {/* Graph Panel - Popout */}
      {isPopout && (
        <div className="fixed inset-4 z-50 flex flex-col bg-white rounded-xl shadow-2xl border border-gray-300 overflow-hidden">
          <GraphPanelContent />
        </div>
      )}

      {/* Popout backdrop */}
      {isPopout && (
        <div
          className="fixed inset-0 bg-black/20 z-40"
          onClick={() => setIsPopout(false)}
        />
      )}

      {/* Resize overlay */}
      {isResizing && (
        <div className="fixed inset-0 z-30 cursor-col-resize" />
      )}
    </div>
  )
}
