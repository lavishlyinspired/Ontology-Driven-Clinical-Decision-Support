'use client'

import { useState, useCallback, useRef } from 'react'
import { MessageSquare, X, Maximize2, Minimize2, ListTree, Activity, Shield, Network, GripVertical } from 'lucide-react'
import ChatInterface from './ChatInterface'
import { ActivityFeed } from './ActivityFeed'
import { DecisionTracePanel } from './DecisionTracePanel'
import { ArgumentationView, DEMO_ARGUMENTATION_CHAIN } from './ArgumentationView'
import { SigmaGraph } from './SigmaGraph'
import { useApp } from '@/contexts/AppContext'
import type { GraphData, GraphNode, TreatmentDecision } from '@/lib/api'

type FloatingTab = 'chat' | 'graph' | 'decisions' | 'activity' | 'arguments'

export function FloatingChatButton() {
  const {
    graphData,
    setGraphData,
    decisionNodes: globalDecisionNodes,
    setDecisionNodes: setGlobalDecisionNodes,
    selectedDecision: globalSelectedDecision,
    setSelectedDecision: setGlobalSelectedDecision,
    argumentationChain: globalArgumentationChain,
    setArgumentationChain: setGlobalArgumentationChain
  } = useApp()

  const [isOpen, setIsOpen] = useState(false)
  const [isMaximized, setIsMaximized] = useState(false)
  const [isMinimized, setIsMinimized] = useState(false)
  const [activeTab, setActiveTab] = useState<FloatingTab>('chat')

  // Position and size state
  const [position, setPosition] = useState({ x: 0, y: 64 }) // Start at top: 64px (4rem)
  const [size, setSize] = useState({ width: 520, height: 0 }) // Height will be calc'd
  const [isDragging, setIsDragging] = useState(false)
  const [isResizing, setIsResizing] = useState(false)
  const dragStart = useRef({ x: 0, y: 0 })
  const resizeStart = useRef({ width: 0, height: 0, mouseX: 0, mouseY: 0 })
  const [activities, setActivities] = useState<Array<{
    id: string
    type: 'query' | 'decision' | 'biomarker' | 'treatment' | 'alert' | 'guideline' | 'system'
    title: string
    description: string
    timestamp: Date
    status: 'info' | 'success' | 'warning' | 'error'
  }>>([
    {
      id: 'welcome',
      type: 'system',
      title: 'LCA Assistant Ready',
      description: 'Your personal lung cancer assistant is ready',
      timestamp: new Date(),
      status: 'success'
    }
  ])

  // Handlers for graph data changes
  const handleGraphDataChange = (data: GraphData) => {
    console.log('[FloatingChat] Graph data received:', data.nodes.length, 'nodes')
    setGraphData(data)  // Update global context
    addActivity('system', 'Graph Updated', `${data.nodes.length} nodes analyzed`, 'success')
  }

  const handleDecisionNodesChange = useCallback((decisions: GraphNode[]) => {
    console.log('[FloatingChat] Decision nodes received:', decisions.length)
    setGlobalDecisionNodes(decisions)  // Update global context
    if (decisions.length > 0) {
      addActivity('decision', 'Decisions Available', `${decisions.length} treatment options found`, 'success')
    }
  }, [setGlobalDecisionNodes])

  const handleDecisionSelect = useCallback((decision: TreatmentDecision | null) => {
    setGlobalSelectedDecision(decision)  // Update global context
    if (decision) {
      const newArgumentationChain = {
        ...DEMO_ARGUMENTATION_CHAIN,
        treatment: decision.treatment || DEMO_ARGUMENTATION_CHAIN.treatment,
        conclusion: `${decision.treatment || 'Treatment'} is recommended based on clinical analysis.`,
        confidence: decision.confidence_score ? Math.round(decision.confidence_score * 100) : 85
      }
      setGlobalArgumentationChain(newArgumentationChain)  // Update global context
      setActiveTab('arguments')
    }
  }, [setGlobalSelectedDecision, setGlobalArgumentationChain])

  const addActivity = (type: string, title: string, description: string, status: 'info' | 'success' | 'warning' | 'error' = 'info') => {
    setActivities(prev => [{
      id: Date.now().toString(),
      type: type as any,
      title,
      description,
      timestamp: new Date(),
      status
    }, ...prev.slice(0, 49)])
  }

  // Drag handlers
  const handleDragStart = (e: React.MouseEvent) => {
    if (isMaximized) return
    setIsDragging(true)
    dragStart.current = {
      x: e.clientX - position.x,
      y: e.clientY - position.y
    }
  }

  const handleDrag = useCallback((e: MouseEvent) => {
    if (!isDragging || isMaximized) return
    const newX = e.clientX - dragStart.current.x
    const newY = e.clientY - dragStart.current.y
    setPosition({ x: Math.max(0, newX), y: Math.max(0, newY) })
  }, [isDragging, isMaximized])

  const handleDragEnd = () => {
    setIsDragging(false)
  }

  // Resize handlers
  const handleResizeStart = (e: React.MouseEvent) => {
    if (isMaximized) return
    e.preventDefault()
    setIsResizing(true)
    resizeStart.current = {
      width: size.width,
      height: size.height,
      mouseX: e.clientX,
      mouseY: e.clientY
    }
  }

  const handleResize = useCallback((e: MouseEvent) => {
    if (!isResizing || isMaximized) return
    const deltaX = e.clientX - resizeStart.current.mouseX
    const deltaY = e.clientY - resizeStart.current.mouseY
    setSize({
      width: Math.max(400, resizeStart.current.width + deltaX),
      height: Math.max(400, resizeStart.current.height + deltaY)
    })
  }, [isResizing, isMaximized])

  const handleResizeEnd = () => {
    setIsResizing(false)
  }

  // Mouse event listeners
  useCallback(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleDrag)
      document.addEventListener('mouseup', handleDragEnd)
    } else {
      document.removeEventListener('mousemove', handleDrag)
      document.removeEventListener('mouseup', handleDragEnd)
    }
    return () => {
      document.removeEventListener('mousemove', handleDrag)
      document.removeEventListener('mouseup', handleDragEnd)
    }
  }, [isDragging, handleDrag])()

  useCallback(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleResize)
      document.addEventListener('mouseup', handleResizeEnd)
    } else {
      document.removeEventListener('mousemove', handleResize)
      document.removeEventListener('mouseup', handleResizeEnd)
    }
    return () => {
      document.removeEventListener('mousemove', handleResize)
      document.removeEventListener('mouseup', handleResizeEnd)
    }
  }, [isResizing, handleResize])()

  if (!isOpen || isMinimized) {
    return (
      <button
        onClick={() => { setIsOpen(true); setIsMinimized(false) }}
        className="fixed bottom-6 right-6 z-50 w-14 h-14 bg-gradient-to-br from-violet-500 to-purple-600 text-white rounded-full shadow-lg hover:shadow-xl hover:scale-110 transition-all flex items-center justify-center group"
        title="Open LCA Assistant"
      >
        <MessageSquare className="w-6 h-6" />
        <span className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full animate-pulse"></span>
      </button>
    )
  }

  const panelStyle = isMaximized
    ? { top: '4rem', bottom: '1.5rem', left: '1.5rem', right: '1.5rem' }
    : {
        top: `${position.y}px`,
        right: '1.5rem',
        bottom: '1.5rem',
        width: `${size.width}px`
      }

  return (
    <div
      className="fixed z-50 bg-zinc-900 rounded-xl shadow-2xl border border-zinc-700 flex flex-col"
      style={panelStyle}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3 border-b border-zinc-700 bg-gradient-to-r from-violet-600 via-purple-600 to-violet-600 rounded-t-xl flex-shrink-0 cursor-move select-none"
        onMouseDown={handleDragStart}
      >
        <div className="flex items-center gap-2">
          <GripVertical className="w-4 h-4 text-white/60" />
          <MessageSquare className="w-5 h-5 text-white" />
          <h3 className="text-sm font-bold text-white">LCA Assistant</h3>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={(e) => { e.stopPropagation(); setIsMinimized(true) }}
            className="p-1.5 hover:bg-white/20 rounded transition-colors"
            title="Minimize to icon"
          >
            <Minimize2 className="w-4 h-4 text-white" />
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); setIsMaximized(!isMaximized) }}
            className="p-1.5 hover:bg-white/20 rounded transition-colors"
            title={isMaximized ? 'Restore' : 'Maximize'}
          >
            <Maximize2 className="w-4 h-4 text-white" />
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); setIsOpen(false) }}
            className="p-1.5 hover:bg-white/20 rounded transition-colors"
            title="Close"
          >
            <X className="w-4 h-4 text-white" />
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex items-center gap-1 p-2 border-b border-zinc-700 bg-zinc-800 flex-shrink-0">
        <button
          onClick={() => setActiveTab('chat')}
          className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-md transition-colors ${
            activeTab === 'chat'
              ? 'bg-violet-600/20 text-violet-400 border border-violet-600/30'
              : 'text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200'
          }`}
        >
          <MessageSquare className="w-4 h-4" />
          <span>Chat</span>
        </button>
        <button
          onClick={() => setActiveTab('graph')}
          className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-md transition-colors ${
            activeTab === 'graph'
              ? 'bg-violet-600/20 text-violet-400 border border-violet-600/30'
              : 'text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200'
          }`}
        >
          <Network className="w-4 h-4" />
          <span>Graph</span>
        </button>
        <button
          onClick={() => setActiveTab('decisions')}
          className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-md transition-colors ${
            activeTab === 'decisions'
              ? 'bg-violet-600/20 text-violet-400 border border-violet-600/30'
              : 'text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200'
          }`}
        >
          <ListTree className="w-4 h-4" />
          <span>Decisions</span>
          {globalDecisionNodes.length > 0 && (
            <span className="ml-1 px-1.5 py-0.5 text-[10px] bg-violet-600 text-white rounded-full">
              {globalDecisionNodes.length}
            </span>
          )}
        </button>
        <button
          onClick={() => setActiveTab('activity')}
          className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-md transition-colors ${
            activeTab === 'activity'
              ? 'bg-violet-600/20 text-violet-400 border border-violet-600/30'
              : 'text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200'
          }`}
        >
          <Activity className="w-4 h-4" />
          <span>Activity</span>
        </button>
        <button
          onClick={() => setActiveTab('arguments')}
          className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-md transition-colors ${
            activeTab === 'arguments'
              ? 'bg-violet-600/20 text-violet-400 border border-violet-600/30'
              : 'text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200'
          }`}
        >
          <Shield className="w-4 h-4" />
          <span>Arguments</span>
        </button>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-hidden">
        <div style={{ display: activeTab === 'chat' ? 'flex' : 'none', height: '100%', flexDirection: 'column' }}>
          <ChatInterface
            onGraphDataChange={handleGraphDataChange}
            onDecisionNodesChange={handleDecisionNodesChange}
          />
        </div>

        <div style={{ display: activeTab === 'graph' ? 'block' : 'none', height: '100%' }}>
          {graphData && graphData.nodes.length > 0 ? (
            <SigmaGraph
              data={{
                nodes: graphData.nodes.map(n => ({
                  id: n.id,
                  label: n.properties?.name as string || n.properties?.treatment as string || n.id.split(':').pop() || n.id,
                  type: n.labels[0],
                  properties: n.properties,
                  color: undefined,
                  size: undefined
                })),
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
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-zinc-500 p-8">
              <Network className="w-16 h-16 mb-4 opacity-50" />
              <p className="text-sm font-medium mb-2">No graph data available</p>
              <p className="text-xs text-center">Start a conversation in the Chat tab to generate and visualize the knowledge graph</p>
            </div>
          )}
        </div>

        <div style={{ display: activeTab === 'decisions' ? 'block' : 'none', height: '100%' }}>
          <DecisionTracePanel
            decision={globalSelectedDecision}
            onDecisionSelect={handleDecisionSelect}
            graphDecisions={globalDecisionNodes}
            className="h-full"
          />
        </div>

        <div style={{ display: activeTab === 'activity' ? 'block' : 'none', height: '100%' }}>
          <ActivityFeed
            activities={activities}
            className="h-full"
            onActivityClick={(activity) => {
              console.log('Activity clicked:', activity)
            }}
          />
        </div>

        <div style={{ display: activeTab === 'arguments' ? 'block' : 'none', height: '100%', overflow: 'auto' }}>
          <div className="p-4">
            {globalArgumentationChain ? (
              <ArgumentationView
                chain={globalArgumentationChain}
                onArgumentClick={(arg) => {
                  console.log('Argument clicked:', arg)
                  addActivity('decision', 'Argument Reviewed', arg.claim, 'info')
                }}
                onReferenceClick={(ref) => {
                  if (ref.startsWith('http')) {
                    window.open(ref, '_blank')
                  }
                }}
              />
            ) : (
              <div className="text-center text-zinc-500 py-8">
                <Shield className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p className="text-sm">No argumentation data available</p>
                <p className="text-xs mt-2">Select a decision to view supporting arguments</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Resize Handle - bottom-right corner */}
      {!isMaximized && (
        <div
          className="absolute bottom-0 right-0 w-4 h-4 cursor-nwse-resize"
          onMouseDown={handleResizeStart}
        >
          <div className="absolute bottom-1 right-1 w-3 h-3 border-r-2 border-b-2 border-violet-500/50"></div>
        </div>
      )}
    </div>
  )
}
