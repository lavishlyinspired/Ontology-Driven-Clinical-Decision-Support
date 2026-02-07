'use client'

import { useState, useCallback, useEffect } from 'react'
import {
  Network,
  MessageSquare,
  ChevronLeft,
  ChevronRight,
  Book,
  Activity,
  ListTree,
  Shield,
  Home,
  BarChart3,
  Users,
  FileText,
  Zap,
  RefreshCw,
  Filter,
  Layers
} from 'lucide-react'
import Link from 'next/link'
import { ContextGraphView } from '@/components/ContextGraphView'
import { SigmaGraph } from '@/components/SigmaGraph'
import { CytoscapeGraph } from '@/components/CytoscapeGraph'
import { OntologyBrowser } from '@/components/OntologyBrowser'
import { ActivityFeed } from '@/components/ActivityFeed'
import { DecisionTracePanel } from '@/components/DecisionTracePanel'
import { ArgumentationView, DEMO_ARGUMENTATION_CHAIN } from '@/components/ArgumentationView'
import { useApp } from '@/contexts/AppContext'
import type { GraphData, GraphNode, TreatmentDecision } from '@/lib/api'
import { getGraphData } from '@/lib/api'

// Navigation items
const navItems = [
  { href: "/", label: "Home", icon: Home },
  { href: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { href: "/patients/analyze", label: "Analyze", icon: Users },
  { href: "/guidelines", label: "Guidelines", icon: FileText },
  { href: "/chat", label: "Assistant", icon: MessageSquare, active: true },
]

export default function ChatPage() {
  const {
    graphData,
    setGraphData,
    decisionNodes,
    setDecisionNodes,
    selectedDecision,
    setSelectedDecision,
    argumentationChain,
    setArgumentationChain,
    isLeftSidebarOpen,
    setIsLeftSidebarOpen,
    isRightSidebarOpen,
    setIsRightSidebarOpen,
    activeLeftTab,
    setActiveLeftTab,
    activeRightTab,
    setActiveRightTab
  } = useApp()

  // Graph explorer state
  const [focusDepth, setFocusDepth] = useState(2)
  const [nodeLimit, setNodeLimit] = useState(100)
  const [showFilters, setShowFilters] = useState(true)
  const [isLoadingGraph, setIsLoadingGraph] = useState(false)
  const [graphUpdateTime, setGraphUpdateTime] = useState<string | null>(null)
  const [graphType, setGraphType] = useState<'sigma' | 'cytoscape'>('sigma')

  // Activity feed state - using types compatible with ActivityFeed component
  type ActivityType = 'query' | 'decision' | 'biomarker' | 'treatment' | 'alert' | 'guideline' | 'system'
  type ActivityStatus = 'info' | 'success' | 'warning' | 'error'

  const [activities, setActivities] = useState<Array<{
    id: string
    type: ActivityType
    title: string
    description: string
    timestamp: Date
    status: ActivityStatus
  }>>([
    {
      id: 'welcome',
      type: 'system',
      title: 'System Ready',
      description: 'LCA Assistant is ready for queries',
      timestamp: new Date(),
      status: 'success'
    }
  ])

  // Add activity helper - maps internal types to ActivityFeed compatible types
  const addActivity = useCallback((internalType: string, title: string, description: string, status: ActivityStatus = 'info') => {
    // Map internal types to ActivityFeed compatible types
    const typeMap: Record<string, ActivityType> = {
      'system': 'system',
      'decision': 'decision',
      'query': 'query',
      'ontology_lookup': 'query',
      'error': 'alert',
      'graph_update': 'system'
    }
    const type = typeMap[internalType] || 'system'

    setActivities(prev => [{
      id: Date.now().toString(),
      type,
      title,
      description,
      timestamp: new Date(),
      status
    }, ...prev.slice(0, 49)])
  }, [])

  // Update graph timestamp when data changes (from floating chat or manual load)
  useEffect(() => {
    if (graphData && graphData.nodes.length > 0) {
      setGraphUpdateTime(new Date().toLocaleTimeString())
    }
  }, [graphData])

  // Load graph data
  const loadGraphData = useCallback(async () => {
    setIsLoadingGraph(true)
    addActivity('system', 'Loading Graph', `Fetching graph with depth ${focusDepth}...`, 'info')
    try {
      const data = await getGraphData(undefined, focusDepth, nodeLimit)
      if (data && data.nodes.length > 0) {
        setGraphData(data)
        addActivity('graph_update', 'Graph Loaded', `${data.nodes.length} nodes, ${data.relationships.length} relationships`, 'success')
      } else {
        addActivity('system', 'Empty Graph', 'No data found. Try a chat query to populate the graph.', 'warning')
      }
    } catch (error) {
      console.error('Failed to load graph data:', error)
      addActivity('error', 'Graph Error', String(error), 'error')
    } finally {
      setIsLoadingGraph(false)
    }
  }, [focusDepth, nodeLimit, setGraphData, addActivity])

  // Handle decision node selection
  const handleDecisionSelect = useCallback((decision: TreatmentDecision | null) => {
    setSelectedDecision(decision)
    if (decision) {
      // Generate argumentation for the decision (demo)
      setArgumentationChain({
        ...DEMO_ARGUMENTATION_CHAIN,
        treatment: decision.treatment || DEMO_ARGUMENTATION_CHAIN.treatment,
        conclusion: `${decision.treatment || 'Treatment'} is recommended based on ${decision.decision_type?.replace(/_/g, ' ') || 'clinical analysis'}.`,
        confidence: decision.confidence_score ? Math.round(decision.confidence_score * 100) : 85
      })
      setActiveRightTab('argumentation')
      addActivity('decision', 'Decision Selected', decision.treatment || decision.decision_type || 'Treatment', 'info')
    }
  }, [setSelectedDecision, setArgumentationChain, setActiveRightTab, addActivity])

  // Handle node click in graph
  const handleNodeClick = useCallback((nodeId: string, labels: string[]) => {
    console.log('[ChatPage] Node clicked:', nodeId, labels)
    addActivity('query', 'Node Inspected', `${labels[0] || 'Node'}: ${nodeId.slice(0, 12)}...`, 'info')

    // If it's a decision node, show decision panel
    if (labels.includes('TreatmentDecision') || labels.includes('Decision') || labels.includes('Inference')) {
      const node = graphData?.nodes.find(n => n.id === nodeId)
      if (node) {
        const decision: TreatmentDecision = {
          id: node.id,
          decision_type: (node.properties.decision_type as string) || 'treatment_recommendation',
          category: (node.properties.category as string) || 'NSCLC',
          status: (node.properties.status as string) || 'recommended',
          reasoning: (node.properties.reasoning as string) || '',
          reasoning_summary: (node.properties.reasoning_summary as string) || '',
          treatment: (node.properties.treatment as string) || '',
          confidence_score: (node.properties.confidence_score as number) || 0.85,
          confidence: (node.properties.confidence as number),
          risk_factors: (node.properties.risk_factors as string[]) || [],
          guidelines_applied: [],
          timestamp: (node.properties.decision_timestamp as string),
          decision_timestamp: (node.properties.decision_timestamp as string),
        }
        handleDecisionSelect(decision)
      }
    }
  }, [graphData, handleDecisionSelect, addActivity])

  // Handle left tab click
  const handleLeftTabClick = useCallback((tab: 'ontology') => {
    console.log('[ChatPage] Left tab clicked:', tab)
    setActiveLeftTab(tab)
  }, [setActiveLeftTab])

  // Handle right tab click
  const handleRightTabClick = useCallback((tab: 'decisions' | 'activity' | 'argumentation') => {
    console.log('[ChatPage] Right tab clicked:', tab)
    setActiveRightTab(tab)
  }, [setActiveRightTab])

  return (
    <div className="gitnexus-layout">
      {/* Top Navigation Bar */}
      <header className="top-nav">
        <div className="nav-brand">
          <Network className="w-6 h-6 text-white" />
          <div className="brand-text">
            <span className="brand-title">ConsensusCare</span>
          </div>
        </div>

        <nav className="nav-links">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={`nav-item ${item.active ? 'nav-item-active' : ''}`}
            >
              <item.icon className="w-4 h-4" />
              <span>{item.label}</span>
            </Link>
          ))}
        </nav>

        <div className="nav-status">
          <div className="status-badge">
            <span className="status-dot"></span>
            <span>Connected</span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="main-area">
        {/* Left Sidebar */}
        <aside className={`left-sidebar ${isLeftSidebarOpen ? '' : 'collapsed'}`}>
          {/* Sidebar Toggle */}
          <button
            onClick={() => setIsLeftSidebarOpen(!isLeftSidebarOpen)}
            className="sidebar-toggle left-toggle"
            title={isLeftSidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
          >
            {isLeftSidebarOpen ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          </button>

          {isLeftSidebarOpen && (
            <>
              {/* Tab Navigation */}
              <div className="sidebar-tabs">
                <button
                  onClick={() => handleLeftTabClick('ontology')}
                  className={`sidebar-tab ${activeLeftTab === 'ontology' ? 'active' : ''}`}
                  title="Ontology Browser"
                >
                  <Book className="w-4 h-4" />
                  <span>Ontology</span>
                </button>
              </div>

              {/* Tab Content */}
              <div className="sidebar-content">
                <div style={{ display: activeLeftTab === 'ontology' ? 'block' : 'none', height: '100%' }}>
                  <OntologyBrowser
                    className="h-full"
                    onTermSelect={(term) => {
                      console.log('Term selected:', term)
                      addActivity('ontology_lookup', 'Term Selected', term.label, 'info')
                    }}
                    onTermApply={(term) => {
                      console.log('Term applied:', term)
                      addActivity('ontology_lookup', 'Term Applied', `${term.label} (${term.snomedCode || 'N/A'})`, 'success')
                    }}
                  />
                </div>
              </div>
            </>
          )}
        </aside>

        {/* Center - Context Graph (Main View) */}
        <main className="center-content">
          <div className="graph-header">
            <div className="graph-title">
              <Network className="w-5 h-5 text-violet-400" />
              <h1>Context Graph</h1>
              {graphData && graphData.nodes.length > 0 ? (
                <div className="flex items-center gap-2">
                  <span className="graph-stats">
                    {graphData.nodes.length} nodes | {graphData.relationships.length} edges
                  </span>
                  {graphUpdateTime && (
                    <span className="text-[10px] text-green-500 bg-green-500/10 px-2 py-0.5 rounded">
                      Updated {graphUpdateTime}
                    </span>
                  )}
                </div>
              ) : (
                <span className="text-xs text-zinc-500">No data - use chat to generate graph</span>
              )}
            </div>
            <div className="graph-actions">
              {/* Graph Type Switcher */}
              <div className="flex items-center gap-1 bg-zinc-800 rounded-lg p-1 mr-2">
                <button
                  onClick={() => setGraphType('sigma')}
                  className={`px-3 py-1 text-xs font-medium rounded transition-all ${
                    graphType === 'sigma'
                      ? 'bg-violet-600 text-white'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                  title="Sigma.js graph"
                >
                  Sigma
                </button>
                <button
                  onClick={() => setGraphType('cytoscape')}
                  className={`px-3 py-1 text-xs font-medium rounded transition-all ${
                    graphType === 'cytoscape'
                      ? 'bg-violet-600 text-white'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                  title="Cytoscape graph"
                >
                  Cytoscape
                </button>
              </div>

              {/* Graph Explorer Controls */}
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={`action-btn ${showFilters ? 'bg-violet-600/20 text-violet-400' : ''}`}
                title="Graph explorer settings"
              >
                <Filter className="w-4 h-4" />
              </button>
              <button
                onClick={loadGraphData}
                className="action-btn"
                title="Refresh graph"
                disabled={isLoadingGraph}
              >
                <RefreshCw className={`w-4 h-4 ${isLoadingGraph ? 'animate-spin' : ''}`} />
              </button>
              <button
                onClick={() => {
                  if (graphData) {
                    setGraphData(null)
                    addActivity('system', 'Graph Cleared', 'Graph data cleared', 'info')
                  }
                }}
                className="action-btn"
                title="Clear graph"
              >
                <Zap className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Graph Explorer Filter Panel */}
          {showFilters && (
            <div className="graph-filters">
              <div className="filter-group">
                <label className="filter-label">
                  <Layers className="w-3 h-3" />
                  <span>Focus Depth</span>
                </label>
                <div className="filter-control">
                  <input
                    type="range"
                    min="1"
                    max="5"
                    value={focusDepth}
                    onChange={(e) => setFocusDepth(parseInt(e.target.value))}
                    className="filter-slider"
                  />
                  <span className="filter-value">{focusDepth}</span>
                </div>
              </div>
              <div className="filter-group">
                <label className="filter-label">
                  <Network className="w-3 h-3" />
                  <span>Max Nodes</span>
                </label>
                <div className="filter-control">
                  <input
                    type="range"
                    min="50"
                    max="500"
                    step="50"
                    value={nodeLimit}
                    onChange={(e) => setNodeLimit(parseInt(e.target.value))}
                    className="filter-slider"
                  />
                  <span className="filter-value">{nodeLimit}</span>
                </div>
              </div>
              <button
                onClick={loadGraphData}
                className="filter-apply-btn"
                disabled={isLoadingGraph}
              >
                {isLoadingGraph ? 'Loading...' : 'Load Graph'}
              </button>
            </div>
          )}

          <div className="graph-container">
            {graphType === 'sigma' ? (
              <SigmaGraph
                data={graphData ? {
                  nodes: graphData.nodes.map(n => {
                    // Extract meaningful label from properties
                    const props = n.properties || {}
                    const label = 
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
                } : { nodes: [], edges: [] }}
                height="100%"
                showLabels={true}
                enableZoom={true}
                enablePan={true}
                layout="force"
              />
            ) : (
              <CytoscapeGraph
                graphData={graphData}
                onNodeClick={handleNodeClick}
                onGraphDataChange={setGraphData}
                height="100%"
                showLegend={true}
                className="h-full"
              />
            )}
          </div>
        </main>

        {/* Right Sidebar */}
        <aside className={`right-sidebar ${isRightSidebarOpen ? '' : 'collapsed'}`}>
          {/* Sidebar Toggle */}
          <button
            onClick={() => setIsRightSidebarOpen(!isRightSidebarOpen)}
            className="sidebar-toggle right-toggle"
            title={isRightSidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
          >
            {isRightSidebarOpen ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>

          {isRightSidebarOpen && (
            <>
              {/* Tab Navigation */}
              <div className="sidebar-tabs">
                <button
                  onClick={() => handleRightTabClick('decisions')}
                  className={`sidebar-tab ${activeRightTab === 'decisions' ? 'active' : ''}`}
                  title="Treatment Decisions"
                >
                  <Shield className="w-4 h-4" />
                  <span>Decisions</span>
                </button>
                <button
                  onClick={() => handleRightTabClick('argumentation')}
                  className={`sidebar-tab ${activeRightTab === 'argumentation' ? 'active' : ''}`}
                  title="Clinical Argumentation"
                >
                  <ListTree className="w-4 h-4" />
                  <span>Arguments</span>
                </button>
                <button
                  onClick={() => handleRightTabClick('activity')}
                  className={`sidebar-tab ${activeRightTab === 'activity' ? 'active' : ''}`}
                  title="Activity Feed"
                >
                  <Activity className="w-4 h-4" />
                  <span>Activity</span>
                </button>
              </div>

              {/* Tab Content */}
              <div className="sidebar-content">
                {/* Decision Trace Panel */}
                <div style={{ display: activeRightTab === 'decisions' ? 'block' : 'none', height: '100%' }}>
                  <DecisionTracePanel
                    decision={selectedDecision}
                    onDecisionSelect={handleDecisionSelect}
                    graphDecisions={decisionNodes}
                  />
                </div>

                {/* Argumentation View */}
                <div style={{ display: activeRightTab === 'argumentation' ? 'block' : 'none', height: '100%' }}>
                  {argumentationChain ? (
                    <ArgumentationView chain={argumentationChain} />
                  ) : (
                    <div className="flex items-center justify-center h-full text-zinc-500 text-sm p-4 text-center">
                      <div>
                        <ListTree className="w-8 h-8 mx-auto mb-2 opacity-30" />
                        <p>No argumentation chain available.</p>
                        <p className="text-xs mt-1">Analyze a patient case or click a decision node to see clinical arguments.</p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Activity Feed */}
                <div style={{ display: activeRightTab === 'activity' ? 'block' : 'none', height: '100%' }}>
                  <ActivityFeed activities={activities} />
                </div>
              </div>
            </>
          )}
        </aside>
      </div>
    </div>
  )
}
