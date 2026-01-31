'use client'

import { createContext, useContext, useState, ReactNode, useCallback } from 'react'
import type { GraphData, GraphNode, TreatmentDecision } from '@/lib/api'

interface ArgumentationChain {
  conclusion: string
  treatment: string
  confidence: number
  arguments: Array<{
    id: string
    type: 'support' | 'attack' | 'neutral'
    source: string
    claim: string
    evidence?: string
    strength: 'strong' | 'moderate' | 'weak'
    references?: string[]
  }>
  counterArguments?: Array<{
    id: string
    type: 'support' | 'attack' | 'neutral'
    source: string
    claim: string
    evidence?: string
    strength: 'strong' | 'moderate' | 'weak'
    references?: string[]
  }>
  patientFactors?: string[]
}

interface AppContextType {
  // Graph data
  graphData: GraphData | null
  setGraphData: (data: GraphData | null) => void

  // Decision tracking
  decisionNodes: GraphNode[]
  setDecisionNodes: (nodes: GraphNode[]) => void
  selectedDecision: TreatmentDecision | null
  setSelectedDecision: (decision: TreatmentDecision | null) => void

  // Argumentation
  argumentationChain: ArgumentationChain | null
  setArgumentationChain: (chain: ArgumentationChain | null) => void

  // Panel visibility
  isLeftSidebarOpen: boolean
  setIsLeftSidebarOpen: (open: boolean) => void
  isRightSidebarOpen: boolean
  setIsRightSidebarOpen: (open: boolean) => void

  // Active left panel tab
  activeLeftTab: 'ontology'
  setActiveLeftTab: (tab: 'ontology') => void

  // Active right panel tab
  activeRightTab: 'decisions' | 'activity' | 'argumentation'
  setActiveRightTab: (tab: 'decisions' | 'activity' | 'argumentation') => void

  // Legacy compatibility
  isGraphPanelOpen: boolean
  setIsGraphPanelOpen: (open: boolean) => void
  isChatPanelOpen: boolean
  setIsChatPanelOpen: (open: boolean) => void
}

const AppContext = createContext<AppContextType | undefined>(undefined)

export function AppProvider({ children }: { children: ReactNode }) {
  // Graph data
  const [graphData, setGraphData] = useState<GraphData | null>(null)

  // Decision tracking
  const [decisionNodes, setDecisionNodes] = useState<GraphNode[]>([])
  const [selectedDecision, setSelectedDecision] = useState<TreatmentDecision | null>(null)

  // Argumentation
  const [argumentationChain, setArgumentationChain] = useState<ArgumentationChain | null>(null)

  // Panel visibility
  const [isLeftSidebarOpen, setIsLeftSidebarOpen] = useState(true)
  const [isRightSidebarOpen, setIsRightSidebarOpen] = useState(true)

  // Active tabs
  const [activeLeftTab, setActiveLeftTab] = useState<'ontology'>('ontology')
  const [activeRightTab, setActiveRightTab] = useState<'decisions' | 'activity' | 'argumentation'>('decisions')

  // Legacy compatibility
  const [isGraphPanelOpen, setIsGraphPanelOpen] = useState(true)
  const [isChatPanelOpen, setIsChatPanelOpen] = useState(true)

  return (
    <AppContext.Provider value={{
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
      setActiveRightTab,
      isGraphPanelOpen,
      setIsGraphPanelOpen,
      isChatPanelOpen,
      setIsChatPanelOpen
    }}>
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const context = useContext(AppContext)
  if (context === undefined) {
    throw new Error('useApp must be used within an AppProvider')
  }
  return context
}
