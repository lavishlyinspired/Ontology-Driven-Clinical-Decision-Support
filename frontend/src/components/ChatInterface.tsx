'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, CheckCircle, XCircle, AlertCircle, Bot, User, Activity, Zap, ChevronDown, ChevronUp, Trash2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { GraphData, GraphNode } from '@/lib/api'
import { McpAppHost } from './McpAppHost'
import { GroundedCitations } from './GroundedCitations'
import LabResultsPanel from './LabResultsPanel'
import MedicationListPanel from './MedicationListPanel'
import MonitoringProtocolPanel from './MonitoringProtocolPanel'
import ClinicalTrialsPanel from './ClinicalTrialsPanel'

interface ChatInterfaceProps {
  onGraphDataChange?: (data: GraphData) => void
  onDecisionNodesChange?: (decisions: GraphNode[]) => void
}

interface McpAppData {
  resourceUri: string
  input: Record<string, unknown>
  result?: {
    content?: Array<{ type: string; text: string }>
    structuredContent?: Record<string, unknown>
  }
  title?: string
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp?: string
  patientData?: any
  complexity?: any
  graphData?: any
  toolCalls?: ToolCall[]
  mcpApp?: McpAppData  // NEW: MCP App attachment
  metadata?: {
    complexity?: string
    workflow?: string
    agentStatus?: AgentStatus[]
    workflowSteps?: WorkflowStep[]
    logs?: LogEntry[]
  }
}

interface ToolCall {
  name: string
  input: Record<string, unknown>
  output?: unknown
}

interface AgentStatus {
  agent: string
  status: 'started' | 'completed' | 'failed'
  duration_ms?: number
  confidence?: number
}

interface WorkflowStep {
  id: string
  content: string
  status: 'active' | 'completed' | 'error' | 'reasoning'
  timestamp: string
}

interface LogEntry {
  id: string
  content: string
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'
  timestamp: string
}

const JSONDisplay = ({ data }: { data: any }) => {
  const [isExpanded, setIsExpanded] = useState(false)
  
  return (
    <div className="bg-slate-800 rounded-xl overflow-hidden my-3">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 hover:bg-slate-700 transition-colors"
      >
        <div className="flex items-center gap-2 text-slate-400 text-xs">
          <Activity className="w-3 h-3" />
          <span>Extracted Patient Data</span>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-4 h-4 text-slate-400" />
        ) : (
          <ChevronDown className="w-4 h-4 text-slate-400" />
        )}
      </button>
      {isExpanded && (
        <div className="p-4 pt-0 overflow-x-auto">
          <pre className="text-sm text-green-400 font-mono">
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

const WorkflowTimeline = ({ steps, isStreaming = false }: { steps: WorkflowStep[], isStreaming?: boolean }) => {
  const [isExpanded, setIsExpanded] = useState(false)

  // Function to parse agent execution messages for better display
  const parseAgentMessage = (content: string) => {
    const agentMatch = content.match(/\[(\d+)\/(\d+)\]\s+(.+)/)
    if (agentMatch) {
      return {
        isAgent: true,
        current: agentMatch[1],
        total: agentMatch[2],
        message: agentMatch[3]
      }
    }
    return { isAgent: false, message: content }
  }

  // Don't render if no steps
  if (!steps || steps.length === 0) {
    return null
  }

  // Count completed vs total steps
  const completedSteps = steps.filter(s => s.status === 'completed').length
  const totalSteps = steps.length

  return (
    <div className="bg-gradient-to-br from-slate-50 to-blue-50 border border-blue-200 rounded-xl my-3 overflow-hidden">
      {/* Collapsible Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 hover:bg-blue-50 transition-colors"
      >
        <div className="flex items-center gap-2 text-xs text-blue-600 font-semibold uppercase tracking-wider">
          <Activity className="w-3.5 h-3.5" />
          <span>Workflow Progress</span>
          <span className="text-blue-400 font-normal normal-case">
            ({completedSteps}/{totalSteps} steps)
          </span>
          {isStreaming && (
            <Loader2 className="w-3 h-3 animate-spin text-blue-500" />
          )}
        </div>
        {isExpanded ? (
          <ChevronUp className="w-4 h-4 text-blue-500" />
        ) : (
          <ChevronDown className="w-4 h-4 text-blue-500" />
        )}
      </button>

      {/* Collapsible Content */}
      {isExpanded && (
        <div className="px-4 pb-4 space-y-1 max-h-64 overflow-y-auto">
          {steps.map((step, idx) => {
            const parsed = parseAgentMessage(step.content)
            return (
              <div key={step.id} className="flex items-start gap-3">
              <div className="flex flex-col items-center">
                {step.status === 'completed' ? (
                  <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" />
                ) : step.status === 'active' ? (
                  <Loader2 className="w-4 h-4 text-blue-500 animate-spin flex-shrink-0 mt-0.5" />
                ) : step.status === 'reasoning' ? (
                  <AlertCircle className="w-4 h-4 text-amber-500 flex-shrink-0 mt-0.5" />
                ) : (
                  <XCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                )}
                {idx < steps.length - 1 && (
                  <div className="w-px h-3 bg-gray-200 my-0.5" />
                )}
              </div>
              <div className="flex-1">
                {parsed.isAgent ? (
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-400 font-mono">
                      [{parsed.current}/{parsed.total}]
                    </span>
                    <span className={`text-sm leading-tight ${
                      step.status === 'active' ? 'text-blue-700 font-medium' :
                      step.status === 'completed' ? 'text-gray-600' :
                      step.status === 'reasoning' ? 'text-amber-600' :
                      'text-red-500'
                    }`}>
                      {parsed.message}
                    </span>
                  </div>
                ) : (
                  <span className={`text-sm leading-tight ${
                    step.status === 'active' ? 'text-blue-700 font-medium' :
                    step.status === 'completed' ? 'text-gray-600' :
                    step.status === 'reasoning' ? 'text-purple-700 font-medium' :
                    'text-red-500'
                  }`}>
                    {step.content}
                  </span>
                )}
              </div>
            </div>
              )
            })}
          </div>
        )}
      </div>
    )
  }

const ComplexityBadge = ({ complexity }: { complexity: any }) => (
  <div className="inline-flex items-center gap-3 bg-gradient-to-r from-orange-50 to-red-50 border-2 border-orange-200 rounded-full px-4 py-2 my-2">
    <div className="flex items-center gap-1.5">
      <Zap className="w-4 h-4 text-orange-600" />
      <span className="text-xs font-semibold text-orange-700">Complexity:</span>
      <span className="text-xs font-bold text-orange-900">{complexity.level}</span>
      {complexity.score && complexity.score !== "N/A" && (
        <span className="text-xs text-orange-600">({complexity.score})</span>
      )}
    </div>
    <div className="h-4 w-px bg-orange-300"></div>
    <div className="flex items-center gap-1.5">
      <Activity className="w-4 h-4 text-blue-600" />
      <span className="text-xs font-semibold text-blue-700">Workflow:</span>
      <span className="text-xs font-bold text-blue-900">{complexity.workflow}</span>
    </div>
  </div>
)

const LogDisplay = ({ logs, isExpanded, onToggle }: { logs: LogEntry[], isExpanded: boolean, onToggle: () => void }) => {
  if (logs.length === 0) return null

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'DEBUG': return 'text-gray-400'
      case 'INFO': return 'text-cyan-400'
      case 'WARNING': return 'text-yellow-400'
      case 'ERROR': return 'text-red-400'
      default: return 'text-gray-300'
    }
  }

  return (
    <div className="bg-slate-900 rounded-xl my-3 overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-3 text-slate-300 hover:bg-slate-800 transition-colors"
      >
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider">
          <Activity className="w-3.5 h-3.5 text-cyan-400" />
          <span>Agent & Workflow Logs ({logs.length})</span>
        </div>
        <span className="text-xs text-slate-500">
          {isExpanded ? '▼ Hide' : '▶ Show'}
        </span>
      </button>
      {isExpanded && (
        <div className="max-h-64 overflow-y-auto p-3 pt-0 space-y-1 font-mono text-xs">
          {logs.map((log) => (
            <div key={log.id} className="flex gap-2">
              <span className="text-slate-500 flex-shrink-0">{log.timestamp}</span>
              <span className={`flex-shrink-0 w-16 ${getLevelColor(log.level)}`}>
                [{log.level}]
              </span>
              <span className="text-slate-300 break-all">{log.content}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function ChatInterface({ onGraphDataChange, onDecisionNodesChange }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: `## ConsensusCare - Clinical Decision Support

Powered by LUCADA ontology, NCCN/NICE/ESMO guidelines, and multi-agent AI workflow with SNOMED-CT, LOINC, RxNorm, and NCI Thesaurus integration.

### What I Can Do

| Category | Example |
|----------|---------|
| **Analyze a Patient** | "68M, stage IIIA adenocarcinoma, EGFR Ex19del+, PS 1" |
| **Guidelines Q&A** | "What is first-line for stage IV NSCLC with PD-L1 80%?" |
| **Query Knowledge Graph** | "How many patients are in the database?" |
| **Drug Interactions** | "Check interactions for osimertinib and warfarin" |
| **Lab Interpretation** | "Interpret lab results for chemotherapy monitoring" |
| **Ontology Lookup** | "Look up SNOMED code 254637007" |
| **Vocabulary Translation** | "Translate SNOMED 254637007 to NCIt" |
| **Value Set Expansion** | "Expand lung cancer histologies value set" |
| **Provenance Tracking** | "Show provenance chain for [decision-id]" |

### Architecture
- **20 specialized agents** with adaptive complexity routing
- **SNOMED-CT + LOINC + RxNorm + NCIt** loaded in Neo4j
- **FHIR R4 Terminology** with $lookup, $translate, $expand
- **PROV-O provenance** tracking for all decisions
- **MCP tool integration** for 60+ clinical tools

Try a question below or describe a patient case to begin.`,
      timestamp: 'Welcome'
    }
  ])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [currentStatus, setCurrentStatus] = useState('')
  const [currentComplexity, setCurrentComplexity] = useState<any>(null)
  const [currentPatientData, setCurrentPatientData] = useState<any>(null)
  const [suggestions, setSuggestions] = useState<string[]>([
    '68M, stage IIIA adenocarcinoma, EGFR Ex19del+, PS 1, COPD',
    'What is first-line treatment for stage IV NSCLC with PD-L1 80%?',
    'What ontologies are loaded in the system?',
    'Expand lung cancer histologies value set',
    'How many patients are in the database?',
  ])
  const [workflowSteps, setWorkflowSteps] = useState<WorkflowStep[]>([])
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [logsExpanded, setLogsExpanded] = useState(false)

  // New state for lab/medication/monitoring/trials
  const [labResults, setLabResults] = useState<any[]>([])
  const [labInterpretations, setLabInterpretations] = useState<any[]>([])
  const [medications, setMedications] = useState<any[]>([])
  const [drugInteractions, setDrugInteractions] = useState<any[]>([])
  const [monitoringProtocol, setMonitoringProtocol] = useState<any>(null)
  const [eligibleTrials, setEligibleTrials] = useState<any[]>([])

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const workflowStepsRef = useRef<WorkflowStep[]>([])
  const logsRef = useRef<LogEntry[]>([])
  
  const sessionId = useRef(crypto.randomUUID()).current

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, currentStatus])

  const sendMessage = async (overrideText?: string) => {
    const messageText = overrideText || input.trim()
    if (!messageText || isStreaming) return

    // Add user message
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: messageText,
      timestamp: new Date().toLocaleTimeString()
    }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsStreaming(true)
    setCurrentStatus('')
    setCurrentComplexity(null)
    setCurrentPatientData(null)
    setSuggestions([])
    setWorkflowSteps([])
    setLogs([])
    setLogsExpanded(false)
    workflowStepsRef.current = []
    logsRef.current = []

    // Create assistant message placeholder
    const assistantId = crypto.randomUUID()
    setStreamingMessageId(assistantId)
    let assistantContent = ''
    let patientData: any = null
    let complexity: any = null

    setMessages(prev => [...prev, {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toLocaleTimeString()
    }])

    try {
      // Stream response via SSE from backend with graph integration
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/api/chat-graph/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: messageText,
          session_id: sessionId,
          include_graph: true,
          auto_expand_entities: true
        })
      })

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) throw new Error('No reader available')

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              
              // Console log all incoming messages for debugging
              console.log('[LCA] SSE Message:', data.type, '|', data.content)

              if (data.type === 'status') {
                // Accumulate workflow steps - mark previous active steps as completed
                setWorkflowSteps(prev => {
                  const updated = prev.map(step =>
                    step.status === 'active' ? { ...step, status: 'completed' as const } : step
                  )
                  const newSteps = [...updated, {
                    id: crypto.randomUUID(),
                    content: data.content,
                    status: 'active' as const,
                    timestamp: new Date().toLocaleTimeString()
                  }]
                  console.log('[LCA] Workflow steps after status:', newSteps.length, '| Content:', data.content)
                  workflowStepsRef.current = newSteps
                  return newSteps
                })
                setCurrentStatus(data.content)
              } else if (data.type === 'reasoning') {
                console.log('[LCA] Adding reasoning step:', data.content)
                // Add reasoning step with special styling
                setWorkflowSteps(prev => {
                  const newSteps = [...prev, {
                    id: crypto.randomUUID(),
                    content: data.content,
                    status: 'reasoning' as const,
                    timestamp: new Date().toLocaleTimeString()
                  }]
                  console.log('[LCA] Workflow steps after reasoning:', newSteps.length)
                  workflowStepsRef.current = newSteps
                  return newSteps
                })
              } else if (data.type === 'progress') {
                // Log progress to console for debugging
                console.log('🔄 [LCA Agent Progress]:', data.content)
                
                // Determine if this is an agent execution message
                const isAgentExecution = data.content.includes('[') && (
                  data.content.includes('Running') ||
                  data.content.includes('completed') ||
                  data.content.includes('SKIPPED') ||
                  data.content.includes('FAILED')
                )
                
                const stepStatus = data.content.includes('FAILED') ? 'error' as const :
                                  data.content.includes('SKIPPED') ? 'reasoning' as const :
                                  data.content.includes('completed') ? 'completed' as const :
                                  'active' as const
                
                console.log('   └─ Status:', stepStatus, '| IsAgent:', isAgentExecution)
                
                // Add as step with appropriate status
                setWorkflowSteps(prev => {
                  const newSteps = [...prev, {
                    id: crypto.randomUUID(),
                    content: data.content,
                    status: stepStatus,
                    timestamp: new Date().toLocaleTimeString()
                  }]
                  console.log('[LCA] Workflow steps after progress:', newSteps.length)
                  workflowStepsRef.current = newSteps
                  return newSteps
                })
                setCurrentStatus(data.content)
              } else if (data.type === 'patient_data') {
                patientData = data.content
                setCurrentPatientData(data.content)
              } else if (data.type === 'complexity') {
                complexity = data.content
                setCurrentComplexity(data.content)
              } else if (data.type === 'recommendation') {
                assistantContent += data.content + '\n\n'
              } else if (data.type === 'text') {
                assistantContent += data.content
              } else if (data.type === 'clinical_summary') {
                assistantContent += `\n## Clinical Summary\n\n${data.content}\n\n`
              } else if (data.type === 'treatment_plan') {
                assistantContent += `\n## Treatment Plan\n\n${data.content}\n\n`
              } else if (data.type === 'suggestions') {
                setSuggestions(data.content)
              } else if (data.type === 'lab_results') {
                // Handle lab results from LabInterpretationAgent
                if (data.content.results) {
                  setLabResults(data.content.results)
                }
                if (data.content.interpretations) {
                  setLabInterpretations(data.content.interpretations)
                }
              } else if (data.type === 'drug_interactions') {
                // Handle drug interactions from MedicationManagementAgent
                if (data.content.interactions) {
                  setDrugInteractions(data.content.interactions)
                }
                // Show alert for severe interactions
                if (data.content.severe_count > 0) {
                  setWorkflowSteps(prev => [...prev, {
                    id: crypto.randomUUID(),
                    content: `⚠️ ${data.content.severe_count} severe drug interaction(s) detected!`,
                    status: 'error' as const,
                    timestamp: new Date().toLocaleTimeString()
                  }])
                }
              } else if (data.type === 'monitoring_protocol') {
                // Handle monitoring protocol from MonitoringCoordinatorAgent
                setMonitoringProtocol(data.content)
              } else if (data.type === 'eligible_trials') {
                // Handle clinical trials from ClinicalTrialsService
                if (data.content.trials) {
                  setEligibleTrials(data.content.trials)
                }
              } else if (data.type === 'error') {
                assistantContent += `\n\n**Error:** ${data.content}\n`
                setWorkflowSteps(prev => [...prev, {
                  id: crypto.randomUUID(),
                  content: data.content,
                  status: 'error' as const,
                  timestamp: new Date().toLocaleTimeString()
                }])
              } else if (data.type === 'log') {
                // Capture log entries for display
                console.log(`[${data.level}] ${data.timestamp}: ${data.content}`)
                setLogs(prev => {
                  const newLogs = [...prev, {
                    id: crypto.randomUUID(),
                    content: data.content,
                    level: data.level || 'INFO',
                    timestamp: data.timestamp || new Date().toLocaleTimeString()
                  }]
                  logsRef.current = newLogs
                  return newLogs
                })
              } else if (data.type === 'tool_call') {
                // MCP tool invocation
                console.log('[LCA] Tool call:', data.content)
                setWorkflowSteps(prev => [...prev, {
                  id: crypto.randomUUID(),
                  content: `🔧 Invoking tool: ${data.content.tool}`,
                  status: 'active' as const,
                  timestamp: new Date().toLocaleTimeString()
                }])
                // Store tool call in message metadata
                setMessages(prev => prev.map(msg =>
                  msg.id === assistantId
                    ? {
                        ...msg,
                        toolCalls: [...(msg.toolCalls || []), {
                          name: data.content.tool,
                          input: data.content.arguments
                        }]
                      }
                    : msg
                ))
              } else if (data.type === 'tool_result') {
                // MCP tool result
                console.log('[LCA] Tool result:', data.content)
                setWorkflowSteps(prev => {
                  const updated = prev.map(step =>
                    step.status === 'active' && step.content.includes('Invoking tool')
                      ? { ...step, status: 'completed' as const }
                      : step
                  )
                  return [...updated, {
                    id: crypto.randomUUID(),
                    content: '✅ Tool execution completed',
                    status: 'completed' as const,
                    timestamp: new Date().toLocaleTimeString()
                  }]
                })
                // Update last tool call with result
                setMessages(prev => prev.map(msg => {
                  if (msg.id === assistantId && msg.toolCalls && msg.toolCalls.length > 0) {
                    const updatedToolCalls = [...msg.toolCalls]
                    updatedToolCalls[updatedToolCalls.length - 1] = {
                      ...updatedToolCalls[updatedToolCalls.length - 1],
                      output: data.content
                    }
                    return { ...msg, toolCalls: updatedToolCalls }
                  }
                  return msg
                }))
              } else if (data.type === 'graph_data') {
                // Graph visualization data
                console.log('[LCA] Received graph_data:', data.content)
                console.log('[LCA] Graph data nodes:', data.content?.nodes?.length || 0)
                console.log('[LCA] Graph data relationships:', data.content?.relationships?.length || 0)
                
                // Store in message for optional inline display
                setMessages(prev => prev.map(msg =>
                  msg.id === assistantId
                    ? { ...msg, graphData: data.content }
                    : msg
                ))
                
                // Notify parent component of graph data change
                if (onGraphDataChange && data.content && data.content.nodes && data.content.relationships) {
                  console.log('[LCA] Calling onGraphDataChange with', data.content.nodes.length, 'nodes')
                  onGraphDataChange(data.content)

                  // Extract decision nodes if present - check both labels array and properties
                  if (onDecisionNodesChange && data.content.nodes) {
                    const decisionNodes = data.content.nodes.filter((node: GraphNode) => {
                      // Check labels array
                      const hasDecisionLabel =
                        node.labels.includes('TreatmentDecision') ||
                        node.labels.includes('Decision') ||
                        node.labels.includes('Inference')

                      // Also check if it's a virtual decision node (has decision_type property)
                      const hasDecisionProperties =
                        node.properties?.decision_type === 'treatment_recommendation' ||
                        (node.properties?.treatment !== undefined && node.labels.includes('TreatmentDecision'))

                      return hasDecisionLabel || hasDecisionProperties
                    })

                    console.log('[LCA] Found', decisionNodes.length, 'decision nodes')
                    console.log('[LCA] Decision nodes:', decisionNodes.map((n: GraphNode) => ({
                      id: n.id,
                      labels: n.labels,
                      treatment: n.properties?.treatment
                    })))

                    // Always notify parent to update the decisions panel
                    onDecisionNodesChange(decisionNodes)
                  }
                } else {
                  console.warn('[LCA] graph_data is missing nodes or relationships:', data.content)
                }
              } else if (data.type === 'tool_use') {
                // Tool call started
                console.log('[LCA] Tool use:', data.name, data.input)
              } else if (data.type === 'tool_result') {
                // Tool call completed
                console.log('[LCA] Tool result:', data.name, data.output)
              } else if (data.type === 'agent_context') {
                // Agent context for transparency
                console.log('[LCA] Agent context:', data.context)
              }

              // Update assistant message with all data
              setMessages(prev => prev.map(msg =>
                msg.id === assistantId
                  ? {
                      ...msg,
                      content: assistantContent,
                      patientData: patientData,
                      complexity: complexity
                    }
                  : msg
              ))
            } catch (e) {
              // Skip invalid JSON
              console.warn('Failed to parse SSE data:', line)
            }
          }
        }
      }
    } catch (error) {
      console.error('Stream error:', error)
      setMessages(prev => prev.map(msg =>
        msg.id === assistantId
          ? { ...msg, content: `**Connection Error**\n\nCouldn't connect to the backend server. Please ensure:\n- Backend is running on http://localhost:8000\n- All services are initialized\n\nError: ${error}` }
          : msg
      ))
    } finally {
      setIsStreaming(false)
      setCurrentStatus('')
      
      // Mark remaining active steps as completed - use ref to get most current state
      const finalWorkflowSteps = workflowStepsRef.current.map(step =>
        step.status === 'active' ? { ...step, status: 'completed' as const } : step
      )
      const finalLogs = logsRef.current
      
      // Log for debugging
      console.log('[LCA] Finalizing - workflow steps:', finalWorkflowSteps.length, 'steps')
      console.log('[LCA] Finalizing - logs:', finalLogs.length, 'logs')
      if (finalWorkflowSteps.length > 0) {
        console.log('[LCA] First step:', finalWorkflowSteps[0].content)
        console.log('[LCA] Last step:', finalWorkflowSteps[finalWorkflowSteps.length - 1].content)
      }
      
      // Update workflow steps state
      setWorkflowSteps(finalWorkflowSteps)
      
      // Save workflow steps and logs to message metadata before clearing streaming state
      setMessages(prev => prev.map(msg => {
        if (msg.id === assistantId) {
          console.log('[LCA] Updating message', assistantId, 'with', finalWorkflowSteps.length, 'workflow steps')
          return {
            ...msg,
            metadata: {
              ...msg.metadata,
              workflowSteps: finalWorkflowSteps,
              logs: finalLogs
            }
          }
        }
        return msg
      }))
      
      setStreamingMessageId(null)

      // Add helpful message if no content was generated
      setMessages(prev => prev.map(msg => {
        if (msg.id === assistantId && !msg.content.trim() && !msg.patientData) {
          return {
            ...msg,
            content: `**No Response Generated**\n\nThe system processed your request but didn't generate recommendations. This might be because:\n- The patient data needs more details\n- The workflow is still being refined\n- Additional context is required\n\nTry providing more specific patient information.`
          }
        }
        return msg
      }))
    }
  }

  const handleSuggestionClick = (suggestion: string) => {
    sendMessage(suggestion)
  }

  const clearChat = () => {
    setMessages([messages[0]]) // Keep welcome message
    setSuggestions([])
    setWorkflowSteps([])
    setLogs([])
    setLogsExpanded(false)
    setCurrentComplexity(null)
    setCurrentPatientData(null)
  }

  return (
    <div className="flex flex-col h-full w-full bg-zinc-900">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-8 space-y-6 bg-gradient-to-b from-zinc-800 to-zinc-900">
        {messages.map((msg, index) => (
          <div
            key={msg.id}
            className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn`}
            style={{ animationDelay: `${index * 50}ms` }}
          >
            {msg.role === 'assistant' && (
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-sm ring-2 ring-violet-900/50">
                <Bot className="w-5 h-5 text-white" />
              </div>
            )}

            <div className="max-w-[85%] flex flex-col gap-2">
              {/* Patient Data Card */}
              {msg.patientData && (
                <div className="animate-fadeIn">
                  <JSONDisplay data={msg.patientData} />
                </div>
              )}

              {/* Complexity Badge */}
              {msg.complexity && (
                <div className="animate-fadeIn">
                  <ComplexityBadge complexity={msg.complexity} />
                </div>
              )}

              {/* Workflow Timeline - shown during streaming and persisted after */}
              {((streamingMessageId === msg.id && workflowSteps.length > 0) || (msg.metadata?.workflowSteps && msg.metadata.workflowSteps.length > 0)) && (
                <div className="animate-fadeIn">
                  <WorkflowTimeline steps={streamingMessageId === msg.id ? workflowSteps : msg.metadata?.workflowSteps || []} isStreaming={streamingMessageId === msg.id && isStreaming} />
                </div>
              )}

              {/* Log Display - shown during streaming for current message */}
              {streamingMessageId === msg.id && logs.length > 0 && (
                <div className="animate-fadeIn">
                  <LogDisplay
                    logs={logs}
                    isExpanded={logsExpanded}
                    onToggle={() => setLogsExpanded(!logsExpanded)}
                  />
                </div>
              )}

              {/* Main Message Bubble */}
              <div
                className={`rounded-2xl p-5 transition-all ${
                  msg.role === 'user'
                    ? 'bg-gradient-to-br from-violet-500 to-purple-600 text-white shadow-md'
                    : 'bg-zinc-800 border border-zinc-700 text-zinc-100 shadow-sm hover:shadow-md'
                }`}
              >
                <div className={`prose prose-sm max-w-none ${
                  msg.role === 'user' ? 'prose-invert' : 'prose-blue'
                }`}>
                  {msg.content ? (
                    msg.role === 'assistant' && msg.content.includes('[[') ? (
                      <GroundedCitations
                        text={msg.content}
                        showTooltips={true}
                        renderAs="inline"
                        onCitationClick={(citation) => {
                          console.log('Citation clicked:', citation)
                          if (citation.url) {
                            window.open(citation.url, '_blank', 'noopener,noreferrer')
                          }
                        }}
                      />
                    ) : (
                      <ReactMarkdown 
                        remarkPlugins={[remarkGfm]}
                        components={{
                          table: ({ node, ...props }) => (
                            <div className="overflow-x-auto my-4">
                              <table className="min-w-full divide-y divide-zinc-600 border border-zinc-600" {...props} />
                            </div>
                          ),
                          thead: ({ node, ...props }) => (
                            <thead className="bg-zinc-700" {...props} />
                          ),
                          tbody: ({ node, ...props }) => (
                            <tbody className="divide-y divide-zinc-600 bg-zinc-800" {...props} />
                          ),
                          tr: ({ node, ...props }) => (
                            <tr {...props} />
                          ),
                          th: ({ node, ...props }) => (
                            <th className="px-4 py-2 text-left text-xs font-medium text-zinc-200 uppercase tracking-wider border border-zinc-600" {...props} />
                          ),
                          td: ({ node, ...props }) => (
                            <td className="px-4 py-2 text-sm text-zinc-100 border border-zinc-600" {...props} />
                          ),
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    )
                  ) : (
                    <div className="flex items-center gap-2 text-gray-400">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-sm">Processing...</span>
                    </div>
                  )}
                </div>

                {/* Tool Calls Display */}
                {msg.toolCalls && msg.toolCalls.length > 0 && (
                  <div className="mt-4 space-y-2">
                    {msg.toolCalls.map((toolCall, idx) => (
                      <div key={idx} className="bg-slate-800 rounded-lg p-3 border border-slate-700">
                        <div className="flex items-center gap-2 mb-2">
                          <Zap className="w-4 h-4 text-yellow-400" />
                          <span className="text-sm font-semibold text-yellow-400">
                            Tool: {toolCall.name}
                          </span>
                        </div>
                        {toolCall.input && Object.keys(toolCall.input).length > 0 && (
                          <details className="text-xs text-gray-400 mb-2">
                            <summary className="cursor-pointer hover:text-gray-300">Input</summary>
                            <pre className="mt-2 p-2 bg-slate-900 rounded overflow-x-auto">
                              {(() => {
                                try {
                                  return JSON.stringify(toolCall.input, null, 2)
                                } catch (error) {
                                  return `Error displaying input: ${error instanceof Error ? error.message : 'Unknown error'}`
                                }
                              })()}
                            </pre>
                          </details>
                        )}
                        {toolCall.output != null && (
                          <details className="text-xs text-gray-400" open>
                            <summary className="cursor-pointer hover:text-gray-300">Result</summary>
                            <pre className="mt-2 p-2 bg-slate-900 rounded overflow-x-auto text-green-400">
                              {(() => {
                                try {
                                  if (typeof toolCall.output === 'string') {
                                    return toolCall.output
                                  } else if (toolCall.output === null || toolCall.output === undefined) {
                                    return 'null'
                                  } else {
                                    return JSON.stringify(toolCall.output, null, 2)
                                  }
                                } catch (error) {
                                  return `Error displaying result: ${error instanceof Error ? error.message : 'Unknown error'}`
                                }
                              })()}
                            </pre>
                          </details>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* MCP App Inline Rendering */}
                {msg.mcpApp && (
                  <div className="mt-4">
                    <McpAppHost
                      resourceUri={msg.mcpApp.resourceUri}
                      toolInput={msg.mcpApp.input}
                      toolResult={msg.mcpApp.result}
                      title={msg.mcpApp.title || 'Interactive App'}
                      height="350px"
                      onToolCall={async (name, args) => {
                        // Forward tool calls to backend
                        try {
                          const response = await fetch('/api/v1/ontology/arguments', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(args)
                          })
                          return await response.json()
                        } catch (error) {
                          console.error('MCP tool call failed:', error)
                          return { error: 'Tool call failed' }
                        }
                      }}
                      onMessage={(text) => {
                        // Add message to input and send
                        setInput(text)
                        // Optionally auto-send
                      }}
                    />
                  </div>
                )}

                <div className={`text-xs mt-3 flex items-center gap-1 ${
                  msg.role === 'user' ? 'text-blue-100' : 'text-gray-400'
                }`}>
                  <span>{msg.timestamp ?? ''}</span>
                </div>
              </div>
            </div>

            {msg.role === 'user' && (
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-gray-400 to-gray-500 flex items-center justify-center shadow-sm ring-2 ring-gray-200">
                <User className="w-5 h-5 text-white" />
              </div>
            )}
          </div>
        ))}        
        
        {/* Streaming indicator - compact bar */}
        {isStreaming && currentStatus && (
          <div className="flex justify-start animate-fadeIn">
            <div className="bg-blue-50 border border-blue-200 rounded-xl px-4 py-2.5 max-w-[80%]">
              <div className="flex items-center gap-2 text-blue-600">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="text-sm font-medium">{currentStatus}</span>
              </div>
            </div>
          </div>
        )}
        
        {/* Suggestions */}
        {suggestions.length > 0 && !isStreaming && (
          <div className="flex justify-start animate-fadeIn">
            <div className="bg-gradient-to-br from-zinc-800 to-zinc-800/80 border border-zinc-700 rounded-2xl p-5 max-w-[80%] shadow-sm">
              <p className="text-sm font-semibold text-zinc-300 mb-3 flex items-center gap-2">
                <span className="text-lg">💡</span>
                <span>Suggested follow-ups:</span>
              </p>
              <div className="space-y-2">
                {suggestions.map((sug, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSuggestionClick(sug)}
                    className="block w-full text-left px-4 py-3 text-sm bg-zinc-900 hover:bg-violet-900/30 text-zinc-200 border border-zinc-700 hover:border-violet-500 rounded-xl transition-all hover:shadow-md hover:scale-[1.02] active:scale-[0.98]"
                  >
                    {sug}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Clinical Data Panels */}
      {(labResults.length > 0 || medications.length > 0 || monitoringProtocol || eligibleTrials.length > 0) && (
        <div className="border-t border-zinc-700 bg-zinc-900 p-6 max-h-96 overflow-y-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Lab Results Panel */}
            {labResults.length > 0 && (
              <LabResultsPanel
                patientId={currentPatientData?.patient_id}
                labResults={labResults}
                labInterpretations={labInterpretations}
              />
            )}

            {/* Medications Panel */}
            {medications.length > 0 && (
              <MedicationListPanel
                patientId={currentPatientData?.patient_id}
                medications={medications}
                interactions={drugInteractions}
              />
            )}

            {/* Monitoring Protocol Panel */}
            {monitoringProtocol && (
              <MonitoringProtocolPanel
                patientId={currentPatientData?.patient_id}
                protocol={monitoringProtocol}
                labResults={labResults}
              />
            )}

            {/* Clinical Trials Panel */}
            {eligibleTrials.length > 0 && (
              <ClinicalTrialsPanel
                patientId={currentPatientData?.patient_id}
                trials={eligibleTrials}
              />
            )}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="border-t border-zinc-700 bg-zinc-800 p-4 shadow-sm">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
            placeholder="Describe a patient or ask a question..."
            className="flex-1 px-4 py-3 border border-zinc-600 bg-zinc-900 text-zinc-100 rounded-xl focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500 transition-all hover:border-zinc-500 text-base placeholder-zinc-500"
            disabled={isStreaming}
          />
          <button
            onClick={clearChat}
            disabled={messages.length <= 1}
            className="px-3 py-3 bg-zinc-700 hover:bg-zinc-600 text-zinc-300 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 shadow-md hover:shadow-lg active:scale-95"
            title="Clear chat"
          >
            <Trash2 className="w-5 h-5" />
          </button>
          <button
            onClick={() => sendMessage()}
            disabled={isStreaming || !input.trim()}
            className="px-6 py-3 bg-gradient-to-r from-violet-500 to-purple-600 text-white rounded-xl hover:from-violet-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 shadow-md hover:shadow-lg active:scale-95 font-medium"
          >
            {isStreaming ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span className="hidden sm:inline">Sending...</span>
              </>
            ) : (
              <>
                <Send className="w-5 h-5" />
                <span className="hidden sm:inline">Send</span>
              </>
            )}
          </button>
        </div>
        <p className="text-xs text-zinc-400 mt-3 flex items-center gap-1">
          <span className="text-violet-400 font-medium">💡 Try:</span>
          <span>"68M stage IIIA adenocarcinoma EGFR+ PS 1" · "What ontologies are loaded?" · "Expand lung cancer biomarkers"</span>
        </p>
      </div>
    </div>
  )
}
