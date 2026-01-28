'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, CheckCircle, XCircle, AlertCircle, Bot, User, Activity, Zap } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp?: string
  patientData?: any
  complexity?: any
  metadata?: {
    complexity?: string
    workflow?: string
    agentStatus?: AgentStatus[]
  }
}

interface AgentStatus {
  agent: string
  status: 'started' | 'completed' | 'failed'
  duration_ms?: number
  confidence?: number
}

const JSONDisplay = ({ data }: { data: any }) => (
  <div className="bg-gray-900 rounded-xl p-4 overflow-x-auto my-3">
    <div className="flex items-center gap-2 mb-2 text-gray-400 text-xs">
      <Activity className="w-3 h-3" />
      <span>Extracted Patient Data</span>
    </div>
    <pre className="text-sm text-green-400 font-mono">
      {JSON.stringify(data, null, 2)}
    </pre>
  </div>
)

const ComplexityBadge = ({ complexity }: { complexity: any }) => (
  <div className="inline-flex items-center gap-3 bg-gradient-to-r from-orange-50 to-red-50 border-2 border-orange-200 rounded-full px-4 py-2 my-2">
    <div className="flex items-center gap-1.5">
      <Zap className="w-4 h-4 text-orange-600" />
      <span className="text-xs font-semibold text-orange-700">Complexity:</span>
      <span className="text-xs font-bold text-orange-900">{complexity.level}</span>
    </div>
    <div className="h-4 w-px bg-orange-300"></div>
    <div className="flex items-center gap-1.5">
      <Activity className="w-4 h-4 text-blue-600" />
      <span className="text-xs font-semibold text-blue-700">Workflow:</span>
      <span className="text-xs font-bold text-blue-900">{complexity.workflow}</span>
    </div>
  </div>
)

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: `# Welcome to LCA Assistant! 🫁

I help with lung cancer treatment decisions using evidence-based guidelines.

**How to use me:**
- Describe a patient case (e.g., "68M, stage IIIA adenocarcinoma, EGFR Ex19del+")
- Ask questions about treatment options
- Explore biomarker-driven therapies

Try describing a patient to get started!`,
      timestamp: 'Welcome'
    }
  ])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [currentStatus, setCurrentStatus] = useState('')
  const [currentComplexity, setCurrentComplexity] = useState<any>(null)
  const [currentPatientData, setCurrentPatientData] = useState<any>(null)
  const [suggestions, setSuggestions] = useState<string[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  
  const sessionId = useRef(crypto.randomUUID()).current

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, currentStatus])

  const sendMessage = async () => {
    if (!input.trim() || isStreaming) return

    // Add user message
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input,
      timestamp: new Date().toLocaleTimeString()
    }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsStreaming(true)
    setCurrentStatus('')
    setCurrentComplexity(null)
    setCurrentPatientData(null)
    setSuggestions([])

    // Create assistant message placeholder
    const assistantId = crypto.randomUUID()
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
      // Stream response via SSE from backend
      const response = await fetch('http://localhost:8000/api/v1/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          session_id: sessionId
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
              
              if (data.type === 'status') {
                setCurrentStatus(data.content)
              } else if (data.type === 'progress') {
                // Show progress updates in the status area
                setCurrentStatus(data.content)
                // Also append to assistant content as a subtle update
                assistantContent += `\n*${data.content}*\n\n`
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
                assistantContent += `\n## 📊 Clinical Summary\n\n${data.content}\n\n`
              } else if (data.type === 'treatment_plan') {
                assistantContent += `\n## 💊 Treatment Plan\n\n${data.content}\n\n`
              } else if (data.type === 'suggestions') {
                setSuggestions(data.content)
              } else if (data.type === 'error') {
                assistantContent += `\n\n⚠️ **Error:** ${data.content}\n`
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
          ? { ...msg, content: `⚠️ **Connection Error**\n\nCouldn't connect to the backend server. Please ensure:\n- Backend is running on http://localhost:8000\n- All services are initialized\n\nError: ${error}` }
          : msg
      ))
    } finally {
      setIsStreaming(false)
      setCurrentStatus('')
      
      // Add helpful message if no content was generated
      setMessages(prev => prev.map(msg => {
        if (msg.id === assistantId && !msg.content.trim() && !msg.patientData) {
          return { 
            ...msg, 
            content: `ℹ️ **No Response Generated**\n\nThe system processed your request but didn't generate recommendations. This might be because:\n- The patient data needs more details\n- The workflow is still being refined\n- Additional context is required\n\nTry providing more specific patient information.`
          }
        }
        return msg
      }))
    }
  }

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion)
    inputRef.current?.focus()
  }

  const clearChat = () => {
    setMessages([messages[0]]) // Keep welcome message
    setSuggestions([])
    setCurrentComplexity(null)
    setCurrentPatientData(null)
  }

  return (
    <div className="flex flex-col h-screen max-w-6xl mx-auto bg-white shadow-2xl">
      {/* Header */}
      <div className="border-b bg-gradient-to-r from-blue-600 via-blue-700 to-blue-600 text-white p-6 shadow-lg">  
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-white/20 backdrop-blur-sm rounded-xl flex items-center justify-center">
              <Bot className="w-7 h-7" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">
                LCA Assistant
              </h1>
              <p className="text-sm text-blue-100">
                Ontology-driven clinical decision support for lung cancer
              </p>
            </div>
          </div>
          <button
            onClick={clearChat}
            className="px-5 py-2.5 bg-white/20 hover:bg-white/30 backdrop-blur-sm rounded-lg text-sm font-medium transition-all hover:scale-105 active:scale-95"
          >
            Clear Chat
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-8 space-y-6 bg-gradient-to-b from-gray-50 to-white">
        {messages.map((msg, index) => (
          <div
            key={msg.id}
            className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn`}
            style={{ animationDelay: `${index * 50}ms` }}
          >
            {msg.role === 'assistant' && (
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-blue-600 to-blue-700 flex items-center justify-center shadow-md ring-2 ring-blue-200">
                <Bot className="w-6 h-6 text-white" />
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
              
              {/* Main Message Bubble */}
              <div
                className={`rounded-2xl p-5 transition-all hover:scale-[1.01] ${
                  msg.role === 'user'
                    ? 'bg-gradient-to-br from-blue-600 to-blue-700 text-white shadow-lg'
                    : 'bg-white border-2 border-gray-200 text-gray-900 shadow-sm hover:shadow-md'
                }`}
              >
                <div className={`prose prose-sm max-w-none ${
                  msg.role === 'user' ? 'prose-invert' : 'prose-blue'
                }`}>
                  {msg.content ? (
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  ) : (
                    <div className="flex items-center gap-2 text-gray-400">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-sm">Processing...</span>
                    </div>
                  )}
                </div>
                <div className={`text-xs mt-3 flex items-center gap-1 ${
                  msg.role === 'user' ? 'text-blue-100' : 'text-gray-400'
                }`}>
                  <span>{msg.timestamp ?? ''}</span>
                </div>
              </div>
            </div>

            {msg.role === 'user' && (
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-gray-400 to-gray-500 flex items-center justify-center shadow-md ring-2 ring-gray-300">
                <User className="w-6 h-6 text-white" />
              </div>
            )}
          </div>
        ))}        
        
        {/* Status Panel */}
        {isStreaming && currentStatus && (
          <div className="flex justify-start animate-fadeIn">
            <div className="bg-gradient-to-r from-blue-50 to-blue-100 border-2 border-blue-300 rounded-2xl p-5 max-w-[80%] shadow-md">
              <div className="flex items-center gap-3 text-blue-700">
                <div className="relative">
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <div className="absolute inset-0 bg-blue-400 rounded-full blur-sm opacity-30 animate-pulse"></div>
                </div>
                <span className="text-sm font-semibold">{currentStatus}</span>
              </div>
              {currentComplexity && (
                <div className="mt-3 text-xs text-blue-600 bg-white/50 rounded-lg px-3 py-2">
                  <span className="font-medium">Complexity:</span> {currentComplexity.level} | <span className="font-medium">Workflow:</span> {currentComplexity.workflow}
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* Suggestions */}
        {suggestions.length > 0 && !isStreaming && (
          <div className="flex justify-start animate-fadeIn">
            <div className="bg-gradient-to-br from-gray-50 to-gray-100 border border-gray-300 rounded-2xl p-5 max-w-[80%] shadow-sm">
              <p className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                <span className="text-lg">💡</span>
                <span>Suggested follow-ups:</span>
              </p>
              <div className="space-y-2">
                {suggestions.map((sug, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSuggestionClick(sug)}
                    className="block w-full text-left px-4 py-3 text-sm bg-white hover:bg-blue-50 border border-gray-200 hover:border-blue-300 rounded-xl transition-all hover:shadow-md hover:scale-[1.02] active:scale-[0.98]"
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

      {/* Input */}
      <div className="border-t bg-gradient-to-r from-gray-50 to-white p-6 shadow-lg">
        <div className="flex gap-3">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
            placeholder="Describe a patient or ask a question..."
            className="flex-1 px-5 py-4 border-2 border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all shadow-sm hover:border-gray-400 text-base"
            disabled={isStreaming}
          />
          <button
            onClick={sendMessage}
            disabled={isStreaming || !input.trim()}
            className="px-8 py-4 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-xl hover:from-blue-700 hover:to-blue-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 shadow-lg hover:shadow-xl hover:scale-105 active:scale-95 font-medium"
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
        <p className="text-xs text-gray-500 mt-3 flex items-center gap-1">
          <span className="text-blue-600 font-medium">💡 Example:</span>
          <span>"68-year-old male, stage IIIA adenocarcinoma, EGFR Ex19del positive, PS 1"</span>
        </p>
      </div>
    </div>
  )
}
