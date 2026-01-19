'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, CheckCircle, XCircle, AlertCircle, Bot, User } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
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
      timestamp: new Date()
    }
  ])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [currentStatus, setCurrentStatus] = useState('')
  const [complexity, setComplexity] = useState<any>(null)
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
      timestamp: new Date()
    }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsStreaming(true)
    setCurrentStatus('')
    setComplexity(null)
    setSuggestions([])

    // Create assistant message placeholder
    const assistantId = crypto.randomUUID()
    let assistantContent = ''
    
    setMessages(prev => [...prev, {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date()
    }])

    try {
      // Stream response via SSE
      const response = await fetch('/api/v1/chat/stream', {
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
              } else if (data.type === 'patient_data') {
                assistantContent += `\n📋 **Extracted Patient Data:**\n\`\`\`json\n${JSON.stringify(data.content, null, 2)}\n\`\`\`\n`
              } else if (data.type === 'complexity') {
                setComplexity(data.content)
                assistantContent += `\n🎯 **Complexity:** ${data.content.level} | **Workflow:** ${data.content.workflow}\n\n`
              } else if (data.type === 'recommendation' || data.type === 'text') {
                assistantContent += data.content
              } else if (data.type === 'suggestions') {
                setSuggestions(data.content)
              } else if (data.type === 'error') {
                assistantContent += `\n\n⚠️ **Error:** ${data.content}\n`
              }
              
              // Update assistant message
              setMessages(prev => prev.map(msg => 
                msg.id === assistantId 
                  ? { ...msg, content: assistantContent }
                  : msg
              ))
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }
    } catch (error) {
      console.error('Stream error:', error)
      setMessages(prev => prev.map(msg => 
        msg.id === assistantId 
          ? { ...msg, content: `⚠️ Error: ${error}` }
          : msg
      ))
    } finally {
      setIsStreaming(false)
      setCurrentStatus('')
    }
  }

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion)
    inputRef.current?.focus()
  }

  const clearChat = () => {
    setMessages([messages[0]]) // Keep welcome message
    setSuggestions([])
    setComplexity(null)
  }

  return (
    <div className="flex flex-col h-screen max-w-5xl mx-auto bg-white">
      {/* Header */}
      <div className="border-b bg-gradient-to-r from-blue-600 to-blue-700 text-white p-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <Bot className="w-7 h-7" />
              LCA Assistant
            </h1>
            <p className="text-sm text-blue-100">
              Ontology-driven clinical decision support for lung cancer
            </p>
          </div>
          <button
            onClick={clearChat}
            className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-sm transition-colors"
          >
            Clear Chat
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-gray-50">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {msg.role === 'assistant' && (
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
            )}
            
            <div
              className={`max-w-[80%] rounded-lg p-4 ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white border border-gray-200 text-gray-900'
              }`}
            >
              <div className="prose prose-sm max-w-none">
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>
              <div className={`text-xs mt-2 ${msg.role === 'user' ? 'text-blue-100' : 'text-gray-500'}`}>
                {msg.timestamp.toLocaleTimeString()}
              </div>
            </div>

            {msg.role === 'user' && (
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center">
                <User className="w-5 h-5 text-gray-700" />
              </div>
            )}
          </div>
        ))}
        
        {/* Status Panel */}
        {isStreaming && currentStatus && (
          <div className="flex justify-start">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 max-w-[80%]">
              <div className="flex items-center gap-2 text-blue-700">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="text-sm font-medium">{currentStatus}</span>
              </div>
              {complexity && (
                <div className="mt-2 text-xs text-blue-600">
                  Complexity: {complexity.level} | Workflow: {complexity.workflow}
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* Suggestions */}
        {suggestions.length > 0 && !isStreaming && (
          <div className="flex justify-start">
            <div className="bg-gray-100 border border-gray-200 rounded-lg p-4 max-w-[80%]">
              <p className="text-sm font-medium text-gray-700 mb-2">💡 Suggested follow-ups:</p>
              <div className="space-y-1">
                {suggestions.map((sug, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSuggestionClick(sug)}
                    className="block w-full text-left px-3 py-2 text-sm bg-white hover:bg-blue-50 border border-gray-200 rounded transition-colors"
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
      <div className="border-t bg-white p-4">
        <div className="flex gap-3">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
            placeholder="Describe a patient or ask a question..."
            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isStreaming}
          />
          <button
            onClick={sendMessage}
            disabled={isStreaming || !input.trim()}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            {isStreaming ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
            <span className="hidden sm:inline">Send</span>
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Example: "68-year-old male, stage IIIA adenocarcinoma, EGFR Ex19del positive, PS 1"
        </p>
      </div>
    </div>
  )
}
