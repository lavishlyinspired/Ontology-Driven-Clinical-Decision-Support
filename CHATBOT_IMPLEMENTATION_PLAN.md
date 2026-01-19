# LCA Conversational Chatbot - Implementation Plan

## 🎯 Goal
Create a Claude Desktop-like chatbot interface that enables natural clinical dialogue with:
- ✅ Real-time streaming responses  
- ✅ Multi-turn conversations with context
- ✅ Agent transparency during execution
- ✅ Medical-grade UI/UX

---

## 🏗️ Architecture

```
Frontend (React)          Backend (FastAPI)           LCA System
─────────────────         ──────────────────          ───────────
Chat UI Component   ←SSE→  Conversation API    ←──→   Orchestrator
├─ Message Thread          ├─ LLM Chain                ├─ Agents
├─ Streaming Display       ├─ Session Store            ├─ Analytics
├─ Agent Status Panel      └─ Response Stream          └─ Ontology
└─ Input Box
```

---

## 📦 Technology Stack

**Frontend:**
- Next.js 14 (existing)
- shadcn/ui chat components
- EventSource API for SSE streaming
- Zustand for conversation state

**Backend:**
- FastAPI SSE endpoint
- LangChain ConversationChain
- Ollama llama3.2 (existing)
- Redis/Dict for session storage

---

## 🎨 UI Design (Claude-style)

```
┌──────────────────────────────────────────────────────┐
│  🫁 LCA Assistant                         [⚙️] [👤]  │
├──────────────────────────────────────────────────────┤
│                                                      │
│  👤 68M, Stage IIIA adenocarcinoma, EGFR Ex19del+   │
│                                                      │
│  🤖 Analyzing patient...                            │
│     ┌────────────────────────────────────────┐     │
│     │ ✅ Complexity: COMPLEX                  │     │
│     │ 🔄 NSCLCAgent executing...             │     │
│     │ ⏱️  12.3s elapsed                       │     │
│     └────────────────────────────────────────┘     │
│                                                      │
│     **Primary: Osimertinib (80mg daily)**          │
│     Evidence: Grade A | ORR: 70-80%                │
│                                                      │
│     Would you like to explore alternatives?         │
│                                                      │
├──────────────────────────────────────────────────────┤
│  💬 Ask follow-up or enter new case...    [🎤] [📎] │
└──────────────────────────────────────────────────────┘
```

---

## 🚀 Implementation

### Phase 1: Backend API (3-4 days)

#### 1. Create Conversation Service
```python
# backend/src/services/conversation_service.py

from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import json

class ConversationService:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2", temperature=0.1)
        self.sessions = {}  # session_id -> memory
        self.lca_service = LungCancerAssistantService()
    
    async def chat_stream(self, session_id: str, message: str):
        """Stream conversational responses"""
        
        # Get or create session
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferMemory()
        
        memory = self.sessions[session_id]
        
        # Classify intent
        intent = await self._classify_intent(message, memory)
        
        if intent == "patient_analysis":
            # Extract patient data and run LCA workflow
            async for chunk in self._stream_patient_analysis(message):
                yield chunk
        else:
            # General Q&A
            async for chunk in self._stream_qa_response(message, memory):
                yield chunk
    
    async def _classify_intent(self, message: str, memory):
        """Determine if message contains patient data or is a question"""
        # Check for clinical keywords
        clinical_indicators = ["patient", "stage", "histology", "year", 
                              "M", "F", "male", "female", "EGFR", "ALK"]
        
        if any(word.lower() in message.lower() for word in clinical_indicators):
            return "patient_analysis"
        return "question"
    
    async def _stream_patient_analysis(self, message: str):
        """Stream LCA workflow execution"""
        
        # Step 1: Extract patient data from natural language
        yield json.dumps({"type": "status", "content": "Extracting patient data..."}) + "\n"
        
        patient_data = await self._extract_patient_data(message)
        
        if not patient_data:
            yield json.dumps({"type": "error", "content": "Could not parse patient data"}) + "\n"
            return
        
        yield json.dumps({"type": "data", "content": patient_data}) + "\n"
        
        # Step 2: Complexity assessment
        yield json.dumps({"type": "status", "content": "Assessing complexity..."}) + "\n"
        complexity = await self.lca_service.assess_complexity(patient_data)
        yield json.dumps({"type": "complexity", "content": complexity}) + "\n"
        
        # Step 3: Run workflow with agent status updates
        yield json.dumps({"type": "status", "content": f"Running {complexity['recommended_workflow']} workflow..."}) + "\n"
        
        # Hook into orchestrator to stream agent updates
        result = await self._run_with_streaming(patient_data)
        
        # Step 4: Stream final recommendation
        yield json.dumps({"type": "recommendation", "content": result}) + "\n"
        
    async def _extract_patient_data(self, message: str) -> dict:
        """Use LLM to extract structured patient data from text"""
        
        extraction_prompt = f"""Extract patient data from this clinical note:

{message}

Return ONLY valid JSON with these fields (use null for missing):
{{
  "patient_id": "AUTO",
  "age": <number>,
  "sex": "M" or "F",
  "tnm_stage": "<stage>",
  "histology_type": "<type>",
  "performance_status": <0-4>,
  "comorbidities": [],
  "biomarker_profile": {{}}
}}"""
        
        response = await self.llm.agenerate([[extraction_prompt]])
        # Parse JSON from response
        # ... implementation
```

#### 2. Create SSE Endpoint
```python
# backend/src/api/routes/chat.py

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

router = APIRouter()
conversation_service = ConversationService()

@router.post("/chat/stream")
async def chat_stream(
    message: str,
    session_id: str = "default"
):
    """Server-Sent Events endpoint for streaming chat"""
    
    async def event_generator():
        async for chunk in conversation_service.chat_stream(session_id, message):
            yield chunk
    
    return EventSourceResponse(event_generator())

@router.get("/chat/sessions/{session_id}/history")
async def get_history(session_id: str):
    """Get conversation history"""
    return conversation_service.get_history(session_id)

@router.delete("/chat/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history"""
    conversation_service.clear_session(session_id)
    return {"status": "cleared"}
```

#### 3. Enhance Orchestrator for Streaming
```python
# backend/src/agents/dynamic_orchestrator.py (additions)

class DynamicWorkflowOrchestrator:
    
    def __init__(self, status_callback=None):
        self.status_callback = status_callback  # Callback for streaming updates
        # ... existing code
    
    async def execute_agent(self, agent_name: str, agent_function, input_data):
        """Execute agent with status streaming"""
        
        # Notify start
        if self.status_callback:
            await self.status_callback({
                "agent": agent_name,
                "status": "started",
                "timestamp": datetime.now().isoformat()
            })
        
        # Execute (existing code)
        execution = await super().execute_agent(agent_name, agent_function, input_data)
        
        # Notify completion
        if self.status_callback:
            await self.status_callback({
                "agent": agent_name,
                "status": "completed",
                "duration_ms": execution.duration_ms,
                "confidence": execution.confidence
            })
        
        return execution
```

---

### Phase 2: Frontend Chat UI (3-4 days)

#### 1. Create Chat Component
```tsx
// frontend/src/components/ChatInterface.tsx

'use client'
import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, CheckCircle } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  metadata?: {
    agentStatus?: AgentStatus[]
    complexity?: string
  }
}

interface AgentStatus {
  agent: string
  status: 'started' | 'completed' | 'failed'
  duration_ms?: number
  confidence?: number
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [currentAgents, setCurrentAgents] = useState<AgentStatus[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  
  const sessionId = useRef(crypto.randomUUID()).current

  const sendMessage = async () => {
    if (!input.trim()) return

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

    // Create assistant message placeholder
    const assistantId = crypto.randomUUID()
    let assistantContent = ''
    
    setMessages(prev => [...prev, {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date()
    }])

    // Stream response via SSE
    const eventSource = new EventSource(
      `/api/chat/stream?message=${encodeURIComponent(input)}&session_id=${sessionId}`
    )

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'status') {
        assistantContent += `\n_${data.content}_\n`
      } else if (data.type === 'agent_update') {
        setCurrentAgents(prev => {
          const updated = [...prev]
          const idx = updated.findIndex(a => a.agent === data.content.agent)
          if (idx >= 0) {
            updated[idx] = data.content
          } else {
            updated.push(data.content)
          }
          return updated
        })
      } else if (data.type === 'complexity') {
        assistantContent += `\n**Complexity: ${data.content.complexity}**\n`
      } else if (data.type === 'recommendation') {
        assistantContent += formatRecommendation(data.content)
      } else if (data.type === 'text') {
        assistantContent += data.content
      }
      
      // Update assistant message
      setMessages(prev => prev.map(msg => 
        msg.id === assistantId 
          ? { ...msg, content: assistantContent }
          : msg
      ))
    }

    eventSource.onerror = () => {
      eventSource.close()
      setIsStreaming(false)
      setCurrentAgents([])
    }
  }

  const formatRecommendation = (rec: any) => {
    return `
**Primary Recommendation: ${rec.recommendations[0]?.treatment_type}**

- Evidence Level: ${rec.recommendations[0]?.evidence_level}
- Confidence: ${(rec.recommendations[0]?.confidence_score * 100).toFixed(0)}%
- Source: ${rec.recommendations[0]?.rule_source}

${rec.mdt_summary}
    `
  }

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto">
      {/* Header */}
      <div className="border-b p-4">
        <h1 className="text-2xl font-bold">🫁 LCA Assistant</h1>
        <p className="text-sm text-gray-600">
          Clinical decision support for lung cancer
        </p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg p-4 ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-900'
              }`}
            >
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          </div>
        ))}
        
        {/* Agent Status Panel */}
        {isStreaming && currentAgents.length > 0 && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="font-semibold mb-2">Agent Execution</h3>
            <div className="space-y-2">
              {currentAgents.map((agent) => (
                <div key={agent.agent} className="flex items-center gap-2 text-sm">
                  {agent.status === 'completed' ? (
                    <CheckCircle className="w-4 h-4 text-green-600" />
                  ) : (
                    <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
                  )}
                  <span>{agent.agent}</span>
                  {agent.duration_ms && (
                    <span className="text-gray-500">({agent.duration_ms}ms)</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t p-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Describe a patient or ask a question..."
            className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isStreaming}
          />
          <button
            onClick={sendMessage}
            disabled={isStreaming || !input.trim()}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {isStreaming ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
          </button>
        </div>
      </div>
    </div>
  )
}
```

#### 2. Add Chat Route
```tsx
// frontend/src/app/chat/page.tsx

import ChatInterface from '@/components/ChatInterface'

export default function ChatPage() {
  return <ChatInterface />
}
```

---

### Phase 3: Enhancements (2-3 days)

1. **Voice Input**: Web Speech API integration
2. **File Upload**: PDF/DOCX patient report parsing
3. **Context Awareness**: Remember previous patients in session
4. **Suggested Questions**: Auto-generate follow-ups
5. **Export**: Download conversation as PDF

---

## 📊 Expected User Flow

```
1. User: "68-year-old male, stage IIIA adenocarcinoma, EGFR Ex19del+"

2. LCA: "Analyzing patient..."
   [Agent Status Panel shows live updates]
   ✅ Complexity: COMPLEX
   🔄 NSCLCAgent executing...
   
3. LCA: "**Primary: Osimertinib**
   Evidence: Grade A | ORR: 70-80%
   
   Would you like to:
   1. Explore alternatives?
   2. Check comorbidity interactions?
   3. Find similar cases?"

4. User: "What about if patient has COPD?"

5. LCA: "Running comorbidity assessment..."
   [Shows ComorbidityAgent execution]
   
   "With COPD present:
   - Osimertinib remains first-line
   - Monitor for ILD (increased risk)
   - Consider pulmonary function before treatment"
```

---

## ⚡ Quick Start (Minimal Version)

If you want to prototype quickly (1-2 days):

```python
# Minimal streaming endpoint
@app.post("/chat/simple")
async def simple_chat(message: str):
    async def stream():
        yield f"data: {{'type':'status','content':'Processing...'}}\n\n"
        
        # Extract patient data with regex
        patient_data = extract_simple(message)
        
        # Run LCA
        result = await lca_service.process_patient(patient_data)
        
        yield f"data: {{'type':'result','content':'{result.mdt_summary}'}}\n\n"
    
    return StreamingResponse(stream(), media_type="text/event-stream")
```

```tsx
// Minimal chat UI
const Chat = () => {
  const [messages, setMessages] = useState([])
  
  const send = async (msg) => {
    const es = new EventSource(`/chat/simple?message=${msg}`)
    es.onmessage = (e) => {
      const data = JSON.parse(e.data)
      setMessages(prev => [...prev, data.content])
    }
  }
  
  return <div>{/* Simple chat interface */}</div>
}
```

---

## 📋 Checklist

- [ ] Backend SSE endpoint
- [ ] Conversation service with LangChain
- [ ] Patient data extraction from text
- [ ] Streaming orchestrator updates
- [ ] Frontend chat UI component
- [ ] Real-time agent status display
- [ ] Session management
- [ ] Markdown rendering
- [ ] Error handling
- [ ] Testing with real clinical scenarios

---

**Estimated Timeline**: 7-10 days for full implementation
**Minimal Prototype**: 2-3 days

Ready to start? Let me know which approach you prefer!
