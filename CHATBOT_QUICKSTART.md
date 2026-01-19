# LCA Chatbot - Quick Start Guide

## 🎉 What's New

You now have a **Claude Desktop-style conversational interface** for LCA!

### Features
✅ Natural language patient input  
✅ Real-time streaming responses  
✅ Live agent execution status  
✅ Automatic patient data extraction  
✅ Complexity-based workflow routing  
✅ Suggested follow-up questions  

---

## 🚀 Getting Started

### 1. Install Dependencies

**Backend:**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### 2. Start Backend

```bash
cd backend
python -m uvicorn src.api.main:app --reload --port 8000
```

API will be available at: `http://localhost:8000`

### 3. Start Frontend

```bash
cd frontend
npm run dev
```

Frontend will be available at: `http://localhost:3000`

### 4. Open Chat Interface

Navigate to: **`http://localhost:3000/chat`**

---

## 💬 How to Use

### Example 1: Simple Patient Analysis
```
You: 68-year-old male, stage IIIA adenocarcinoma, EGFR Ex19del positive, PS 1

LCA: 🔍 Extracting patient data...
     📊 Assessing complexity...
     ⚙️ Running integrated workflow...
     
     ## Primary Recommendation: Osimertinib
     - Evidence Level: Grade A
     - Confidence: 85%
     - Source: NCCN 2025
```

### Example 2: With Comorbidities
```
You: 72M, stage IV adenocarcinoma, EGFR+, PS 2, COPD and diabetes

LCA: [Streams analysis with ComorbidityAgent execution visible]
     
     **Primary: Osimertinib with dose monitoring**
     - Comorbidity risk score: Moderate
     - Contraindications: Monitor for ILD
```

### Example 3: Biomarker-Driven
```
You: 65F, stage IIIB adenocarcinoma, PD-L1 75%, no driver mutations

LCA: [Executes BiomarkerAgent]
     
     **Primary: Pembrolizumab + Chemotherapy**
     - Based on high PD-L1 expression
     - Evidence: KEYNOTE-189 trial
```

### Example 4: Follow-up Questions
```
You: What are alternative treatments?

LCA: Based on the previous patient analysis:
     
     **Alternatives:**
     1. Carboplatin + Pemetrexed (Evidence: A, 75%)
     2. Radiation therapy (Evidence: B, 65%)
```

---

## 🎨 Chat Interface Features

### Real-Time Status Updates
While processing, you'll see:
```
🔍 Extracting patient data...
📊 Assessing complexity...
⚙️ Running integrated workflow...
✅ Analysis complete (12,345ms)
```

### Complexity Display
```
🎯 Complexity: COMPLEX | Workflow: integrated
```

### Suggested Follow-ups
After analysis, clickable suggestions appear:
- Show alternative treatments
- Assess comorbidity interactions
- Find similar cases
- Explain the reasoning

### Session Management
- Conversations are session-based
- Each browser tab = new session
- Click "Clear Chat" to reset

---

## 📊 What Happens Behind the Scenes

```
1. User Input → Natural Language Parser
   ↓
2. Regex-based Patient Data Extraction
   ↓
3. LCA Service → Complexity Assessment
   ↓
4. Dynamic Workflow Orchestrator
   ├─ SIMPLE → Basic workflow (4 agents)
   └─ COMPLEX → Advanced workflow (10+ agents)
   ↓
5. Streaming Response to Frontend
   ├─ Status updates
   ├─ Agent execution tracking
   └─ Final recommendations
   ↓
6. Suggested Follow-ups Generated
```

---

## 🔧 API Endpoints

### Chat Streaming
```http
POST /api/v1/chat/stream
Content-Type: application/json

{
  "message": "68M, stage IIIA adenocarcinoma, EGFR+",
  "session_id": "optional-session-id"
}

Response: text/event-stream
data: {"type":"status","content":"Processing..."}
data: {"type":"complexity","content":{"level":"complex"}}
data: {"type":"recommendation","content":"..."}
```

### Get History
```http
GET /api/v1/chat/sessions/{session_id}/history

Response: {
  "session_id": "abc123",
  "messages": [...],
  "message_count": 5
}
```

### Clear Session
```http
DELETE /api/v1/chat/sessions/{session_id}

Response: {"status":"success"}
```

---

## 🧪 Testing

### Test with CLI
```bash
# Test patient data extraction
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"68M, stage IIIA adenocarcinoma, EGFR Ex19del+","session_id":"test"}'
```

### Test in Browser
1. Open `http://localhost:3000/chat`
2. Type: "65-year-old female, stage IV adenocarcinoma, PD-L1 80%"
3. Watch the streaming response!

---

## 🐛 Troubleshooting

### Backend not streaming?
Check CORS settings in `backend/src/api/main.py`:
```python
allow_origins=["http://localhost:3000"]
```

### Frontend not receiving events?
Check browser console for errors. Ensure:
- Backend is running on port 8000
- Frontend is running on port 3000
- No CORS errors

### Patient data not extracted?
The regex patterns in `conversation_service.py` look for:
- Age: "68-year-old" or "68M"
- Sex: "male", "female", "M", "F"
- Stage: "stage IIIA" or "IIIA"
- Histology: "adenocarcinoma", "squamous cell", "small cell"

Make sure your input includes these!

---

## 🚀 Next Steps

### Enhancements You Can Add

1. **Voice Input**
   - Use Web Speech API
   - Add microphone button to chat input

2. **File Upload**
   - Parse PDF patient reports
   - Extract data from clinical notes

3. **LLM Integration**
   - Replace regex with LLM-based extraction
   - Use Ollama for Q&A responses

4. **Agent Transparency**
   - Show live agent execution graph
   - Display confidence scores per agent

5. **Export**
   - Download conversation as PDF
   - Export recommendations to EHR format

---

## 📖 Example Conversations

### Full Clinical Scenario
```
User: I have a 68-year-old male patient with stage IIIA adenocarcinoma. 
      He has EGFR exon 19 deletion. Performance status is 1. 
      He also has COPD and is a former smoker.

LCA: 🔍 Extracting patient data...
     
     📋 Extracted Patient Data:
     {
       "age": 68,
       "sex": "M",
       "tnm_stage": "IIIA",
       "histology_type": "Adenocarcinoma",
       "performance_status": 1,
       "biomarker_profile": {
         "egfr_mutation": true,
         "egfr_mutation_type": "Ex19del"
       },
       "comorbidities": ["COPD"]
     }
     
     📊 Assessing complexity...
     🎯 Complexity: COMPLEX | Workflow: integrated
     
     ⚙️ Running integrated workflow...
     ✅ Analysis complete (18,234ms)
     
     ## Primary Recommendation: Osimertinib
     
     - Evidence Level: Grade A
     - Confidence: 90%
     - Source: NCCN NSCLC Guidelines 2025
     - Intent: Curative
     - Expected ORR: 70-80%
     
     **Comorbidity Considerations:**
     - COPD present: Monitor for ILD
     - Risk score: Moderate
     
     **Clinical Summary:**
     This patient has resectable stage IIIA NSCLC with EGFR Ex19del mutation.
     First-line osimertinib is strongly recommended based on Grade A evidence.
     Given COPD history, close monitoring for interstitial lung disease is advised.
     
     💡 Suggested follow-ups:
     - Show alternative treatments
     - Assess comorbidity interactions
     - Find similar cases
     - Explain the reasoning

User: What about if the patient refuses targeted therapy?

LCA: If targeted therapy is refused, alternative options include:
     
     **Alternative Recommendations:**
     1. Carboplatin + Pemetrexed chemotherapy
        - Evidence: Grade A
        - Standard first-line for stage III NSCLC
     
     2. Concurrent chemoradiation
        - Evidence: Grade B
        - For locally advanced disease
     
     However, I strongly recommend discussing the benefits of osimertinib
     given the EGFR mutation and superior outcomes vs chemotherapy.
```

---

## ✅ Checklist

- [x] Backend conversation service created
- [x] SSE streaming endpoint implemented
- [x] Patient data extraction with regex
- [x] Frontend chat UI component
- [x] Markdown rendering for responses
- [x] Real-time status updates
- [x] Session management
- [x] Suggested follow-ups
- [x] Error handling
- [x] Integration with existing LCA service

---

**Ready to chat with your patients!** 🎉

Try it now: `http://localhost:3000/chat`
