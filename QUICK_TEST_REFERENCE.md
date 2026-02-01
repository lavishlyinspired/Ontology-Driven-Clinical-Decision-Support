# 🚀 **LCA Frontend Testing Quick Reference Card**

## **Immediate Test Setup (2 minutes)**

```powershell
# 1. Start all services
cd h:/akash/git/CoherencePLM/Version22
.venv/Scripts/Activate.ps1

# Terminal 1: Backend
python start_backend.py

# Terminal 2: Frontend  
cd frontend
npm run dev

# 3. Verify: http://localhost:3000 (Frontend) & http://localhost:8000/health (Backend)
```

---

## **⚡ 5-Minute Quick Tests**

### **Test 1: Basic Function ✅**
```
Input: "Hello, can you help me with a lung cancer case?"
Expected: Professional greeting + offer to help + asks for patient details
```

### **Test 2: Simple Case 🏥**
```
Input: "65-year-old male with 2cm lung nodule in right upper lobe"
Expected: Acknowledges case + asks follow-up questions about staging
```

### **Test 3: Memory Test 🧠**
```
1. "Patient is 58-year-old female"
2. "She has chest pain and weight loss" 
3. "What's her risk profile based on what I told you?"
Expected: References 58-year-old female from earlier
```

### **Test 4: Enhanced Features 🚀**
```
Input: "Patient has ground-glass opacities"
Expected: Follow-up questions appear automatically
Check: Look for follow-up suggestions in UI
```

---

## **🎯 Core Functionality Tests (15 minutes)**

### **Staging Test**
```
Input: "Help me stage: 3.2cm tumor, mediastinal nodes, no mets"
Expected: T2aN2M0, Stage IIIA discussion
```

### **Biomarker Test**
```
Input: "Adenocarcinoma with EGFR exon 19 deletion, PD-L1 75%"
Expected: EGFR TKI recommendation, immunotherapy discussion
```

### **Complex Case Test**
```
Input: "72-year-old, 50 pack-years, 4.5cm mass + liver lesions"
Expected: Stage IV, palliative approach, treatment options
```

---

## **🔧 Automated Testing**

### **Run All Tests**
```bash
python run_frontend_conversation_tests.py
```

### **Run Single Test**
```bash
python run_frontend_conversation_tests.py --test-id CONV_001
```

### **List Available Tests**
```bash
python run_frontend_conversation_tests.py --list
```

---

## **✅ Success Checkmarks**

| Feature | Test | ✓ |
|---------|------|---|
| **Basic Chat** | Responds to greeting | ☐ |
| **Medical Knowledge** | Provides staging info | ☐ |
| **Memory** | Remembers patient details | ☐ |
| **Follow-ups** | Generates smart questions | ☐ |
| **Complex Cases** | Handles multi-step analysis | ☐ |
| **Error Handling** | Manages unclear input | ☐ |
| **Performance** | Responds in <5 seconds | ☐ |

---

## **🚨 Common Issues & Quick Fixes**

| Problem | Quick Fix |
|---------|-----------|
| **No response** | Check `http://localhost:8000/health` |
| **Slow responses** | Restart backend service |
| **Memory issues** | Clear browser cache |
| **Import errors** | Re-run `.venv/Scripts/Activate.ps1` |

---

## **📊 Test Result Interpretation**

### **✅ Good Signs**
- Responds within 5 seconds
- Remembers previous conversation
- Generates relevant follow-up questions
- Provides medical staging/analysis
- Professional tone throughout

### **❌ Red Flags**
- No response or timeout
- Doesn't remember patient details
- Generic/non-medical responses
- Contradictory information
- System errors in console

---

## **🎯 Quick Quality Check**

**30-Second Test**: Type `"Help me with a 65-year-old smoker with lung cancer"`

**Good Response Should Include**:
- Acknowledgment of patient details
- Request for more specific information
- Relevant medical questions
- Professional medical language
- Follow-up suggestions

---

## **📞 Emergency Contacts**

| Issue Type | Action |
|------------|--------|
| **System Down** | Check all services running |
| **Medical Errors** | Review conversation_service.py |
| **UI Problems** | Check frontend console logs |
| **Performance** | Monitor system resources |

---

*Keep this card handy during testing sessions! 🧪*