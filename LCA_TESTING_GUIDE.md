# LCA Assistant Testing Guide
## Comprehensive Test Questions for All Functionalities

**Date:** February 1, 2026
**Version:** 22.0
**Purpose:** Validate MCP integration and verify all LCA assistant capabilities

---

## ✅ Fixed Issues

### 1. MCP Apps Integration
**Problem:** MCP apps were not integrated with the LCA assistant chat interface. The MCP server existed but was isolated (stdio-only for Claude Desktop).

**Solution Implemented:**
- Created `MCPToolInvoker` class in [backend/src/services/mcp_client.py](backend/src/services/mcp_client.py)
- Integrated MCP tool detection in conversation service
- Added streaming support for MCP tool calls and results
- Updated frontend to display tool executions with collapsible input/output

**Benefits:**
- All 60+ MCP tools are now accessible from the web chat interface
- Detailed explanatory responses for all tool results
- Real-time tool execution progress in workflow timeline
- Tool call transparency (users see what tools are being invoked)

---

## 📋 Test Categories

### Category 1: Patient Analysis (Core Workflow)
**Tests the 11-agent integrated workflow for patient case analysis**

#### Test 1.1: Simple NSCLC Case
```
68 year old male, stage IIIA adenocarcinoma, EGFR exon 19 deletion, ECOG PS 1
```

**Expected:**
- ✓ Patient data extraction
- ✓ Complexity assessment
- ✓ 11-agent workflow execution
- ✓ Treatment recommendation
- ✓ Biomarker-driven precision medicine suggestions

#### Test 1.2: Complex SCLC Case with Comorbidities
```
72F, extensive stage SCLC, PS 2, comorbidities: COPD (FEV1 45%), atrial fibrillation on warfarin, CKD stage 3
```

**Expected:**
- ✓ Comorbidity agent activation
- ✓ Drug interaction assessment
- ✓ Dose adjustment recommendations
- ✓ Risk stratification
- ✓ Treatment alternatives for high-risk patients

#### Test 1.3: Biomarker-Negative NSCLC
```
65 year old female, stage IV adenocarcinoma, EGFR negative, ALK negative, ROS1 negative, PD-L1 65%, no actionable mutations
```

**Expected:**
- ✓ Biomarker agent assessment
- ✓ Immunotherapy recommendation (high PD-L1)
- ✓ Clinical trial matching
- ✓ Chemotherapy + immunotherapy combinations

---

### Category 2: Follow-Up Questions
**Tests context-aware follow-up handling**

#### Test 2.1: Treatment Alternatives
After any patient analysis, ask:
```
What are alternative treatment options for this patient?
```

**Expected:**
- ✓ Recalls patient context
- ✓ Provides 2-3 alternative regimens
- ✓ Explains trade-offs
- ✓ Evidence-based reasoning

#### Test 2.2: Side Effect Assessment
```
What are the main side effects I should watch for with this treatment?
```

**Expected:**
- ✓ Lists common adverse events
- ✓ Management strategies
- ✓ When to seek medical attention
- ✓ Monitoring recommendations

#### Test 2.3: Prognosis Inquiry
```
What is the expected prognosis for this patient?
```

**Expected:**
- ✓ Survival estimates (median OS, PFS)
- ✓ Stage-specific data
- ✓ Biomarker impact on outcomes
- ✓ Uncertainty quantification

---

### Category 3: MCP Tool Invocation
**Tests the newly integrated MCP tool functionality**

#### Test 3.1: Survival Analysis
```
Analyze survival data for stage IIIA EGFR+ patients
```

**Expected:**
- ✓ Detects `analyze_survival_data` tool
- ✓ Shows tool invocation in workflow
- ✓ Returns Kaplan-Meier estimates
- ✓ Provides explanatory context

#### Test 3.2: Find Similar Patients
```
Find similar patients to a 68M with stage IIIA adenocarcinoma EGFR exon 19 deletion
```

**Expected:**
- ✓ Detects `find_similar_patients` tool
- ✓ Returns top 5 similar cases
- ✓ Shows similarity scores
- ✓ Explains matching criteria

#### Test 3.3: Clinical Trial Matching
```
Match clinical trials for stage IV NSCLC with KRAS G12C mutation
```

**Expected:**
- ✓ Detects `match_clinical_trials` tool
- ✓ Returns matching trials from ClinicalTrials.gov
- ✓ Shows eligibility criteria
- ✓ Provides trial contact info

#### Test 3.4: Biomarker Pathway Analysis
```
Get biomarker pathways for EGFR mutations
```

**Expected:**
- ✓ Detects `get_biomarker_pathways` tool
- ✓ Returns pathway information
- ✓ Shows affected biological processes
- ✓ Explains therapeutic implications

#### Test 3.5: Lab Result Interpretation
```
Interpret lab results for a lung cancer patient
```

**Expected:**
- ✓ Detects `interpret_lab_results` tool
- ✓ Requests specific lab values
- ✓ Provides clinical interpretation
- ✓ Flags abnormal results

#### Test 3.6: Generate Clinical Report
```
Generate clinical report for current patient
```

**Expected:**
- ✓ Detects `generate_clinical_report` tool
- ✓ Creates structured MDT summary
- ✓ Includes all relevant clinical data
- ✓ Export-ready format

---

### Category 4: General Q&A
**Tests general medical knowledge and LCA capabilities**

#### Test 4.1: Guideline Questions
```
What are the NCCN guidelines for treating stage IIIA NSCLC?
```

**Expected:**
- ✓ Guideline summary
- ✓ Treatment options by scenario
- ✓ Evidence levels
- ✓ Recent updates

#### Test 4.2: Biomarker Explanation
```
Explain the difference between EGFR exon 19 deletions and exon 21 L858R mutations
```

**Expected:**
- ✓ Molecular biology explanation
- ✓ Clinical significance
- ✓ Treatment response differences
- ✓ Prognosis implications

#### Test 4.3: Staging Clarification
```
What does TNM stage T2aN1M0 mean for lung cancer?
```

**Expected:**
- ✓ TNM breakdown
- ✓ Tumor size interpretation
- ✓ Node involvement
- ✓ Overall stage group

---

### Category 5: Edge Cases & Error Handling

#### Test 5.1: Incomplete Patient Data
```
65 year old male, lung cancer
```

**Expected:**
- ✓ Identifies missing fields
- ✓ Requests: stage, histology, biomarkers
- ✓ Helpful error message
- ✓ Suggests what to provide

#### Test 5.2: Contradictory Information
```
Stage IA small cell lung cancer (SCLC)
```

**Expected:**
- ✓ Flags contradiction (SCLC usually uses limited/extensive staging)
- ✓ Asks for clarification
- ✓ Explains staging systems

#### Test 5.3: Unsupported MCP Tool Request
```
Use the quantum analyzer tool to predict outcomes
```

**Expected:**
- ✓ Gracefully handles unknown tool
- ✓ Lists available tool categories
- ✓ Suggests valid alternatives

---

### Category 6: Multi-Turn Conversations

#### Test 6.1: Iterative Refinement
1. Start with basic case:
   ```
   68M, stage IIIA adenocarcinoma
   ```
2. Add biomarker data:
   ```
   EGFR exon 19 deletion positive
   ```
3. Add comorbidity:
   ```
   Patient also has severe COPD with FEV1 40%
   ```

**Expected:**
- ✓ Updates recommendations at each step
- ✓ Maintains context across turns
- ✓ Adjusts for new information
- ✓ Explains changes in recommendation

#### Test 6.2: Comparative Analysis
1. Analyze patient 1:
   ```
   65F, stage IV adenocarcinoma, EGFR+, PS 1
   ```
2. Compare to patient 2:
   ```
   How would the recommendation differ for a similar patient who is EGFR negative but PD-L1 80%?
   ```

**Expected:**
- ✓ Compares treatment strategies
- ✓ Highlights key differences
- ✓ Explains biomarker-driven decisions
- ✓ Discusses prognosis differences

---

### Category 7: Advanced Analytics (MCP Tools)

#### Test 7.1: Risk Stratification
```
Stratify risk for stage IIIA patients based on biomarker status
```

**Expected:**
- ✓ Invokes `stratify_risk` tool
- ✓ Returns risk categories
- ✓ Shows prognostic factors
- ✓ Recommends treatment intensity

#### Test 7.2: Counterfactual Analysis
```
What would happen if we used chemotherapy instead of targeted therapy for this EGFR+ patient?
```

**Expected:**
- ✓ Invokes `analyze_counterfactuals` tool
- ✓ Compares outcomes
- ✓ Shows survival differences
- ✓ Explains why targeted therapy is preferred

#### Test 7.3: Uncertainty Quantification
```
Quantify the uncertainty in the survival estimate for this patient
```

**Expected:**
- ✓ Invokes `quantify_uncertainty` tool
- ✓ Shows confidence intervals
- ✓ Explains sources of uncertainty
- ✓ Monte Carlo simulation results

---

### Category 8: Graph Queries & Neo4j Integration

#### Test 8.1: Graph Query
```
Query the knowledge graph for all treatment pathways for stage IIIA NSCLC
```

**Expected:**
- ✓ Invokes `execute_graph_query` tool
- ✓ Returns graph visualization data
- ✓ Shows nodes and relationships
- ✓ Displays in graph panel

#### Test 8.2: Ontology Mapping
```
Map the concept "adenocarcinoma" to SNOMED-CT
```

**Expected:**
- ✓ Invokes `validate_ontology` tool
- ✓ Returns SNOMED code
- ✓ Shows concept hierarchy
- ✓ Lists synonyms and relationships

---

### Category 9: Export & Reporting

#### Test 9.1: Patient Data Export
```
Export patient data for P001 in FHIR format
```

**Expected:**
- ✓ Invokes `export_patient_data` tool
- ✓ Returns FHIR-compliant JSON
- ✓ Includes all clinical data
- ✓ Validates against FHIR schema

#### Test 9.2: MDT Summary Generation
```
Generate an MDT summary for this patient
```

**Expected:**
- ✓ Invokes `generate_mdt_summary` tool
- ✓ Structured clinical summary
- ✓ Treatment recommendations
- ✓ Discussion points for team

---

## 🔍 How to Verify MCP Integration is Working

### Visual Indicators:
1. **Workflow Timeline** should show:
   - "🔧 Invoking tool: [tool_name]"
   - "✅ Tool execution completed"

2. **Message Display** should show:
   - Collapsible tool call sections with yellow ⚡ icon
   - Input arguments (expandable)
   - Result output (expandable, green text)

3. **Response Text** should include:
   - Tool result explanation
   - Structured markdown formatting
   - Clinical interpretation

### Console Logs (Browser DevTools):
```
[LCA] Tool call: { tool: "analyze_survival_data", arguments: {...} }
[LCA] Tool result: { status: "success", result: {...} }
```

---

## 🐛 Troubleshooting

### Issue 1: MCP Tools Not Detected
**Symptom:** Questions like "Find similar patients" go to general Q&A instead of invoking tools

**Fix:**
- Check [backend/src/services/conversation_service.py:116-175](backend/src/services/conversation_service.py#L116-L175) for intent classification
- Verify patterns in `_classify_intent` method
- Try more explicit phrasing: "Use the find_similar_patients tool"

### Issue 2: Tool Invocation Fails
**Symptom:** "Tool execution failed" error

**Possible Causes:**
1. **Neo4j not running:** Check `curl http://localhost:7474`
2. **Components not initialized:** First tool call may take longer
3. **Missing arguments:** Check tool schema requirements

**Fix:**
- Verify Neo4j connection in [.env](/.env#L43-L46)
- Check backend logs for detailed error
- Ensure patient data exists in Neo4j

### Issue 3: Frontend Not Displaying Tool Calls
**Symptom:** Tool executes but doesn't show in UI

**Fix:**
- Check browser console for SSE events
- Verify `tool_call` and `tool_result` event handlers in [ChatInterface.tsx:465-533](frontend/src/components/ChatInterface.tsx#L465-L533)
- Clear browser cache and reload

### Issue 4: "Components will load on first use"
**Symptom:** First tool call takes 30+ seconds

**Expected Behavior:** This is normal - MCP server initializes components lazily:
- Loads LUCADA ontology
- Connects to Neo4j
- Initializes agents
- Subsequent calls are fast

---

## 📊 Success Metrics

After testing, you should observe:

1. **Patient Analysis:**
   - ✅ 100% success rate for valid patient data
   - ✅ <5 second response time
   - ✅ All 11 agents execute successfully

2. **MCP Tool Integration:**
   - ✅ Tools detected from natural language
   - ✅ Tool calls visible in UI
   - ✅ Results displayed with explanations
   - ✅ No MCP-related errors in logs

3. **Follow-Up Handling:**
   - ✅ Context maintained across conversation
   - ✅ Relevant suggestions provided
   - ✅ Accurate responses to clinical questions

4. **Error Handling:**
   - ✅ Graceful degradation for missing data
   - ✅ Helpful error messages
   - ✅ Recovery suggestions

---

## 🚀 Quick Start Testing Sequence

**Recommended testing order:**

1. **Basic Patient Analysis** (Test 1.1)
2. **MCP Tool: Find Similar Patients** (Test 3.2)
3. **Follow-Up Question** (Test 2.1)
4. **MCP Tool: Survival Analysis** (Test 3.1)
5. **Complex Case with Comorbidities** (Test 1.2)

This sequence tests:
- Core workflow ✅
- MCP integration ✅
- Context awareness ✅
- Advanced analytics ✅
- Complex reasoning ✅

---

## 📝 Test Results Template

```markdown
### Test Results - [Date]

**Tester:** [Name]
**Environment:** Development / Production

| Test ID | Description | Status | Notes |
|---------|-------------|--------|-------|
| 1.1 | Simple NSCLC | ✅ / ❌ | |
| 1.2 | Complex SCLC | ✅ / ❌ | |
| 3.1 | Survival Analysis Tool | ✅ / ❌ | |
| 3.2 | Find Similar Patients | ✅ / ❌ | |
| ... | ... | ... | ... |

**Overall Assessment:**
- MCP Integration: ✅ / ❌
- Response Quality: ✅ / ❌
- Performance: ✅ / ❌

**Issues Found:**
1. [Issue description]
2. [Issue description]
```

---

## 🔗 Related Files

**Backend:**
- [backend/src/services/mcp_client.py](backend/src/services/mcp_client.py) - MCP tool invoker
- [backend/src/services/conversation_service.py](backend/src/services/conversation_service.py) - Chat service with MCP integration
- [backend/src/mcp_server/lca_mcp_server.py](backend/src/mcp_server/lca_mcp_server.py) - MCP server with 60+ tools

**Frontend:**
- [frontend/src/components/ChatInterface.tsx](frontend/src/components/ChatInterface.tsx) - Chat UI with tool display

**Configuration:**
- [.env](/.env) - Environment variables (Neo4j, Ollama, etc.)

---

## 💡 Tips for Effective Testing

1. **Use Specific Patient Data:** Include age, stage, histology, biomarkers for best results
2. **Try Natural Language:** Don't just use exact tool names - test intent detection
3. **Check Browser Console:** Look for SSE events and error messages
4. **Monitor Backend Logs:** Use `LOG_LEVEL=DEBUG` for detailed tracing
5. **Test Edge Cases:** Invalid data, missing fields, contradictions
6. **Multi-Turn Conversations:** Test context retention across multiple messages
7. **Graph Visualization:** Check if graph data appears in the graph panel

---

## 🎯 Next Steps

After completing initial testing:

1. **Performance Optimization**
   - Profile slow tool calls
   - Implement caching for frequent queries
   - Optimize Neo4j queries

2. **Enhanced Explanations**
   - Improve `_explain_tool_result` method
   - Add more clinical context
   - Include evidence citations

3. **Additional MCP Tools**
   - Drug interaction checker
   - Radiation therapy planner
   - Genomic variant interpreter

4. **UI Improvements**
   - Tool call badges in timeline
   - Collapsible tool history
   - Export tool results

---

**Happy Testing! 🧪**

For questions or issues, check:
- Backend logs: `logs/lca_system.log`
- Frontend console: Browser DevTools → Console
- Neo4j browser: http://localhost:7474
