# LCA Assistant Testing Guide
## Comprehensive Test Coverage - Phase 1 to Phase 6

**Date:** February 1, 2026
**Version:** 22.0
**Purpose:** Complete testing coverage for all LCA assistant capabilities including MCP Tools, MCP Apps, Clustering, and Citations

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: Core Patient Analysis](#phase-1-core-patient-analysis)
3. [Phase 2: Follow-Up Handling](#phase-2-follow-up-handling)
4. [Phase 3: MCP Tool Integration](#phase-3-mcp-tool-integration)
5. [Phase 4: MCP App Integration](#phase-4-mcp-app-integration)
6. [Phase 5: Clustering Analysis](#phase-5-clustering-analysis)
7. [Phase 6: Citations Enhancement](#phase-6-citations-enhancement)
8. [Edge Cases & Error Handling](#edge-cases--error-handling)
9. [Multi-Turn Conversations](#multi-turn-conversations)
10. [Advanced Analytics](#advanced-analytics)
11. [Graph Queries & Neo4j](#graph-queries--neo4j)
12. [Export & Reporting](#export--reporting)
13. [Verification Checklist](#verification-checklist)
14. [Troubleshooting](#troubleshooting)

---

## Overview

The LCA (Lung Cancer Assistant) system provides clinical decision support through:
- **11-Agent Integrated Workflow** for patient analysis
- **60+ MCP Tools** for specialized clinical operations
- **4 Interactive MCP Apps** for data visualization
- **Patient Clustering** for cohort analysis
- **Grounded Citations** for evidence-based recommendations

---

## Phase 1: Core Patient Analysis

### Test 1.1: Simple NSCLC Case
**Query:**
```
68 year old male, stage IIIA adenocarcinoma, EGFR exon 19 deletion, ECOG PS 1
```

**Expected Results:**
- [ ] Patient data extraction shows all fields
- [ ] Complexity assessment: MODERATE or COMPLEX
- [ ] Workflow selection: integrated workflow
- [ ] Agent pipeline execution visible in timeline
- [ ] Primary recommendation: Osimertinib (EGFR+ targeted therapy)
- [ ] Evidence level: Category 1 / Level A
- [ ] Execution time displayed

**Verification Points:**
- [ ] `patient_data` SSE event with extracted fields
- [ ] `complexity` SSE event with level/workflow
- [ ] `progress` SSE events showing agent steps
- [ ] Final `text` SSE event with formatted recommendations

---

### Test 1.2: Complex SCLC Case with Comorbidities
**Query:**
```
72F, extensive stage SCLC, PS 2, comorbidities: COPD (FEV1 45%), atrial fibrillation on warfarin, CKD stage 3
```

**Expected Results:**
- [ ] SCLC staging recognized (Extensive Stage)
- [ ] Comorbidity agent activation
- [ ] Drug interaction assessment for warfarin
- [ ] Dose adjustment recommendations for CKD
- [ ] Treatment: Platinum-etoposide chemotherapy
- [ ] Risk stratification for PS 2

---

### Test 1.3: Biomarker-Negative NSCLC
**Query:**
```
65 year old female, stage IV adenocarcinoma, EGFR negative, ALK negative, ROS1 negative, PD-L1 65%, no actionable mutations
```

**Expected Results:**
- [ ] High PD-L1 detected (>=50%)
- [ ] Immunotherapy recommendation (pembrolizumab)
- [ ] Clinical trial matching for PD-L1 high
- [ ] Chemotherapy + immunotherapy combinations considered

---

### Test 1.4: Early Stage Resectable NSCLC
**Query:**
```
58M, stage IA2 adenocarcinoma, EGFR negative, PS 0, no comorbidities
```

**Expected Results:**
- [ ] Surgical recommendation (lobectomy)
- [ ] Adjuvant therapy not recommended for stage IA
- [ ] Good prognosis indicated
- [ ] Follow-up surveillance recommendations

---

## Phase 2: Follow-Up Handling

### Test 2.1: Treatment Alternatives
**After Test 1.1, ask:**
```
What are alternative treatment options for this patient?
```

**Expected Results:**
- [ ] Context maintained from previous analysis
- [ ] 2-4 alternative regimens provided
- [ ] Evidence levels for each alternative
- [ ] Trade-off explanations (efficacy vs. toxicity)
- [ ] No new patient analysis triggered

---

### Test 2.2: Side Effect Assessment
**Query:**
```
What are the main side effects I should watch for with this treatment?
```

**Expected Results:**
- [ ] Lists common adverse events for recommended treatment
- [ ] Management strategies provided
- [ ] Monitoring recommendations
- [ ] When to seek medical attention

---

### Test 2.3: Prognosis Inquiry
**Query:**
```
What is the expected prognosis for this patient?
```

**Expected Results:**
- [ ] Survival estimates (median OS, PFS)
- [ ] Stage-specific survival data
- [ ] Biomarker impact on outcomes
- [ ] Performance status considerations
- [ ] Uncertainty acknowledgment

---

### Test 2.4: Biomarker Details
**Query:**
```
Tell me more about the EGFR mutation and treatment options
```

**Expected Results:**
- [ ] EGFR mutation biology explained
- [ ] Available TKI options listed
- [ ] Generation differences (1st, 2nd, 3rd gen TKIs)
- [ ] Resistance mechanisms mentioned

---

## Phase 3: MCP Tool Integration

### Test 3.1: Survival Analysis Tool
**Query:**
```
Analyze survival data for stage IIIA EGFR+ patients
```

**Expected Results:**
- [ ] Tool invocation: `analyze_survival_data`
- [ ] Tool call visible in workflow timeline
- [ ] Kaplan-Meier estimates returned
- [ ] Explanatory clinical context
- [ ] Collapsible tool input/output display

---

### Test 3.2: Find Similar Patients Tool
**Query:**
```
Find similar patients to a 68M with stage IIIA adenocarcinoma EGFR exon 19 deletion
```

**Expected Results:**
- [ ] Tool invocation: `find_similar_patients`
- [ ] Returns top 5 similar cases (or explains if none found)
- [ ] Similarity scores shown
- [ ] Matching criteria explained

---

### Test 3.3: Clinical Trial Matching Tool
**Query:**
```
Match clinical trials for stage IV NSCLC with KRAS G12C mutation
```

**Expected Results:**
- [ ] Tool invocation: `match_clinical_trials`
- [ ] Matching trials from ClinicalTrials.gov
- [ ] Eligibility criteria displayed
- [ ] Contact information provided
- [ ] Match scores shown

---

### Test 3.4: Biomarker Pathway Tool
**Query:**
```
Get biomarker pathways for EGFR mutations
```

**Expected Results:**
- [ ] Tool invocation: `get_biomarker_pathways`
- [ ] Pathway information returned
- [ ] Affected biological processes
- [ ] Therapeutic implications

---

### Test 3.5: Lab Result Interpretation Tool
**Query:**
```
Interpret lab results for a lung cancer patient
```

**Expected Results:**
- [ ] Tool invocation: `interpret_lab_results`
- [ ] Requests specific lab values if not provided
- [ ] Clinical interpretation
- [ ] Flagged abnormal results

---

### Test 3.6: Clinical Report Generation Tool
**Query:**
```
Generate clinical report for current patient
```

**Expected Results:**
- [ ] Tool invocation: `generate_clinical_report`
- [ ] Structured MDT summary
- [ ] All relevant clinical data included
- [ ] Export-ready format

---

## Phase 4: MCP App Integration

### Test 4.1: Treatment Comparison App
**Query:**
```
Compare treatments for this patient
```

**Expected Results:**
- [ ] Intent detected: `mcp_app`
- [ ] Status message: "Loading treatment comparison tool..."
- [ ] `mcp_app` SSE event with resourceUri
- [ ] Interactive comparison UI loads
- [ ] Treatment data populated (ORR, PFS, OS, toxicity)
- [ ] Text explanation accompanies visualization

**Visual Verification:**
- [ ] Side-by-side treatment comparison
- [ ] Evidence levels visible
- [ ] NCCN category shown

---

### Test 4.2: Survival Curves App
**Query:**
```
Show survival curves
```

**Alternative triggers:**
- "Show Kaplan Meier analysis"
- "Display survival data"

**Expected Results:**
- [ ] Kaplan-Meier curves rendered
- [ ] Multiple treatment arms shown
- [ ] Median survival values
- [ ] Curve separation visible
- [ ] Legend with treatment names

---

### Test 4.3: Guideline Tree App
**Query:**
```
Show guideline tree
```

**Alternative triggers:**
- "NCCN decision tree"
- "Show decision pathways"

**Expected Results:**
- [ ] Interactive decision tree loads
- [ ] Patient-specific pathway highlighted
- [ ] Stage/histology/biomarker considered
- [ ] NCCN version displayed
- [ ] Expandable/collapsible nodes

---

### Test 4.4: Clinical Trial Matcher App
**Query:**
```
Match clinical trials
```

**Alternative triggers:**
- "Find clinical trials"
- "Trial search"

**Expected Results:**
- [ ] Trial matcher interface loads
- [ ] Matching trials displayed
- [ ] Eligibility criteria shown
- [ ] Match scores visible
- [ ] Contact information provided

---

## Phase 5: Clustering Analysis

### Test 5.1: Basic Clustering
**Query:**
```
Cluster all patients
```

**Expected Results:**
- [ ] Intent detected: `clustering_analysis`
- [ ] Status: "Analyzing patient cohorts..."
- [ ] Patients fetched from Neo4j
- [ ] `cluster_info` SSE events for each cluster
- [ ] Summary with cohort descriptions

---

### Test 5.2: Clinical Rules Clustering
**Query:**
```
Cluster patients using clinical rules
```

**Expected Results:**
- [ ] ClusteringMethod: CLINICAL_RULES
- [ ] Clinically meaningful clusters
- [ ] Stage-based groupings
- [ ] Biomarker-based groupings
- [ ] Outcomes summary per cluster

---

### Test 5.3: K-Means Clustering
**Query:**
```
Cluster patients using k-means algorithm
```

**Expected Results:**
- [ ] ClusteringMethod: KMEANS
- [ ] Feature importance shown
- [ ] Silhouette score displayed
- [ ] Cluster characteristics
- [ ] Statistical quality metrics

---

### Test 5.4: Stage-Specific Clustering
**Query:**
```
Cluster all stage IV patients
```

**Expected Results:**
- [ ] Filters to stage IV patients
- [ ] Sub-cohorts within stage IV
- [ ] Biomarker-driven groups
- [ ] Treatment response patterns
- [ ] Outcomes comparison

---

## Phase 6: Citations Enhancement

### Test 6.1: NCCN Guideline Citations
**After patient analysis, check:**
```
Recommendations should include [[Guideline:NCCN]]
```

**Expected Results:**
- [ ] NCCN citation badge appears
- [ ] Tooltip with guideline details
- [ ] Link to NCCN reference

---

### Test 6.2: Clinical Trial Citations
**For EGFR+ patient:**
```
Osimertinib should include [[Trial:FLAURA]]
```

**For ALK+ patient:**
```
Alectinib should include [[Trial:ALEX]]
```

**For high PD-L1 patient:**
```
Pembrolizumab should include [[Trial:KEYNOTE-024]]
```

**Expected Results:**
- [ ] Trial citation badges appear
- [ ] Correct trial matched to drug
- [ ] Tooltips with trial information

---

### Test 6.3: Ontology Citations
**Check for:**
```
Adenocarcinoma should include [[Ontology:SNOMED]]
```

**Expected Results:**
- [ ] SNOMED-CT citation badge
- [ ] Concept code reference
- [ ] Terminology linkage

---

## Edge Cases & Error Handling

### Test E.1: Incomplete Patient Data
**Query:**
```
65 year old male, lung cancer
```

**Expected Results:**
- [ ] Partial extraction acknowledged
- [ ] Missing fields listed (stage, histology)
- [ ] Helpful hints provided
- [ ] No crash or empty response

---

### Test E.2: Contradictory Information
**Query:**
```
Stage IA small cell lung cancer (SCLC)
```

**Expected Results:**
- [ ] Flags staging contradiction
- [ ] SCLC uses limited/extensive staging
- [ ] Asks for clarification
- [ ] Provides staging system explanation

---

### Test E.3: Unsupported Tool Request
**Query:**
```
Use the quantum analyzer tool to predict outcomes
```

**Expected Results:**
- [ ] Graceful handling of unknown tool
- [ ] Available tool categories listed
- [ ] Valid alternatives suggested
- [ ] No error thrown

---

### Test E.4: Empty Neo4j Database
**When no patients in database:**

**Expected Results:**
- [ ] Clustering returns helpful message
- [ ] Explains minimum patient requirement
- [ ] Suggests adding patient data
- [ ] No crash

---

## Multi-Turn Conversations

### Test M.1: Iterative Refinement
**Sequence:**
```
1. "68M, stage IIIA adenocarcinoma"
2. "EGFR exon 19 deletion positive"
3. "Patient also has severe COPD with FEV1 40%"
```

**Expected Results:**
- [ ] Each turn updates analysis
- [ ] Context maintained across turns
- [ ] Recommendations refined with new info
- [ ] Changes explained at each step

---

### Test M.2: Comparative Analysis
**Sequence:**
```
1. "65F, stage IV adenocarcinoma, EGFR+, PS 1"
2. "How would the recommendation differ for a similar patient who is EGFR negative but PD-L1 80%?"
```

**Expected Results:**
- [ ] Comparative analysis performed
- [ ] Key differences highlighted
- [ ] Treatment strategy differences explained
- [ ] Prognosis comparison provided

---

## Advanced Analytics

### Test A.1: Risk Stratification
**Query:**
```
Stratify risk for stage IIIA patients based on biomarker status
```

**Expected Results:**
- [ ] Risk categories returned
- [ ] Prognostic factors shown
- [ ] Treatment intensity recommendations

---

### Test A.2: Counterfactual Analysis
**Query:**
```
What would happen if we used chemotherapy instead of targeted therapy for this EGFR+ patient?
```

**Expected Results:**
- [ ] Counterfactual comparison
- [ ] Outcome differences shown
- [ ] Explanation of why targeted therapy is preferred

---

### Test A.3: Uncertainty Quantification
**Query:**
```
Quantify the uncertainty in the survival estimate for this patient
```

**Expected Results:**
- [ ] Confidence intervals shown
- [ ] Sources of uncertainty explained
- [ ] Monte Carlo simulation results (if available)

---

## Graph Queries & Neo4j

### Test G.1: Knowledge Graph Query
**Query:**
```
Query the knowledge graph for all treatment pathways for stage IIIA NSCLC
```

**Expected Results:**
- [ ] Tool invocation: `execute_graph_query`
- [ ] Graph visualization data returned
- [ ] Nodes and relationships shown
- [ ] Displays in graph panel

---

### Test G.2: Ontology Mapping
**Query:**
```
Map the concept "adenocarcinoma" to SNOMED-CT
```

**Expected Results:**
- [ ] Tool invocation: `validate_ontology`
- [ ] SNOMED code returned
- [ ] Concept hierarchy shown
- [ ] Synonyms and relationships listed

---

## Export & Reporting

### Test R.1: FHIR Export
**Query:**
```
Export patient data for P001 in FHIR format
```

**Expected Results:**
- [ ] Tool invocation: `export_patient_data`
- [ ] FHIR-compliant JSON returned
- [ ] All clinical data included
- [ ] Validates against FHIR schema

---

### Test R.2: MDT Summary Generation
**Query:**
```
Generate an MDT summary for this patient
```

**Expected Results:**
- [ ] Tool invocation: `generate_mdt_summary`
- [ ] Structured clinical summary
- [ ] Treatment recommendations
- [ ] Discussion points for MDT

---

## Verification Checklist

### Backend Verification
- [ ] Python syntax valid: `python -m py_compile backend/src/services/conversation_service.py`
- [ ] Imports resolve correctly
- [ ] No runtime errors on startup
- [ ] Logging working correctly

### Frontend Verification
- [ ] SSE connection established
- [ ] All event types handled (text, status, progress, patient_data, complexity, mcp_app, cluster_info, tool_call, tool_result)
- [ ] MCP apps load in iframe
- [ ] Citations render as badges
- [ ] Workflow timeline updates in real-time

### Integration Verification
- [ ] Backend serves chat stream correctly
- [ ] Frontend receives and parses SSE events
- [ ] MCP tools invokable from chat
- [ ] Neo4j queries execute successfully
- [ ] Clustering produces valid results

---

## Troubleshooting

### Issue 1: MCP Tools Not Detected
**Symptom:** Questions go to general Q&A instead of invoking tools

**Fix:**
- Check `_classify_intent()` patterns in conversation_service.py
- Verify pattern matching with explicit test
- Try more explicit phrasing: "Use the find_similar_patients tool"

---

### Issue 2: Tool Invocation Fails
**Symptom:** "Tool execution failed" error

**Checks:**
1. Neo4j running: `curl http://localhost:7474`
2. Ollama running: `curl http://localhost:11434`
3. MCP server initialized
4. Required arguments provided

---

### Issue 3: MCP App Not Loading
**Symptom:** App iframe shows blank or error

**Checks:**
1. HTML files exist in `/frontend/public/mcp-apps/`
2. ResourceUri path correct
3. Data passed correctly
4. Browser console for JS errors

---

### Issue 4: Clustering Returns Empty
**Symptom:** "Not enough patient data"

**Checks:**
1. Neo4j database has >=5 patients
2. Patient nodes have required properties
3. Graph connection successful
4. Query permissions correct

---

### Issue 5: Citations Not Rendering
**Symptom:** Raw [[Type:ID]] text shown instead of badges

**Checks:**
1. GroundedCitations component imported
2. Citation parsing regex correct
3. Badge styles applied
4. Check for escaped characters

---

## Quick Start Testing

```bash
# 1. Start backend
cd backend
python -m uvicorn src.api.main:app --reload --port 8000

# 2. Start frontend
cd frontend
npm run dev

# 3. Open http://localhost:3000

# 4. Run test sequence:
# a) Patient Analysis
"68M, stage IIIA adenocarcinoma, EGFR+"

# b) MCP App
"Compare treatments"

# c) Clustering
"Cluster all patients"

# d) Follow-up
"What are the side effects?"

# e) Tool Invocation
"Find similar patients"
```

---

## Test Results Template

```markdown
### Test Results - [Date]

**Tester:** [Name]
**Environment:** Development / Staging / Production
**Backend Version:** [commit hash]
**Frontend Version:** [commit hash]

#### Phase 1: Core Patient Analysis
| Test ID | Description | Status | Notes |
|---------|-------------|--------|-------|
| 1.1 | Simple NSCLC | Pass/Fail | |
| 1.2 | Complex SCLC | Pass/Fail | |
| 1.3 | Biomarker-Negative | Pass/Fail | |
| 1.4 | Early Stage | Pass/Fail | |

#### Phase 2: Follow-Up Handling
| Test ID | Description | Status | Notes |
|---------|-------------|--------|-------|
| 2.1 | Alternatives | Pass/Fail | |
| 2.2 | Side Effects | Pass/Fail | |
| 2.3 | Prognosis | Pass/Fail | |
| 2.4 | Biomarkers | Pass/Fail | |

#### Phase 3: MCP Tool Integration
| Test ID | Description | Status | Notes |
|---------|-------------|--------|-------|
| 3.1 | Survival Analysis | Pass/Fail | |
| 3.2 | Similar Patients | Pass/Fail | |
| 3.3 | Trial Matching | Pass/Fail | |
| 3.4 | Biomarker Pathways | Pass/Fail | |
| 3.5 | Lab Interpretation | Pass/Fail | |
| 3.6 | Report Generation | Pass/Fail | |

#### Phase 4: MCP App Integration
| Test ID | Description | Status | Notes |
|---------|-------------|--------|-------|
| 4.1 | Treatment Comparison | Pass/Fail | |
| 4.2 | Survival Curves | Pass/Fail | |
| 4.3 | Guideline Tree | Pass/Fail | |
| 4.4 | Trial Matcher | Pass/Fail | |

#### Phase 5: Clustering Analysis
| Test ID | Description | Status | Notes |
|---------|-------------|--------|-------|
| 5.1 | Basic Clustering | Pass/Fail | |
| 5.2 | Clinical Rules | Pass/Fail | |
| 5.3 | K-Means | Pass/Fail | |
| 5.4 | Stage-Specific | Pass/Fail | |

#### Phase 6: Citations Enhancement
| Test ID | Description | Status | Notes |
|---------|-------------|--------|-------|
| 6.1 | NCCN Citations | Pass/Fail | |
| 6.2 | Trial Citations | Pass/Fail | |
| 6.3 | Ontology Citations | Pass/Fail | |

#### Overall Assessment
- **Core Functionality:** Pass/Fail
- **MCP Integration:** Pass/Fail
- **Clustering:** Pass/Fail
- **Citations:** Pass/Fail
- **Performance:** Acceptable/Needs Work
- **User Experience:** Good/Needs Improvement

#### Issues Found
1. [Description of issue]
2. [Description of issue]

#### Recommendations
1. [Recommendation]
2. [Recommendation]
```

---

## Success Criteria

For the system to be considered production-ready:

1. **Patient Analysis:** 100% success rate for valid patient data
2. **Response Time:** <5 seconds for typical queries
3. **MCP Tools:** All 60+ tools accessible and functional
4. **MCP Apps:** All 4 apps load and render correctly
5. **Clustering:** Produces meaningful cohorts with >=5 patients
6. **Citations:** Appropriate citations attached to recommendations
7. **Error Handling:** Graceful degradation for edge cases
8. **Context Retention:** Maintains context across 10+ turns

---

**Document Version:** 2.0
**Last Updated:** February 1, 2026
**Author:** LCA Development Team
