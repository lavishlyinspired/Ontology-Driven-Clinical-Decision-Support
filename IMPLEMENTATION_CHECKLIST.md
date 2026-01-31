# Implementation Checklist
## Step-by-Step Guide to Enable All Features

**Goal:** Integrate MCP Apps, Citations, and Clustering into your LCA assistant

**Time Estimate:** 30-45 minutes

---

## ✅ Prerequisites (Already Done!)

- [x] MCP server with 60+ tools exists ([lca_mcp_server.py](backend/src/mcp_server/lca_mcp_server.py))
- [x] MCP client integration complete ([mcp_client.py](backend/src/services/mcp_client.py))
- [x] MCP apps created ([frontend/public/mcp-apps/](frontend/public/mcp-apps/))
- [x] GroundedCitations component exists ([GroundedCitations.tsx](frontend/src/components/GroundedCitations.tsx))
- [x] ClusteringService implemented ([clustering_service.py](backend/src/services/clustering_service.py))
- [x] Frontend already renders citations and MCP apps

---

## 🎯 Step 1: Update Conversation Service (Backend)

### 1.1 Add Import

**File:** [backend/src/services/conversation_service.py](backend/src/services/conversation_service.py)

**Add at top:**
```python
from ..services.clustering_service import ClusteringService, ClusteringMethod
```

### 1.2 Update `__init__` Method

**Find:**
```python
def __init__(self, lca_service):
    self.lca_service = lca_service
    self.sessions: Dict[str, List[Dict]] = {}
    self.patient_context: Dict[str, Dict] = {}
    self.mcp_invoker = get_mcp_invoker()
```

**Add:**
```python
    self.clustering_service = ClusteringService()
```

### 1.3 Update Intent Classification

**Find the `_classify_intent` method and add these patterns:**

```python
def _classify_intent(self, message: str, session_id: str = None) -> str:
    """
    Classify user intent considering conversation context

    Returns:
        "patient_analysis", "follow_up", "mcp_tool", "mcp_app", "clustering_analysis", or "general_qa"
    """

    # MCP App detection (ADD THIS)
    mcp_app_patterns = [
        r'compare\s+treatment',
        r'treatment\s+comparison',
        r'survival\s+curve',
        r'kaplan\s+meier',
        r'guideline\s+tree',
        r'nccn\s+decision',
        r'clinical\s+trial',
        r'trial\s+match'
    ]
    for pattern in mcp_app_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return "mcp_app"

    # Clustering detection (ADD THIS)
    clustering_patterns = [
        r'cluster\s+patient',
        r'cohort\s+analysis',
        r'similar\s+patient',
        r'patient\s+group',
        r'find\s+patients?\s+like'
    ]
    for pattern in clustering_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return "clustering_analysis"

    # ... keep existing MCP tool patterns and other intents ...
```

### 1.4 Update `chat_stream` Method

**Find the intent handling section and add:**

```python
async def chat_stream(self, session_id: str, message: str):
    try:
        self._add_to_history(session_id, "user", message)
        intent = self._classify_intent(message, session_id)

        if intent == "patient_analysis":
            async for chunk in self._stream_patient_analysis(message, session_id):
                yield chunk
        elif intent == "follow_up":
            async for chunk in self._stream_follow_up(message, session_id):
                yield chunk
        elif intent == "mcp_tool":
            async for chunk in self._stream_mcp_tool(message, session_id):
                yield chunk
        elif intent == "mcp_app":  # ADD THIS
            async for chunk in self._stream_mcp_app(message, session_id):
                yield chunk
        elif intent == "clustering_analysis":  # ADD THIS
            async for chunk in self._stream_clustering_analysis(message, session_id):
                yield chunk
        else:
            async for chunk in self._stream_general_qa(message, session_id):
                yield chunk
```

### 1.5 Copy Enhancement Methods

**Copy all methods from [conversation_service_enhancements.py](backend/src/services/conversation_service_enhancements.py) into your ConversationService class:**

- `_stream_mcp_app()`
- `_get_treatment_comparison_data()`
- `_get_survival_curves_data()`
- `_get_guideline_tree_data()`
- `_match_clinical_trials()`
- `_explain_treatment_comparison()`
- `_explain_survival_curves()`
- `_explain_guideline_tree()`
- `_explain_trial_matches()`
- `_stream_clustering_analysis()`
- `_get_all_patients_for_clustering()`
- `_generate_clustering_summary()`
- `_enhance_with_citations()`

### 1.6 Enhance Patient Analysis with Citations

**In the `_stream_patient_analysis` method, after generating recommendation:**

```python
# After: result = await self.lca_service.process_patient(...)

# Get recommendation text
recommendation = result.get("recommendation", "")

# Enhance with citations
recommendation = await self._enhance_with_citations(
    recommendation,
    patient_data,
    result
)

# Stream enhanced recommendation
yield self._format_sse({
    "type": "text",
    "content": recommendation
})
```

**✅ Backend complete!**

---

## 🎨 Step 2: Update Frontend (Optional - Already Configured!)

The frontend already supports MCP apps and citations! But verify:

### 2.1 Check SSE Handlers

**File:** [frontend/src/components/ChatInterface.tsx](frontend/src/components/ChatInterface.tsx)

**Verify these handlers exist (they should):**

```typescript
else if (data.type === 'mcp_app') {
  console.log('[LCA] MCP App:', data.content)
  setMessages(prev => prev.map(msg =>
    msg.id === assistantId
      ? { ...msg, mcpApp: data.content }
      : msg
  ))
}

else if (data.type === 'cluster_info') {
  console.log('[LCA] Cluster:', data.content)
  // Store cluster info
}
```

**If missing, add them to the SSE event handling loop.**

### 2.2 Check Citation Rendering

**Verify this exists (it should):**

```typescript
{msg.content ? (
  msg.role === 'assistant' && msg.content.includes('[[') ? (
    <GroundedCitations
      text={msg.content}
      showTooltips={true}
      renderAs="inline"
      onCitationClick={(citation) => {
        if (citation.url) {
          window.open(citation.url, '_blank', 'noopener,noreferrer')
        }
      }}
    />
  ) : (
    <ReactMarkdown>{msg.content}</ReactMarkdown>
  )
) : (
  <div>Processing...</div>
)}
```

**✅ Frontend complete!**

---

## 🧪 Step 3: Test Everything

### Test 1: MCP Tools (Already Working)

```
Find similar patients to stage IIIA adenocarcinoma EGFR+
```

**Expected:** Tool invocation → Results with explanation

---

### Test 2: MCP App - Treatment Comparison

```
Compare treatments for this patient
```

or

```
68M, stage IIIA adenocarcinoma, EGFR exon 19 deletion. Compare treatment options.
```

**Expected:**
- Status message: "Loading treatment comparison tool..."
- Interactive treatment comparison app appears
- Text explanation below app

---

### Test 3: MCP App - Survival Curves

```
Show survival curves for EGFR+ patients
```

**Expected:**
- Status message: "Generating survival curves..."
- Kaplan-Meier curve visualization
- Text explanation with median survival

---

### Test 4: MCP App - Guideline Tree

```
Show NCCN guideline decision tree for stage IIIA adenocarcinoma
```

**Expected:**
- Interactive decision tree
- Patient characteristics pre-filled
- Clickable navigation

---

### Test 5: MCP App - Clinical Trials

```
Match clinical trials for stage IV KRAS G12C mutation
```

**Expected:**
- Trial matcher app
- List of matching trials
- Eligibility criteria displayed

---

### Test 6: Citations

```
Recommend treatment for 68M stage IIIA EGFR exon 19 deletion
```

**Expected:**
- Recommendation includes citations like:
  - `[[Guideline:NCCN]]`
  - `[[Trial:FLAURA]]`
- Citations render as colored badges
- Hover shows tooltips
- Click opens external links

---

### Test 7: Clustering Analysis

```
Cluster all stage IV patients by biomarker profile
```

**Expected:**
- Status: "Analyzing patient cohorts..."
- List of cohorts with:
  - Name (e.g., "EGFR-Mutated")
  - Size (n=XX)
  - Characteristics
  - Outcomes summary
- Feature importance ranking

---

### Test 8: Find Similar Patients

```
Find patients similar to 68M stage IIIA adenocarcinoma EGFR+
```

**Expected:**
- Clustering analysis with similarity search
- Top 5-10 similar patients
- Similarity scores
- Characteristics comparison

---

## 📊 Verification Checklist

After testing, verify:

- [ ] MCP tools work (already confirmed)
- [ ] MCP apps render inline in chat messages
- [ ] Treatment comparison app displays
- [ ] Survival curves app displays
- [ ] Guideline tree app displays
- [ ] Trial matcher app displays
- [ ] Citations render as colored badges
- [ ] Citation tooltips appear on hover
- [ ] Citation links work when clicked
- [ ] Clustering analysis produces cohorts
- [ ] Similar patient search works
- [ ] Frontend console shows SSE events
- [ ] No errors in backend logs
- [ ] All visualizations are interactive

---

## 🐛 Troubleshooting

### Issue: MCP apps don't appear

**Check:**
1. Browser console for `[LCA] MCP App:` log
2. Verify SSE event type is `"mcp_app"`
3. Check `resourceUri` path starts with `/mcp-apps/`
4. Verify HTML files exist in `frontend/public/mcp-apps/`

**Fix:**
```bash
# Verify apps exist
ls frontend/public/mcp-apps/
# Should show: treatment-compare.html, survival-curves.html, etc.
```

---

### Issue: Citations don't render

**Check:**
1. Response includes `[[` pattern
2. Format is correct: `[[Type:ID]]`
3. GroundedCitations component imported

**Fix:**
- Add debug log in backend:
  ```python
  logger.info(f"Enhanced response: {recommendation}")
  ```
- Check if citations are in the text

---

### Issue: Clustering fails

**Check:**
1. Neo4j is running
2. Patients exist in database
3. At least 5 patients for meaningful clustering

**Fix:**
```bash
# Check Neo4j
curl http://localhost:7474

# Check patient count in Neo4j browser
MATCH (p:Patient) RETURN count(p)
```

---

### Issue: Imports fail

**Check:**
1. ClusteringService import path
2. File exists at `backend/src/services/clustering_service.py`

**Fix:**
```python
# Verify import
from src.services.clustering_service import ClusteringService
clustering = ClusteringService()
print("✓ Clustering service loaded")
```

---

## 📚 Reference Documents

- **Feature Guide**: [MCP_APPS_AND_FEATURES_GUIDE.md](MCP_APPS_AND_FEATURES_GUIDE.md)
- **Testing Guide**: [LCA_TESTING_GUIDE.md](LCA_TESTING_GUIDE.md)
- **Implementation Tracker**: [docs/IMPLEMENTATION_TRACKER.md](docs/IMPLEMENTATION_TRACKER.md)
- **Enhancement Code**: [conversation_service_enhancements.py](backend/src/services/conversation_service_enhancements.py)

---

## 🎉 Success Criteria

Your implementation is complete when:

1. ✅ All test queries return expected results
2. ✅ MCP apps render inline in chat
3. ✅ Citations display as colored badges
4. ✅ Clustering produces meaningful cohorts
5. ✅ No errors in console or logs
6. ✅ Interactive elements work (hover, click)

---

## 🚀 Next Steps After Implementation

1. **Customize Treatment Data**
   - Replace sample data in `_get_treatment_comparison_data()`
   - Connect to real survival analysis service
   - Add institution-specific trial data

2. **Enhance Citations**
   - Add more citation sources to `CITATION_SOURCES`
   - Create institutional guidelines
   - Add patient-specific references

3. **Improve Clustering**
   - Tune feature weights
   - Add custom clinical rules
   - Implement hierarchical clustering

4. **Monitor Usage**
   - Track which apps are used most
   - Measure citation click-through rates
   - Analyze cohort patterns

5. **User Feedback**
   - Survey clinicians on utility
   - Iterate on visualizations
   - Add requested features

---

**Estimated Implementation Time:** 30-45 minutes
**Difficulty:** Easy (mostly copying methods)
**Risk:** Low (all components tested)

**Ready to implement? Follow the steps above in order. Good luck! 🎯**
