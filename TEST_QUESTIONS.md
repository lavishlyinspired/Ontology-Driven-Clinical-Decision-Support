# LCA Assistant Test Questions

Use these questions to test different complexity levels and capabilities of the Lung Cancer Assistant.

## Simple (SIMPLE) - Early Stage NSCLC

### Test Case 1: Stage IA NSCLC
```
55M, stage IA adenocarcinoma, PS 0
```
**Expected**: Surgery recommendation (lobectomy), high confidence

### Test Case 2: Stage IB NSCLC
```
62 year old female, stage IB squamous cell carcinoma, ECOG 1
```
**Expected**: Surgical resection, possibly adjuvant chemotherapy

## Moderate (MODERATE) - Locally Advanced

### Test Case 3: Stage IIIA with EGFR (Your screenshot case)
```
68M, stage IIIA adenocarcinoma, EGFR Ex19del+
```
**Expected**: Osimertinib (1st-line EGFR TKI) based on FLAURA trial, curative intent

### Test Case 4: Stage IIIA without biomarkers
```
70 year old male, stage IIIA adenocarcinoma, PS 1, PD-L1 45%
```
**Expected**: Concurrent chemoradiotherapy + durvalumab (PACIFIC regimen)

### Test Case 5: Stage IIIB with comorbidities
```
65F, stage IIIB adenocarcinoma, PS 2, COPD, diabetes
```
**Expected**: Sequential chemoradiotherapy (safer for PS 2), comorbidity alerts

## Complex (COMPLEX) - Advanced Stage

### Test Case 6: Stage IV NSCLC with high PD-L1
```
58 year old female, stage IV adenocarcinoma, PD-L1 75%, PS 1
```
**Expected**: Pembrolizumab monotherapy, high confidence based on KEYNOTE-024

### Test Case 7: Stage IV with ALK rearrangement
```
45M, stage IV adenocarcinoma, ALK positive, PS 0
```
**Expected**: Alectinib (1st-line ALK inhibitor), excellent CNS activity mentioned

### Test Case 8: Stage IV with ROS1
```
52 year old male, stage IV adenocarcinoma, ROS1 positive
```
**Expected**: Entrectinib, good CNS penetration

### Test Case 9: Stage IV Squamous Cell
```
67M, stage IV squamous cell carcinoma, PD-L1 20%, PS 1
```
**Expected**: Pembrolizumab + platinum doublet chemotherapy

## Critical (CRITICAL) - Complex Multi-factorial

### Test Case 10: Poor PS with multiple comorbidities
```
78F, stage IV adenocarcinoma, PS 3, CKD, diabetes, hypertension, EGFR L858R
```
**Expected**: Cautious approach, possibly single-agent or BSC discussion, full comorbidity analysis

### Test Case 11: Young patient with SCLC
```
42M, small cell lung cancer, limited stage, PS 1
```
**Expected**: SCLC-specific treatment (chemotherapy + radiotherapy), urgent MDT discussion

### Test Case 12: Extensive SCLC
```
65 year old male, SCLC extensive stage, PS 2, brain metastases
```
**Expected**: Platinum-etoposide + immunotherapy, prophylactic cranial irradiation consideration

---

## Follow-up Questions to Test

After any patient analysis, try these follow-ups:

1. **Alternative treatments**: "Show alternative treatments" or "What are other options?"
2. **Reasoning**: "Explain the reasoning" or "Why was this recommended?"
3. **Similar cases**: "Find similar cases" or "Any similar patients?"
4. **Comorbidities**: "Assess comorbidity interactions" or "What about drug interactions?"
5. **Biomarkers**: "Tell me more about biomarkers" or "What biomarker tests should be done?"
6. **Prognosis**: "What's the prognosis?" or "Survival outlook?"
7. **Clinical trials**: "Check clinical trial eligibility" or "Any trials available?"

---

## Questions to Test General Q&A

```
What are the NICE guidelines for lung cancer treatment?
```

```
When should I test for ALK rearrangement?
```

```
What is the difference between NSCLC and SCLC treatment?
```

```
How does PD-L1 expression affect treatment choice?
```

---

## Expected Workflow Steps (Visible in Timeline)

For a typical patient analysis, you should see these workflow steps:

1. "Extracting patient data from your message..."
2. "Checking existing patient records in Neo4j..." (if Neo4j enabled)
3. "Assessing case complexity..."
4. "Running integrated/basic workflow..."
5. "Matched guidelines: R1, R2..." (specific rules)
6. "Analysis complete (XXXms)"

---

## Troubleshooting

### No recommendations generated?
- Check that all required fields are present: age, sex, stage, histology
- Verify biomarker format: `EGFR Ex19del+` not `EGFR+` alone
- Check backend logs for agent execution errors

### Follow-ups not working?
- Make sure you've done an initial patient analysis first
- Use the suggested follow-up buttons or similar phrases
- Check that session ID is maintained between messages

### Neo4j not persisting?
- Verify `CHAT_USE_NEO4J=true` in environment
- Check Neo4j is running on localhost:7687
- Look for "Connecting to Neo4j..." in backend logs
