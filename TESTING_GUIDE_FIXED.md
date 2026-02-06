# Testing Guide - Fixed EGFR+ Recommendations

## Quick Test Cases

### Test Case 1: EGFR+ Stage IIIA (Primary Bug Fix)
**Input:**
```
68-year-old male, stage IIIA adenocarcinoma, EGFR Ex19del positive, PS 1
```

**Expected Recommendations:**
- ✅ **R8A: Adjuvant EGFR TKI (osimertinib)** - PRIMARY recommendation
- R6: Chemoradiotherapy
- R3: Radiotherapy  
- R1: Chemotherapy

**Evidence:** ADAURA trial - 89% 3-year DFS vs 53% with placebo

---

### Test Case 2: EGFR+ Stage IV (Advanced Disease)
**Input:**
```
65-year-old female, stage IV adenocarcinoma, EGFR L858R mutation, PS 1
```

**Expected Recommendations:**
- ✅ **R8: EGFR TKI (osimertinib)** - for advanced disease
- R1: Chemotherapy

---

### Test Case 3: ALK+ Stage IV
**Input:**
```
55-year-old male, stage IV adenocarcinoma, ALK rearrangement positive, PS 0
```

**Expected Recommendations:**
- ✅ **R9: ALK inhibitor (alectinib/lorlatinib)**
- R1: Chemotherapy

---

### Test Case 4: High PD-L1 Stage IV (No Driver Mutation)
**Input:**
```
70-year-old male, stage IV squamous cell carcinoma, PD-L1 85%, PS 1
```

**Expected Recommendations:**
- ✅ **R7: Immunotherapy (pembrolizumab monotherapy)**
- R1: Chemotherapy

---

### Test Case 5: Early Stage (Surgery Candidate)
**Input:**
```
60-year-old female, stage IB adenocarcinoma, EGFR wild-type, PS 0
```

**Expected Recommendations:**
- ✅ **R2: Surgical resection**

---

## Testing via Frontend (localhost:3000)

1. Open http://localhost:3000
2. Paste test case into chat
3. Submit and wait for response
4. Verify recommendations appear
5. Check that R8A appears for EGFR+ Stage IIIA patients

## Testing via Python Script

```bash
python test_patient_case.py
```

This will:
- Initialize the LCA service
- Load all 11 guideline rules
- Process test patient (EGFR+ Stage IIIA)
- Display matched recommendations

## What Was Fixed

**Problem:** Histology matching was case-sensitive
- Patient data: `"adenocarcinoma"` (lowercase)
- Matcher looking for: `"Adenocarcinoma"` (uppercase)
- Result: NO MATCH ❌

**Solution:** Made matching case-insensitive
- Convert both strings to lowercase before comparison
- Result: MATCH ✅

**Files Changed:**
- `backend/src/ontology/guideline_rules.py` - Fixed `_is_nsclc()` and `_is_sclc()`

## Verification

Run the backend logs and look for:
```
[INFO] ✓ Found 4 applicable guidelines
[INFO]   - R8A: AdjuvantTargetedTherapy
```

If you see R8A in the list, the fix is working! 🎉
