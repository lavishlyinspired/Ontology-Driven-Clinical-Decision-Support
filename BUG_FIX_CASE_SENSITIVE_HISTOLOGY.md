# Bug Fix: Case-Sensitive Histology Matching

## Issue
Patient cases with EGFR+ Stage IIIA adenocarcinoma were not receiving appropriate treatment recommendations despite having matching rules (R8A) in the system.

## Root Cause
The `_is_nsclc()` method in `guideline_rules.py` was performing case-sensitive string matching:

```python
# BEFORE (BUG):
def _is_nsclc(self, histology: str) -> bool:
    nsclc_types = [
        "NonSmallCellCarcinoma",
        "Adenocarcinoma",  # Capital 'A'
        "SquamousCellCarcinoma",
        "LargeCellCarcinoma"
    ]
    return any(h in histology for h in nsclc_types)
```

When patient data had `histology_type: "adenocarcinoma"` (lowercase), the check for `"Adenocarcinoma" in "adenocarcinoma"` failed due to case mismatch.

## Solution
Made histology type matching case-insensitive:

```python
# AFTER (FIXED):
def _is_nsclc(self, histology: str) -> bool:
    nsclc_types = [
        "NonSmallCellCarcinoma",
        "Adenocarcinoma",
        "SquamousCellCarcinoma",
        "LargeCellCarcinoma"
    ]
    # Make case-insensitive comparison
    histology_lower = histology.lower()
    return any(h.lower() in histology_lower for h in nsclc_types)
```

Also fixed `_is_sclc()` for consistency:
```python
def _is_sclc(self, histology: str) -> bool:
    return "smallcell" in histology.lower()
```

## Test Results

### Before Fix:
```
[INFO] Classifying with guideline rules...
[INFO] ✓ Found 0 applicable guidelines
[WARNING] ⚠️ Rule engine generated zero recommendations
```

### After Fix:
```
[INFO] Classifying with guideline rules...
[INFO] ✓ Found 4 applicable guidelines
[INFO]   - R6: Chemoradiotherapy
[INFO]   - R8A: AdjuvantTargetedTherapy  ✓ CORRECT!
[INFO]   - R3: Radiotherapy
[INFO]   - R1: Chemotherapy
```

## Test Case
**Patient:** 68-year-old male, Stage IIIA adenocarcinoma, EGFR Ex19del positive, PS 1

**Expected:** R8A rule should match (Adjuvant EGFR TKI - osimertinib)

**Result:** ✅ R8A now matches correctly

## Files Modified
1. `backend/src/ontology/guideline_rules.py`:
   - Fixed `_is_nsclc()` method (lines 570-579)
   - Fixed `_is_sclc()` method (lines 581-583)

2. `backend/src/services/lca_service.py`:
   - Fixed logging bug (line 338) that was trying to slice a dict

## Impact
This fix ensures that:
- All NSCLC histology types are recognized regardless of case
- EGFR+ Stage IIIA patients receive appropriate adjuvant TKI recommendations
- Future patient data with varying case formats will match correctly

## Related
- Addresses "No Response Generated" issue reported in production
- Ensures ADAURA trial guidelines (R8A) are properly applied
