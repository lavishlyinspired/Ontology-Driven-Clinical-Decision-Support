"""
Ontology API Routes
Provides endpoints for LUCADA ontology lookup, validation, and exploration
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ontology", tags=["ontology"])


# =============================================================================
# SNOMED-CT Mappings from LUCADA Ontology
# =============================================================================

SNOMED_MAPPINGS = {
    # Histology Types
    "nonsmallcellcarcinoma": {"code": "254637007", "display": "Non-small cell lung cancer", "type": "Histology"},
    "nsclc": {"code": "254637007", "display": "Non-small cell lung cancer", "type": "Histology"},
    "smallcellcarcinoma": {"code": "254632001", "display": "Small cell carcinoma of lung", "type": "Histology"},
    "sclc": {"code": "254632001", "display": "Small cell carcinoma of lung", "type": "Histology"},
    "adenocarcinoma": {"code": "35917007", "display": "Adenocarcinoma", "type": "Histology"},
    "squamouscellcarcinoma": {"code": "59367005", "display": "Squamous cell carcinoma", "type": "Histology"},
    "squamous": {"code": "59367005", "display": "Squamous cell carcinoma", "type": "Histology"},
    "largecellcarcinoma": {"code": "67101007", "display": "Large cell carcinoma", "type": "Histology"},
    "carcinosarcoma": {"code": "128885008", "display": "Carcinosarcoma", "type": "Histology"},

    # Biomarkers
    "egfr": {"code": "426964009", "display": "EGFR gene mutation analysis", "type": "Biomarker"},
    "egfr mutation": {"code": "426964009", "display": "EGFR gene mutation", "type": "Biomarker"},
    "alk": {"code": "830151004", "display": "ALK gene rearrangement", "type": "Biomarker"},
    "alk rearrangement": {"code": "830151004", "display": "ALK gene rearrangement", "type": "Biomarker"},
    "pdl1": {"code": "783555007", "display": "PD-L1 expression", "type": "Biomarker"},
    "pd-l1": {"code": "783555007", "display": "PD-L1 expression", "type": "Biomarker"},
    "ros1": {"code": "840533007", "display": "ROS1 gene rearrangement", "type": "Biomarker"},
    "kras": {"code": "702840005", "display": "KRAS gene mutation", "type": "Biomarker"},

    # Performance Status
    "performance status 0": {"code": "373803006", "display": "WHO performance status grade 0", "type": "PerformanceStatus"},
    "ps 0": {"code": "373803006", "display": "Fully active", "type": "PerformanceStatus"},
    "performance status 1": {"code": "373804000", "display": "WHO performance status grade 1", "type": "PerformanceStatus"},
    "ps 1": {"code": "373804000", "display": "Restricted but ambulatory", "type": "PerformanceStatus"},
    "performance status 2": {"code": "373805004", "display": "WHO performance status grade 2", "type": "PerformanceStatus"},
    "ps 2": {"code": "373805004", "display": "Ambulatory, self-care capable", "type": "PerformanceStatus"},
    "performance status 3": {"code": "373806003", "display": "WHO performance status grade 3", "type": "PerformanceStatus"},
    "ps 3": {"code": "373806003", "display": "Limited self-care", "type": "PerformanceStatus"},
    "performance status 4": {"code": "373807007", "display": "WHO performance status grade 4", "type": "PerformanceStatus"},
    "ps 4": {"code": "373807007", "display": "Completely disabled", "type": "PerformanceStatus"},

    # Procedures
    "surgery": {"code": "387713003", "display": "Surgical procedure", "type": "Procedure"},
    "lobectomy": {"code": "173171007", "display": "Lobectomy of lung", "type": "Procedure"},
    "pneumonectomy": {"code": "49795001", "display": "Pneumonectomy", "type": "Procedure"},
    "chemotherapy": {"code": "367336001", "display": "Chemotherapy", "type": "Procedure"},
    "radiotherapy": {"code": "108290001", "display": "Radiation therapy", "type": "Procedure"},
    "radiation": {"code": "108290001", "display": "Radiation therapy", "type": "Procedure"},
    "immunotherapy": {"code": "76334006", "display": "Immunotherapy", "type": "Procedure"},
    "palliative care": {"code": "103735009", "display": "Palliative care", "type": "Procedure"},

    # Clinical Findings
    "lung cancer": {"code": "363358000", "display": "Malignant neoplasm of lung", "type": "ClinicalFinding"},
    "malignant neoplasm of lung": {"code": "363358000", "display": "Malignant neoplasm of lung", "type": "ClinicalFinding"},
    "patient": {"code": "116154003", "display": "Patient", "type": "Patient"},

    # Comorbidities
    "copd": {"code": "13645005", "display": "Chronic obstructive pulmonary disease", "type": "Comorbidity"},
    "diabetes": {"code": "73211009", "display": "Diabetes mellitus", "type": "Comorbidity"},
    "cardiovascular disease": {"code": "49601007", "display": "Cardiovascular disease", "type": "Comorbidity"},
    "dementia": {"code": "52448006", "display": "Dementia", "type": "Comorbidity"},
}


# =============================================================================
# Ontology Hierarchy
# =============================================================================

ONTOLOGY_HIERARCHY = {
    "Histology": {
        "children": ["NonSmallCellCarcinoma", "SmallCellCarcinoma", "Carcinosarcoma"],
        "snomed": None
    },
    "NonSmallCellCarcinoma": {
        "parent": "Histology",
        "children": ["Adenocarcinoma", "SquamousCellCarcinoma", "LargeCellCarcinoma"],
        "snomed": "254637007"
    },
    "SmallCellCarcinoma": {
        "parent": "Histology",
        "children": [],
        "snomed": "254632001"
    },
    "Adenocarcinoma": {
        "parent": "NonSmallCellCarcinoma",
        "children": [],
        "snomed": "35917007"
    },
    "SquamousCellCarcinoma": {
        "parent": "NonSmallCellCarcinoma",
        "children": [],
        "snomed": "59367005"
    },
    "LargeCellCarcinoma": {
        "parent": "NonSmallCellCarcinoma",
        "children": [],
        "snomed": "67101007"
    },
    "Biomarker": {
        "children": ["EGFRMutation", "ALKRearrangement", "PDL1Expression", "ROSRearrangement", "KRASMutation"],
        "snomed": None
    },
    "EGFRMutation": {
        "parent": "Biomarker",
        "children": ["EGFRPositive", "EGFRNegative"],
        "snomed": "426964009"
    },
    "ALKRearrangement": {
        "parent": "Biomarker",
        "children": ["ALKPositive", "ALKNegative"],
        "snomed": "830151004"
    },
    "PDL1Expression": {
        "parent": "Biomarker",
        "children": ["PDL1High", "PDL1Low", "PDL1Negative"],
        "snomed": "783555007"
    },
    "PerformanceStatus": {
        "children": ["WHOPerfStatusGrade0", "WHOPerfStatusGrade1", "WHOPerfStatusGrade2", "WHOPerfStatusGrade3", "WHOPerfStatusGrade4"],
        "snomed": None
    },
    "TherapeuticProcedure": {
        "children": ["Surgery", "Chemotherapy", "Radiotherapy", "Immunotherapy", "Chemoradiotherapy", "PalliativeCare"],
        "snomed": None
    },
    "Surgery": {
        "parent": "TherapeuticProcedure",
        "children": ["Lobectomy", "Pneumonectomy"],
        "snomed": "387713003"
    },
    "Comorbidity": {
        "children": ["COPD", "Diabetes", "CardiovascularDisease", "Dementia"],
        "snomed": None
    }
}


# =============================================================================
# Pydantic Models
# =============================================================================

class SNOMEDConcept(BaseModel):
    code: str
    display: str
    type: str


class OntologySearchResult(BaseModel):
    term: str
    matches: List[Dict[str, Any]]


class HierarchyNode(BaseModel):
    name: str
    snomed: Optional[str]
    parent: Optional[str]
    children: List[str]


class ValidationResult(BaseModel):
    valid: bool
    errors: List[Dict[str, str]]
    warnings: List[Dict[str, str]]
    snomed_mappings: Dict[str, str]


class PatientDataInput(BaseModel):
    histology_type: Optional[str] = None
    tnm_stage: Optional[str] = None
    performance_status: Optional[int] = None
    biomarkers: Optional[Dict[str, Any]] = None
    comorbidities: Optional[List[str]] = None
    age: Optional[int] = None
    sex: Optional[str] = None


class ArgumentationResult(BaseModel):
    patient_scenario: Dict[str, Any]
    arguments: List[Dict[str, Any]]
    decisions: List[Dict[str, Any]]
    treatment_plans: List[Dict[str, Any]]


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/search", response_model=OntologySearchResult)
async def search_ontology(
    term: str = Query(..., description="Search term"),
    domain: Optional[str] = Query(None, description="Filter by domain: Histology, Biomarker, Procedure, etc.")
):
    """
    Search LUCADA ontology for matching clinical concepts.
    Returns SNOMED-CT codes and descriptions.
    """
    term_lower = term.lower().strip()
    matches = []

    for key, value in SNOMED_MAPPINGS.items():
        # Check if term matches
        if term_lower in key or key in term_lower:
            # Apply domain filter if specified
            if domain and value.get("type", "").lower() != domain.lower():
                continue
            matches.append({
                "term": key,
                "snomed_code": value["code"],
                "display": value["display"],
                "type": value["type"]
            })

    # Sort by relevance (exact matches first)
    matches.sort(key=lambda x: (x["term"] != term_lower, len(x["term"])))

    return OntologySearchResult(term=term, matches=matches[:10])


@router.get("/snomed/{code}", response_model=SNOMEDConcept)
async def lookup_snomed(code: str):
    """
    Look up a SNOMED-CT code and return its details.
    """
    for key, value in SNOMED_MAPPINGS.items():
        if value["code"] == code:
            return SNOMEDConcept(
                code=value["code"],
                display=value["display"],
                type=value["type"]
            )

    raise HTTPException(status_code=404, detail=f"SNOMED code {code} not found in LUCADA ontology")


@router.get("/hierarchy/{concept}", response_model=HierarchyNode)
async def get_hierarchy(
    concept: str,
    direction: str = Query("children", description="Direction: 'children' or 'parents'")
):
    """
    Get ontology hierarchy for a clinical concept.
    Returns parent and/or children in the class hierarchy.
    """
    # Normalize concept name
    concept_normalized = concept.replace(" ", "").replace("-", "")

    # Find matching concept
    for key, value in ONTOLOGY_HIERARCHY.items():
        if key.lower() == concept_normalized.lower():
            return HierarchyNode(
                name=key,
                snomed=value.get("snomed"),
                parent=value.get("parent"),
                children=value.get("children", [])
            )

    raise HTTPException(status_code=404, detail=f"Concept '{concept}' not found in ontology hierarchy")


@router.get("/hierarchy", response_model=Dict[str, HierarchyNode])
async def get_full_hierarchy():
    """
    Get the complete ontology hierarchy.
    """
    result = {}
    for key, value in ONTOLOGY_HIERARCHY.items():
        result[key] = HierarchyNode(
            name=key,
            snomed=value.get("snomed"),
            parent=value.get("parent"),
            children=value.get("children", [])
        )
    return result


@router.post("/validate", response_model=ValidationResult)
async def validate_patient_data(data: PatientDataInput):
    """
    Validate patient clinical data against LUCADA ontology.
    Returns validation errors and SNOMED-CT code mappings.
    """
    errors = []
    warnings = []
    snomed_mappings = {}

    # Validate histology type
    if data.histology_type:
        histology_lower = data.histology_type.lower().replace(" ", "")
        valid_histologies = ["adenocarcinoma", "squamouscellcarcinoma", "largecellcarcinoma",
                           "smallcellcarcinoma", "nonsmallcellcarcinoma", "nsclc", "sclc", "carcinosarcoma"]
        if histology_lower not in valid_histologies:
            errors.append({
                "field": "histology_type",
                "message": f"Invalid histology type: {data.histology_type}",
                "suggestion": "Valid types: Adenocarcinoma, SquamousCellCarcinoma, LargeCellCarcinoma, SmallCellCarcinoma"
            })
        else:
            # Map to SNOMED
            for key, value in SNOMED_MAPPINGS.items():
                if key.replace(" ", "") == histology_lower:
                    snomed_mappings["histology_type"] = value["code"]
                    break

    # Validate TNM stage
    if data.tnm_stage:
        valid_stages = ["IA1", "IA2", "IA3", "IB", "IIA", "IIB", "IIIA", "IIIB", "IIIC", "IVA", "IVB",
                       "I", "II", "III", "IV", "1", "2", "3", "4"]
        stage_upper = data.tnm_stage.upper().replace(" ", "")
        if stage_upper not in valid_stages:
            errors.append({
                "field": "tnm_stage",
                "message": f"Invalid TNM stage: {data.tnm_stage}",
                "suggestion": "Valid stages: IA1-IA3, IB, IIA, IIB, IIIA-IIIC, IVA, IVB"
            })

    # Validate performance status
    if data.performance_status is not None:
        if data.performance_status < 0 or data.performance_status > 4:
            errors.append({
                "field": "performance_status",
                "message": f"Invalid performance status: {data.performance_status}",
                "suggestion": "Valid values: 0 (fully active) to 4 (completely disabled)"
            })
        else:
            ps_key = f"ps {data.performance_status}"
            if ps_key in SNOMED_MAPPINGS:
                snomed_mappings["performance_status"] = SNOMED_MAPPINGS[ps_key]["code"]

    # Validate biomarkers
    if data.biomarkers:
        for marker, value in data.biomarkers.items():
            marker_lower = marker.lower().replace("-", "").replace("_", "")
            valid_markers = ["egfr", "alk", "pdl1", "ros1", "kras", "braf", "met", "ret", "ntrk", "her2"]
            if marker_lower not in valid_markers:
                warnings.append({
                    "field": f"biomarkers.{marker}",
                    "message": f"Unknown biomarker: {marker}",
                    "suggestion": "Known biomarkers: EGFR, ALK, PD-L1, ROS1, KRAS, BRAF, MET, RET, NTRK, HER2"
                })
            else:
                # Map to SNOMED if available
                if marker_lower in SNOMED_MAPPINGS:
                    snomed_mappings[f"biomarker_{marker_lower}"] = SNOMED_MAPPINGS[marker_lower]["code"]

            # Validate PD-L1 score if numeric
            if marker_lower == "pdl1" and isinstance(value, (int, float)):
                if value < 0 or value > 100:
                    errors.append({
                        "field": f"biomarkers.{marker}",
                        "message": f"Invalid PD-L1 score: {value}",
                        "suggestion": "PD-L1 TPS should be 0-100%"
                    })

    # Validate age
    if data.age is not None:
        if data.age < 0 or data.age > 120:
            errors.append({
                "field": "age",
                "message": f"Invalid age: {data.age}",
                "suggestion": "Age should be between 0 and 120"
            })
        elif data.age < 18:
            warnings.append({
                "field": "age",
                "message": "Pediatric patient - lung cancer treatment guidelines may differ",
                "suggestion": "Consider pediatric oncology consultation"
            })

    # Validate sex
    if data.sex:
        if data.sex.upper() not in ["M", "F", "MALE", "FEMALE", "U", "UNKNOWN"]:
            errors.append({
                "field": "sex",
                "message": f"Invalid sex: {data.sex}",
                "suggestion": "Valid values: M, F, Male, Female, U, Unknown"
            })

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        snomed_mappings=snomed_mappings
    )


@router.post("/arguments", response_model=ArgumentationResult)
async def get_guideline_arguments(data: PatientDataInput):
    """
    Get argumentation-based treatment recommendations.
    Uses the LUCADA argumentation domain to derive treatment decisions.
    """
    arguments = []
    decisions = []
    treatment_plans = []

    # Build patient scenario
    patient_scenario = {
        "histology": data.histology_type,
        "stage": data.tnm_stage,
        "performance_status": data.performance_status,
        "biomarkers": data.biomarkers or {},
        "comorbidities": data.comorbidities or [],
        "age": data.age,
        "sex": data.sex
    }

    # Determine treatment arguments based on clinical factors
    stage = (data.tnm_stage or "").upper()
    ps = data.performance_status or 0
    histology = (data.histology_type or "").lower()
    biomarkers = data.biomarkers or {}

    # Stage-based arguments
    if stage in ["IA1", "IA2", "IA3", "IB", "IIA", "IIB"]:
        arguments.append({
            "type": "supporting",
            "target": "Surgery",
            "strength": "strong",
            "evidence": "Grade A",
            "rationale": f"Early stage ({stage}) is eligible for surgical resection",
            "source": "NCCN NSCLC Guidelines 2025"
        })
        if ps <= 1:
            decisions.append({
                "treatment": "Surgical Resection",
                "intent": "curative",
                "confidence": 0.9
            })
            treatment_plans.append({
                "type": "Lobectomy",
                "intent": "curative",
                "evidence_level": "Grade A",
                "guideline": "NCCN-NSCLC-2025"
            })

    elif stage in ["IIIA", "IIIB"]:
        arguments.append({
            "type": "supporting",
            "target": "Chemoradiotherapy",
            "strength": "strong",
            "evidence": "Grade A",
            "rationale": f"Locally advanced stage ({stage}) benefits from concurrent chemoradiotherapy",
            "source": "NCCN NSCLC Guidelines 2025"
        })
        decisions.append({
            "treatment": "Concurrent Chemoradiotherapy",
            "intent": "curative",
            "confidence": 0.85
        })
        treatment_plans.append({
            "type": "Chemoradiotherapy",
            "intent": "curative",
            "evidence_level": "Grade A",
            "guideline": "NCCN-NSCLC-2025"
        })

    elif stage in ["IV", "IVA", "IVB"]:
        arguments.append({
            "type": "supporting",
            "target": "Systemic Therapy",
            "strength": "strong",
            "evidence": "Grade A",
            "rationale": f"Metastatic stage ({stage}) requires systemic therapy",
            "source": "NCCN NSCLC Guidelines 2025"
        })

    # Biomarker-based arguments
    egfr = biomarkers.get("EGFR") or biomarkers.get("egfr")
    if egfr and str(egfr).lower() in ["positive", "mutated", "+"]:
        arguments.append({
            "type": "supporting",
            "target": "EGFR TKI",
            "strength": "strong",
            "evidence": "Grade A",
            "rationale": "EGFR mutation positive - targeted therapy indicated",
            "source": "KEYNOTE-024, FLAURA"
        })
        decisions.append({
            "treatment": "Osimertinib",
            "intent": "palliative" if "IV" in stage else "adjuvant",
            "confidence": 0.95
        })
        treatment_plans.append({
            "type": "EGFR TKI (Osimertinib)",
            "intent": "targeted",
            "evidence_level": "Grade A",
            "guideline": "NCCN-NSCLC-2025"
        })

    pdl1 = biomarkers.get("PD-L1") or biomarkers.get("pdl1") or biomarkers.get("PDL1")
    if pdl1:
        try:
            pdl1_score = float(str(pdl1).replace("%", "").replace(">", "").replace(">=", ""))
            if pdl1_score >= 50:
                arguments.append({
                    "type": "supporting",
                    "target": "Pembrolizumab Monotherapy",
                    "strength": "strong",
                    "evidence": "Grade A",
                    "rationale": f"PD-L1 ≥50% ({pdl1_score}%) - single-agent immunotherapy indicated",
                    "source": "KEYNOTE-024, KEYNOTE-042"
                })
                decisions.append({
                    "treatment": "Pembrolizumab",
                    "intent": "palliative",
                    "confidence": 0.9
                })
                treatment_plans.append({
                    "type": "Pembrolizumab Monotherapy",
                    "intent": "immunotherapy",
                    "evidence_level": "Grade A",
                    "guideline": "NCCN-NSCLC-2025"
                })
            elif pdl1_score >= 1:
                arguments.append({
                    "type": "supporting",
                    "target": "Pembrolizumab + Chemotherapy",
                    "strength": "moderate",
                    "evidence": "Grade A",
                    "rationale": f"PD-L1 1-49% ({pdl1_score}%) - combination immunotherapy indicated",
                    "source": "KEYNOTE-189"
                })
                decisions.append({
                    "treatment": "Pembrolizumab + Platinum Doublet",
                    "intent": "palliative",
                    "confidence": 0.85
                })
        except ValueError:
            pass

    # Performance status arguments
    if ps >= 3:
        arguments.append({
            "type": "opposing",
            "target": "Aggressive Treatment",
            "strength": "strong",
            "evidence": "Grade B",
            "rationale": f"Poor performance status (PS {ps}) - consider best supportive care",
            "source": "NCCN NSCLC Guidelines"
        })

    # Comorbidity arguments
    if data.comorbidities:
        for comorbidity in data.comorbidities:
            if "copd" in comorbidity.lower():
                arguments.append({
                    "type": "caution",
                    "target": "Pneumonectomy",
                    "strength": "moderate",
                    "evidence": "Grade C",
                    "rationale": "COPD comorbidity - assess pulmonary reserve before surgery",
                    "source": "Clinical Practice"
                })

    return ArgumentationResult(
        patient_scenario=patient_scenario,
        arguments=arguments,
        decisions=decisions,
        treatment_plans=treatment_plans
    )


@router.get("/domains")
async def get_ontology_domains():
    """
    Get list of available ontology domains.
    """
    return {
        "domains": [
            {"name": "Histology", "description": "Tumor histology types (NSCLC, SCLC, subtypes)", "icon": "🔬"},
            {"name": "Biomarker", "description": "Molecular biomarkers (EGFR, ALK, PD-L1, etc.)", "icon": "🧬"},
            {"name": "PerformanceStatus", "description": "WHO/ECOG performance status grades", "icon": "📊"},
            {"name": "Procedure", "description": "Therapeutic procedures (surgery, chemo, radio)", "icon": "💊"},
            {"name": "Comorbidity", "description": "Patient comorbidities", "icon": "⚕️"},
            {"name": "TNMStaging", "description": "TNM staging classification", "icon": "📈"},
        ]
    }
