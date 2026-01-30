"""
Classification Agent (Agent 3 of 6)
Applies LUCADA ontology rules and NICE guidelines.

Responsibilities:
- Load and apply LUCADA ontology rules via Owlready2
- Apply NICE guideline recommendations
- Return ClassificationResult with PatientScenarios
- Perform ontology-based reasoning

Tools: query_ontology(), apply_nice_rules(), get_guideline_recommendations()
Data Sources: OWL ontology file, guideline rules
NEVER: Direct Neo4j writes
"""

from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

# Centralized logging
from ..logging_config import get_logger, log_agent_action

logger = get_logger(__name__)

from ..db.models import (
    PatientFactWithCodes,
    ClassificationResult,
    TreatmentRecommendation,
    EvidenceLevel,
    TreatmentIntent
)


class PatientScenario(str, Enum):
    """LUCADA classification scenarios from ontology"""
    EARLY_STAGE_OPERABLE = "early_stage_operable"
    EARLY_STAGE_INOPERABLE = "early_stage_inoperable"
    LOCALLY_ADVANCED_RESECTABLE = "locally_advanced_resectable"
    LOCALLY_ADVANCED_UNRESECTABLE = "locally_advanced_unresectable"
    METASTATIC_GOOD_PS = "metastatic_good_ps"
    METASTATIC_POOR_PS = "metastatic_poor_ps"
    SCLC_LIMITED = "sclc_limited"
    SCLC_EXTENSIVE = "sclc_extensive"
    UNKNOWN = "unknown"


class ClassificationAgent:
    """
    Agent 3: Classification Agent
    Applies LUCADA ontology rules and NICE guidelines.
    READ-ONLY: Never writes to Neo4j.
    """

    # NICE guideline recommendations by scenario (from final.md)
    NICE_RECOMMENDATIONS: Dict[PatientScenario, List[Dict[str, Any]]] = {
        PatientScenario.EARLY_STAGE_OPERABLE: [
            {
                "treatment": "Surgical resection (lobectomy preferred)",
                "evidence_level": EvidenceLevel.GRADE_A,
                "intent": TreatmentIntent.CURATIVE,
                "guideline_ref": "NICE NG122 1.4.1",
                "rationale": "Lobectomy is the preferred surgical approach for early-stage NSCLC with good PS"
            },
            {
                "treatment": "SABR if unfit for surgery",
                "evidence_level": EvidenceLevel.GRADE_A,
                "intent": TreatmentIntent.CURATIVE,
                "guideline_ref": "NICE NG122 1.4.9",
                "rationale": "SABR is alternative curative option for medically inoperable patients"
            }
        ],
        PatientScenario.EARLY_STAGE_INOPERABLE: [
            {
                "treatment": "SABR (stereotactic ablative radiotherapy)",
                "evidence_level": EvidenceLevel.GRADE_A,
                "intent": TreatmentIntent.CURATIVE,
                "guideline_ref": "NICE NG122 1.4.9",
                "rationale": "SABR offers curative potential for inoperable early-stage disease"
            },
            {
                "treatment": "Conventional radical radiotherapy",
                "evidence_level": EvidenceLevel.GRADE_B,
                "intent": TreatmentIntent.CURATIVE,
                "guideline_ref": "NICE NG122 1.4.12",
                "rationale": "Alternative when SABR not available or suitable"
            }
        ],
        PatientScenario.LOCALLY_ADVANCED_RESECTABLE: [
            {
                "treatment": "Chemoradiotherapy followed by surgery",
                "evidence_level": EvidenceLevel.GRADE_B,
                "intent": TreatmentIntent.CURATIVE,
                "guideline_ref": "NICE NG122 1.4.21",
                "rationale": "Trimodality therapy for selected Stage IIIA patients"
            },
            {
                "treatment": "Concurrent chemoradiotherapy",
                "evidence_level": EvidenceLevel.GRADE_A,
                "intent": TreatmentIntent.CURATIVE,
                "guideline_ref": "NICE NG122 1.4.18",
                "rationale": "Standard for unresectable locally advanced NSCLC"
            }
        ],
        PatientScenario.LOCALLY_ADVANCED_UNRESECTABLE: [
            {
                "treatment": "Concurrent chemoradiotherapy + durvalumab",
                "evidence_level": EvidenceLevel.GRADE_A,
                "intent": TreatmentIntent.CURATIVE,
                "guideline_ref": "NICE NG122 1.4.18 + TA798",
                "rationale": "Durvalumab consolidation after chemoRT improves OS in unresectable Stage III"
            },
            {
                "treatment": "Sequential chemoradiotherapy if unfit for concurrent",
                "evidence_level": EvidenceLevel.GRADE_B,
                "intent": TreatmentIntent.CURATIVE,
                "guideline_ref": "NICE NG122 1.4.20",
                "rationale": "For patients who cannot tolerate concurrent treatment"
            }
        ],
        PatientScenario.METASTATIC_GOOD_PS: [
            {
                "treatment": "Pembrolizumab monotherapy (PD-L1 ≥50%)",
                "evidence_level": EvidenceLevel.GRADE_A,
                "intent": TreatmentIntent.PALLIATIVE,
                "guideline_ref": "NICE TA531",
                "rationale": "First-line for high PD-L1 expression without targetable mutations"
            },
            {
                "treatment": "Pembrolizumab + chemotherapy",
                "evidence_level": EvidenceLevel.GRADE_A,
                "intent": TreatmentIntent.PALLIATIVE,
                "guideline_ref": "NICE TA683",
                "rationale": "First-line combination for metastatic NSCLC regardless of PD-L1"
            },
            {
                "treatment": "Targeted therapy if driver mutation present",
                "evidence_level": EvidenceLevel.GRADE_A,
                "intent": TreatmentIntent.PALLIATIVE,
                "guideline_ref": "NICE TA654 (EGFR), TA670 (ALK)",
                "rationale": "Osimertinib for EGFR+, Alectinib for ALK+"
            }
        ],
        PatientScenario.METASTATIC_POOR_PS: [
            {
                "treatment": "Best supportive care",
                "evidence_level": EvidenceLevel.GRADE_A,
                "intent": TreatmentIntent.SUPPORTIVE,
                "guideline_ref": "NICE NG122 1.5.1",
                "rationale": "Symptom control and quality of life focus for poor PS patients"
            },
            {
                "treatment": "Single-agent chemotherapy if PS improves",
                "evidence_level": EvidenceLevel.GRADE_C,
                "intent": TreatmentIntent.PALLIATIVE,
                "guideline_ref": "NICE NG122 1.5.3",
                "rationale": "May consider if performance status improves to PS 2"
            }
        ],
        PatientScenario.SCLC_LIMITED: [
            {
                "treatment": "Concurrent chemoradiotherapy",
                "evidence_level": EvidenceLevel.GRADE_A,
                "intent": TreatmentIntent.CURATIVE,
                "guideline_ref": "NICE NG122 1.6.1",
                "rationale": "Platinum-etoposide with concurrent thoracic RT for limited SCLC"
            },
            {
                "treatment": "Prophylactic cranial irradiation",
                "evidence_level": EvidenceLevel.GRADE_A,
                "intent": TreatmentIntent.CURATIVE,
                "guideline_ref": "NICE NG122 1.6.3",
                "rationale": "Reduces brain metastasis risk in responders"
            }
        ],
        PatientScenario.SCLC_EXTENSIVE: [
            {
                "treatment": "Platinum-etoposide + atezolizumab",
                "evidence_level": EvidenceLevel.GRADE_A,
                "intent": TreatmentIntent.PALLIATIVE,
                "guideline_ref": "NICE TA638",
                "rationale": "First-line for extensive SCLC"
            },
            {
                "treatment": "Palliative radiotherapy for symptoms",
                "evidence_level": EvidenceLevel.GRADE_A,
                "intent": TreatmentIntent.PALLIATIVE,
                "guideline_ref": "NICE NG122 1.6.5",
                "rationale": "For symptom control in extensive disease"
            }
        ],
        PatientScenario.UNKNOWN: [
            {
                "treatment": "MDT discussion required",
                "evidence_level": EvidenceLevel.GRADE_C,
                "intent": TreatmentIntent.UNKNOWN,
                "guideline_ref": "NICE NG122 1.1",
                "rationale": "Insufficient information for classification"
            }
        ]
    }

    def __init__(self, ontology_path: Optional[str] = None):
        self.name = "ClassificationAgent"
        self.version = "1.0.0"
        self.ontology_path = ontology_path
        self.ontology = None

    def execute(self, patient: PatientFactWithCodes) -> ClassificationResult:
        """
        Execute classification: determine patient scenario and recommendations.
        
        Args:
            patient: Patient data with SNOMED codes
            
        Returns:
            ClassificationResult with scenario, recommendations, and confidence
        """
        logger.info(f"[{self.name}] Classifying patient {patient.patient_id}...")

        # Determine patient scenario
        scenario, scenario_confidence = self._classify_scenario(patient)
        
        # Get NICE guideline recommendations for this scenario
        recommendations = self._get_recommendations(scenario, patient)
        
        # Build reasoning chain
        reasoning = self._build_reasoning(patient, scenario)

        result = ClassificationResult(
            patient_id=patient.patient_id,
            scenario=scenario.value,
            scenario_confidence=scenario_confidence,
            recommendations=recommendations,
            reasoning_chain=reasoning,
            ontology_concepts_matched=self._get_matched_concepts(patient),
            guideline_refs=self._get_guideline_refs(recommendations)
        )

        logger.info(f"[{self.name}] ✓ Classified {patient.patient_id} as {scenario.value}")
        return result

    def _classify_scenario(self, patient: PatientFactWithCodes) -> Tuple[PatientScenario, float]:
        """
        Classify patient into a clinical scenario.
        Uses ontology rules and clinical criteria.
        """
        stage = patient.tnm_stage
        ps = patient.performance_status
        histology = patient.histology_type

        # Check for SCLC first
        if str(histology) == "SmallCellCarcinoma":
            if stage in ["I", "IA", "IB", "II", "IIA", "IIB", "IIIA"]:
                return PatientScenario.SCLC_LIMITED, 0.95
            else:
                return PatientScenario.SCLC_EXTENSIVE, 0.95

        # NSCLC staging scenarios
        if stage in ["I", "IA", "IB", "II", "IIA", "IIB"]:
            # Early stage
            if ps is not None and int(ps) <= 1:
                return PatientScenario.EARLY_STAGE_OPERABLE, 0.9
            else:
                return PatientScenario.EARLY_STAGE_INOPERABLE, 0.85

        elif stage in ["IIIA"]:
            # Locally advanced - potentially resectable
            if ps is not None and int(ps) <= 1:
                return PatientScenario.LOCALLY_ADVANCED_RESECTABLE, 0.85
            else:
                return PatientScenario.LOCALLY_ADVANCED_UNRESECTABLE, 0.85

        elif stage in ["IIIB", "IIIC"]:
            # Locally advanced unresectable
            return PatientScenario.LOCALLY_ADVANCED_UNRESECTABLE, 0.9

        elif stage in ["IV", "IVA", "IVB"]:
            # Metastatic
            if ps is not None and int(ps) <= 1:
                return PatientScenario.METASTATIC_GOOD_PS, 0.9
            else:
                return PatientScenario.METASTATIC_POOR_PS, 0.9

        return PatientScenario.UNKNOWN, 0.5

    def _get_recommendations(
        self, 
        scenario: PatientScenario, 
        patient: PatientFactWithCodes
    ) -> List[Dict[str, Any]]:
        """Get treatment recommendations for a scenario."""
        scenario_recs = self.NICE_RECOMMENDATIONS.get(scenario, [])
        
        recommendations = []
        for i, rec in enumerate(scenario_recs):
            recommendations.append({
                "rank": i + 1,
                "treatment": rec["treatment"],
                "evidence_level": rec["evidence_level"].value if hasattr(rec["evidence_level"], 'value') else str(rec["evidence_level"]),
                "intent": rec["intent"].value if hasattr(rec["intent"], 'value') else str(rec["intent"]),
                "guideline_reference": rec["guideline_ref"],
                "rationale": rec["rationale"],
                "contraindications": self._check_contraindications(patient, rec["treatment"]),
                "requires_biomarker": self._requires_biomarker(rec["treatment"])
            })
        
        return recommendations

    def _check_contraindications(self, patient: PatientFactWithCodes, treatment: str) -> List[str]:
        """Check for contraindications based on patient factors."""
        contraindications = []
        
        ps = patient.performance_status
        if ps is not None:
            ps_val = int(ps)
            
            if "surgery" in treatment.lower() and ps_val >= 3:
                contraindications.append("Poor performance status (PS ≥3) may preclude surgery")
            
            if "concurrent chemo" in treatment.lower() and ps_val >= 2:
                contraindications.append("PS ≥2 may require dose reduction or sequential approach")
            
            if "chemotherapy" in treatment.lower() and ps_val >= 4:
                contraindications.append("PS 4 generally contraindicates systemic therapy")
        
        return contraindications

    def _requires_biomarker(self, treatment: str) -> Optional[str]:
        """Check if treatment requires specific biomarker testing."""
        biomarker_treatments = {
            "pembrolizumab monotherapy": "PD-L1 ≥50%",
            "targeted therapy": "EGFR/ALK/ROS1/BRAF testing",
            "osimertinib": "EGFR mutation positive",
            "alectinib": "ALK rearrangement positive",
            "atezolizumab": "None required for SCLC",
        }
        
        treatment_lower = treatment.lower()
        for key, biomarker in biomarker_treatments.items():
            if key in treatment_lower:
                return biomarker
        
        return None

    def _build_reasoning(self, patient: PatientFactWithCodes, scenario: PatientScenario) -> List[str]:
        """Build chain of reasoning for classification."""
        reasoning = []
        
        reasoning.append(f"Patient presents with {patient.histology_type} lung cancer")
        reasoning.append(f"TNM Stage: {patient.tnm_stage}")
        
        if patient.performance_status is not None:
            reasoning.append(f"ECOG Performance Status: {patient.performance_status}")
        
        if patient.laterality:
            reasoning.append(f"Laterality: {patient.laterality}")
        
        reasoning.append(f"Classified scenario: {scenario.value}")
        
        if scenario in [PatientScenario.EARLY_STAGE_OPERABLE, PatientScenario.EARLY_STAGE_INOPERABLE]:
            reasoning.append("Early stage disease - curative intent treatment indicated")
        elif scenario in [PatientScenario.LOCALLY_ADVANCED_RESECTABLE, PatientScenario.LOCALLY_ADVANCED_UNRESECTABLE]:
            reasoning.append("Locally advanced disease - multimodality treatment indicated")
        elif scenario in [PatientScenario.METASTATIC_GOOD_PS, PatientScenario.METASTATIC_POOR_PS]:
            reasoning.append("Metastatic disease - palliative/systemic therapy indicated")
        
        return reasoning

    def _get_matched_concepts(self, patient: PatientFactWithCodes) -> List[str]:
        """Get list of ontology concepts matched."""
        concepts = []
        
        if patient.snomed_histology_code:
            concepts.append(f"SCTID:{patient.snomed_histology_code}")
        if patient.snomed_stage_code:
            concepts.append(f"SCTID:{patient.snomed_stage_code}")
        if patient.snomed_ps_code:
            concepts.append(f"SCTID:{patient.snomed_ps_code}")
        
        return concepts

    def _get_guideline_refs(self, recommendations: List[Dict[str, Any]]) -> List[str]:
        """Extract unique guideline references."""
        refs = set()
        for rec in recommendations:
            if rec.get("guideline_reference"):
                refs.add(rec["guideline_reference"])
        return list(refs)

    def query_ontology(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Query LUCADA ontology for concept details."""
        # Placeholder for Owlready2 ontology queries
        # Would load from self.ontology_path
        return None

    def apply_nice_rules(self, scenario: PatientScenario) -> List[Dict[str, Any]]:
        """Apply NICE guideline rules for a scenario."""
        return self.NICE_RECOMMENDATIONS.get(scenario, [])
