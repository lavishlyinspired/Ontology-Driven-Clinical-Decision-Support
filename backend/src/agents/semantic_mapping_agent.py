"""
Semantic Mapping Agent (Agent 2 of 6)
Maps clinical concepts to SNOMED-CT codes.

Responsibilities:
- Map clinical concepts to SNOMED-CT codes
- Use local SNOMED module for mapping
- Validate mappings with confidence scores
- Return PatientFactWithCodes

Tools: map_to_snomed(), get_snomed_hierarchy(), validate_mapping()
Data Sources: SNOMED-CT via local module
NEVER: Direct Neo4j writes
"""

from typing import Dict, Any, Optional, Tuple, List
import logging

from ..db.models import PatientFact, PatientFactWithCodes
from ..ontology.snomed_loader import SNOMEDLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticMappingAgent:
    """
    Agent 2: Semantic Mapping Agent
    Maps clinical concepts to SNOMED-CT codes.
    READ-ONLY: Never writes to Neo4j.
    """

    # SNOMED-CT code mappings for lung cancer (from Patient_Records_SNOMED_Guide.md)
    HISTOLOGY_SNOMED = {
        "Adenocarcinoma": "35917007",
        "SquamousCellCarcinoma": "59367005",
        "LargeCellCarcinoma": "67101007",
        "SmallCellCarcinoma": "254632001",
        "Carcinosarcoma": "128885008",
        "NonSmallCellCarcinoma_NOS": "254637007",
    }

    STAGE_SNOMED = {
        "IA": "424132000",
        "IB": "424132000",
        "IIA": "425048006",
        "IIB": "425048006",
        "IIIA": "422968005",
        "IIIB": "422968005",
        "IV": "423121009",
    }

    PERFORMANCE_STATUS_SNOMED = {
        0: "373803006",
        1: "373804000",
        2: "373805004",
        3: "373806003",
        4: "373807007",
    }

    LATERALITY_SNOMED = {
        "Right": "39607008",
        "Left": "44029006",
        "Bilateral": "51185008",
    }

    DIAGNOSIS_SNOMED = {
        "Malignant Neoplasm of Lung": "363358000",
        "NSCLC": "254637007",
        "SCLC": "254632001",
    }

    def __init__(self, snomed_loader: Optional[SNOMEDLoader] = None):
        self.name = "SemanticMappingAgent"
        self.version = "1.0.0"
        self.snomed_loader = snomed_loader

    def execute(self, patient_fact: PatientFact) -> Tuple[PatientFactWithCodes, float]:
        """
        Execute semantic mapping: add SNOMED-CT codes to patient data.
        
        Args:
            patient_fact: Validated patient data
            
        Returns:
            Tuple of (PatientFactWithCodes, overall confidence score)
        """
        logger.info(f"[{self.name}] Mapping patient {patient_fact.patient_id} to SNOMED-CT...")

        # Map each clinical concept
        histology_code, hist_conf = self.map_to_snomed("histology", patient_fact.histology_type)
        stage_code, stage_conf = self.map_to_snomed("stage", patient_fact.tnm_stage)
        ps_code, ps_conf = self.map_to_snomed("performance_status", patient_fact.performance_status)
        lat_code, lat_conf = self.map_to_snomed("laterality", patient_fact.laterality)
        diag_code, diag_conf = self.map_to_snomed("diagnosis", patient_fact.diagnosis)

        # Calculate overall confidence
        confidences = [hist_conf, stage_conf, ps_conf, lat_conf, diag_conf]
        overall_confidence = sum(confidences) / len(confidences)

        # Create PatientFactWithCodes
        patient_with_codes = PatientFactWithCodes(
            **patient_fact.model_dump(),
            snomed_diagnosis_code=diag_code,
            snomed_histology_code=histology_code,
            snomed_stage_code=stage_code,
            snomed_ps_code=ps_code,
            snomed_laterality_code=lat_code,
            mapping_confidence=overall_confidence
        )

        logger.info(f"[{self.name}] ✓ Mapped {patient_fact.patient_id} with confidence {overall_confidence:.2f}")
        return patient_with_codes, overall_confidence

    def map_to_snomed(self, concept_type: str, value: Any) -> Tuple[Optional[str], float]:
        """
        Map a clinical concept to its SNOMED-CT code.
        
        Args:
            concept_type: Type of concept (histology, stage, etc.)
            value: The value to map
            
        Returns:
            Tuple of (SNOMED code or None, confidence score)
        """
        if value is None:
            return None, 0.0

        try:
            if concept_type == "histology":
                code = self.HISTOLOGY_SNOMED.get(str(value))
                confidence = 1.0 if code else 0.5
                return code, confidence

            elif concept_type == "stage":
                code = self.STAGE_SNOMED.get(str(value))
                confidence = 1.0 if code else 0.5
                return code, confidence

            elif concept_type == "performance_status":
                ps_value = int(value) if not isinstance(value, int) else value
                code = self.PERFORMANCE_STATUS_SNOMED.get(ps_value)
                confidence = 1.0 if code else 0.5
                return code, confidence

            elif concept_type == "laterality":
                code = self.LATERALITY_SNOMED.get(str(value))
                confidence = 1.0 if code else 0.5
                return code, confidence

            elif concept_type == "diagnosis":
                # Try exact match first
                code = self.DIAGNOSIS_SNOMED.get(str(value))
                if code:
                    return code, 1.0
                
                # Default to generic lung cancer
                return "363358000", 0.8

            else:
                return None, 0.0

        except Exception as e:
            logger.warning(f"[{self.name}] Mapping failed for {concept_type}={value}: {e}")
            return None, 0.0

    def get_snomed_hierarchy(self, sctid: str) -> List[str]:
        """
        Get ancestor concepts in SNOMED hierarchy.
        Useful for subsumption reasoning.
        """
        # Simplified hierarchy for lung cancer concepts
        hierarchy_map = {
            # Histology hierarchy
            "35917007": ["254637007", "363358000"],  # Adenocarcinoma → NSCLC → Lung cancer
            "59367005": ["254637007", "363358000"],  # Squamous → NSCLC → Lung cancer
            "67101007": ["254637007", "363358000"],  # Large cell → NSCLC → Lung cancer
            "128885008": ["254637007", "363358000"],  # Carcinosarcoma → NSCLC → Lung cancer
            "254632001": ["363358000"],  # SCLC → Lung cancer
            "254637007": ["363358000"],  # NSCLC → Lung cancer
        }
        return hierarchy_map.get(sctid, [])

    def validate_mapping(self, sctid: str, expected_type: str) -> bool:
        """
        Validate that a SNOMED code matches expected semantic type.
        """
        type_codes = {
            "histology": list(self.HISTOLOGY_SNOMED.values()),
            "stage": list(self.STAGE_SNOMED.values()),
            "performance_status": list(self.PERFORMANCE_STATUS_SNOMED.values()),
            "laterality": list(self.LATERALITY_SNOMED.values()),
            "diagnosis": list(self.DIAGNOSIS_SNOMED.values()),
        }
        
        expected_codes = type_codes.get(expected_type, [])
        return sctid in expected_codes

    def is_nsclc_subtype(self, histology_code: str) -> bool:
        """
        Check if a histology code represents an NSCLC subtype.
        Important for guideline rule matching.
        """
        nsclc_subtypes = [
            "35917007",   # Adenocarcinoma
            "59367005",   # Squamous cell
            "67101007",   # Large cell
            "128885008",  # Carcinosarcoma
            "254637007",  # NSCLC NOS
        ]
        return histology_code in nsclc_subtypes

    def is_sclc(self, histology_code: str) -> bool:
        """Check if a histology code represents SCLC."""
        return histology_code == "254632001"
