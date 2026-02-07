"""
Ingestion Agent (Agent 1 of 6)
Validates and normalizes raw patient data.

Responsibilities:
- Validate raw patient data against schema
- Normalize TNM staging (e.g., "Stage IIA" → "IIA")
- Calculate derived fields (age group, etc.)
- Return PatientFact object or validation errors

Tools: validate_schema(), normalize_tnm(), calculate_age_group()
NEVER: Direct Neo4j writes
"""

from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import re

# Centralized logging
from ..logging_config import get_logger, log_agent_action

logger = get_logger(__name__)

from ..db.models import PatientFact, Sex, TNMStage, HistologyType, PerformanceStatus, Laterality


class IngestionAgent:
    """
    Agent 1: Ingestion Agent
    Validates and normalizes raw patient data.
    READ-ONLY: Never writes to Neo4j.
    """

    REQUIRED_FIELDS = ["sex", "age", "tnm_stage", "histology_type", "performance_status"]

    TNM_NORMALIZATION = {
        # Stage I variants
        "stage ia": "IA", "stage IA": "IA", "1a": "IA", "ia": "IA", "IA": "IA", "I A": "IA",
        "stage ib": "IB", "stage IB": "IB", "1b": "IB", "ib": "IB", "IB": "IB", "I B": "IB",
        "stage i": "IA", "stage 1": "IA", "1": "IA", "i": "IA", "I": "IA",
        # Stage II variants
        "stage iia": "IIA", "stage IIA": "IIA", "2a": "IIA", "iia": "IIA", "IIA": "IIA", "II A": "IIA",
        "stage iib": "IIB", "stage IIB": "IIB", "2b": "IIB", "iib": "IIB", "IIB": "IIB", "II B": "IIB",
        "stage ii": "IIA", "stage 2": "IIA", "2": "IIA", "ii": "IIA", "II": "IIA",
        # Stage III variants
        "stage iiia": "IIIA", "stage IIIA": "IIIA", "3a": "IIIA", "iiia": "IIIA", "IIIA": "IIIA", "III A": "IIIA",
        "stage iiib": "IIIB", "stage IIIB": "IIIB", "3b": "IIIB", "iiib": "IIIB", "IIIB": "IIIB", "III B": "IIIB",
        "stage iii": "IIIA", "stage 3": "IIIA", "3": "IIIA", "iii": "IIIA", "III": "IIIA",
        # Stage IV variants
        "stage iv": "IV", "stage IV": "IV", "4": "IV", "iv": "IV", "IV": "IV",
    }

    HISTOLOGY_NORMALIZATION = {
        "adenocarcinoma": "Adenocarcinoma",
        "squamous cell carcinoma": "SquamousCellCarcinoma",
        "squamous cell": "SquamousCellCarcinoma",
        "squamous": "SquamousCellCarcinoma",
        "large cell carcinoma": "LargeCellCarcinoma",
        "large cell": "LargeCellCarcinoma",
        "small cell carcinoma": "SmallCellCarcinoma",
        "small cell": "SmallCellCarcinoma",
        "sclc": "SmallCellCarcinoma",
        "carcinosarcoma": "Carcinosarcoma",
        "nsclc": "NonSmallCellCarcinoma_NOS",
        "non-small cell": "NonSmallCellCarcinoma_NOS",
        "non small cell": "NonSmallCellCarcinoma_NOS",
    }

    def __init__(self):
        self.name = "IngestionAgent"
        self.version = "1.0.0"

    def execute(self, raw_data: Dict[str, Any]) -> Tuple[Optional[PatientFact], List[str]]:
        """
        Execute ingestion: validate and normalize patient data.

        Args:
            raw_data: Raw patient data dictionary

        Returns:
            Tuple of (PatientFact or None, list of error messages)
        """
        logger.info("=" * 60)
        logger.info(f"[{self.name}] STARTING EXECUTION")
        logger.info("=" * 60)
        logger.info(f"[{self.name}] INPUT DATA:")
        for key, value in raw_data.items():
            logger.info(f"  • {key}: {value}")
        errors = []

        # Step 1: Validate required fields
        validation_errors = self.validate_schema(raw_data)
        if validation_errors:
            errors.extend(validation_errors)
            return None, errors

        # Step 1b: Ontology-based validation (non-blocking warnings)
        ontology_warnings = self._validate_against_ontology(raw_data)
        if ontology_warnings:
            for w in ontology_warnings:
                logger.warning(f"[{self.name}] Ontology warning: {w}")

        # Step 2: Normalize data
        try:
            normalized_data = self._normalize_data(raw_data)
        except Exception as e:
            errors.append(f"Normalization failed: {str(e)}")
            return None, errors

        # Step 3: Create PatientFact
        try:
            # Generate default name if not provided
            patient_name = normalized_data.get("name", f"Patient_{normalized_data.get('patient_id', 'Unknown')}")
            
            patient_fact = PatientFact(
                patient_id=normalized_data.get("patient_id"),
                name=patient_name,
                sex=normalized_data["sex"],
                age_at_diagnosis=normalized_data["age"],
                tnm_stage=normalized_data["tnm_stage"],
                histology_type=normalized_data["histology_type"],
                performance_status=normalized_data["performance_status"],
                laterality=normalized_data.get("laterality", "Right"),
                fev1_percent=normalized_data.get("fev1_percent"),
                diagnosis=normalized_data.get("diagnosis", "Malignant Neoplasm of Lung"),
                comorbidities=normalized_data.get("comorbidities", []),
                notes=normalized_data.get("notes")
            )
            
            logger.info(f"[{self.name}] ✓ Patient {patient_fact.patient_id} ingested successfully")
            logger.info(f"[{self.name}] OUTPUT PatientFact:")
            logger.info(f"  • patient_id: {patient_fact.patient_id}")
            logger.info(f"  • name: {patient_fact.name}")
            logger.info(f"  • sex: {patient_fact.sex}")
            logger.info(f"  • age: {patient_fact.age_at_diagnosis}")
            logger.info(f"  • stage: {patient_fact.tnm_stage}")
            logger.info(f"  • histology: {patient_fact.histology_type}")
            logger.info(f"  • PS: {patient_fact.performance_status}")
            logger.info(f"  • comorbidities: {patient_fact.comorbidities}")
            logger.info("=" * 60)
            # Attach ontology warnings (non-blocking) as metadata
            patient_fact.ontology_warnings = ontology_warnings

            logger.info(f"[{self.name}] EXECUTION COMPLETE - SUCCESS")
            logger.info("=" * 60)
            return patient_fact, []

        except Exception as e:
            logger.error(f"[{self.name}] EXECUTION FAILED: {str(e)}")
            errors.append(f"Failed to create PatientFact: {str(e)}")
            return None, errors

    def validate_schema(self, data: Dict[str, Any]) -> List[str]:
        """Validate raw data against required schema."""
        errors = []

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")

        # Validate age
        age = data.get("age")
        if age is not None:
            if not isinstance(age, (int, float)) or age < 0 or age > 120:
                errors.append(f"Invalid age: {age}. Must be between 0 and 120.")

        # Validate sex
        sex = data.get("sex", "").upper()
        if sex not in ["M", "F", "U", "MALE", "FEMALE"]:
            errors.append(f"Invalid sex: {data.get('sex')}. Must be M, F, or U.")

        # Validate performance status
        ps = data.get("performance_status")
        if ps is not None:
            if not isinstance(ps, int) or ps < 0 or ps > 4:
                errors.append(f"Invalid performance status: {ps}. Must be 0-4.")

        return errors

    def normalize_tnm(self, tnm_input: str) -> str:
        """Normalize TNM staging to standard format."""
        if not tnm_input:
            raise ValueError("TNM stage is required")

        normalized = tnm_input.strip()
        
        # Check exact match first
        if normalized in self.TNM_NORMALIZATION:
            return self.TNM_NORMALIZATION[normalized]
        
        # Try lowercase match
        lower = normalized.lower()
        if lower in self.TNM_NORMALIZATION:
            return self.TNM_NORMALIZATION[lower]
        
        # Try with "stage " prefix removed
        if lower.startswith("stage "):
            stage_part = lower[6:].strip()
            if stage_part in self.TNM_NORMALIZATION:
                return self.TNM_NORMALIZATION[stage_part]

        # If nothing matches, try to extract stage pattern
        match = re.search(r'(I{1,3}V?|IV)[AB]?', normalized, re.IGNORECASE)
        if match:
            extracted = match.group(0).upper()
            if extracted in ["IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IV"]:
                return extracted

        raise ValueError(f"Cannot normalize TNM stage: {tnm_input}")

    def normalize_histology(self, histology_input: str) -> str:
        """Normalize histology type to standard format."""
        if not histology_input:
            raise ValueError("Histology type is required")

        normalized = histology_input.strip().lower()
        
        # Check exact match
        if normalized in self.HISTOLOGY_NORMALIZATION:
            return self.HISTOLOGY_NORMALIZATION[normalized]
        
        # Check if already in correct format
        valid_types = [e.value for e in HistologyType]
        if histology_input in valid_types:
            return histology_input
        
        # Fuzzy match
        for key, value in self.HISTOLOGY_NORMALIZATION.items():
            if key in normalized or normalized in key:
                return value

        raise ValueError(f"Cannot normalize histology: {histology_input}")

    def calculate_age_group(self, age: int) -> str:
        """Calculate age group for cohort analysis."""
        if age < 50:
            return "<50"
        elif age < 60:
            return "50-59"
        elif age < 70:
            return "60-69"
        elif age < 80:
            return "70-79"
        else:
            return "80+"

    # Valid ontology histology classes (from LUCADA + SNOMED)
    VALID_HISTOLOGY_CLASSES = {
        "Adenocarcinoma", "SquamousCellCarcinoma", "LargeCellCarcinoma",
        "SmallCellCarcinoma", "Carcinosarcoma", "NonSmallCellCarcinoma_NOS",
        "Mesothelioma", "Carcinoid", "AdenosquamousCarcinoma",
        "LargeCell_Neuroendocrine", "Sarcomatoid",
    }

    VALID_TNM_STAGES = {"IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IIIC", "IV"}

    KNOWN_GENE_SYMBOLS = {
        "EGFR", "ALK", "ROS1", "BRAF", "KRAS", "NTRK", "MET", "RET",
        "HER2", "ERBB2", "PIK3CA", "STK11", "TP53", "NF1", "PTEN",
    }

    def _validate_against_ontology(self, raw_data: Dict[str, Any]) -> List[str]:
        """
        Validate raw data against ontology constraints (SHACL-style).

        Non-blocking: returns warnings but does not reject data.
        Checks:
        - Histology against known ontology classes
        - TNM stage validity
        - Performance status range (0-4)
        - Biomarker gene symbols against known genes
        """
        warnings = []

        # Check histology
        histology = raw_data.get("histology_type", "")
        normalized_hist = self.HISTOLOGY_NORMALIZATION.get(
            histology.strip().lower(), histology
        )
        if normalized_hist and normalized_hist not in self.VALID_HISTOLOGY_CLASSES:
            warnings.append(
                f"Histology '{histology}' (normalized: '{normalized_hist}') "
                f"not in known ontology classes"
            )

        # Check TNM stage
        tnm = raw_data.get("tnm_stage", "")
        try:
            normalized_tnm = self.normalize_tnm(tnm)
            if normalized_tnm not in self.VALID_TNM_STAGES:
                warnings.append(f"TNM stage '{normalized_tnm}' not in valid set")
        except ValueError:
            warnings.append(f"TNM stage '{tnm}' could not be normalized")

        # Check PS
        ps = raw_data.get("performance_status")
        if ps is not None:
            try:
                ps_val = int(ps)
                if ps_val < 0 or ps_val > 4:
                    warnings.append(f"Performance status {ps_val} outside valid range 0-4")
            except (ValueError, TypeError):
                warnings.append(f"Performance status '{ps}' is not a valid integer")

        # Check biomarker gene symbols
        biomarkers = raw_data.get("biomarker_profile", {})
        if isinstance(biomarkers, dict):
            for marker_type in biomarkers:
                gene = marker_type.upper().split("_")[0].split("-")[0]
                if gene not in self.KNOWN_GENE_SYMBOLS and gene != "PD":
                    warnings.append(
                        f"Biomarker gene '{marker_type}' not in known gene symbols"
                    )

        return warnings

    def _normalize_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize all patient data fields."""
        normalized = dict(raw_data)

        # Generate patient ID if not provided
        if not normalized.get("patient_id"):
            import uuid
            normalized["patient_id"] = str(uuid.uuid4())[:8].upper()

        # Normalize sex
        sex = normalized.get("sex", "U").upper()
        if sex in ["MALE", "M"]:
            normalized["sex"] = "M"
        elif sex in ["FEMALE", "F"]:
            normalized["sex"] = "F"
        else:
            normalized["sex"] = "U"

        # Normalize TNM stage
        normalized["tnm_stage"] = self.normalize_tnm(normalized.get("tnm_stage", "IV"))

        # Normalize histology
        normalized["histology_type"] = self.normalize_histology(
            normalized.get("histology_type", "NonSmallCellCarcinoma_NOS")
        )

        # Normalize performance status
        ps = normalized.get("performance_status", 0)
        normalized["performance_status"] = max(0, min(4, int(ps)))

        # Normalize laterality
        laterality = normalized.get("laterality", "Right")
        if laterality.lower() in ["right", "r"]:
            normalized["laterality"] = "Right"
        elif laterality.lower() in ["left", "l"]:
            normalized["laterality"] = "Left"
        elif laterality.lower() in ["bilateral", "both", "b"]:
            normalized["laterality"] = "Bilateral"
        else:
            normalized["laterality"] = "Right"

        # Normalize FEV1
        fev1 = normalized.get("fev1_percent")
        if fev1 is not None:
            normalized["fev1_percent"] = float(fev1)

        # Calculate age group (derived field)
        age = normalized.get("age", 0)
        normalized["age_group"] = self.calculate_age_group(age)

        return normalized
