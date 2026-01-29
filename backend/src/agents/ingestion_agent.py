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
import logging

from ..db.models import PatientFact, Sex, TNMStage, HistologyType, PerformanceStatus, Laterality

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        logger.info(f"[{self.name}] Processing patient data...")
        errors = []

        # Step 1: Validate required fields
        validation_errors = self.validate_schema(raw_data)
        if validation_errors:
            errors.extend(validation_errors)
            return None, errors

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
            return patient_fact, []

        except Exception as e:
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
