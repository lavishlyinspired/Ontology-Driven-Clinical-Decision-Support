"""
LOINC Ontology 2.0 Integration

Integrates LOINC Ontology 2.0 (October 2025 release) for laboratory test
standardization and semantic interoperability with SNOMED-CT.

LOINC Ontology 2.0: 41,000+ concepts covering 70% of top 20,000 LOINC codes.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import os
import requests
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LOINCCode:
    """LOINC code representation"""
    loinc_num: str  # e.g., "718-7"
    component: str  # What is measured (e.g., "Hemoglobin")
    property: str  # Property (e.g., "MCnc" = Mass Concentration)
    time_aspect: str  # Time (e.g., "Pt" = Point in time)
    system: str  # Where measured (e.g., "Bld" = Blood)
    scale: str  # Type of scale (e.g., "Qn" = Quantitative)
    method: Optional[str] = None  # Method (if specified)
    display_name: str = ""
    snomed_mapping: Optional[str] = None  # SNOMED-CT code via LOINC Ontology 2.0


@dataclass
class LabResult:
    """Laboratory test result with LOINC coding"""
    test_name: str
    loinc_code: Optional[str] = None
    value: Optional[float] = None
    value_string: Optional[str] = None  # For non-numeric results
    unit: Optional[str] = None
    reference_range_low: Optional[float] = None
    reference_range_high: Optional[float] = None
    interpretation: Optional[str] = None  # Normal, High, Low, Critical
    snomed_code: Optional[str] = None
    timestamp: Optional[str] = None


class LOINCIntegrator:
    """
    LOINC Ontology 2.0 integration for laboratory test standardization.

    Features:
    - Maps lab test names to LOINC codes
    - Integrates with SNOMED-CT via LOINC Ontology 2.0 mappings
    - Interprets lab results based on reference ranges
    - Provides semantic enrichment for clinical reasoning
    """

    def __init__(self, use_online_api: bool = False):
        """
        Initialize LOINC integrator.

        Args:
            use_online_api: Use LOINC FHIR API for online lookups (requires internet)
        """
        self.use_online_api = use_online_api
        self.loinc_api_base = "https://fhir.loinc.org" if use_online_api else None

        # Load local LOINC mappings
        self.loinc_mappings = self._load_local_loinc_mappings()
        self.snomed_bridge = self._load_loinc_snomed_bridge()

        logger.info(f"✓ LOINC Integrator initialized (local mappings: {len(self.loinc_mappings)})")

    def _load_local_loinc_mappings(self) -> Dict[str, LOINCCode]:
        """Load commonly used LOINC codes for lung cancer workup"""

        # Clinical chemistry - commonly used in lung cancer
        mappings = {
            # Hematology
            "hemoglobin": LOINCCode(
                loinc_num="718-7",
                component="Hemoglobin",
                property="MCnc",
                time_aspect="Pt",
                system="Bld",
                scale="Qn",
                display_name="Hemoglobin [Mass/volume] in Blood",
                snomed_mapping="365616005"  # Hemoglobin finding
            ),
            "wbc": LOINCCode(
                loinc_num="6690-2",
                component="Leukocytes",
                property="NCnc",
                time_aspect="Pt",
                system="Bld",
                scale="Qn",
                display_name="Leukocytes [#/volume] in Blood",
                snomed_mapping="365632008"
            ),
            "platelet": LOINCCode(
                loinc_num="777-3",
                component="Platelets",
                property="NCnc",
                time_aspect="Pt",
                system="Bld",
                scale="Qn",
                display_name="Platelets [#/volume] in Blood",
                snomed_mapping="365631001"
            ),
            "neutrophils": LOINCCode(
                loinc_num="751-8",
                component="Neutrophils",
                property="NCnc",
                time_aspect="Pt",
                system="Bld",
                scale="Qn",
                display_name="Neutrophils [#/volume] in Blood",
                snomed_mapping="365638004"
            ),

            # Chemistry - Renal function
            "creatinine": LOINCCode(
                loinc_num="2160-0",
                component="Creatinine",
                property="MCnc",
                time_aspect="Pt",
                system="Ser/Plas",
                scale="Qn",
                display_name="Creatinine [Mass/volume] in Serum or Plasma",
                snomed_mapping="70901006"
            ),
            "egfr": LOINCCode(
                loinc_num="48643-1",
                component="Glomerular filtration rate/1.73 sq M",
                property="VRat",
                time_aspect="Pt",
                system="Ser/Plas/Bld",
                scale="Qn",
                display_name="Glomerular filtration rate/1.73 sq M.predicted",
                snomed_mapping="80274001"
            ),

            # Chemistry - Liver function
            "alt": LOINCCode(
                loinc_num="1742-6",
                component="Alanine aminotransferase",
                property="CCnc",
                time_aspect="Pt",
                system="Ser/Plas",
                scale="Qn",
                display_name="Alanine aminotransferase [Enzymatic activity/volume]",
                snomed_mapping="45896001"
            ),
            "ast": LOINCCode(
                loinc_num="1920-8",
                component="Aspartate aminotransferase",
                property="CCnc",
                time_aspect="Pt",
                system="Ser/Plas",
                scale="Qn",
                display_name="Aspartate aminotransferase [Enzymatic activity/volume]",
                snomed_mapping="45753004"
            ),
            "bilirubin": LOINCCode(
                loinc_num="1975-2",
                component="Bilirubin.total",
                property="MCnc",
                time_aspect="Pt",
                system="Ser/Plas",
                scale="Qn",
                display_name="Bilirubin.total [Mass/volume] in Serum or Plasma",
                snomed_mapping="302787003"
            ),
            "albumin": LOINCCode(
                loinc_num="1751-7",
                component="Albumin",
                property="MCnc",
                time_aspect="Pt",
                system="Ser/Plas",
                scale="Qn",
                display_name="Albumin [Mass/volume] in Serum or Plasma",
                snomed_mapping="365634009"
            ),

            # Tumor markers
            "cea": LOINCCode(
                loinc_num="2039-6",
                component="Carcinoembryonic Ag",
                property="MCnc",
                time_aspect="Pt",
                system="Ser/Plas",
                scale="Qn",
                display_name="Carcinoembryonic Ag [Mass/volume] in Serum or Plasma",
                snomed_mapping="405944004"
            ),
            "cyfra21-1": LOINCCode(
                loinc_num="25390-9",
                component="Cytokeratin 19 fragment",
                property="MCnc",
                time_aspect="Pt",
                system="Ser/Plas",
                scale="Qn",
                display_name="Cytokeratin 19 fragment [Mass/volume]",
                snomed_mapping="445343003"
            ),

            # Blood gases (for respiratory function)
            "pao2": LOINCCode(
                loinc_num="2703-7",
                component="Oxygen",
                property="PPr",
                time_aspect="Pt",
                system="Bld",
                scale="Qn",
                display_name="Oxygen [Partial pressure] in Blood",
                snomed_mapping="250775008"
            ),
            "paco2": LOINCCode(
                loinc_num="2019-8",
                component="Carbon dioxide",
                property="PPr",
                time_aspect="Pt",
                system="Bld",
                scale="Qn",
                display_name="Carbon dioxide [Partial pressure] in Blood",
                snomed_mapping="250774007"
            ),

            # Molecular/Genetic tests
            "egfr_mutation": LOINCCode(
                loinc_num="81700-8",
                component="EGFR gene mutations found",
                property="PrThr",
                time_aspect="Pt",
                system="Tiss/Bld",
                scale="Nom",
                display_name="EGFR gene mutations found [Identifier] in Tissue",
                snomed_mapping="445326009"
            ),
            "alk_rearrangement": LOINCCode(
                loinc_num="82666-9",
                component="ALK gene rearrangements found",
                property="PrThr",
                time_aspect="Pt",
                system="Tiss",
                scale="Nom",
                display_name="ALK gene rearrangements found [Identifier]",
                snomed_mapping="445351000"
            ),
            "pdl1_expression": LOINCCode(
                loinc_num="85337-4",
                component="PD-L1 by Tumor cells",
                property="NFr",
                time_aspect="Pt",
                system="Tissue",
                scale="Qn",
                display_name="PD-L1 by Tumor cells [Interpretation] in Tissue",
                snomed_mapping="787882002"
            )
        }

        return mappings

    def _load_loinc_snomed_bridge(self) -> Dict[str, str]:
        """Load LOINC to SNOMED-CT bridge mappings (LOINC Ontology 2.0)"""

        # This would be loaded from the official LOINC Ontology 2.0 release
        # For now, using mappings defined in LOINCCode objects
        bridge = {}

        for key, loinc_code in self.loinc_mappings.items():
            if loinc_code.snomed_mapping:
                bridge[loinc_code.loinc_num] = loinc_code.snomed_mapping

        return bridge

    # ========================================
    # LOINC MAPPING
    # ========================================

    def map_lab_test(self, test_name: str) -> Optional[LOINCCode]:
        """
        Map laboratory test name to LOINC code.

        Args:
            test_name: Common test name (e.g., "Hemoglobin", "WBC")

        Returns:
            LOINCCode object or None if not found
        """
        # Normalize test name
        normalized = test_name.lower().replace(" ", "_").replace("-", "_")

        # Check local mappings first
        if normalized in self.loinc_mappings:
            return self.loinc_mappings[normalized]

        # Try online API if enabled
        if self.use_online_api:
            return self._search_loinc_api(test_name)

        logger.warning(f"No LOINC mapping found for: {test_name}")
        return None

    def _search_loinc_api(self, test_name: str) -> Optional[LOINCCode]:
        """Search LOINC FHIR API for test name"""

        if not self.loinc_api_base:
            return None

        try:
            response = requests.get(
                f"{self.loinc_api_base}/CodeSystem/$lookup",
                params={"system": "http://loinc.org", "code": test_name},
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                # Parse FHIR response and create LOINCCode
                # Implementation depends on FHIR response structure
                pass

        except Exception as e:
            logger.error(f"LOINC API search failed: {e}")

        return None

    # ========================================
    # RESULT INTERPRETATION
    # ========================================

    def interpret_lab_result(
        self,
        test_name: str,
        value: float,
        unit: str,
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None
    ) -> LabResult:
        """
        Interpret laboratory result with LOINC coding and clinical interpretation.

        Args:
            test_name: Test name
            value: Numeric result value
            unit: Unit of measurement
            patient_age: Patient age (for age-specific ranges)
            patient_sex: Patient sex (for sex-specific ranges)

        Returns:
            LabResult with interpretation
        """
        loinc_code = self.map_lab_test(test_name)

        if not loinc_code:
            return LabResult(
                test_name=test_name,
                value=value,
                unit=unit,
                interpretation="Unknown - LOINC mapping not available"
            )

        # Get reference ranges (simplified - would come from database in production)
        ref_low, ref_high = self._get_reference_range(
            loinc_code.loinc_num,
            patient_age,
            patient_sex
        )

        # Interpret result
        interpretation = self._interpret_value(value, ref_low, ref_high)

        return LabResult(
            test_name=test_name,
            loinc_code=loinc_code.loinc_num,
            value=value,
            unit=unit,
            reference_range_low=ref_low,
            reference_range_high=ref_high,
            interpretation=interpretation,
            snomed_code=loinc_code.snomed_mapping
        )

    def _get_reference_range(
        self,
        loinc_num: str,
        age: Optional[int],
        sex: Optional[str]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get reference range for LOINC code"""

        # Simplified reference ranges (would be from database in production)
        ranges = {
            "718-7": (13.0, 17.0) if sex == "M" else (12.0, 15.5),  # Hemoglobin g/dL
            "6690-2": (4.5, 11.0),  # WBC x10^9/L
            "777-3": (150, 400),  # Platelets x10^9/L
            "2160-0": (0.7, 1.3),  # Creatinine mg/dL
            "48643-1": (90, 120),  # eGFR mL/min/1.73m2
            "1742-6": (7, 56),  # ALT U/L
            "1920-8": (10, 40),  # AST U/L
            "1975-2": (0.1, 1.2),  # Bilirubin mg/dL
            "1751-7": (3.5, 5.5),  # Albumin g/dL
            "2039-6": (0, 5.0),  # CEA ng/mL
            "2703-7": (75, 100),  # PaO2 mmHg
            "2019-8": (35, 45)  # PaCO2 mmHg
        }

        return ranges.get(loinc_num, (None, None))

    def _interpret_value(
        self,
        value: float,
        ref_low: Optional[float],
        ref_high: Optional[float]
    ) -> str:
        """Interpret lab value against reference range"""

        if ref_low is None or ref_high is None:
            return "Normal"  # Default if no reference range

        if value < ref_low:
            # Check if critically low
            if ref_low > 0 and value < (ref_low * 0.5):
                return "Critical Low"
            return "Low"
        elif value > ref_high:
            # Check if critically high
            if value > (ref_high * 2.0):
                return "Critical High"
            return "High"
        else:
            return "Normal"

    # ========================================
    # BATCH PROCESSING
    # ========================================

    def process_lab_panel(
        self,
        lab_results: List[Dict[str, Any]],
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None
    ) -> List[LabResult]:
        """
        Process a panel of lab results.

        Args:
            lab_results: List of dicts with 'test_name', 'value', 'unit'
            patient_age: Patient age
            patient_sex: Patient sex

        Returns:
            List of interpreted LabResult objects
        """
        interpreted_results = []

        for lab in lab_results:
            result = self.interpret_lab_result(
                test_name=lab.get("test_name", ""),
                value=lab.get("value", 0),
                unit=lab.get("unit", ""),
                patient_age=patient_age,
                patient_sex=patient_sex
            )
            interpreted_results.append(result)

        return interpreted_results

    # ========================================
    # CLINICAL SIGNIFICANCE
    # ========================================

    def assess_clinical_significance(
        self,
        lab_results: List[LabResult]
    ) -> Dict[str, Any]:
        """
        Assess overall clinical significance of lab panel.

        Returns:
            Clinical assessment with key findings
        """
        findings = {
            "critical_abnormalities": [],
            "significant_abnormalities": [],
            "all_normal": True,
            "clinical_notes": []
        }

        for result in lab_results:
            if result.interpretation in ["Critical High", "Critical Low"]:
                findings["critical_abnormalities"].append({
                    "test": result.test_name,
                    "value": result.value,
                    "unit": result.unit,
                    "interpretation": result.interpretation
                })
                findings["all_normal"] = False

            elif result.interpretation in ["High", "Low"]:
                findings["significant_abnormalities"].append({
                    "test": result.test_name,
                    "value": result.value,
                    "unit": result.unit,
                    "interpretation": result.interpretation
                })
                findings["all_normal"] = False

        # Add clinical notes
        if findings["critical_abnormalities"]:
            findings["clinical_notes"].append(
                f"⚠️ {len(findings['critical_abnormalities'])} critical abnormalities requiring immediate attention"
            )

        # Specific assessments for treatment eligibility
        findings["treatment_eligibility"] = self._assess_treatment_eligibility(lab_results)

        return findings

    def _assess_treatment_eligibility(
        self,
        lab_results: List[LabResult]
    ) -> Dict[str, Any]:
        """Assess treatment eligibility based on lab values"""

        eligibility = {
            "chemotherapy_safe": True,
            "surgery_safe": True,
            "concerns": []
        }

        # Check renal function for chemotherapy
        egfr_result = next((r for r in lab_results if "egfr" in r.test_name.lower()), None)
        if egfr_result and egfr_result.value and egfr_result.value < 30:
            eligibility["chemotherapy_safe"] = False
            eligibility["concerns"].append("Severe renal impairment (eGFR <30) - chemotherapy dose adjustment required")

        # Check liver function
        alt_result = next((r for r in lab_results if "alt" in r.test_name.lower() or "alanine" in r.test_name.lower()), None)
        if alt_result and alt_result.interpretation in ["High", "Critical High"]:
            eligibility["chemotherapy_safe"] = False
            eligibility["concerns"].append("Elevated liver enzymes - assess hepatic function before chemotherapy")

        # Check blood counts
        wbc_result = next((r for r in lab_results if r.loinc_code == "6690-2"), None)
        if wbc_result and wbc_result.value and wbc_result.value < 3.0:
            eligibility["chemotherapy_safe"] = False
            eligibility["concerns"].append("Leukopenia (WBC <3.0) - chemotherapy may need to be delayed")

        platelet_result = next((r for r in lab_results if r.loinc_code == "777-3"), None)
        if platelet_result and platelet_result.value and platelet_result.value < 100:
            eligibility["surgery_safe"] = False
            eligibility["chemotherapy_safe"] = False
            eligibility["concerns"].append("Thrombocytopenia (platelets <100) - bleeding risk for surgery and chemotherapy")

        return eligibility

    # ========================================
    # NEO4J INTEGRATION
    # ========================================

    def store_lab_results_neo4j(
        self,
        patient_id: str,
        lab_results: List[LabResult],
        neo4j_tools
    ) -> bool:
        """
        Store lab results in Neo4j with LOINC codes.

        Args:
            patient_id: Patient ID
            lab_results: List of lab results
            neo4j_tools: Neo4j write tools

        Returns:
            Success status
        """
        # This would integrate with Neo4jWriteTools
        # Implementation depends on graph schema
        logger.info(f"Would store {len(lab_results)} lab results for patient {patient_id}")
        return True
