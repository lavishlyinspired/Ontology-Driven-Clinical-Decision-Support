"""
Lab Interpretation Specialist Agent

Specialized agent for interpreting laboratory results using LOINC codes
and providing clinical context for treatment decisions.

Implements:
- CTCAE toxicity grading
- Treatment-specific reference ranges
- Critical value detection
- Lab trend analysis
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

# Centralized logging
from ..logging_config import get_logger, log_agent_action

logger = get_logger(__name__)

from ..db.models import PatientFactWithCodes
from .negotiation_protocol import AgentProposal
from ..services.loinc_service import LOINCService, get_loinc_service, LabInterpretation


@dataclass
class LabInterpretationResult:
    """Result of lab interpretation analysis"""
    patient_id: str
    interpretations: List[Dict[str, Any]]  # Changed from LabInterpretation to Dict
    critical_values: List[Dict[str, Any]]
    recommendations: List[str]
    requires_action: bool
    action_priority: str  # "immediate", "urgent", "routine", "none"
    dose_adjustments_needed: List[Dict[str, Any]]
    follow_up_labs: List[str]  # LOINC codes for follow-up
    clinical_summary: str
    confidence: float


class LabInterpretationAgent:
    """
    Specialized agent for laboratory result interpretation.

    Key features:
    - Interprets lab values with LOINC codes
    - Flags critical values requiring immediate action
    - Provides treatment-specific reference ranges
    - Detects trends in serial lab values
    - Generates lab panels for specific workflows
    """

    def __init__(self):
        self.agent_id = "lab_interpretation_specialist"
        self.agent_type = "LabInterpretationAgent"
        self.loinc_service = get_loinc_service()
        self.version = "1.0.0"

    def _get_patient_attr(self, patient, attr: str, default=None):
        """Helper to safely get patient attributes from dict or object"""
        if isinstance(patient, dict):
            return patient.get(attr, default)
        return getattr(patient, attr, default)

    def execute(
        self,
        patient: PatientFactWithCodes,
        lab_results: Optional[List[Dict[str, Any]]] = None,
        treatment_context: Optional[Dict[str, Any]] = None
    ) -> LabInterpretationResult:
        """
        Interpret laboratory results with clinical context.

        Args:
            patient: Patient with clinical data
            lab_results: List of lab results with format:
                [{loinc_code: str, value: float, units: str, date: str}]
            treatment_context: Optional treatment context for reference ranges
                {current_treatment: str, regimen: str}

        Returns:
            LabInterpretationResult with interpretations and recommendations
        """
        patient_id = self._get_patient_attr(patient, 'patient_id', 'Unknown')
        logger.info(f"Lab Interpretation Agent analyzing patient {patient_id}")

        # Extract lab results if not provided
        if lab_results is None:
            lab_results = self._extract_lab_results(patient)

        if not lab_results:
            logger.info(f"No lab results found for patient {patient_id}")
            return LabInterpretationResult(
                patient_id=patient_id,
                interpretations=[],
                critical_values=[],
                recommendations=[],
                requires_action=False,
                action_priority="none",
                dose_adjustments_needed=[],
                follow_up_labs=[],
                clinical_summary="No laboratory results available for interpretation.",
                confidence=1.0
            )

        # Get patient context for interpretation
        sex = self._get_patient_attr(patient, 'sex', 'unknown')
        age = self._get_patient_attr(patient, 'age', 0)
        is_smoker = self._get_patient_attr(patient, 'smoking_status', 'unknown') in ['current', 'former']

        # Interpret each lab result
        interpretations = []
        critical_values = []

        for lab in lab_results:
            try:
                interpretation = self.loinc_service.interpret_lab_result(
                    loinc_code=lab.get('loinc_code'),
                    value=lab.get('value'),
                    unit=lab.get('units', lab.get('unit', '')),
                    patient_context={
                        'sex': sex,
                        'age': age,
                        'is_smoker': is_smoker,
                        'treatment': treatment_context.get('current_treatment') if treatment_context else None
                    }
                )

                interpretations.append(interpretation)

                # Flag critical values
                if interpretation.get('interpretation') in ['critical_low', 'critical_high']:
                    critical_values.append({
                        'loinc_code': interpretation.get('loinc_code'),
                        'value': interpretation.get('value'),
                        'unit': interpretation.get('unit'),
                        'interpretation': interpretation.get('interpretation'),
                        'severity': 'critical',
                        'date': lab.get('date', datetime.now().isoformat())
                    })
                elif interpretation.get('interpretation') in ['high', 'low']:
                    # Check if it's treatment-related toxicity (Grade 2+)
                    severity = self._assess_toxicity_grade(interpretation)
                    if severity and severity >= 2:
                        critical_values.append({
                            'loinc_code': interpretation.get('loinc_code'),
                            'value': interpretation.get('value'),
                            'unit': interpretation.get('unit'),
                            'interpretation': interpretation.get('interpretation'),
                            'severity': f'grade{severity}',
                            'date': lab.get('date', datetime.now().isoformat())
                        })

            except Exception as e:
                logger.error(f"Failed to interpret lab {lab.get('loinc_code')}: {e}")

        # Generate recommendations
        recommendations = self._generate_recommendations(
            interpretations,
            critical_values,
            patient,
            treatment_context
        )

        # Determine action priority
        action_priority = self._determine_action_priority(critical_values)
        requires_action = action_priority in ['immediate', 'urgent']

        # Assess dose adjustments
        dose_adjustments = self._assess_dose_adjustments(
            interpretations,
            treatment_context
        )

        # Recommend follow-up labs
        follow_up_labs = self._recommend_follow_up_labs(
            critical_values,
            treatment_context
        )

        # Generate clinical summary
        clinical_summary = self._generate_clinical_summary(
            interpretations,
            critical_values,
            recommendations
        )

        # Calculate confidence
        confidence = self._calculate_confidence(interpretations)

        result = LabInterpretationResult(
            patient_id=patient_id,
            interpretations=interpretations,
            critical_values=critical_values,
            recommendations=recommendations,
            requires_action=requires_action,
            action_priority=action_priority,
            dose_adjustments_needed=dose_adjustments,
            follow_up_labs=follow_up_labs,
            clinical_summary=clinical_summary,
            confidence=confidence
        )

        logger.info(f"Lab interpretation complete: {len(interpretations)} results, "
                   f"{len(critical_values)} critical, priority={action_priority}")

        return result

    def _extract_lab_results(self, patient) -> List[Dict[str, Any]]:
        """Extract lab results from patient data"""
        # Try to extract from patient object/dict
        lab_results = self._get_patient_attr(patient, 'lab_results', [])
        labs = self._get_patient_attr(patient, 'labs', [])

        return lab_results or labs or []

    def _assess_toxicity_grade(self, interpretation: Dict) -> Optional[int]:
        """
        Assess CTCAE toxicity grade based on lab value.

        Returns:
            1-5 toxicity grade, or None if not applicable
        """
        loinc = interpretation.get('loinc_code')
        value = interpretation.get('value')
        ref_range = interpretation.get('reference_range', {})

        # Hematologic toxicity (CTCAE 5.0)
        if loinc == "718-7":  # Hemoglobin
            if value < 8.0:
                return 3  # Grade 3: Hgb < 8.0 g/dL
            elif value < 10.0:
                return 2  # Grade 2: Hgb < 10.0 g/dL
            elif value < ref_range.get('low', 12.0):
                return 1

        elif loinc == "777-3":  # Platelets
            if value < 25:
                return 4  # Grade 4: < 25 x10^9/L
            elif value < 50:
                return 3  # Grade 3: < 50 x10^9/L
            elif value < 75:
                return 2  # Grade 2: < 75 x10^9/L
            elif value < 100:
                return 1

        elif loinc == "6690-2":  # WBC
            if value < 1.0:
                return 4  # Grade 4: < 1.0 x10^9/L
            elif value < 2.0:
                return 3  # Grade 3: < 2.0 x10^9/L
            elif value < 3.0:
                return 2

        elif loinc == "751-8":  # ANC
            if value < 0.5:
                return 4  # Grade 4: < 0.5 x10^9/L
            elif value < 1.0:
                return 3  # Grade 3: < 1.0 x10^9/L
            elif value < 1.5:
                return 2

        # Hepatic toxicity
        elif loinc in ["1742-6", "1920-8"]:  # ALT, AST
            uln = ref_range.get('high', 40)
            if value > 20 * uln:
                return 4  # Grade 4: > 20x ULN
            elif value > 5 * uln:
                return 3  # Grade 3: > 5x ULN
            elif value > 3 * uln:
                return 2  # Grade 2: > 3x ULN
            elif value > uln:
                return 1

        # Renal toxicity
        elif loinc == "2160-0":  # Creatinine
            baseline = ref_range.get('baseline', 1.0)
            if value > 3 * baseline:
                return 3  # Grade 3: > 3x baseline
            elif value > 1.5 * baseline:
                return 2

        return None

    def _generate_recommendations(
        self,
        interpretations: List[Dict],  # Changed from LabInterpretation to Dict
        critical_values: List[Dict],
        patient,
        treatment_context: Optional[Dict]
    ) -> List[str]:
        """Generate clinical recommendations based on lab results"""
        recommendations = []

        # Critical value recommendations
        for critical in critical_values:
            if critical['severity'] == 'critical':
                recommendations.append(
                    f"🔴 IMMEDIATE ACTION: {critical['loinc_code']} is critically "
                    f"{'low' if 'low' in critical['interpretation'] else 'high'}. "
                    f"Consider emergency intervention."
                )

            elif critical['severity'] in ['grade3', 'grade4']:
                recommendations.append(
                    f"⚠️ URGENT: {critical['loinc_code']} shows Grade {critical['severity'][-1]} toxicity. "
                    f"Consider dose reduction or treatment hold."
                )

            elif critical['severity'] == 'grade2':
                recommendations.append(
                    f"⚠️ {critical['loinc_code']} shows Grade 2 toxicity. "
                    f"Monitor closely and consider dose adjustment."
                )

        # Treatment-specific recommendations
        if treatment_context:
            current_treatment = treatment_context.get('current_treatment', '')

            # EGFR TKI specific monitoring
            if 'osimertinib' in current_treatment.lower() or 'erlotinib' in current_treatment.lower():
                # Check for hepatotoxicity
                alt_results = [i for i in interpretations if i.get('loinc_code') == "1742-6"]
                if alt_results and alt_results[0].get('interpretation') in ['high', 'critical_high']:
                    recommendations.append(
                        "Consider dose hold for EGFR TKI-related hepatotoxicity. "
                        "Resume at reduced dose when ALT < 3x ULN."
                    )

            # Immunotherapy specific monitoring
            if 'pembrolizumab' in current_treatment.lower() or 'nivolumab' in current_treatment.lower():
                # Check for immune-related AEs
                tsh_results = [i for i in interpretations if i.get('loinc_code') == "3016-3"]
                if tsh_results and tsh_results[0].get('interpretation') in ['high', 'low']:
                    recommendations.append(
                        "Possible immune-related thyroid dysfunction. "
                        "Consider endocrine consultation and corticosteroids if symptomatic."
                    )

            # Chemotherapy specific monitoring
            if 'carboplatin' in current_treatment.lower() or 'cisplatin' in current_treatment.lower():
                # Check for bone marrow suppression
                plt_results = [i for i in interpretations if i.get('loinc_code') == "777-3"]
                if plt_results and plt_results[0].get('value') < 100:
                    recommendations.append(
                        "Thrombocytopenia detected. Consider dose reduction or growth factor support."
                    )

        # Add standard LOINC recommendations
        for interp in interpretations:
            recommendations.extend(interp.get('recommendations', []))

        return list(set(recommendations))  # Remove duplicates

    def _determine_action_priority(self, critical_values: List[Dict]) -> str:
        """Determine urgency of action needed"""
        if not critical_values:
            return "none"

        severities = [cv['severity'] for cv in critical_values]

        if 'critical' in severities:
            return "immediate"  # Within minutes to hours
        elif 'grade4' in severities or 'grade3' in severities:
            return "urgent"  # Within 24 hours
        elif 'grade2' in severities:
            return "routine"  # Within 1 week
        else:
            return "none"

    def _assess_dose_adjustments(
        self,
        interpretations: List[Dict],  # Changed from LabInterpretation to Dict
        treatment_context: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Assess if dose adjustments are needed based on labs"""
        adjustments = []

        if not treatment_context:
            return adjustments

        current_treatment = treatment_context.get('current_treatment', '')

        # Carboplatin dose adjustment based on CrCl
        if 'carboplatin' in current_treatment.lower():
            crcl_results = [i for i in interpretations if 'creatinine clearance' in i.get('loinc_code', '')]
            if crcl_results and crcl_results[0].get('value') < 60:
                adjustments.append({
                    'drug': 'Carboplatin',
                    'current_dose': 'AUC 6',
                    'recommended_dose': 'AUC 5',
                    'rationale': f'CrCl {crcl_results[0].get("value")} mL/min < 60',
                    'evidence': 'Grade A - Calvert formula adjustment'
                })

        # Pemetrexed contraindication
        if 'pemetrexed' in current_treatment.lower():
            crcl_results = [i for i in interpretations if 'creatinine clearance' in i.get('loinc_code', '')]
            if crcl_results and crcl_results[0].get('value') < 45:
                adjustments.append({
                    'drug': 'Pemetrexed',
                    'current_dose': 'Standard',
                    'recommended_dose': 'CONTRAINDICATED',
                    'rationale': f'CrCl {crcl_results[0].get("value")} mL/min < 45',
                    'evidence': 'Grade A - FDA contraindication'
                })

        return adjustments

    def _recommend_follow_up_labs(
        self,
        critical_values: List[Dict],
        treatment_context: Optional[Dict]
    ) -> List[str]:
        """Recommend follow-up labs based on critical values"""
        follow_up = []

        # If critical values exist, recheck soon
        if critical_values:
            critical_loinc_codes = [cv['loinc_code'] for cv in critical_values]
            follow_up.extend(critical_loinc_codes)

        # Treatment-specific monitoring
        if treatment_context:
            current_treatment = treatment_context.get('current_treatment', '')

            if 'cisplatin' in current_treatment.lower():
                follow_up.extend([
                    "2160-0",  # Creatinine
                    "2823-3",  # Potassium
                    "2028-9"   # Magnesium
                ])

            if 'immunotherapy' in current_treatment.lower():
                follow_up.extend([
                    "3016-3",  # TSH
                    "1742-6",  # ALT
                    "1920-8"   # AST
                ])

        return list(set(follow_up))

    def _generate_clinical_summary(
        self,
        interpretations: List[Dict],  # Changed from LabInterpretation to Dict
        critical_values: List[Dict],
        recommendations: List[str]
    ) -> str:
        """Generate a concise clinical summary"""
        summary_parts = []

        if not interpretations:
            return "No laboratory results available for interpretation."

        # Count abnormalities
        normal_count = sum(1 for i in interpretations if i.get('interpretation') == 'normal')
        abnormal_count = len(interpretations) - normal_count

        summary_parts.append(
            f"Analyzed {len(interpretations)} laboratory results: "
            f"{normal_count} normal, {abnormal_count} abnormal."
        )

        if critical_values:
            summary_parts.append(
                f"\n⚠️ {len(critical_values)} critical value(s) detected requiring attention."
            )

        if recommendations:
            summary_parts.append(
                f"\n📋 {len(recommendations)} clinical recommendation(s) generated."
            )

        return " ".join(summary_parts)

    def _calculate_confidence(self, interpretations: List[Dict]) -> float:
        """Calculate confidence in interpretation"""
        if not interpretations:
            return 1.0

        # Confidence based on number of interpretations with known reference ranges
        interpretations_with_ranges = sum(
            1 for i in interpretations if i.get('reference_range')
        )

        confidence = interpretations_with_ranges / len(interpretations)
        return round(confidence, 2)

    def generate_lab_panel(
        self,
        panel_type: str,
        patient: Optional[PatientFactWithCodes] = None
    ) -> Dict[str, Any]:
        """
        Generate a predefined lab panel for specific workflows.

        Args:
            panel_type: Type of panel (baseline_staging, molecular_testing, etc.)
            patient: Optional patient context

        Returns:
            Dictionary with lab panel details
        """
        logger.info(f"Generating lab panel: {panel_type}")

        return self.loinc_service.get_lung_cancer_panel(panel_type)

    def flag_critical_values(
        self,
        lab_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Flag critical values from a list of lab results.

        Args:
            lab_results: List of lab results

        Returns:
            List of critical values with flags
        """
        critical = []

        for lab in lab_results:
            interpretation = self.loinc_service.interpret_lab_result(
                loinc_code=lab.get('loinc_code'),
                value=lab.get('value'),
                unit=lab.get('units', '')
            )

            if interpretation.get('interpretation') in ['critical_low', 'critical_high']:
                critical.append({
                    'loinc_code': interpretation.get('loinc_code'),
                    'value': interpretation.get('value'),
                    'unit': interpretation.get('unit'),
                    'interpretation': interpretation.get('interpretation'),
                    'severity': 'critical',
                    'clinical_significance': interpretation.get('clinical_significance')
                })

        return critical


# Singleton instance
_lab_interpretation_agent = None


def get_lab_interpretation_agent() -> LabInterpretationAgent:
    """Get singleton LabInterpretationAgent instance"""
    global _lab_interpretation_agent
    if _lab_interpretation_agent is None:
        _lab_interpretation_agent = LabInterpretationAgent()
    return _lab_interpretation_agent
