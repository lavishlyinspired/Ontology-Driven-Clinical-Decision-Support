"""
Medication Management Specialist Agent

Specialized agent for comprehensive medication management using RxNorm codes,
including drug-drug interaction checking, therapeutic alternative identification,
and medication safety assessment.

Implements:
- RxNorm drug formulary integration
- Drug-drug interaction detection
- Therapeutic alternative recommendations
- Contraindication assessment
- Medication safety scoring
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

# Centralized logging
from ..logging_config import get_logger, log_agent_action

logger = get_logger(__name__)

from ..db.models import PatientFactWithCodes
from .negotiation_protocol import AgentProposal
from ..services.rxnorm_service import RXNORMService, get_rxnorm_service


@dataclass
class DrugInteractionWarning:
    """Drug-drug interaction warning"""
    drug1: str
    drug2: str
    drug1_rxcui: str
    drug2_rxcui: str
    severity: str  # "SEVERE", "MODERATE", "MILD"
    mechanism: str
    clinical_effect: str
    recommendation: str
    evidence_level: str


@dataclass
class MedicationRecommendation:
    """Result of medication management analysis"""
    patient_id: str
    recommended_medications: List[Dict[str, Any]]
    drug_interactions: List[DrugInteractionWarning]
    contraindications: List[Dict[str, Any]]
    therapeutic_alternatives: Dict[str, List[Dict[str, Any]]]
    safety_score: float  # 0.0 (unsafe) to 1.0 (safe)
    recommendations: List[str]
    requires_clinical_review: bool
    clinical_summary: str
    confidence: float


class MedicationManagementAgent:
    """
    Specialized agent for medication management and safety.

    Key features:
    - Validates medication selection against RxNorm formulary
    - Checks drug-drug interactions across patient medication list
    - Identifies therapeutic alternatives within same class
    - Assesses contraindications based on patient comorbidities
    - Provides dosing and administration guidance
    """

    def __init__(self):
        self.agent_id = "medication_management_specialist"
        self.agent_type = "MedicationManagementAgent"
        self.rxnorm_service = get_rxnorm_service()
        self.version = "1.0.0"

        # Severity weights for safety scoring
        self.severity_weights = {
            "SEVERE": 0.0,     # Major safety concern
            "MODERATE": 0.5,   # Moderate concern
            "MILD": 0.8,       # Minor concern
            "NONE": 1.0        # No concern
        }

    def _get_patient_attr(self, patient, attr: str, default=None):
        """Helper to safely get patient attributes from dict or object"""
        if isinstance(patient, dict):
            return patient.get(attr, default)
        return getattr(patient, attr, default)

    def execute(
        self,
        patient: PatientFactWithCodes,
        medications: Optional[List[Dict[str, Any]]] = None,
        proposed_treatment: Optional[str] = None
    ) -> MedicationRecommendation:
        """
        Analyze medications for safety and provide recommendations.

        Args:
            patient: Patient with clinical data
            medications: List of current medications with format:
                [{drug_name: str, dose: str, route: str, frequency: str}]
            proposed_treatment: Optional proposed new treatment to validate

        Returns:
            MedicationRecommendation with safety analysis and recommendations
        """
        patient_id = self._get_patient_attr(patient, 'patient_id', 'Unknown')
        logger.info(f"Medication Management Agent analyzing patient {patient_id}")

        # Extract medications if not provided
        if medications is None:
            medications = self._extract_medications(patient)

        # Add proposed treatment to medication list for interaction checking
        if proposed_treatment:
            medications = list(medications) + [{
                'drug_name': proposed_treatment,
                'dose': 'Proposed',
                'route': 'TBD',
                'frequency': 'TBD',
                'is_proposed': True
            }]

        if not medications:
            logger.info(f"No medications found for patient {patient_id}")
            return MedicationRecommendation(
                patient_id=patient_id,
                recommended_medications=[],
                drug_interactions=[],
                contraindications=[],
                therapeutic_alternatives={},
                safety_score=1.0,
                recommendations=[],
                requires_clinical_review=False,
                clinical_summary="No medications to analyze.",
                confidence=1.0
            )

        # Validate medications and get details
        recommended_medications = []
        for med in medications:
            drug_name = med.get('drug_name', '')
            try:
                drug_details = self.rxnorm_service.get_drug_details(drug_name)
                if drug_details:
                    recommended_medications.append({
                        'drug_name': drug_details.get('name', drug_name),
                        'rxcui': drug_details.get('rxcui', ''),
                        'drug_class': drug_details.get('class', 'Unknown'),
                        'dose': med.get('dose', drug_details.get('typical_dose', '')),
                        'route': med.get('route', drug_details.get('route', '')),
                        'frequency': med.get('frequency', drug_details.get('frequency', '')),
                        'is_proposed': med.get('is_proposed', False)
                    })
            except Exception as e:
                logger.error(f"Failed to get details for {drug_name}: {e}")

        # Check drug-drug interactions
        drug_interactions = self._check_drug_interactions(recommended_medications)

        # Assess contraindications
        contraindications = self._assess_contraindications(
            recommended_medications,
            patient
        )

        # Find therapeutic alternatives for problematic medications
        therapeutic_alternatives = self._find_therapeutic_alternatives(
            recommended_medications,
            drug_interactions,
            contraindications
        )

        # Calculate safety score
        safety_score = self._calculate_safety_score(
            drug_interactions,
            contraindications
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            recommended_medications,
            drug_interactions,
            contraindications,
            therapeutic_alternatives
        )

        # Determine if clinical review required
        requires_clinical_review = (
            any(di.severity == "SEVERE" for di in drug_interactions) or
            len(contraindications) > 0 or
            safety_score < 0.7
        )

        # Generate clinical summary
        clinical_summary = self._generate_clinical_summary(
            recommended_medications,
            drug_interactions,
            contraindications,
            safety_score
        )

        result = MedicationRecommendation(
            patient_id=patient_id,
            recommended_medications=recommended_medications,
            drug_interactions=drug_interactions,
            contraindications=contraindications,
            therapeutic_alternatives=therapeutic_alternatives,
            safety_score=safety_score,
            recommendations=recommendations,
            requires_clinical_review=requires_clinical_review,
            clinical_summary=clinical_summary,
            confidence=0.95  # High confidence in RxNorm data
        )

        logger.info(f"Medication analysis complete: {len(recommended_medications)} medications, "
                   f"{len(drug_interactions)} interactions, safety_score={safety_score:.2f}")

        return result

    def _extract_medications(self, patient) -> List[Dict[str, Any]]:
        """Extract medications from patient data"""
        medications = self._get_patient_attr(patient, 'medications', [])
        current_meds = self._get_patient_attr(patient, 'current_medications', [])

        return medications or current_meds or []

    def _check_drug_interactions(
        self,
        medications: List[Dict[str, Any]]
    ) -> List[DrugInteractionWarning]:
        """Check for drug-drug interactions across medication list"""
        if len(medications) < 2:
            return []

        drug_names = [med['drug_name'] for med in medications]

        try:
            interaction_results = self.rxnorm_service.check_drug_interactions(drug_names)

            warnings = []
            for interaction in interaction_results.get('interactions', []):
                warnings.append(DrugInteractionWarning(
                    drug1=interaction['drug1_name'],
                    drug2=interaction['drug2_name'],
                    drug1_rxcui=interaction['drug1_rxcui'],
                    drug2_rxcui=interaction['drug2_rxcui'],
                    severity=interaction['severity'],
                    mechanism=interaction['mechanism'],
                    clinical_effect=interaction['clinical_effect'],
                    recommendation=interaction['recommendation'],
                    evidence_level=interaction.get('evidence_level', 'Grade C')
                ))

            # Sort by severity
            severity_order = {"SEVERE": 0, "MODERATE": 1, "MILD": 2}
            warnings.sort(key=lambda w: severity_order.get(w.severity, 3))

            return warnings

        except Exception as e:
            logger.error(f"Failed to check drug interactions: {e}")
            return []

    def _assess_contraindications(
        self,
        medications: List[Dict[str, Any]],
        patient
    ) -> List[Dict[str, Any]]:
        """Assess contraindications based on patient comorbidities and conditions"""
        contraindications = []

        # Get patient comorbidities
        comorbidities = self._get_patient_attr(patient, 'comorbidities', [])
        renal_function = self._get_patient_attr(patient, 'renal_function', None)
        hepatic_function = self._get_patient_attr(patient, 'hepatic_function', None)
        age = self._get_patient_attr(patient, 'age', 0)

        for med in medications:
            drug_name = med['drug_name'].lower()

            # Pemetrexed: Contraindicated in severe renal impairment
            if 'pemetrexed' in drug_name:
                if renal_function and renal_function.get('crcl', 100) < 45:
                    contraindications.append({
                        'drug': med['drug_name'],
                        'reason': f'CrCl {renal_function.get("crcl")} mL/min < 45',
                        'severity': 'ABSOLUTE',
                        'recommendation': 'Contraindicated - select alternative chemotherapy'
                    })

            # Cisplatin: Caution in renal impairment
            if 'cisplatin' in drug_name:
                if renal_function and renal_function.get('crcl', 100) < 60:
                    contraindications.append({
                        'drug': med['drug_name'],
                        'reason': 'Renal impairment - increased nephrotoxicity risk',
                        'severity': 'RELATIVE',
                        'recommendation': 'Consider carboplatin alternative or dose reduction'
                    })

            # Immunotherapy: Caution in autoimmune disease
            if any(x in drug_name for x in ['pembrolizumab', 'nivolumab', 'atezolizumab']):
                if any('autoimmune' in str(c).lower() for c in comorbidities):
                    contraindications.append({
                        'drug': med['drug_name'],
                        'reason': 'Active autoimmune disease',
                        'severity': 'RELATIVE',
                        'recommendation': 'Requires rheumatology consultation, increased immune-related AE monitoring'
                    })

            # EGFR TKIs: Dose adjustment in hepatic impairment
            if any(x in drug_name for x in ['osimertinib', 'erlotinib', 'gefitinib']):
                if hepatic_function and hepatic_function.get('child_pugh', 'A') in ['B', 'C']:
                    contraindications.append({
                        'drug': med['drug_name'],
                        'reason': f'Hepatic impairment (Child-Pugh {hepatic_function.get("child_pugh")})',
                        'severity': 'RELATIVE',
                        'recommendation': 'Consider dose reduction or alternate TKI'
                    })

            # Elderly considerations
            if age > 75:
                if 'cisplatin' in drug_name:
                    contraindications.append({
                        'drug': med['drug_name'],
                        'reason': 'Age > 75 years - increased toxicity risk',
                        'severity': 'RELATIVE',
                        'recommendation': 'Consider carboplatin alternative or reduced dose intensity'
                    })

        return contraindications

    def _find_therapeutic_alternatives(
        self,
        medications: List[Dict[str, Any]],
        drug_interactions: List[DrugInteractionWarning],
        contraindications: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Find therapeutic alternatives for problematic medications"""
        alternatives = {}

        # Find alternatives for medications with SEVERE interactions
        severe_interaction_drugs = set()
        for interaction in drug_interactions:
            if interaction.severity == "SEVERE":
                severe_interaction_drugs.add(interaction.drug1)
                severe_interaction_drugs.add(interaction.drug2)

        # Find alternatives for contraindicated medications
        contraindicated_drugs = {ci['drug'] for ci in contraindications if ci['severity'] == 'ABSOLUTE'}

        # Combine problematic medications
        problematic_drugs = severe_interaction_drugs | contraindicated_drugs

        for drug_name in problematic_drugs:
            try:
                alt_results = self.rxnorm_service.get_therapeutic_alternatives(drug_name)
                if alt_results and alt_results.get('alternatives'):
                    alternatives[drug_name] = alt_results['alternatives']
            except Exception as e:
                logger.error(f"Failed to find alternatives for {drug_name}: {e}")

        return alternatives

    def _calculate_safety_score(
        self,
        drug_interactions: List[DrugInteractionWarning],
        contraindications: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall medication safety score (0.0 = unsafe, 1.0 = safe)
        """
        if not drug_interactions and not contraindications:
            return 1.0

        score_components = []

        # Interaction safety component
        if drug_interactions:
            interaction_scores = [
                self.severity_weights.get(di.severity, 0.5)
                for di in drug_interactions
            ]
            # Use minimum (worst) interaction score
            score_components.append(min(interaction_scores))
        else:
            score_components.append(1.0)

        # Contraindication safety component
        if contraindications:
            # ABSOLUTE contraindication = 0.0, RELATIVE = 0.5
            ci_scores = [
                0.0 if ci['severity'] == 'ABSOLUTE' else 0.5
                for ci in contraindications
            ]
            score_components.append(min(ci_scores))
        else:
            score_components.append(1.0)

        # Overall score is weighted average
        overall_score = sum(score_components) / len(score_components)

        return round(overall_score, 2)

    def _generate_recommendations(
        self,
        medications: List[Dict[str, Any]],
        drug_interactions: List[DrugInteractionWarning],
        contraindications: List[Dict[str, Any]],
        therapeutic_alternatives: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """Generate clinical recommendations for medication management"""
        recommendations = []

        # SEVERE interaction recommendations
        severe_interactions = [di for di in drug_interactions if di.severity == "SEVERE"]
        for interaction in severe_interactions:
            recommendations.append(
                f"🔴 SEVERE INTERACTION: {interaction.drug1} + {interaction.drug2}. "
                f"{interaction.clinical_effect}. "
                f"RECOMMENDATION: {interaction.recommendation}"
            )

            # Suggest alternatives if available
            if interaction.drug1 in therapeutic_alternatives:
                alts = therapeutic_alternatives[interaction.drug1]
                if alts:
                    alt_names = ', '.join([alt['name'] for alt in alts[:3]])
                    recommendations.append(
                        f"  → Consider alternatives to {interaction.drug1}: {alt_names}"
                    )

        # MODERATE interaction recommendations
        moderate_interactions = [di for di in drug_interactions if di.severity == "MODERATE"]
        if moderate_interactions:
            recommendations.append(
                f"⚠️ MODERATE INTERACTIONS: {len(moderate_interactions)} identified. "
                f"Monitor closely for {', '.join(set(di.clinical_effect for di in moderate_interactions[:2]))}."
            )

        # Contraindication recommendations
        absolute_ci = [ci for ci in contraindications if ci['severity'] == 'ABSOLUTE']
        for ci in absolute_ci:
            recommendations.append(
                f"🚫 CONTRAINDICATED: {ci['drug']} - {ci['reason']}. "
                f"RECOMMENDATION: {ci['recommendation']}"
            )

            # Suggest alternatives if available
            if ci['drug'] in therapeutic_alternatives:
                alts = therapeutic_alternatives[ci['drug']]
                if alts:
                    alt_names = ', '.join([alt['name'] for alt in alts[:3]])
                    recommendations.append(
                        f"  → Alternatives: {alt_names}"
                    )

        relative_ci = [ci for ci in contraindications if ci['severity'] == 'RELATIVE']
        for ci in relative_ci:
            recommendations.append(
                f"⚠️ CAUTION: {ci['drug']} - {ci['reason']}. {ci['recommendation']}"
            )

        # Medication adherence recommendations
        complex_regimens = [med for med in medications if 'TKI' in med.get('drug_class', '')]
        if complex_regimens:
            recommendations.append(
                "💊 ADHERENCE: Targeted therapies require strict adherence for efficacy. "
                "Provide patient education on dosing schedule and food interactions."
            )

        # Supportive care recommendations
        if any('chemotherapy' in med.get('drug_class', '').lower() for med in medications):
            recommendations.append(
                "💊 SUPPORTIVE CARE: Ensure antiemetic prophylaxis (5-HT3 antagonist + NK1 antagonist + dexamethasone) "
                "and growth factor support per ASCO guidelines."
            )

        return recommendations

    def _generate_clinical_summary(
        self,
        medications: List[Dict[str, Any]],
        drug_interactions: List[DrugInteractionWarning],
        contraindications: List[Dict[str, Any]],
        safety_score: float
    ) -> str:
        """Generate a concise clinical summary"""
        summary_parts = []

        summary_parts.append(
            f"Analyzed {len(medications)} medication(s) with overall safety score: {safety_score:.2f}/1.00."
        )

        if drug_interactions:
            severe_count = sum(1 for di in drug_interactions if di.severity == "SEVERE")
            moderate_count = sum(1 for di in drug_interactions if di.severity == "MODERATE")
            mild_count = len(drug_interactions) - severe_count - moderate_count

            summary_parts.append(
                f"\n⚠️ Drug Interactions: {severe_count} SEVERE, {moderate_count} MODERATE, {mild_count} MILD."
            )

        if contraindications:
            absolute_count = sum(1 for ci in contraindications if ci['severity'] == 'ABSOLUTE')
            relative_count = len(contraindications) - absolute_count

            summary_parts.append(
                f"\n🚫 Contraindications: {absolute_count} ABSOLUTE, {relative_count} RELATIVE."
            )

        if safety_score < 0.5:
            summary_parts.append("\n🔴 URGENT: Medication regimen requires immediate clinical review.")
        elif safety_score < 0.7:
            summary_parts.append("\n⚠️ CAUTION: Medication regimen requires careful monitoring.")
        else:
            summary_parts.append("\n✓ Medication regimen appears acceptable with monitoring.")

        return " ".join(summary_parts)

    def validate_medication(
        self,
        drug_name: str,
        patient: Optional[PatientFactWithCodes] = None
    ) -> Dict[str, Any]:
        """
        Validate a single medication selection.

        Args:
            drug_name: Name of medication to validate
            patient: Optional patient context

        Returns:
            Validation result with drug details and safety assessment
        """
        logger.info(f"Validating medication: {drug_name}")

        try:
            drug_details = self.rxnorm_service.get_drug_details(drug_name)

            if not drug_details:
                return {
                    'valid': False,
                    'reason': 'Drug not found in RxNorm formulary'
                }

            # Basic validation passed
            result = {
                'valid': True,
                'drug_name': drug_details.get('name'),
                'rxcui': drug_details.get('rxcui'),
                'drug_class': drug_details.get('class'),
                'typical_dose': drug_details.get('typical_dose'),
                'route': drug_details.get('route'),
                'monitoring': drug_details.get('monitoring', [])
            }

            # Check contraindications if patient provided
            if patient:
                contraindications = self._assess_contraindications([{
                    'drug_name': drug_name,
                    'drug_class': drug_details.get('class', '')
                }], patient)

                if contraindications:
                    result['contraindications'] = contraindications
                    result['requires_review'] = True

            return result

        except Exception as e:
            logger.error(f"Failed to validate {drug_name}: {e}")
            return {
                'valid': False,
                'reason': str(e)
            }


# Singleton instance
_medication_management_agent = None


def get_medication_management_agent() -> MedicationManagementAgent:
    """Get singleton MedicationManagementAgent instance"""
    global _medication_management_agent
    if _medication_management_agent is None:
        _medication_management_agent = MedicationManagementAgent()
    return _medication_management_agent
