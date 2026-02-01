"""
Monitoring Coordinator Specialist Agent

Specialized agent for coordinating laboratory-drug monitoring using integrated
LOINC, RxNorm, and Lab-Drug services. Generates monitoring protocols,
assesses dose adjustments, and predicts lab effects.

Implements:
- Treatment-specific monitoring protocol generation
- Lab-based dose adjustment recommendations
- Expected lab effect prediction
- Monitoring schedule creation
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Centralized logging
from ..logging_config import get_logger, log_agent_action

logger = get_logger(__name__)

from ..db.models import PatientFactWithCodes
from .negotiation_protocol import AgentProposal
from ..services.loinc_service import LOINCService, get_loinc_service
from ..services.rxnorm_service import RXNORMService, get_rxnorm_service
from ..services.lab_drug_service import LabDrugService, get_lab_drug_service


@dataclass
class MonitoringProtocol:
    """Monitoring protocol for treatment regimen"""
    protocol_id: str
    patient_id: str
    regimen: str
    protocol_name: str
    frequency: str  # "weekly", "q3weeks", "monthly", etc.
    duration: str  # "4 weeks", "indefinite", etc.
    lab_tests: List[Dict[str, Any]]  # LOINC codes with schedules
    baseline_labs: List[str]  # LOINC codes for baseline
    monitoring_labs: List[str]  # LOINC codes for ongoing monitoring
    alert_thresholds: Dict[str, Any]  # Critical values to watch
    created_date: datetime
    status: str  # "active", "completed", "discontinued"


@dataclass
class DoseAdjustment:
    """Dose adjustment recommendation"""
    drug_name: str
    current_dose: str
    recommended_dose: str
    rationale: str
    lab_trigger: Dict[str, Any]  # Which lab triggered adjustment
    severity: str  # "immediate", "urgent", "routine"
    evidence_level: str


@dataclass
class MonitoringRecommendation:
    """Result of monitoring coordination analysis"""
    patient_id: str
    monitoring_protocol: Optional[MonitoringProtocol]
    dose_adjustments: List[DoseAdjustment]
    predicted_lab_effects: Dict[str, List[Dict[str, Any]]]
    follow_up_schedule: List[Dict[str, Any]]
    recommendations: List[str]
    requires_immediate_action: bool
    clinical_summary: str
    confidence: float


class MonitoringCoordinatorAgent:
    """
    Specialized agent for coordinating lab-drug monitoring.

    Key features:
    - Generates monitoring protocols for treatment regimens
    - Assesses current labs against treatment-specific safety thresholds
    - Recommends dose adjustments based on lab values
    - Predicts expected lab changes for given drugs
    - Creates monitoring schedules (baseline, weekly, monthly)
    """

    def __init__(self):
        self.agent_id = "monitoring_coordinator_specialist"
        self.agent_type = "MonitoringCoordinatorAgent"
        self.loinc_service = get_loinc_service()
        self.rxnorm_service = get_rxnorm_service()
        self.lab_drug_service = get_lab_drug_service()
        self.version = "1.0.0"

    def _get_patient_attr(self, patient, attr: str, default=None):
        """Helper to safely get patient attributes from dict or object"""
        if isinstance(patient, dict):
            return patient.get(attr, default)
        return getattr(patient, attr, default)

    def execute(
        self,
        patient: PatientFactWithCodes,
        current_treatment: Optional[str] = None,
        lab_results: Optional[List[Dict[str, Any]]] = None,
        medications: Optional[List[Dict[str, Any]]] = None
    ) -> MonitoringRecommendation:
        """
        Generate monitoring protocol and assess dose adjustments.

        Args:
            patient: Patient with clinical data
            current_treatment: Current treatment regimen
            lab_results: Recent lab results for dose adjustment assessment
            medications: Current medication list

        Returns:
            MonitoringRecommendation with protocol and dose adjustments
        """
        patient_id = self._get_patient_attr(patient, 'patient_id', 'Unknown')
        logger.info(f"Monitoring Coordinator Agent analyzing patient {patient_id}")

        # Extract treatment if not provided
        if current_treatment is None:
            current_treatment = self._extract_treatment(patient, medications)

        if not current_treatment:
            logger.info(f"No treatment regimen found for patient {patient_id}")
            return MonitoringRecommendation(
                patient_id=patient_id,
                monitoring_protocol=None,
                dose_adjustments=[],
                predicted_lab_effects={},
                follow_up_schedule=[],
                recommendations=["No active treatment regimen. Protocol will be generated when treatment starts."],
                requires_immediate_action=False,
                clinical_summary="No active treatment requiring monitoring.",
                confidence=1.0
            )

        # Generate monitoring protocol
        monitoring_protocol = self._create_monitoring_protocol(
            patient_id,
            current_treatment
        )

        # Assess dose adjustments if lab results available
        dose_adjustments = []
        if lab_results:
            dose_adjustments = self._assess_dose_adjustments(
                current_treatment,
                lab_results,
                medications
            )

        # Predict lab effects for current medications
        predicted_lab_effects = self._predict_lab_effects(
            current_treatment,
            medications
        )

        # Create follow-up schedule
        follow_up_schedule = self._create_follow_up_schedule(
            monitoring_protocol,
            dose_adjustments
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            monitoring_protocol,
            dose_adjustments,
            predicted_lab_effects
        )

        # Determine if immediate action required
        requires_immediate_action = any(
            da.severity == "immediate" for da in dose_adjustments
        )

        # Generate clinical summary
        clinical_summary = self._generate_clinical_summary(
            monitoring_protocol,
            dose_adjustments,
            requires_immediate_action
        )

        result = MonitoringRecommendation(
            patient_id=patient_id,
            monitoring_protocol=monitoring_protocol,
            dose_adjustments=dose_adjustments,
            predicted_lab_effects=predicted_lab_effects,
            follow_up_schedule=follow_up_schedule,
            recommendations=recommendations,
            requires_immediate_action=requires_immediate_action,
            clinical_summary=clinical_summary,
            confidence=0.90
        )

        logger.info(f"Monitoring coordination complete: Protocol created, "
                   f"{len(dose_adjustments)} dose adjustments, "
                   f"immediate_action={requires_immediate_action}")

        return result

    def _extract_treatment(
        self,
        patient,
        medications: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Extract treatment regimen from patient or medications"""
        # Try to get from patient
        treatment = self._get_patient_attr(patient, 'current_treatment', None)
        if treatment:
            return treatment

        # Try to construct from medications
        if medications:
            drug_names = [med.get('drug_name', '') for med in medications]
            return ' + '.join(drug_names)

        return ""

    def _create_monitoring_protocol(
        self,
        patient_id: str,
        regimen: str
    ) -> Optional[MonitoringProtocol]:
        """Create monitoring protocol for treatment regimen"""
        try:
            # Get protocol from lab-drug service
            protocol_data = self.lab_drug_service.get_monitoring_protocol(regimen)

            if not protocol_data or not protocol_data.get('protocol_name'):
                logger.warning(f"No standard protocol found for regimen: {regimen}")
                # Create generic protocol
                return self._create_generic_protocol(patient_id, regimen)

            # Build lab test schedule
            lab_tests = []
            for lab in protocol_data.get('lab_tests', []):
                lab_tests.append({
                    'loinc_code': lab['loinc_code'],
                    'loinc_name': lab['loinc_name'],
                    'frequency': lab['frequency'],
                    'category': lab.get('category', 'unknown')
                })

            # Extract baseline and monitoring labs
            baseline_labs = [
                lab['loinc_code'] for lab in protocol_data.get('baseline_labs', [])
            ]
            monitoring_labs = [
                lab['loinc_code'] for lab in protocol_data.get('monitoring_labs', [])
            ]

            # Build alert thresholds
            alert_thresholds = protocol_data.get('alert_thresholds', {})

            protocol = MonitoringProtocol(
                protocol_id=f"{patient_id}_{regimen.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}",
                patient_id=patient_id,
                regimen=regimen,
                protocol_name=protocol_data['protocol_name'],
                frequency=protocol_data.get('frequency', 'variable'),
                duration=protocol_data.get('duration', 'indefinite'),
                lab_tests=lab_tests,
                baseline_labs=baseline_labs,
                monitoring_labs=monitoring_labs,
                alert_thresholds=alert_thresholds,
                created_date=datetime.now(),
                status='active'
            )

            return protocol

        except Exception as e:
            logger.error(f"Failed to create monitoring protocol: {e}")
            return self._create_generic_protocol(patient_id, regimen)

    def _create_generic_protocol(
        self,
        patient_id: str,
        regimen: str
    ) -> MonitoringProtocol:
        """Create generic monitoring protocol for unknown regimens"""
        # Standard lung cancer monitoring panel
        generic_labs = [
            {'loinc_code': '718-7', 'loinc_name': 'Hemoglobin', 'frequency': 'weekly', 'category': 'hematology'},
            {'loinc_code': '777-3', 'loinc_name': 'Platelets', 'frequency': 'weekly', 'category': 'hematology'},
            {'loinc_code': '6690-2', 'loinc_name': 'WBC', 'frequency': 'weekly', 'category': 'hematology'},
            {'loinc_code': '1742-6', 'loinc_name': 'ALT', 'frequency': 'q3weeks', 'category': 'chemistry'},
            {'loinc_code': '1920-8', 'loinc_name': 'AST', 'frequency': 'q3weeks', 'category': 'chemistry'},
            {'loinc_code': '2160-0', 'loinc_name': 'Creatinine', 'frequency': 'q3weeks', 'category': 'chemistry'},
        ]

        return MonitoringProtocol(
            protocol_id=f"{patient_id}_generic_{datetime.now().strftime('%Y%m%d')}",
            patient_id=patient_id,
            regimen=regimen,
            protocol_name="Generic Lung Cancer Treatment Monitoring",
            frequency="weekly initially, then q3weeks",
            duration="during treatment",
            lab_tests=generic_labs,
            baseline_labs=['718-7', '777-3', '6690-2', '1742-6', '1920-8', '2160-0'],
            monitoring_labs=['718-7', '777-3', '6690-2', '1742-6', '2160-0'],
            alert_thresholds={
                '718-7': {'critical_low': 8.0, 'low': 10.0},
                '777-3': {'critical_low': 25, 'low': 50},
                '1742-6': {'high': 120, 'critical_high': 200}
            },
            created_date=datetime.now(),
            status='active'
        )

    def _assess_dose_adjustments(
        self,
        regimen: str,
        lab_results: List[Dict[str, Any]],
        medications: Optional[List[Dict[str, Any]]]
    ) -> List[DoseAdjustment]:
        """Assess if dose adjustments needed based on lab results"""
        adjustments = []

        # Extract drug names from regimen
        drugs = self._extract_drugs_from_regimen(regimen, medications)

        for drug in drugs:
            try:
                # Convert lab results to format expected by lab_drug_service
                lab_dict = {
                    lab['loinc_code']: {
                        'value': lab['value'],
                        'units': lab.get('units', '')
                    }
                    for lab in lab_results
                }

                # Assess dose adjustment
                assessment = self.lab_drug_service.assess_dose_for_labs(
                    drug_name=drug,
                    lab_results=lab_dict
                )

                if assessment and assessment.get('adjustment_needed'):
                    # Find triggering lab
                    trigger_loinc = assessment.get('trigger_loinc', '')
                    trigger_lab = next(
                        (lab for lab in lab_results if lab['loinc_code'] == trigger_loinc),
                        {}
                    )

                    adjustments.append(DoseAdjustment(
                        drug_name=drug,
                        current_dose=assessment.get('current_dose', 'Standard'),
                        recommended_dose=assessment.get('recommended_dose', 'Adjust'),
                        rationale=assessment.get('rationale', ''),
                        lab_trigger={
                            'loinc_code': trigger_loinc,
                            'value': trigger_lab.get('value'),
                            'units': trigger_lab.get('units')
                        },
                        severity=assessment.get('severity', 'routine'),
                        evidence_level=assessment.get('evidence_level', 'Grade B')
                    ))

            except Exception as e:
                logger.error(f"Failed to assess dose adjustment for {drug}: {e}")

        # Sort by severity
        severity_order = {"immediate": 0, "urgent": 1, "routine": 2}
        adjustments.sort(key=lambda a: severity_order.get(a.severity, 3))

        return adjustments

    def _predict_lab_effects(
        self,
        regimen: str,
        medications: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Predict expected lab changes for medications"""
        predicted_effects = {}

        drugs = self._extract_drugs_from_regimen(regimen, medications)

        for drug in drugs:
            try:
                effects = self.lab_drug_service.check_drug_lab_effects(drug)

                if effects and effects.get('lab_effects'):
                    predicted_effects[drug] = effects['lab_effects']

            except Exception as e:
                logger.error(f"Failed to predict lab effects for {drug}: {e}")

        return predicted_effects

    def _extract_drugs_from_regimen(
        self,
        regimen: str,
        medications: Optional[List[Dict[str, Any]]]
    ) -> List[str]:
        """Extract individual drug names from regimen string"""
        if medications:
            return [med.get('drug_name', '') for med in medications if med.get('drug_name')]

        # Parse regimen string (e.g., "Carboplatin + Pemetrexed + Pembrolizumab")
        drugs = [drug.strip() for drug in regimen.split('+')]
        return drugs

    def _create_follow_up_schedule(
        self,
        protocol: Optional[MonitoringProtocol],
        dose_adjustments: List[DoseAdjustment]
    ) -> List[Dict[str, Any]]:
        """Create follow-up lab schedule"""
        if not protocol:
            return []

        schedule = []
        today = datetime.now()

        # Add baseline labs (within 1 week)
        if protocol.baseline_labs:
            schedule.append({
                'date': today + timedelta(days=3),
                'type': 'baseline',
                'loinc_codes': protocol.baseline_labs,
                'description': 'Baseline laboratory assessment before treatment'
            })

        # Add monitoring labs based on frequency
        frequency_map = {
            'weekly': 7,
            'q2weeks': 14,
            'q3weeks': 21,
            'monthly': 30
        }

        for lab in protocol.lab_tests:
            frequency = lab.get('frequency', 'monthly')
            interval_days = frequency_map.get(frequency, 30)

            # Schedule next 3 monitoring points
            for i in range(1, 4):
                schedule.append({
                    'date': today + timedelta(days=interval_days * i),
                    'type': 'monitoring',
                    'loinc_codes': [lab['loinc_code']],
                    'description': f"{lab['loinc_name']} - {frequency} monitoring"
                })

        # Add urgent follow-ups for dose adjustments
        for adjustment in dose_adjustments:
            if adjustment.severity in ['immediate', 'urgent']:
                days_ahead = 1 if adjustment.severity == 'immediate' else 7
                schedule.append({
                    'date': today + timedelta(days=days_ahead),
                    'type': 'recheck',
                    'loinc_codes': [adjustment.lab_trigger.get('loinc_code')],
                    'description': f"Recheck {adjustment.lab_trigger.get('loinc_code')} after {adjustment.drug_name} dose adjustment"
                })

        # Sort by date
        schedule.sort(key=lambda s: s['date'])

        return schedule

    def _generate_recommendations(
        self,
        protocol: Optional[MonitoringProtocol],
        dose_adjustments: List[DoseAdjustment],
        predicted_effects: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """Generate clinical recommendations for monitoring"""
        recommendations = []

        # Protocol recommendations
        if protocol:
            recommendations.append(
                f"📋 MONITORING PROTOCOL: {protocol.protocol_name} initiated. "
                f"Baseline labs required within 1 week."
            )

            recommendations.append(
                f"🔬 FREQUENCY: {protocol.frequency} monitoring for {protocol.duration}."
            )

        # Dose adjustment recommendations
        for adjustment in dose_adjustments:
            if adjustment.severity == "immediate":
                recommendations.append(
                    f"🔴 IMMEDIATE: {adjustment.drug_name} dose adjustment required. "
                    f"{adjustment.current_dose} → {adjustment.recommended_dose}. "
                    f"Rationale: {adjustment.rationale}"
                )
            elif adjustment.severity == "urgent":
                recommendations.append(
                    f"⚠️ URGENT: Consider {adjustment.drug_name} dose adjustment within 48 hours. "
                    f"{adjustment.rationale}"
                )
            else:
                recommendations.append(
                    f"💊 {adjustment.drug_name}: {adjustment.rationale}"
                )

        # Predicted effect recommendations
        if predicted_effects:
            high_risk_effects = []
            for drug, effects in predicted_effects.items():
                for effect in effects:
                    if effect.get('frequency') in ['very_common', 'common']:
                        high_risk_effects.append(f"{effect.get('loinc_name')} from {drug}")

            if high_risk_effects:
                recommendations.append(
                    f"⚠️ EXPECTED TOXICITIES: Monitor closely for {', '.join(high_risk_effects[:3])}."
                )

        # General monitoring recommendations
        recommendations.append(
            "📊 DOCUMENTATION: Record all lab values in patient chart with trend analysis."
        )

        return recommendations

    def _generate_clinical_summary(
        self,
        protocol: Optional[MonitoringProtocol],
        dose_adjustments: List[DoseAdjustment],
        requires_immediate_action: bool
    ) -> str:
        """Generate concise clinical summary"""
        summary_parts = []

        if protocol:
            summary_parts.append(
                f"Monitoring protocol '{protocol.protocol_name}' created for {protocol.regimen}. "
                f"{len(protocol.lab_tests)} lab tests scheduled at {protocol.frequency} frequency."
            )

        if dose_adjustments:
            summary_parts.append(
                f"\n⚠️ {len(dose_adjustments)} dose adjustment(s) recommended based on current labs."
            )

        if requires_immediate_action:
            summary_parts.append(
                "\n🔴 IMMEDIATE ACTION REQUIRED: Critical lab values detected requiring urgent dose modification."
            )
        elif dose_adjustments:
            summary_parts.append(
                "\n💊 Routine dose adjustments recommended - review within 1 week."
            )
        else:
            summary_parts.append(
                "\n✓ No dose adjustments required at this time. Continue monitoring per protocol."
            )

        return " ".join(summary_parts)


# Singleton instance
_monitoring_coordinator_agent = None


def get_monitoring_coordinator_agent() -> MonitoringCoordinatorAgent:
    """Get singleton MonitoringCoordinatorAgent instance"""
    global _monitoring_coordinator_agent
    if _monitoring_coordinator_agent is None:
        _monitoring_coordinator_agent = MonitoringCoordinatorAgent()
    return _monitoring_coordinator_agent
