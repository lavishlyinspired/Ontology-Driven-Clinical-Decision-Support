"""
Comorbidity Assessment Agent

Specialized agent for evaluating treatment safety based on
patient comorbidities, organ function, and drug interactions.
"""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

from ..db.models import PatientFactWithCodes
from .negotiation_protocol import AgentProposal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComorbidityProfile:
    """Patient comorbidity profile"""
    # Cardiac
    heart_disease: bool = False
    heart_failure_nyha_class: Optional[int] = None  # NYHA I-IV
    arrhythmia: bool = False
    hypertension: bool = False

    # Pulmonary (critical for lung cancer)
    copd: bool = False
    fev1_percent: Optional[float] = None
    interstitial_lung_disease: bool = False
    pulmonary_fibrosis: bool = False

    # Renal
    chronic_kidney_disease: bool = False
    egfr: Optional[float] = None  # mL/min/1.73m²
    dialysis: bool = False

    # Hepatic
    hepatic_impairment: bool = False
    cirrhosis: bool = False
    hepatitis: bool = False

    # Metabolic
    diabetes: bool = False
    diabetes_controlled: bool = True

    # Autoimmune
    autoimmune_disease: bool = False
    inflammatory_bowel_disease: bool = False
    rheumatoid_arthritis: bool = False

    # Infectious
    hiv: bool = False
    hiv_controlled: bool = True
    tuberculosis: bool = False

    # Other
    thromboembolic_history: bool = False
    bleeding_disorder: bool = False
    osteoporosis: bool = False

    # Medications (for drug interactions)
    current_medications: List[str] = None

    def __post_init__(self):
        if self.current_medications is None:
            self.current_medications = []


@dataclass
class SafetyAssessment:
    """Safety assessment for a treatment"""
    treatment: str
    overall_safety: str  # "Safe", "Caution", "Contraindicated"
    risk_score: float  # 0.0 (low) to 1.0 (high)
    contraindications: List[str]
    warnings: List[str]
    dose_adjustments: List[str]
    monitoring_requirements: List[str]
    alternative_if_unsafe: Optional[str] = None


class ComorbidityAgent:
    """
    Specialized agent for comorbidity-based safety assessment.

    Evaluates:
    - Treatment contraindications
    - Organ function requirements
    - Drug interactions
    - Dose adjustments needed
    - Monitoring requirements
    """

    def __init__(self):
        self.agent_id = "comorbidity_specialist"
        self.agent_type = "ComorbidityAgent"

        # Drug interaction database (simplified)
        self.drug_interactions = self._initialize_drug_interactions()

    def execute(
        self,
        patient: PatientFactWithCodes,
        treatment: str,
        comorbidity_profile: Optional[ComorbidityProfile] = None
    ) -> SafetyAssessment:
        """
        Assess treatment safety based on comorbidities.

        Args:
            patient: Patient data
            treatment: Proposed treatment
            comorbidity_profile: Detailed comorbidity information

        Returns:
            SafetyAssessment with contraindications and recommendations
        """
        logger.info(f"Comorbidity Agent assessing safety for {treatment}")

        # Extract or create comorbidity profile
        if comorbidity_profile is None:
            comorbidity_profile = self._extract_comorbidities(patient)

        # Initialize assessment
        contraindications = []
        warnings = []
        dose_adjustments = []
        monitoring = []

        # Assess based on treatment type
        if "chemotherapy" in treatment.lower() or "platinum" in treatment.lower():
            safety = self._assess_chemotherapy_safety(
                comorbidity_profile, contraindications, warnings,
                dose_adjustments, monitoring
            )
        elif any(drug in treatment.lower() for drug in ["osimertinib", "alectinib", "crizotinib"]):
            safety = self._assess_tki_safety(
                treatment, comorbidity_profile, contraindications,
                warnings, dose_adjustments, monitoring
            )
        elif "immunotherapy" in treatment.lower() or "pembrolizumab" in treatment.lower():
            safety = self._assess_immunotherapy_safety(
                comorbidity_profile, contraindications, warnings,
                dose_adjustments, monitoring
            )
        elif "surgery" in treatment.lower():
            safety = self._assess_surgery_safety(
                comorbidity_profile, contraindications, warnings,
                dose_adjustments, monitoring
            )
        elif "radiation" in treatment.lower():
            safety = self._assess_radiation_safety(
                comorbidity_profile, contraindications, warnings,
                dose_adjustments, monitoring
            )
        else:
            # General assessment
            safety = self._assess_general_safety(
                comorbidity_profile, contraindications, warnings,
                dose_adjustments, monitoring
            )

        # Calculate risk score
        risk_score = self._calculate_risk_score(contraindications, warnings)

        # Determine overall safety
        if contraindications:
            overall_safety = "Contraindicated"
        elif len(warnings) > 2 or risk_score > 0.6:
            overall_safety = "Caution"
        else:
            overall_safety = "Safe"

        # Suggest alternative if contraindicated
        alternative = None
        if overall_safety == "Contraindicated":
            alternative = self._suggest_alternative(treatment, comorbidity_profile)

        return SafetyAssessment(
            treatment=treatment,
            overall_safety=overall_safety,
            risk_score=risk_score,
            contraindications=contraindications,
            warnings=warnings,
            dose_adjustments=dose_adjustments,
            monitoring_requirements=monitoring,
            alternative_if_unsafe=alternative
        )

    # ========================================
    # TREATMENT-SPECIFIC ASSESSMENTS
    # ========================================

    def _assess_chemotherapy_safety(
        self,
        profile: ComorbidityProfile,
        contraindications: List[str],
        warnings: List[str],
        dose_adjustments: List[str],
        monitoring: List[str]
    ) -> str:
        """Assess safety of platinum-based chemotherapy"""

        # Renal function - CRITICAL for platinum compounds
        if profile.egfr is not None:
            if profile.egfr < 30:
                contraindications.append(
                    "Severe renal impairment (eGFR <30) - avoid cisplatin, consider carboplatin with dose reduction"
                )
                dose_adjustments.append("Carboplatin AUC adjustment for CrCl <60")
            elif profile.egfr < 60:
                warnings.append(f"Moderate renal impairment (eGFR {profile.egfr}) - cisplatin nephrotoxicity risk")
                monitoring.append("Monitor renal function before each cycle")

        # Cardiac function - anthracyclines, taxanes
        if profile.heart_failure_nyha_class and profile.heart_failure_nyha_class >= 3:
            contraindications.append(
                "NYHA Class III-IV heart failure - avoid anthracyclines"
            )
        elif profile.heart_disease:
            warnings.append("Pre-existing cardiac disease - monitor for cardiotoxicity")
            monitoring.append("Baseline and periodic echocardiogram")

        # Pulmonary function
        if profile.fev1_percent and profile.fev1_percent < 40:
            warnings.append(f"Severe pulmonary impairment (FEV1 {profile.fev1_percent}%) - increased toxicity risk")

        if profile.interstitial_lung_disease:
            contraindications.append(
                "Pre-existing ILD - high risk of bleomycin pulmonary toxicity"
            )

        # Hepatic function
        if profile.cirrhosis:
            warnings.append("Hepatic impairment - reduce taxane doses by 20-25%")
            dose_adjustments.append("Paclitaxel/docetaxel dose reduction for bilirubin >1.5x ULN")

        # Neuropathy risk
        if profile.diabetes and not profile.diabetes_controlled:
            warnings.append("Uncontrolled diabetes - increased risk of peripheral neuropathy from taxanes")

        # Thromboembolic risk
        if profile.thromboembolic_history:
            warnings.append("History of VTE - consider prophylactic anticoagulation during chemotherapy")
            monitoring.append("Monitor for signs of thromboembolism")

        # Bone marrow function
        monitoring.extend([
            "CBC with differential before each cycle",
            "Assess for febrile neutropenia risk (age >65, prior chemo)"
        ])

        return "chemotherapy"

    def _assess_tki_safety(
        self,
        treatment: str,
        profile: ComorbidityProfile,
        contraindications: List[str],
        warnings: List[str],
        dose_adjustments: List[str],
        monitoring: List[str]
    ) -> str:
        """Assess safety of tyrosine kinase inhibitors"""

        # Hepatic function - TKIs are hepatically metabolized
        if profile.hepatic_impairment or profile.cirrhosis:
            if profile.cirrhosis:
                contraindications.append(
                    "Severe hepatic impairment (Child-Pugh C) - avoid or reduce TKI dose"
                )
                dose_adjustments.append("Consider 50% dose reduction for moderate-severe hepatic impairment")
            else:
                warnings.append("Hepatic impairment - monitor LFTs closely")

            monitoring.append("LFTs every 2 weeks for first month, then monthly")

        # Interstitial lung disease - MAJOR risk with EGFR TKIs
        if profile.interstitial_lung_disease or profile.pulmonary_fibrosis:
            contraindications.append(
                "Pre-existing ILD - osimertinib/gefitinib ILD risk 2-5%, potentially fatal"
            )

        # Cardiac - QT prolongation risk
        if profile.arrhythmia:
            warnings.append("Cardiac arrhythmia - TKIs may prolong QTc interval")
            monitoring.append("Baseline and periodic ECG monitoring")

        # GI perforation risk with VEGF inhibitors
        if "bevacizumab" in treatment.lower():
            if profile.inflammatory_bowel_disease:
                contraindications.append("IBD - bevacizumab GI perforation risk")

        # Drug interactions - many TKIs metabolized by CYP3A4
        monitoring.append("Review medications for CYP3A4 interactions (avoid strong inhibitors/inducers)")

        return "tki"

    def _assess_immunotherapy_safety(
        self,
        profile: ComorbidityProfile,
        contraindications: List[str],
        warnings: List[str],
        dose_adjustments: List[str],
        monitoring: List[str]
    ) -> str:
        """Assess safety of immune checkpoint inhibitors"""

        # Autoimmune conditions - MAJOR concern
        if profile.autoimmune_disease:
            warnings.append(
                "Pre-existing autoimmune disease - increased risk of immune-related adverse events (irAEs)"
            )
            monitoring.append("Close monitoring for autoimmune flares")

        if profile.inflammatory_bowel_disease:
            warnings.append(
                "IBD history - high risk of immune-mediated colitis (10-20%)"
            )
            monitoring.append("Monitor for diarrhea/colitis symptoms")

        # Organ transplant recipients - absolute contraindication
        # (Would check in patient history)

        # Hepatitis - risk of reactivation
        if profile.hepatitis:
            warnings.append("Hepatitis history - risk of viral reactivation")
            monitoring.append("Monitor HBV/HCV viral loads if applicable")

        # Standard irAE monitoring
        monitoring.extend([
            "Baseline and periodic TSH (thyroiditis risk)",
            "LFTs (hepatitis risk)",
            "Blood glucose (diabetes risk)",
            "Monitor for pneumonitis, colitis, dermatitis, hypophysitis"
        ])

        # Corticosteroid availability
        warnings.append("Ensure access to corticosteroids for irAE management")

        return "immunotherapy"

    def _assess_surgery_safety(
        self,
        profile: ComorbidityProfile,
        contraindications: List[str],
        warnings: List[str],
        dose_adjustments: List[str],
        monitoring: List[str]
    ) -> str:
        """Assess surgical safety"""

        # Cardiac risk - Revised Cardiac Risk Index
        cardiac_risk_factors = sum([
            profile.heart_disease,
            profile.heart_failure_nyha_class is not None,
            profile.diabetes,
            profile.chronic_kidney_disease
        ])

        if cardiac_risk_factors >= 3:
            warnings.append(
                f"High cardiac risk ({cardiac_risk_factors} risk factors) - cardiology consult recommended"
            )
        elif cardiac_risk_factors >= 1:
            warnings.append(f"Moderate cardiac risk - perioperative monitoring")

        # Pulmonary function - CRITICAL for thoracic surgery
        if profile.fev1_percent:
            if profile.fev1_percent < 40:
                contraindications.append(
                    f"Severe pulmonary impairment (FEV1 {profile.fev1_percent}%) - high surgical risk"
                )
            elif profile.fev1_percent < 60:
                warnings.append(
                    f"Moderate pulmonary impairment (FEV1 {profile.fev1_percent}%) - consider limited resection"
                )

        if profile.copd:
            warnings.append("COPD - increased postoperative pulmonary complication risk")
            monitoring.append("Optimize pulmonary function preoperatively")

        # Bleeding risk
        if profile.bleeding_disorder:
            warnings.append("Bleeding disorder - hematology consult for perioperative management")

        # Anticoagulation
        if profile.thromboembolic_history:
            warnings.append("On anticoagulation - bridging protocol required")

        return "surgery"

    def _assess_radiation_safety(
        self,
        profile: ComorbidityProfile,
        contraindications: List[str],
        warnings: List[str],
        dose_adjustments: List[str],
        monitoring: List[str]
    ) -> str:
        """Assess radiation therapy safety"""

        # Pulmonary fibrosis - radiation pneumonitis risk
        if profile.interstitial_lung_disease or profile.pulmonary_fibrosis:
            warnings.append("Pre-existing ILD - high risk of radiation pneumonitis")
            monitoring.append("Close monitoring for dyspnea, cough during/after RT")

        # Connective tissue disorders - increased toxicity
        if profile.autoimmune_disease:
            warnings.append("Autoimmune disease - may increase radiation toxicity")

        # Prior radiation
        # (Would check treatment history)

        return "radiation"

    def _assess_general_safety(
        self,
        profile: ComorbidityProfile,
        contraindications: List[str],
        warnings: List[str],
        dose_adjustments: List[str],
        monitoring: List[str]
    ) -> str:
        """General safety assessment"""

        if profile.egfr and profile.egfr < 30:
            warnings.append("Severe renal impairment - review drug dosing")

        if profile.hepatic_impairment:
            warnings.append("Hepatic impairment - review drug metabolism")

        return "general"

    # ========================================
    # UTILITIES
    # ========================================

    def _extract_comorbidities(self, patient: PatientFactWithCodes) -> ComorbidityProfile:
        """Extract comorbidity data from patient record"""

        comorbidities = patient.comorbidities if hasattr(patient, 'comorbidities') else []

        return ComorbidityProfile(
            heart_disease="heart_disease" in comorbidities,
            copd="copd" in comorbidities,
            diabetes="diabetes" in comorbidities,
            chronic_kidney_disease="ckd" in comorbidities,
            hepatic_impairment="hepatic_impairment" in comorbidities,
            autoimmune_disease="autoimmune" in comorbidities,
            interstitial_lung_disease="ild" in comorbidities,
            fev1_percent=patient.fev1_percent if hasattr(patient, 'fev1_percent') else None,
            egfr=getattr(patient, 'egfr', None)
        )

    def _calculate_risk_score(
        self,
        contraindications: List[str],
        warnings: List[str]
    ) -> float:
        """Calculate overall risk score"""

        # Contraindications = 1.0, warnings = 0.3 each
        score = len(contraindications) * 1.0 + len(warnings) * 0.3

        # Cap at 1.0
        return min(score, 1.0)

    def _suggest_alternative(
        self,
        treatment: str,
        profile: ComorbidityProfile
    ) -> Optional[str]:
        """Suggest alternative treatment if contraindicated"""

        alternatives = {
            "cisplatin": "Carboplatin (better renal safety profile)",
            "chemotherapy": "Single-agent therapy or best supportive care",
            "immunotherapy": "Chemotherapy (if no autoimmune contraindication)",
            "surgery": "Stereotactic radiotherapy (SABR/SBRT)",
            "osimertinib": "Alternative EGFR TKI with hepatology consult"
        }

        for key, alt in alternatives.items():
            if key.lower() in treatment.lower():
                return alt

        return None

    def _initialize_drug_interactions(self) -> Dict[str, List[str]]:
        """Initialize drug interaction database"""

        return {
            "osimertinib": [
                "Strong CYP3A4 inducers (rifampin, phenytoin) - decrease efficacy",
                "QT-prolonging drugs - additive cardiac risk"
            ],
            "alectinib": [
                "CYP3A substrates - monitor for interactions"
            ],
            "pembrolizumab": [
                "Corticosteroids - may decrease efficacy (use only for irAE management)"
            ]
        }

    def generate_safety_summary(
        self,
        assessment: SafetyAssessment
    ) -> str:
        """Generate human-readable safety summary"""

        summary_parts = [
            f"Safety Assessment for {assessment.treatment}:",
            f"Overall Safety: {assessment.overall_safety}",
            f"Risk Score: {assessment.risk_score:.2f}/1.00"
        ]

        if assessment.contraindications:
            summary_parts.append("\n⚠️ CONTRAINDICATIONS:")
            for ci in assessment.contraindications:
                summary_parts.append(f"  - {ci}")

        if assessment.warnings:
            summary_parts.append("\n⚡ WARNINGS:")
            for w in assessment.warnings:
                summary_parts.append(f"  - {w}")

        if assessment.dose_adjustments:
            summary_parts.append("\n💊 DOSE ADJUSTMENTS:")
            for da in assessment.dose_adjustments:
                summary_parts.append(f"  - {da}")

        if assessment.monitoring_requirements:
            summary_parts.append("\n🔍 MONITORING:")
            for m in assessment.monitoring_requirements:
                summary_parts.append(f"  - {m}")

        if assessment.alternative_if_unsafe:
            summary_parts.append(f"\n✅ ALTERNATIVE: {assessment.alternative_if_unsafe}")

        return "\n".join(summary_parts)
