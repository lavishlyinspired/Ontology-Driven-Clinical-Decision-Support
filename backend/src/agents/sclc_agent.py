"""
SCLC-Specific Treatment Agent

Specialized agent for Small Cell Lung Cancer patients.
Implements NCCN guidelines specific to SCLC staging and management.
"""

from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum

# Centralized logging
from ..logging_config import get_logger, log_agent_action

logger = get_logger(__name__)


class SCLCStage(Enum):
    """SCLC staging system (Limited vs Extensive)"""
    LIMITED = "limited"  # Confined to one hemithorax
    EXTENSIVE = "extensive"  # Beyond one hemithorax


@dataclass
class SCLCProposal:
    """SCLC-specific treatment proposal"""
    agent_id: str = "sclc_agent"
    treatment: str = ""
    confidence: float = 0.0
    evidence_level: str = ""
    treatment_intent: str = ""
    rationale: str = ""
    sclc_stage: str = ""
    prophylactic_cranial_irradiation: bool = False
    risk_score: float = 0.0


class SCLCAgent:
    """
    Specialized Agent for SCLC Treatment Recommendations

    SCLC represents ~15% of lung cancer cases but has distinct biology and treatment:
    - Rapid doubling time (aggressive growth)
    - High metastatic potential
    - Chemosensitive but quickly develops resistance
    - Limited vs Extensive stage classification

    Treatment paradigm:
    - Limited: Concurrent chemoradiotherapy + PCI
    - Extensive: Platinum-based chemotherapy + immunotherapy
    """

    def __init__(self):
        self.agent_id = "sclc_agent"

    def execute(self, patient) -> SCLCProposal:
        """
        Generate SCLC-specific treatment proposal

        Args:
            patient: PatientFactWithCodes

        Returns:
            SCLCProposal with treatment recommendation
        """

        # Determine SCLC staging (Limited vs Extensive)
        sclc_stage = self._determine_sclc_stage(patient.tnm_stage)

        if sclc_stage == SCLCStage.LIMITED:
            return self._limited_stage_sclc(patient)
        else:  # EXTENSIVE
            return self._extensive_stage_sclc(patient)

    def _determine_sclc_stage(self, tnm_stage: str) -> SCLCStage:
        """
        Convert TNM stage to SCLC Limited vs Extensive

        Limited: Confined to one hemithorax (roughly T1-T4, N0-N3, M0)
        Extensive: Beyond one hemithorax (M1 or contralateral involvement)
        """
        if tnm_stage in ["IV", "IVA", "IVB"]:
            return SCLCStage.EXTENSIVE

        # IIIB with contralateral nodes or pleural effusion
        if tnm_stage == "IIIB":
            # In practice, would need more clinical details
            # Default to LIMITED for T4N2-3 or T1-4N3 without distant mets
            return SCLCStage.LIMITED

        # Stages I-IIIA generally LIMITED
        return SCLCStage.LIMITED

    def _limited_stage_sclc(self, patient) -> SCLCProposal:
        """
        Limited-Stage SCLC Treatment

        Standard: Concurrent platinum/etoposide + radiotherapy
        Goal: Curative intent
        5-year survival: 20-25% with optimal treatment
        """

        ps = patient.performance_status

        if ps <= 1:
            # Good PS - concurrent chemoradiotherapy
            return SCLCProposal(
                treatment="Concurrent cisplatin/etoposide + thoracic radiotherapy (45 Gy)",
                confidence=0.95,
                evidence_level="Grade A",
                treatment_intent="curative",
                rationale="Limited-stage SCLC with good PS - concurrent chemoRT standard",
                sclc_stage="limited",
                prophylactic_cranial_irradiation=True,  # Recommend PCI if complete response
                risk_score=0.3
            )

        elif ps == 2:
            # PS 2 - sequential approach safer
            return SCLCProposal(
                treatment="Sequential cisplatin/etoposide then thoracic radiotherapy",
                confidence=0.80,
                evidence_level="Grade B",
                treatment_intent="curative",
                rationale="Limited-stage SCLC with PS 2 - sequential approach to reduce toxicity",
                sclc_stage="limited",
                prophylactic_cranial_irradiation=True,
                risk_score=0.5
            )

        else:  # PS 3-4
            # Poor PS - chemotherapy alone
            return SCLCProposal(
                treatment="Carboplatin/etoposide (chemotherapy only)",
                confidence=0.70,
                evidence_level="Grade C",
                treatment_intent="palliative",
                rationale="Limited-stage SCLC with poor PS - chemotherapy alone",
                sclc_stage="limited",
                prophylactic_cranial_irradiation=False,
                risk_score=0.7
            )

    def _extensive_stage_sclc(self, patient) -> SCLCProposal:
        """
        Extensive-Stage SCLC Treatment

        Standard: Platinum/etoposide + atezolizumab (IMpower133 regimen)
        Alternative: Durvalumab + platinum/etoposide (CASPIAN)
        Goal: Prolonged survival, symptom control
        Median OS: 12-13 months with immunotherapy
        """

        ps = patient.performance_status

        if ps <= 2:
            # Good/moderate PS - immunotherapy + chemotherapy
            return SCLCProposal(
                treatment="Atezolizumab + carboplatin/etoposide (IMpower133 regimen)",
                confidence=0.92,
                evidence_level="Grade A",
                treatment_intent="palliative",
                rationale="Extensive-stage SCLC - immunotherapy + chemotherapy improves OS (13 vs 10.3 mo)",
                sclc_stage="extensive",
                prophylactic_cranial_irradiation=False,  # Generally not done in extensive stage
                risk_score=0.4
            )

        else:  # PS 3-4
            # Poor PS - best supportive care or single-agent chemotherapy
            return SCLCProposal(
                treatment="Best supportive care (consider single-agent carboplatin or etoposide)",
                confidence=0.65,
                evidence_level="Grade C",
                treatment_intent="palliative",
                rationale="Extensive-stage SCLC with poor PS - focus on quality of life",
                sclc_stage="extensive",
                prophylactic_cranial_irradiation=False,
                risk_score=0.8
            )

    def generate_sclc_specific_notes(self, proposal: SCLCProposal) -> Dict[str, Any]:
        """
        Generate SCLC-specific clinical notes

        Returns:
            Dictionary with additional SCLC considerations
        """

        notes = {
            "aggressive_biology": "SCLC doubling time ~30 days - prompt treatment initiation critical",
            "chemosensitivity": "Initial response rate 60-80%, but resistance develops quickly",
            "brain_metastases": "30% have brain mets at diagnosis, 50-60% develop during course",
            "prophylactic_cranial_irradiation": None,
            "surveillance": "CT chest q3-4 months for 2 years, then q6 months"
        }

        if proposal.prophylactic_cranial_irradiation:
            notes["prophylactic_cranial_irradiation"] = (
                "Recommend PCI if complete/good partial response achieved. "
                "PCI reduces brain metastases from 59% to 33% and improves 3-year survival "
                "(15% vs 21%, p<0.001). Discuss neurocognitive risks with patient."
            )
        else:
            notes["prophylactic_cranial_irradiation"] = (
                "PCI not routinely recommended in extensive stage. "
                "Consider brain MRI surveillance instead."
            )

        # Stage-specific notes
        if proposal.sclc_stage == "limited":
            notes["expected_outcome"] = {
                "complete_response_rate": "60-70%",
                "median_survival": "18-24 months",
                "5_year_survival": "20-25%",
                "note": "Cure possible in limited stage with optimal treatment"
            }
        else:  # extensive
            notes["expected_outcome"] = {
                "response_rate": "60-70%",
                "median_survival": "10-13 months (with immunotherapy)",
                "1_year_survival": "50%",
                "2_year_survival": "20-30%",
                "note": "Extensive stage is incurable; focus on prolonging survival and QOL"
            }

        return notes
