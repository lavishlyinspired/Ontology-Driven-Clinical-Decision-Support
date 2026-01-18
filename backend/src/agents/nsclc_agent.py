"""
NSCLC-Specific Treatment Agent

Specialized agent for Non-Small Cell Lung Cancer patients.
Implements NCCN/ASCO/ESMO guidelines specific to NSCLC subtypes.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from .classification_agent import ClassificationAgent
from ..ontology.guideline_rules import GuidelineRule, EvidenceLevel, TreatmentIntent


class NSCLCSubtype(Enum):
    """NSCLC histological subtypes"""
    ADENOCARCINOMA = "Adenocarcinoma"
    SQUAMOUS_CELL = "Squamous Cell Carcinoma"
    LARGE_CELL = "Large Cell Carcinoma"
    NSCLC_NOS = "NSCLC Not Otherwise Specified"


@dataclass
class NSCLCProposal:
    """NSCLC-specific treatment proposal"""
    agent_id: str = "nsclc_agent"
    treatment: str = ""
    confidence: float = 0.0
    evidence_level: str = ""
    treatment_intent: str = ""
    rationale: str = ""
    subtype_specific: bool = False
    biomarker_driven: bool = False
    risk_score: float = 0.0


class NSCLCAgent:
    """
    Specialized Agent for NSCLC Treatment Recommendations

    Focuses on:
    - Adenocarcinoma-specific pathways
    - Squamous cell carcinoma considerations
    - Stage-specific NSCLC protocols
    - Coordination with BiomarkerAgent for driver mutations
    """

    def __init__(self):
        self.agent_id = "nsclc_agent"
        self.classification_agent = ClassificationAgent()

    def execute(self, patient, biomarker_profile: Dict[str, Any] = None) -> NSCLCProposal:
        """
        Generate NSCLC-specific treatment proposal

        Args:
            patient: PatientFactWithCodes
            biomarker_profile: Optional biomarker data

        Returns:
            NSCLCProposal with treatment recommendation
        """

        # Determine NSCLC subtype
        subtype = self._determine_subtype(patient.histology_type)

        # Check for biomarker-driven pathways first
        if biomarker_profile and self._has_actionable_biomarkers(biomarker_profile):
            # Defer to BiomarkerAgent for driver mutation cases
            return NSCLCProposal(
                treatment="Biomarker-directed therapy (see BiomarkerAgent)",
                confidence=0.95,
                evidence_level="Grade A",
                treatment_intent="curative",
                rationale="Actionable driver mutation detected",
                subtype_specific=False,
                biomarker_driven=True,
                risk_score=0.2
            )

        # Stage-based recommendations for NSCLC without driver mutations
        stage = patient.tnm_stage
        ps = patient.performance_status

        if stage in ["IA", "IB"]:
            return self._early_stage_nsclc(patient, subtype)
        elif stage in ["IIA", "IIB"]:
            return self._locally_advanced_resectable(patient, subtype)
        elif stage in ["IIIA", "IIIB", "IIIC"]:
            return self._locally_advanced_unresectable(patient, subtype)
        elif stage in ["IV", "IVA", "IVB"]:
            return self._metastatic_nsclc(patient, subtype, biomarker_profile)
        else:
            return NSCLCProposal(
                treatment="Classification required",
                confidence=0.5,
                evidence_level="Grade C",
                treatment_intent="unknown",
                rationale="Unable to classify stage",
                risk_score=0.5
            )

    def _determine_subtype(self, histology: str) -> NSCLCSubtype:
        """Determine NSCLC subtype from histology"""
        histology_lower = histology.lower()

        if "adenocarcinoma" in histology_lower:
            return NSCLCSubtype.ADENOCARCINOMA
        elif "squamous" in histology_lower:
            return NSCLCSubtype.SQUAMOUS_CELL
        elif "large cell" in histology_lower:
            return NSCLCSubtype.LARGE_CELL
        else:
            return NSCLCSubtype.NSCLC_NOS

    def _has_actionable_biomarkers(self, biomarker_profile: Dict[str, Any]) -> bool:
        """Check for actionable driver mutations"""
        actionable_markers = [
            "EGFR", "ALK", "ROS1", "BRAF", "MET", "RET", "NTRK", "KRAS"
        ]
        return any(marker in biomarker_profile for marker in actionable_markers)

    def _early_stage_nsclc(self, patient, subtype: NSCLCSubtype) -> NSCLCProposal:
        """Stage I NSCLC recommendations"""
        if patient.performance_status <= 1:
            return NSCLCProposal(
                treatment="Surgical resection (lobectomy preferred)",
                confidence=0.90,
                evidence_level="Grade A",
                treatment_intent="curative",
                rationale="Early-stage NSCLC with good PS - surgery is standard of care",
                subtype_specific=False,
                risk_score=0.2
            )
        else:
            return NSCLCProposal(
                treatment="Stereotactic body radiotherapy (SBRT)",
                confidence=0.85,
                evidence_level="Grade A",
                treatment_intent="curative",
                rationale="Early-stage NSCLC with poor PS - SBRT alternative",
                subtype_specific=False,
                risk_score=0.3
            )

    def _locally_advanced_resectable(self, patient, subtype: NSCLCSubtype) -> NSCLCProposal:
        """Stage II NSCLC recommendations"""
        return NSCLCProposal(
            treatment="Surgery followed by adjuvant chemotherapy",
            confidence=0.88,
            evidence_level="Grade A",
            treatment_intent="curative",
            rationale="Resectable locally advanced NSCLC - multimodality therapy",
            subtype_specific=False,
            risk_score=0.3
        )

    def _locally_advanced_unresectable(self, patient, subtype: NSCLCSubtype) -> NSCLCProposal:
        """Stage III NSCLC recommendations"""
        if patient.performance_status <= 1:
            return NSCLCProposal(
                treatment="Concurrent chemoradiotherapy followed by durvalumab",
                confidence=0.90,
                evidence_level="Grade A",
                treatment_intent="curative",
                rationale="Unresectable Stage III NSCLC - PACIFIC regimen",
                subtype_specific=False,
                risk_score=0.4
            )
        else:
            return NSCLCProposal(
                treatment="Sequential chemoradiotherapy",
                confidence=0.75,
                evidence_level="Grade B",
                treatment_intent="curative",
                rationale="Stage III NSCLC with PS 2 - sequential approach safer",
                subtype_specific=False,
                risk_score=0.5
            )

    def _metastatic_nsclc(self, patient, subtype: NSCLCSubtype, biomarker_profile) -> NSCLCProposal:
        """Stage IV NSCLC recommendations"""

        # PD-L1 based if no driver mutation
        if biomarker_profile and "PD-L1" in biomarker_profile:
            pdl1_value = self._extract_pdl1_value(biomarker_profile["PD-L1"])

            if pdl1_value >= 50:
                return NSCLCProposal(
                    treatment="Pembrolizumab monotherapy",
                    confidence=0.92,
                    evidence_level="Grade A",
                    treatment_intent="palliative",
                    rationale="Metastatic NSCLC with PD-L1 ≥50% - pembrolizumab first-line",
                    subtype_specific=False,
                    biomarker_driven=True,
                    risk_score=0.3
                )
            elif pdl1_value >= 1:
                return NSCLCProposal(
                    treatment="Pembrolizumab + platinum-based chemotherapy",
                    confidence=0.88,
                    evidence_level="Grade A",
                    treatment_intent="palliative",
                    rationale="Metastatic NSCLC with PD-L1 1-49% - combination therapy",
                    subtype_specific=False,
                    biomarker_driven=True,
                    risk_score=0.4
                )

        # Default platinum doublet
        if subtype == NSCLCSubtype.SQUAMOUS_CELL:
            return NSCLCProposal(
                treatment="Carboplatin + paclitaxel",
                confidence=0.80,
                evidence_level="Grade A",
                treatment_intent="palliative",
                rationale="Metastatic squamous NSCLC - platinum doublet",
                subtype_specific=True,
                risk_score=0.5
            )
        else:  # Adenocarcinoma or other
            return NSCLCProposal(
                treatment="Carboplatin + pemetrexed",
                confidence=0.82,
                evidence_level="Grade A",
                treatment_intent="palliative",
                rationale="Metastatic non-squamous NSCLC - platinum/pemetrexed",
                subtype_specific=True,
                risk_score=0.5
            )

    def _extract_pdl1_value(self, pdl1_data) -> float:
        """Extract PD-L1 percentage value"""
        if isinstance(pdl1_data, (int, float)):
            return float(pdl1_data)
        elif isinstance(pdl1_data, str):
            # Parse "45%" -> 45.0
            return float(pdl1_data.rstrip('%'))
        return 0.0
