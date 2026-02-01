"""
Biomarker Specialist Agent

Specialized agent for biomarker-driven treatment selection based on
molecular profiling (EGFR, ALK, ROS1, BRAF, PD-L1, etc.)

Implements precision medicine guidelines from NCCN, ASCO, ESMO 2025.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Centralized logging
from ..logging_config import get_logger, log_agent_action

logger = get_logger(__name__)

from ..db.models import PatientFactWithCodes, ClassificationResult, TreatmentRecommendation, EvidenceLevel, TreatmentIntent
from .negotiation_protocol import AgentProposal


@dataclass
class BiomarkerProfile:
    """Patient biomarker profile"""
    egfr_mutation: Optional[str] = None
    egfr_mutation_type: Optional[str] = None  # Ex19del, L858R, T790M, etc.
    alk_rearrangement: Optional[str] = None
    ros1_rearrangement: Optional[str] = None
    braf_mutation: Optional[str] = None
    met_exon14_skipping: Optional[str] = None
    ret_rearrangement: Optional[str] = None
    kras_mutation: Optional[str] = None
    pdl1_tps: Optional[float] = None  # PD-L1 tumor proportion score (%)
    tmb_score: Optional[float] = None  # Tumor mutational burden
    her2_mutation: Optional[str] = None
    ntrk_fusion: Optional[str] = None


class BiomarkerAgent:
    """
    Specialized agent for biomarker-driven treatment recommendations.

    Key features:
    - Prioritizes targeted therapies for actionable mutations
    - Implements NCCN/ASCO/ESMO 2025 guidelines for precision medicine
    - Handles biomarker testing recommendations
    - Provides mutation-specific dosing and resistance considerations
    """

    def __init__(self):
        self.agent_id = "biomarker_specialist"
        self.agent_type = "BiomarkerAgent"
        self.guidelines_version = "2025"

    def _get_patient_attr(self, patient, attr: str, default=None):
        """Helper to safely get patient attributes from dict or object"""
        if isinstance(patient, dict):
            return patient.get(attr, default)
        return getattr(patient, attr, default)

    def _is_positive(self, value) -> bool:
        """Check if a biomarker value indicates positive status (handles bool, str, etc.)"""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ['positive', 'true', 'yes', '+', '1']
        return bool(value)

    def execute(
        self,
        patient: PatientFactWithCodes,
        biomarker_profile: Optional[BiomarkerProfile] = None
    ) -> AgentProposal:
        """
        Generate biomarker-driven treatment recommendation.

        Args:
            patient: Patient with clinical data and codes
            biomarker_profile: Molecular biomarker results (BiomarkerProfile or dict)

        Returns:
            AgentProposal with biomarker-guided treatment
        """
        # Handle patient as dict or object
        patient_id = patient.patient_id if hasattr(patient, 'patient_id') else patient.get('patient_id', 'Unknown')
        logger.info(f"Biomarker Agent analyzing patient {patient_id}")

        # Convert dict biomarker_profile to BiomarkerProfile object
        if biomarker_profile is not None and isinstance(biomarker_profile, dict):
            # Handle boolean values - convert True/False to "Positive"/"Negative"
            def to_status(val):
                if val is None:
                    return None
                if isinstance(val, bool):
                    return "Positive" if val else "Negative"
                return val

            biomarker_profile = BiomarkerProfile(
                egfr_mutation=to_status(biomarker_profile.get('egfr_mutation')),
                egfr_mutation_type=biomarker_profile.get('egfr_mutation_type'),
                alk_rearrangement=to_status(biomarker_profile.get('alk_rearrangement')),
                ros1_rearrangement=to_status(biomarker_profile.get('ros1_rearrangement')),
                braf_mutation=to_status(biomarker_profile.get('braf_mutation')),
                met_exon14_skipping=to_status(biomarker_profile.get('met_exon14_skipping') or biomarker_profile.get('met_exon14')),
                ret_rearrangement=to_status(biomarker_profile.get('ret_rearrangement')),
                kras_mutation=to_status(biomarker_profile.get('kras_mutation')),
                pdl1_tps=biomarker_profile.get('pdl1_tps'),
                tmb_score=biomarker_profile.get('tmb_score'),
                her2_mutation=to_status(biomarker_profile.get('her2_mutation')),
                ntrk_fusion=to_status(biomarker_profile.get('ntrk_fusion'))
            )

        # Extract biomarker data if not provided
        if biomarker_profile is None:
            biomarker_profile = self._extract_biomarkers(patient)

        # Check for actionable mutations (in order of priority)
        proposal = None

        # 1. EGFR mutations (highest priority for NSCLC)
        if self._is_positive(biomarker_profile.egfr_mutation):
            proposal = self._egfr_pathway(patient, biomarker_profile)

        # 2. ALK rearrangements
        elif self._is_positive(biomarker_profile.alk_rearrangement):
            proposal = self._alk_pathway(patient, biomarker_profile)

        # 3. ROS1 rearrangements
        elif self._is_positive(biomarker_profile.ros1_rearrangement):
            proposal = self._ros1_pathway(patient, biomarker_profile)

        # 4. BRAF V600E mutations
        elif biomarker_profile.braf_mutation == "V600E" or self._is_positive(biomarker_profile.braf_mutation):
            proposal = self._braf_pathway(patient, biomarker_profile)

        # 5. MET exon 14 skipping
        elif self._is_positive(biomarker_profile.met_exon14_skipping):
            proposal = self._met_pathway(patient, biomarker_profile)

        # 6. RET rearrangements
        elif self._is_positive(biomarker_profile.ret_rearrangement):
            proposal = self._ret_pathway(patient, biomarker_profile)

        # 7. NTRK fusions (rare but highly actionable)
        elif self._is_positive(biomarker_profile.ntrk_fusion):
            proposal = self._ntrk_pathway(patient, biomarker_profile)

        # 8. High PD-L1 expression (≥50%)
        elif biomarker_profile.pdl1_tps and biomarker_profile.pdl1_tps >= 50:
            proposal = self._high_pdl1_pathway(patient, biomarker_profile)

        # 9. Moderate PD-L1 expression (1-49%)
        elif biomarker_profile.pdl1_tps and 1 <= biomarker_profile.pdl1_tps < 50:
            proposal = self._moderate_pdl1_pathway(patient, biomarker_profile)

        # 10. No actionable biomarkers
        else:
            proposal = self._biomarker_negative_pathway(patient, biomarker_profile)

        logger.info(f"Biomarker Agent recommends: {proposal.treatment}")
        return proposal

    # ========================================
    # TARGETED THERAPY PATHWAYS
    # ========================================

    def _egfr_pathway(
        self,
        patient: PatientFactWithCodes,
        biomarkers: BiomarkerProfile
    ) -> AgentProposal:
        """EGFR mutation-positive pathway"""

        # Determine specific EGFR TKI based on mutation type
        if biomarkers.egfr_mutation_type == "T790M":
            # Resistance mutation - third-generation TKI
            treatment = "Osimertinib (3rd-gen EGFR TKI)"
            rationale = (
                "EGFR T790M resistance mutation detected. "
                "Osimertinib is FDA-approved for T790M-positive NSCLC with superior CNS penetration."
            )
        elif biomarkers.egfr_mutation_type in ["Ex19del", "L858R"]:
            # Common sensitizing mutations - treatment approach depends on stage
            tnm_stage = self._get_patient_attr(patient, 'tnm_stage', 'IV')
            
            if tnm_stage == "IIIA":
                treatment = "Osimertinib (consider multimodal approach)"
                rationale = (
                    f"EGFR {biomarkers.egfr_mutation_type} sensitizing mutation in stage IIIA disease. "
                    "Consider: 1) Neoadjuvant osimertinib if resectable, 2) Osimertinib first-line if unresectable, "
                    "or 3) Concurrent chemoradiation followed by adjuvant osimertinib. "
                    "Multidisciplinary team evaluation recommended for optimal sequencing."
                )
            else:
                treatment = "Osimertinib (1st-line)"
                rationale = (
                    f"EGFR {biomarkers.egfr_mutation_type} sensitizing mutation. "
                    "Osimertinib first-line per FLAURA trial (median PFS 18.9 months vs 10.2 for 1st-gen TKIs)."
                )
        else:
            # Other EGFR mutations
            treatment = "EGFR TKI (mutation-specific selection)"
            rationale = (
                f"EGFR mutation detected ({biomarkers.egfr_mutation_type or 'unspecified'}). "
                "Targeted therapy recommended with mutation-specific TKI selection."
            )

        # Risk assessment
        risk_score = 0.2  # Low risk - highly effective targeted therapy
        contraindications = []

        # Check for ILD risk factors
        comorbidities = self._get_patient_attr(patient, 'comorbidities', [])
        if "interstitial_lung_disease" in comorbidities:
            contraindications.append("Pre-existing ILD - EGFR TKI use with caution")
            risk_score = 0.5
        
        tnm_stage = self._get_patient_attr(patient, 'tnm_stage', 'IV')

        return AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            treatment=treatment,
            confidence=0.95,  # High confidence for actionable mutation
            evidence_level="Grade A",
            treatment_intent="Curative" if tnm_stage in ["I", "II", "III", "IIIA", "IIIB"] else "Palliative",
            rationale=rationale,
            guideline_reference="NCCN NSCLC 2025.1, ASCO EGFR Guidelines",
            contraindications=contraindications,
            risk_score=risk_score,
            supporting_data={
                "biomarker": "EGFR",
                "mutation_type": biomarkers.egfr_mutation_type,
                "median_pfs": "18.9 months (FLAURA trial)" if tnm_stage != "IIIA" else "Varies by approach",
                "response_rate": "80%" if tnm_stage != "IIIA" else "70-95% (depends on resectability)",
                "cns_penetration": "Excellent (crosses blood-brain barrier)",
                "stage_specific_note": "Consider multimodal approach with MDT evaluation" if tnm_stage == "IIIA" else ""
            },
            expected_benefit="High response rate (70-80%), prolonged PFS (18+ months)" if tnm_stage != "IIIA" else "High response rate, consider resection + systemic therapy sequencing"
        )

    def _alk_pathway(
        self,
        patient: PatientFactWithCodes,
        biomarkers: BiomarkerProfile
    ) -> AgentProposal:
        """ALK rearrangement-positive pathway"""

        treatment = "Alectinib (first-line ALK inhibitor)"
        rationale = (
            "ALK rearrangement detected. Alectinib first-line per ALEX trial "
            "(median PFS 34.8 months vs 10.9 for crizotinib). Excellent CNS activity."
        )

        contraindications = []
        risk_score = 0.2

        # Check hepatic function
        comorbidities = self._get_patient_attr(patient, 'comorbidities', [])
        if "hepatic_impairment" in comorbidities:
            contraindications.append("Hepatic impairment - monitor liver function closely")
            risk_score = 0.4
        
        tnm_stage = self._get_patient_attr(patient, 'tnm_stage', 'IV')

        return AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            treatment=treatment,
            confidence=0.95,
            evidence_level="Grade A",
            treatment_intent="Curative" if tnm_stage in ["I", "II", "III"] else "Palliative",
            rationale=rationale,
            guideline_reference="NCCN NSCLC 2025, ESMO ALK+ Guidelines",
            contraindications=contraindications,
            risk_score=risk_score,
            supporting_data={
                "biomarker": "ALK",
                "median_pfs": "34.8 months (ALEX)",
                "response_rate": "83%",
                "cns_response_rate": "81%",
                "alternative_options": ["Brigatinib", "Lorlatinib (2nd-gen)"]
            },
            expected_benefit="Very high response rate (83%), extended PFS (34+ months)"
        )

    def _ros1_pathway(
        self,
        patient: PatientFactWithCodes,
        biomarkers: BiomarkerProfile
    ) -> AgentProposal:
        """ROS1 rearrangement-positive pathway"""

        treatment = "Entrectinib (ROS1 inhibitor)"
        rationale = (
            "ROS1 rearrangement detected. Entrectinib recommended with CNS activity "
            "(ORR 77%, intracranial ORR 55%)."
        )
        
        tnm_stage = self._get_patient_attr(patient, 'tnm_stage', 'IV')

        return AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            treatment=treatment,
            confidence=0.93,
            evidence_level="Grade A",
            treatment_intent="Curative" if tnm_stage in ["I", "II", "III"] else "Palliative",
            rationale=rationale,
            guideline_reference="NCCN NSCLC 2025, FDA ROS1 Guidelines",
            contraindications=[],
            risk_score=0.25,
            supporting_data={
                "biomarker": "ROS1",
                "response_rate": "77%",
                "intracranial_response": "55%",
                "median_dor": "24.6 months"
            },
            expected_benefit="High response rate (77%), durable responses"
        )

    def _braf_pathway(
        self,
        patient: PatientFactWithCodes,
        biomarkers: BiomarkerProfile
    ) -> AgentProposal:
        """BRAF V600E mutation pathway"""

        treatment = "Dabrafenib + Trametinib (BRAF/MEK inhibition)"
        rationale = (
            "BRAF V600E mutation detected. Combination dabrafenib/trametinib "
            "per FDA approval (ORR 64%, median PFS 10.9 months)."
        )

        return AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            treatment=treatment,
            confidence=0.90,
            evidence_level="Grade A",
            treatment_intent="Palliative",
            rationale=rationale,
            guideline_reference="NCCN NSCLC 2025, FDA BRAF Guidelines",
            contraindications=[],
            risk_score=0.3,
            supporting_data={
                "biomarker": "BRAF V600E",
                "response_rate": "64%",
                "median_pfs": "10.9 months",
                "combination_required": True
            },
            expected_benefit="Good response rate (64%), manageable toxicity"
        )

    def _met_pathway(
        self,
        patient: PatientFactWithCodes,
        biomarkers: BiomarkerProfile
    ) -> AgentProposal:
        """MET exon 14 skipping pathway"""

        treatment = "Capmatinib or Tepotinib (MET inhibitor)"
        rationale = (
            "MET exon 14 skipping mutation detected. MET inhibitors recommended "
            "(ORR 40-50%, responses observed across all PD-L1 levels)."
        )

        return AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            treatment=treatment,
            confidence=0.88,
            evidence_level="Grade A",
            treatment_intent="Palliative",
            rationale=rationale,
            guideline_reference="NCCN NSCLC 2025, FDA METex14 Guidelines",
            contraindications=[],
            risk_score=0.3,
            supporting_data={
                "biomarker": "MET exon 14",
                "response_rate": "40-50%",
                "median_dor": "9.7 months"
            }
        )

    def _ret_pathway(
        self,
        patient: PatientFactWithCodes,
        biomarkers: BiomarkerProfile
    ) -> AgentProposal:
        """RET rearrangement pathway"""

        treatment = "Selpercatinib (RET inhibitor)"
        rationale = (
            "RET rearrangement detected. Selpercatinib recommended "
            "(ORR 64%, median PFS 16.5 months in treatment-naive patients)."
        )

        return AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            treatment=treatment,
            confidence=0.90,
            evidence_level="Grade A",
            treatment_intent="Palliative",
            rationale=rationale,
            guideline_reference="NCCN NSCLC 2025, FDA RET Guidelines",
            contraindications=[],
            risk_score=0.25,
            supporting_data={
                "biomarker": "RET",
                "response_rate": "64%",
                "median_pfs": "16.5 months"
            }
        )

    def _ntrk_pathway(
        self,
        patient: PatientFactWithCodes,
        biomarkers: BiomarkerProfile
    ) -> AgentProposal:
        """NTRK fusion pathway"""

        treatment = "Larotrectinib or Entrectinib (TRK inhibitor)"
        rationale = (
            "NTRK fusion detected (rare but highly actionable). "
            "TRK inhibitors show remarkable response rates (75-80%) across tumor types."
        )

        return AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            treatment=treatment,
            confidence=0.92,
            evidence_level="Grade A",
            treatment_intent="Palliative",
            rationale=rationale,
            guideline_reference="NCCN NSCLC 2025, FDA NTRK Guidelines",
            contraindications=[],
            risk_score=0.2,
            supporting_data={
                "biomarker": "NTRK",
                "response_rate": "75-80%",
                "tumor_agnostic": True
            },
            expected_benefit="Exceptionally high response rate, tumor-agnostic approval"
        )

    # ========================================
    # IMMUNOTHERAPY PATHWAYS
    # ========================================

    def _high_pdl1_pathway(
        self,
        patient: PatientFactWithCodes,
        biomarkers: BiomarkerProfile
    ) -> AgentProposal:
        """High PD-L1 expression (≥50%) pathway"""

        treatment = "Pembrolizumab monotherapy (first-line)"
        rationale = (
            f"PD-L1 TPS ≥50% ({biomarkers.pdl1_tps}%). "
            "Pembrolizumab monotherapy per KEYNOTE-024 "
            "(median OS 26.3 vs 13.4 months with chemotherapy)."
        )

        contraindications = []
        risk_score = 0.3

        # Check for autoimmune conditions
        comorbidities = self._get_patient_attr(patient, 'comorbidities', [])
        if any(c in comorbidities for c in ["autoimmune_disease", "inflammatory_bowel_disease"]):
            contraindications.append("Autoimmune condition - increased risk of immune-related AEs")
            risk_score = 0.6

        return AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            treatment=treatment,
            confidence=0.90,
            evidence_level="Grade A",
            treatment_intent="Palliative",
            rationale=rationale,
            guideline_reference="NCCN NSCLC 2025, KEYNOTE-024",
            contraindications=contraindications,
            risk_score=risk_score,
            supporting_data={
                "biomarker": "PD-L1",
                "pdl1_tps": biomarkers.pdl1_tps,
                "median_os": "26.3 months",
                "response_rate": "45%"
            },
            expected_benefit="Improved OS vs chemotherapy, better tolerability"
        )

    def _moderate_pdl1_pathway(
        self,
        patient: PatientFactWithCodes,
        biomarkers: BiomarkerProfile
    ) -> AgentProposal:
        """Moderate PD-L1 expression (1-49%) pathway"""

        # Check performance status for chemotherapy eligibility
        ps = self._get_patient_attr(patient, 'performance_status', 2)
        if ps <= 1:
            treatment = "Pembrolizumab + Platinum doublet chemotherapy"
            rationale = (
                f"PD-L1 TPS 1-49% ({biomarkers.pdl1_tps}%), PS {ps}. "
                "Chemo-immunotherapy per KEYNOTE-189 (median OS 22.0 vs 10.7 months)."
            )
            confidence = 0.88
        else:
            treatment = "Carboplatin-based chemotherapy (consider immunotherapy if PS improves)"
            rationale = (
                f"PD-L1 TPS 1-49% ({biomarkers.pdl1_tps}%), PS {ps}. "
                "Chemotherapy preferred due to performance status. Reassess for immunotherapy if PS improves."
            )
            confidence = 0.75

        return AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            treatment=treatment,
            confidence=confidence,
            evidence_level="Grade A",
            treatment_intent="Palliative",
            rationale=rationale,
            guideline_reference="NCCN NSCLC 2025, KEYNOTE-189",
            contraindications=[],
            risk_score=0.4,
            supporting_data={
                "biomarker": "PD-L1",
                "pdl1_tps": biomarkers.pdl1_tps,
                "combination_benefit": "OS 22.0 vs 10.7 months"
            }
        )

    def _biomarker_negative_pathway(
        self,
        patient: PatientFactWithCodes,
        biomarkers: BiomarkerProfile
    ) -> AgentProposal:
        """No actionable biomarkers - recommend standard therapy"""

        ps = self._get_patient_attr(patient, 'performance_status', 2)
        if ps <= 1:
            treatment = "Platinum-based chemotherapy doublet"
            rationale = (
                "No actionable molecular alterations detected. "
                "Standard platinum-based chemotherapy recommended. "
                "Consider comprehensive genomic profiling if not already performed."
            )
            confidence = 0.70
        else:
            treatment = "Single-agent chemotherapy or best supportive care"
            rationale = (
                f"No actionable biomarkers, PS {ps}. "
                "Single-agent therapy or best supportive care recommended."
            )
            confidence = 0.65

        return AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            treatment=treatment,
            confidence=confidence,
            evidence_level="Grade B",
            treatment_intent="Palliative",
            rationale=rationale,
            guideline_reference="NCCN NSCLC 2025",
            contraindications=[],
            risk_score=0.4,
            supporting_data={
                "recommendation": "Consider comprehensive genomic profiling",
                "alternative": "Clinical trial enrollment"
            }
        )

    # ========================================
    # UTILITIES
    # ========================================

    def _extract_biomarkers(self, patient: PatientFactWithCodes) -> BiomarkerProfile:
        """Extract biomarker data from patient record"""
        
        # Handle dict input
        if isinstance(patient, dict):
            return BiomarkerProfile(
                egfr_mutation=patient.get('egfr_mutation_status') or patient.get('egfr_mutation'),
                egfr_mutation_type=patient.get('egfr_mutation_type'),
                alk_rearrangement=patient.get('alk_rearrangement'),
                ros1_rearrangement=patient.get('ros1_rearrangement'),
                braf_mutation=patient.get('braf_mutation'),
                met_exon14_skipping=patient.get('met_exon14') or patient.get('met_exon14_skipping'),
                ret_rearrangement=patient.get('ret_rearrangement'),
                kras_mutation=patient.get('kras_mutation'),
                pdl1_tps=patient.get('pdl1_score') or patient.get('pdl1_tps'),
                tmb_score=patient.get('tmb_score'),
                her2_mutation=patient.get('her2_mutation'),
                ntrk_fusion=patient.get('ntrk_fusion')
            )

        return BiomarkerProfile(
            egfr_mutation=patient.egfr_mutation_status if hasattr(patient, 'egfr_mutation_status') else None,
            egfr_mutation_type=patient.egfr_mutation_type if hasattr(patient, 'egfr_mutation_type') else None,
            alk_rearrangement=patient.alk_rearrangement if hasattr(patient, 'alk_rearrangement') else None,
            ros1_rearrangement=patient.ros1_rearrangement if hasattr(patient, 'ros1_rearrangement') else None,
            braf_mutation=patient.braf_mutation if hasattr(patient, 'braf_mutation') else None,
            met_exon14_skipping=patient.met_exon14 if hasattr(patient, 'met_exon14') else None,
            ret_rearrangement=patient.ret_rearrangement if hasattr(patient, 'ret_rearrangement') else None,
            kras_mutation=patient.kras_mutation if hasattr(patient, 'kras_mutation') else None,
            pdl1_tps=patient.pdl1_score if hasattr(patient, 'pdl1_score') else None,
            tmb_score=patient.tmb_score if hasattr(patient, 'tmb_score') else None,
            her2_mutation=patient.her2_mutation if hasattr(patient, 'her2_mutation') else None,
            ntrk_fusion=patient.ntrk_fusion if hasattr(patient, 'ntrk_fusion') else None
        )

    def recommend_biomarker_testing(
        self,
        patient: PatientFactWithCodes
    ) -> List[str]:
        """
        Recommend which biomarker tests should be ordered.

        Returns:
            List of recommended tests
        """
        recommended_tests = []
        
        histology_type = self._get_patient_attr(patient, 'histology_type', '')
        tnm_stage = self._get_patient_attr(patient, 'tnm_stage', 'IV')

        # For NSCLC, comprehensive panel is standard
        if "NonSmallCell" in histology_type or "Adenocarcinoma" in histology_type:
            recommended_tests.extend([
                "EGFR mutation analysis (exons 18-21)",
                "ALK rearrangement (IHC or FISH)",
                "ROS1 rearrangement (IHC or FISH)",
                "BRAF V600E mutation",
                "PD-L1 expression (22C3 or SP263 assay)",
                "Comprehensive genomic profiling (NGS panel recommended)"
            ])

            # Add additional markers for advanced disease
            if tnm_stage in ["IV", "IVA", "IVB"]:
                recommended_tests.extend([
                    "MET exon 14 skipping",
                    "RET rearrangement",
                    "KRAS mutation",
                    "HER2 mutation",
                    "NTRK fusion",
                    "Tumor mutational burden (TMB)"
                ])

        # For squamous histology
        elif "Squamous" in histology_type:
            recommended_tests.extend([
                "PD-L1 expression",
                "Consider comprehensive genomic profiling"
            ])

        return recommended_tests
