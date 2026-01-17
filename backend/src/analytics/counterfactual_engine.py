"""
Counterfactual Reasoning Engine

Implements "what-if" analysis for treatment decisions by
simulating alternative patient scenarios.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from copy import deepcopy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CounterfactualScenario:
    """A counterfactual scenario with modified patient characteristics"""
    scenario_id: str
    description: str
    modified_attributes: Dict[str, Any]
    original_value: Any
    counterfactual_value: Any
    clinical_significance: str


@dataclass
class CounterfactualAnalysis:
    """Results of counterfactual analysis"""
    patient_id: str
    original_scenario: Dict[str, Any]
    original_recommendation: str
    counterfactuals: List[Dict[str, Any]]
    actionable_interventions: List[str]
    sensitivity_analysis: Dict[str, Any]


class CounterfactualEngine:
    """
    Counterfactual reasoning for "what-if" treatment analysis.

    Features:
    - Biomarker counterfactuals (what if patient had different mutations?)
    - Performance status counterfactuals (what if PS improved?)
    - Stage counterfactuals (what if detected earlier?)
    - Comorbidity counterfactuals (what if no contraindications?)
    """

    def __init__(self, workflow=None):
        """
        Initialize counterfactual engine.

        Args:
            workflow: LCA workflow for running scenarios
        """
        self.workflow = workflow

    def analyze_counterfactuals(
        self,
        patient: Dict[str, Any],
        current_recommendation: str
    ) -> CounterfactualAnalysis:
        """
        Generate and analyze counterfactual scenarios.

        Args:
            patient: Current patient data
            current_recommendation: Current treatment recommendation

        Returns:
            CounterfactualAnalysis with scenarios and insights
        """
        logger.info(f"Generating counterfactual scenarios for patient {patient.get('patient_id')}")

        counterfactuals = []

        # 1. Biomarker counterfactuals
        counterfactuals.extend(self._biomarker_counterfactuals(patient, current_recommendation))

        # 2. Performance status counterfactuals
        counterfactuals.extend(self._performance_status_counterfactuals(patient, current_recommendation))

        # 3. Stage counterfactuals
        counterfactuals.extend(self._stage_counterfactuals(patient, current_recommendation))

        # 4. Comorbidity counterfactuals
        counterfactuals.extend(self._comorbidity_counterfactuals(patient, current_recommendation))

        # Identify actionable interventions
        actionable = self._identify_actionable_interventions(counterfactuals)

        # Sensitivity analysis
        sensitivity = self._sensitivity_analysis(counterfactuals)

        return CounterfactualAnalysis(
            patient_id=patient.get('patient_id', 'UNKNOWN'),
            original_scenario={
                "stage": patient.get('tnm_stage'),
                "ps": patient.get('performance_status'),
                "biomarkers": self._extract_biomarkers(patient)
            },
            original_recommendation=current_recommendation,
            counterfactuals=counterfactuals,
            actionable_interventions=actionable,
            sensitivity_analysis=sensitivity
        )

    # ========================================
    # BIOMARKER COUNTERFACTUALS
    # ========================================

    def _biomarker_counterfactuals(
        self,
        patient: Dict[str, Any],
        current_recommendation: str
    ) -> List[Dict[str, Any]]:
        """Generate biomarker-based counterfactuals"""

        scenarios = []

        # Scenario 1: What if EGFR positive?
        if patient.get('egfr_mutation') != "Positive":
            cf_patient = deepcopy(patient)
            cf_patient['egfr_mutation'] = "Positive"
            cf_patient['egfr_mutation_type'] = "Ex19del"

            scenarios.append({
                "id": "CF-EGFR+",
                "description": "If patient had EGFR mutation",
                "modified_attribute": "egfr_mutation",
                "original_value": patient.get('egfr_mutation', 'Negative'),
                "counterfactual_value": "Positive (Ex19del)",
                "expected_recommendation": "Osimertinib (first-line EGFR TKI)",
                "expected_benefit": "Median PFS 18.9 months vs 10.2 months with chemotherapy",
                "clinical_impact": "HIGH",
                "actionability": "Recommend comprehensive genomic testing if not done",
                "evidence": "FLAURA trial, Grade A evidence"
            })

        # Scenario 2: What if ALK positive?
        if patient.get('alk_rearrangement') != "Positive":
            scenarios.append({
                "id": "CF-ALK+",
                "description": "If patient had ALK rearrangement",
                "modified_attribute": "alk_rearrangement",
                "original_value": patient.get('alk_rearrangement', 'Negative'),
                "counterfactual_value": "Positive",
                "expected_recommendation": "Alectinib (first-line ALK inhibitor)",
                "expected_benefit": "Median PFS 34.8 months, excellent CNS activity",
                "clinical_impact": "HIGH",
                "actionability": "Ensure ALK testing performed (IHC or FISH)",
                "evidence": "ALEX trial, Grade A evidence"
            })

        # Scenario 3: What if PD-L1 ≥50%?
        current_pdl1 = patient.get('pdl1_tps', 0)
        if current_pdl1 < 50:
            scenarios.append({
                "id": "CF-PDL1-High",
                "description": "If patient had high PD-L1 expression (≥50%)",
                "modified_attribute": "pdl1_tps",
                "original_value": f"{current_pdl1}%",
                "counterfactual_value": "≥50%",
                "expected_recommendation": "Pembrolizumab monotherapy",
                "expected_benefit": "Median OS 26.3 vs 13.4 months with chemotherapy",
                "clinical_impact": "HIGH",
                "actionability": "PD-L1 testing is standard - results guide IO eligibility",
                "evidence": "KEYNOTE-024, Grade A evidence"
            })

        # Scenario 4: What if biomarker negative?
        if any(patient.get(bm) == "Positive" for bm in ['egfr_mutation', 'alk_rearrangement']):
            scenarios.append({
                "id": "CF-Biomarker-Neg",
                "description": "If no actionable mutations",
                "modified_attribute": "all_biomarkers",
                "original_value": "Actionable mutation present",
                "counterfactual_value": "All negative",
                "expected_recommendation": "Platinum-based chemotherapy",
                "expected_benefit": "Standard chemotherapy outcomes (ORR 20-30%)",
                "clinical_impact": "MODERATE",
                "actionability": "Highlights importance of comprehensive molecular testing",
                "evidence": "NCCN Guidelines"
            })

        return scenarios

    # ========================================
    # PERFORMANCE STATUS COUNTERFACTUALS
    # ========================================

    def _performance_status_counterfactuals(
        self,
        patient: Dict[str, Any],
        current_recommendation: str
    ) -> List[Dict[str, Any]]:
        """Generate performance status counterfactuals"""

        scenarios = []
        current_ps = patient.get('performance_status', 1)

        # Scenario: What if PS improved?
        if current_ps >= 2:
            scenarios.append({
                "id": "CF-PS-Improved",
                "description": f"If performance status improved from {current_ps} to 0-1",
                "modified_attribute": "performance_status",
                "original_value": current_ps,
                "counterfactual_value": 1,
                "expected_recommendation": "More aggressive treatment options available",
                "expected_benefit": "Eligible for combination chemotherapy, immunotherapy",
                "clinical_impact": "HIGH",
                "actionability": "Supportive care to improve PS: transfusions, pain control, nutrition",
                "evidence": "ECOG PS 0-1 required for most clinical trials"
            })

        # Scenario: What if PS declined?
        if current_ps <= 1:
            scenarios.append({
                "id": "CF-PS-Declined",
                "description": f"If performance status declined from {current_ps} to 3-4",
                "modified_attribute": "performance_status",
                "original_value": current_ps,
                "counterfactual_value": 3,
                "expected_recommendation": "Best supportive care, palliative focus",
                "expected_benefit": "Symptom management priority over cytotoxic therapy",
                "clinical_impact": "HIGH",
                "actionability": "Close PS monitoring, early supportive care integration",
                "evidence": "ASCO Guidelines - PS 3-4 chemotherapy not recommended"
            })

        return scenarios

    # ========================================
    # STAGE COUNTERFACTUALS
    # ========================================

    def _stage_counterfactuals(
        self,
        patient: Dict[str, Any],
        current_recommendation: str
    ) -> List[Dict[str, Any]]:
        """Generate stage-based counterfactuals"""

        scenarios = []
        current_stage = patient.get('tnm_stage', 'IV')

        # Scenario: What if detected earlier?
        if current_stage in ['IV', 'IVA', 'IVB', 'IIIB']:
            scenarios.append({
                "id": "CF-Early-Detection",
                "description": f"If disease detected at Stage I instead of {current_stage}",
                "modified_attribute": "tnm_stage",
                "original_value": current_stage,
                "counterfactual_value": "I",
                "expected_recommendation": "Surgical resection with curative intent",
                "expected_benefit": "5-year survival 70-90% vs <10% for Stage IV",
                "clinical_impact": "CRITICAL",
                "actionability": "Highlights importance of lung cancer screening (LDCT for high-risk)",
                "evidence": "NLST trial - 20% mortality reduction with LDCT screening"
            })

        # Scenario: What if locally advanced instead of metastatic?
        if current_stage in ['IV', 'IVA', 'IVB']:
            scenarios.append({
                "id": "CF-Locally-Advanced",
                "description": f"If disease was Stage IIIA instead of {current_stage}",
                "modified_attribute": "tnm_stage",
                "original_value": current_stage,
                "counterfactual_value": "IIIA",
                "expected_recommendation": "Chemoradiotherapy with curative intent",
                "expected_benefit": "Potential for cure vs palliative approach",
                "clinical_impact": "HIGH",
                "actionability": "Emphasizes importance of thorough staging (PET/CT, brain MRI)",
                "evidence": "PACIFIC trial - durvalumab consolidation post-CRT"
            })

        return scenarios

    # ========================================
    # COMORBIDITY COUNTERFACTUALS
    # ========================================

    def _comorbidity_counterfactuals(
        self,
        patient: Dict[str, Any],
        current_recommendation: str
    ) -> List[Dict[str, Any]]:
        """Generate comorbidity-based counterfactuals"""

        scenarios = []
        comorbidities = patient.get('comorbidities', [])

        # Scenario: What if no renal impairment?
        if patient.get('egfr', 100) < 60:
            scenarios.append({
                "id": "CF-Normal-Renal",
                "description": "If normal renal function (eGFR ≥60)",
                "modified_attribute": "egfr",
                "original_value": f"{patient.get('egfr')} mL/min",
                "counterfactual_value": "≥60 mL/min",
                "expected_recommendation": "Full-dose cisplatin option available",
                "expected_benefit": "Cisplatin superior to carboplatin in some settings",
                "clinical_impact": "MODERATE",
                "actionability": "Optimize renal function before therapy, adequate hydration",
                "evidence": "Cisplatin dose reductions needed for CrCl <60"
            })

        # Scenario: What if no ILD?
        if "ild" in comorbidities or "interstitial_lung_disease" in comorbidities:
            scenarios.append({
                "id": "CF-No-ILD",
                "description": "If no pre-existing interstitial lung disease",
                "modified_attribute": "ild_status",
                "original_value": "ILD present",
                "counterfactual_value": "No ILD",
                "expected_recommendation": "EGFR TKIs and immunotherapy options expanded",
                "expected_benefit": "Reduced risk of potentially fatal ILD/pneumonitis",
                "clinical_impact": "HIGH",
                "actionability": "Careful risk-benefit discussion for TKI/IO in ILD patients",
                "evidence": "EGFR TKI ILD incidence 2-5%, higher with pre-existing ILD"
            })

        return scenarios

    # ========================================
    # ACTIONABILITY ANALYSIS
    # ========================================

    def _identify_actionable_interventions(
        self,
        counterfactuals: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify which counterfactuals suggest actionable interventions"""

        actionable = []

        for cf in counterfactuals:
            if cf.get('clinical_impact') in ['HIGH', 'CRITICAL']:
                actionable.append(cf.get('actionability', ''))

        return [a for a in actionable if a]  # Remove empty strings

    def _sensitivity_analysis(
        self,
        counterfactuals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze which factors have highest impact on recommendations"""

        impact_scores = {
            "biomarkers": 0,
            "performance_status": 0,
            "stage": 0,
            "comorbidities": 0
        }

        for cf in counterfactuals:
            impact = cf.get('clinical_impact', 'LOW')
            cf_id = cf.get('id', '')

            if 'EGFR' in cf_id or 'ALK' in cf_id or 'PDL1' in cf_id:
                category = "biomarkers"
            elif 'PS' in cf_id:
                category = "performance_status"
            elif 'Stage' in cf_id or 'Detection' in cf_id:
                category = "stage"
            else:
                category = "comorbidities"

            # Score: CRITICAL=3, HIGH=2, MODERATE=1, LOW=0
            score = {'CRITICAL': 3, 'HIGH': 2, 'MODERATE': 1, 'LOW': 0}.get(impact, 0)
            impact_scores[category] += score

        # Rank factors by impact
        ranked = sorted(impact_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "most_influential_factor": ranked[0][0],
            "impact_scores": impact_scores,
            "ranked_factors": [{"factor": f, "score": s} for f, s in ranked],
            "interpretation": f"{ranked[0][0]} has the greatest impact on treatment recommendations"
        }

    def _extract_biomarkers(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        """Extract biomarker information from patient"""

        return {
            "egfr": patient.get('egfr_mutation', 'Unknown'),
            "alk": patient.get('alk_rearrangement', 'Unknown'),
            "ros1": patient.get('ros1_rearrangement', 'Unknown'),
            "pdl1_tps": patient.get('pdl1_tps', 'Unknown')
        }

    # ========================================
    # REPORTING
    # ========================================

    def generate_counterfactual_report(
        self,
        analysis: CounterfactualAnalysis
    ) -> str:
        """Generate human-readable counterfactual report"""

        lines = [
            f"Counterfactual Analysis for Patient {analysis.patient_id}",
            "=" * 70,
            "",
            f"Current Scenario:",
            f"  Stage: {analysis.original_scenario.get('stage')}",
            f"  Performance Status: {analysis.original_scenario.get('ps')}",
            f"  Biomarkers: {analysis.original_scenario.get('biomarkers')}",
            f"  Recommendation: {analysis.original_recommendation}",
            "",
            f"Alternative Scenarios ({len(analysis.counterfactuals)}):",
            ""
        ]

        for cf in analysis.counterfactuals:
            lines.extend([
                f"Scenario: {cf.get('description')}",
                f"  Change: {cf.get('original_value')} → {cf.get('counterfactual_value')}",
                f"  Expected Outcome: {cf.get('expected_recommendation')}",
                f"  Benefit: {cf.get('expected_benefit')}",
                f"  Impact: {cf.get('clinical_impact')}",
                f"  Action: {cf.get('actionability')}",
                ""
            ])

        lines.extend([
            "Sensitivity Analysis:",
            f"  Most influential factor: {analysis.sensitivity_analysis['most_influential_factor']}",
            f"  {analysis.sensitivity_analysis['interpretation']}",
            "",
            "Actionable Interventions:",
        ])

        for action in analysis.actionable_interventions:
            lines.append(f"  • {action}")

        return "\n".join(lines)
