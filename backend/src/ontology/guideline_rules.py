"""
Clinical Guideline Rules Engine
Implements NICE Lung Cancer Guidelines as OWL 2 class expressions
Based on Sesen et al. paper Section 4
"""

import types
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from owlready2 import *
import logging

logger = logging.getLogger(__name__)


@dataclass
class GuidelineRule:
    """
    Represents a clinical guideline rule.

    Structure from paper:
    - Head (Antecedent): Patient eligibility criteria
    - Body (Consequent): Recommended treatment action
    """
    rule_id: str
    name: str
    source: str
    description: str
    owl_expression: str
    recommended_treatment: str
    treatment_intent: str  # "Curative" or "Palliative"
    evidence_level: str  # "Grade A", "Grade B", "Grade C"
    contraindications: List[str]
    survival_benefit: Optional[str] = None


class GuidelineRuleEngine:
    """
    Engine for creating and executing clinical guideline rules.

    Implements the Patient Scenario pattern from Figure 3:
    1. Define Patient Scenario class with OWL expression
    2. Reasoner classifies patients into scenarios
    3. Scenario links to Decision and Treatment Plan
    """

    # NICE Lung Cancer Guidelines 2011 (from paper)
    NICE_GUIDELINES = [
        GuidelineRule(
            rule_id="R1",
            name="ChemoRule001_AdvancedNSCLC",
            source="NICE Lung Cancer 2011 - CG121",
            description="Offer chemotherapy to patients with stage III or IV NSCLC and good performance status (WHO 0-1)",
            owl_expression="""
                (has_clinical_finding some
                    (NeoplasticDisease and
                        ((has_pre_tnm_staging value "III") or
                         (has_pre_tnm_staging value "IIIA") or
                         (has_pre_tnm_staging value "IIIB") or
                         (has_pre_tnm_staging value "IV")) and
                        (has_histology some NonSmallCellCarcinoma)
                    )
                )
                and
                (has_performance_status some (WHOPerfStatusGrade0 or WHOPerfStatusGrade1))
            """,
            recommended_treatment="Chemotherapy",
            treatment_intent="Palliative",
            evidence_level="Grade A",
            contraindications=["Poor renal function", "Severe comorbidities"],
            survival_benefit="3-4 months median survival improvement"
        ),

        GuidelineRule(
            rule_id="R2",
            name="SurgeryRule001_EarlyStageNSCLC",
            source="NICE Lung Cancer 2011 - CG121",
            description="Offer surgery to patients with stage I-II NSCLC and good performance status",
            owl_expression="""
                (has_clinical_finding some
                    (NeoplasticDisease and
                        ((has_pre_tnm_staging value "IA") or
                         (has_pre_tnm_staging value "IB") or
                         (has_pre_tnm_staging value "IIA") or
                         (has_pre_tnm_staging value "IIB")) and
                        (has_histology some NonSmallCellCarcinoma)
                    )
                )
                and
                (has_performance_status some (WHOPerfStatusGrade0 or WHOPerfStatusGrade1))
            """,
            recommended_treatment="Surgery",
            treatment_intent="Curative",
            evidence_level="Grade A",
            contraindications=["Inadequate lung function (FEV1 <40%)", "Cardiovascular disease"],
            survival_benefit="60-80% 5-year survival for Stage I"
        ),

        GuidelineRule(
            rule_id="R3",
            name="RadioRule001_InoperableNSCLC",
            source="NICE Lung Cancer 2011 - CG121",
            description="Offer radical radiotherapy for stage I-III NSCLC unsuitable for surgery with PS 0-2",
            owl_expression="""
                (has_clinical_finding some
                    (NeoplasticDisease and
                        ((has_pre_tnm_staging value "IA") or
                         (has_pre_tnm_staging value "IB") or
                         (has_pre_tnm_staging value "IIA") or
                         (has_pre_tnm_staging value "IIB") or
                         (has_pre_tnm_staging value "IIIA")) and
                        (has_histology some NonSmallCellCarcinoma)
                    )
                )
                and
                (has_performance_status some
                    (WHOPerfStatusGrade0 or WHOPerfStatusGrade1 or WHOPerfStatusGrade2))
            """,
            recommended_treatment="Radiotherapy",
            treatment_intent="Curative",
            evidence_level="Grade B",
            contraindications=["Previous chest radiotherapy", "Large tumor volume"],
            survival_benefit="40-50% 5-year survival for Stage I with SABR"
        ),

        GuidelineRule(
            rule_id="R4",
            name="PalliativeRule001_AdvancedDiseasePoorPS",
            source="NICE Lung Cancer 2011 - CG121",
            description="Offer palliative care for patients with advanced disease and poor performance status (WHO 3-4)",
            owl_expression="""
                (has_clinical_finding some
                    (NeoplasticDisease and
                        ((has_pre_tnm_staging value "IIIB") or
                         (has_pre_tnm_staging value "IV"))
                    )
                )
                and
                (has_performance_status some (WHOPerfStatusGrade3 or WHOPerfStatusGrade4))
            """,
            recommended_treatment="PalliativeCare",
            treatment_intent="Palliative",
            evidence_level="Grade C",
            contraindications=[],
            survival_benefit="Focus on quality of life, not survival extension"
        ),

        GuidelineRule(
            rule_id="R5",
            name="SCLCChemoRule001",
            source="NICE Lung Cancer 2011 - CG121",
            description="Offer chemotherapy for SCLC with good performance status (WHO 0-2)",
            owl_expression="""
                (has_clinical_finding some
                    (NeoplasticDisease and
                        (has_histology some SmallCellCarcinoma)
                    )
                )
                and
                (has_performance_status some
                    (WHOPerfStatusGrade0 or WHOPerfStatusGrade1 or WHOPerfStatusGrade2))
            """,
            recommended_treatment="Chemotherapy",
            treatment_intent="Curative",
            evidence_level="Grade A",
            contraindications=["Severe comorbidities"],
            survival_benefit="Limited disease: 20-25% 2-year survival; Extensive: 10-15 months median"
        ),

        GuidelineRule(
            rule_id="R6",
            name="ChemoRadioRule001_LocallyAdvancedNSCLC",
            source="NICE Lung Cancer 2011 - CG121",
            description="Offer concurrent chemoradiotherapy for stage IIIA/IIIB NSCLC with good PS",
            owl_expression="""
                (has_clinical_finding some
                    (NeoplasticDisease and
                        ((has_pre_tnm_staging value "IIIA") or
                         (has_pre_tnm_staging value "IIIB")) and
                        (has_histology some NonSmallCellCarcinoma)
                    )
                )
                and
                (has_performance_status some (WHOPerfStatusGrade0 or WHOPerfStatusGrade1))
            """,
            recommended_treatment="Chemoradiotherapy",
            treatment_intent="Curative",
            evidence_level="Grade A",
            contraindications=["Large pleural effusion", "Inadequate lung function"],
            survival_benefit="15-20% 5-year survival"
        ),

        GuidelineRule(
            rule_id="R7",
            name="ImmunoRule001_AdvancedNSCLC_HighPDL1",
            source="Contemporary Immunotherapy Guidelines 2025",
            description="Offer immunotherapy (PD-L1 inhibitor) for advanced NSCLC with PD-L1 ≥50% and PS 0-1",
            owl_expression="""
                (has_clinical_finding some
                    (NeoplasticDisease and
                        ((has_pre_tnm_staging value "IV") or
                         (has_pre_tnm_staging value "IIIB")) and
                        (has_histology some NonSmallCellCarcinoma) and
                        (has_biomarker some PDL1High)
                    )
                )
                and
                (has_performance_status some (WHOPerfStatusGrade0 or WHOPerfStatusGrade1))
            """,
            recommended_treatment="Immunotherapy",
            treatment_intent="Palliative",
            evidence_level="Grade A",
            contraindications=["Autoimmune disease", "Organ transplant recipients"],
            survival_benefit="Median survival >12 months vs 9 months with chemotherapy"
        ),

        # Targeted Therapy Rules (Modern Precision Medicine)
        GuidelineRule(
            rule_id="R8",
            name="EGFRTKIRule001_EGFRPositiveNSCLC",
            source="ESMO/ASCO Precision Medicine Guidelines 2025",
            description="Offer EGFR TKI (osimertinib) for EGFR-mutated advanced NSCLC",
            owl_expression="""
                (has_clinical_finding some
                    (NeoplasticDisease and
                        ((has_pre_tnm_staging value "IIIB") or
                         (has_pre_tnm_staging value "IV")) and
                        (has_histology some NonSmallCellCarcinoma) and
                        (has_biomarker some EGFRPositive)
                    )
                )
                and
                (has_performance_status some (WHOPerfStatusGrade0 or WHOPerfStatusGrade1 or WHOPerfStatusGrade2))
            """,
            recommended_treatment="TargetedTherapy",
            treatment_intent="Palliative",
            evidence_level="Grade A",
            contraindications=["Severe hepatic impairment", "Interstitial lung disease"],
            survival_benefit="Median PFS 18-21 months vs 10-12 months with chemotherapy"
        ),

        GuidelineRule(
            rule_id="R9",
            name="ALKInhibitorRule001_ALKPositiveNSCLC",
            source="ESMO/ASCO Precision Medicine Guidelines 2025",
            description="Offer ALK inhibitor (alectinib/lorlatinib) for ALK-rearranged advanced NSCLC",
            owl_expression="""
                (has_clinical_finding some
                    (NeoplasticDisease and
                        ((has_pre_tnm_staging value "IIIB") or
                         (has_pre_tnm_staging value "IV")) and
                        (has_histology some NonSmallCellCarcinoma) and
                        (has_biomarker some ALKPositive)
                    )
                )
                and
                (has_performance_status some (WHOPerfStatusGrade0 or WHOPerfStatusGrade1 or WHOPerfStatusGrade2))
            """,
            recommended_treatment="TargetedTherapy",
            treatment_intent="Palliative",
            evidence_level="Grade A",
            contraindications=["Severe bradycardia", "QT prolongation"],
            survival_benefit="Median PFS >34 months with first-line alectinib"
        ),

        GuidelineRule(
            rule_id="R10",
            name="ChemoImmunoRule001_PDL1Low",
            source="Contemporary Combined Therapy Guidelines 2025",
            description="Offer chemo-immunotherapy combination for advanced NSCLC with PD-L1 1-49%",
            owl_expression="""
                (has_clinical_finding some
                    (NeoplasticDisease and
                        (has_pre_tnm_staging value "IV") and
                        (has_histology some NonSmallCellCarcinoma) and
                        (has_biomarker some PDL1Low)
                    )
                )
                and
                (has_performance_status some (WHOPerfStatusGrade0 or WHOPerfStatusGrade1))
            """,
            recommended_treatment="ChemoImmunotherapy",
            treatment_intent="Palliative",
            evidence_level="Grade A",
            contraindications=["Autoimmune disease", "Poor renal function"],
            survival_benefit="Median OS 22 months vs 11 months with chemotherapy alone"
        ),
    ]

    # FEV1 thresholds for surgery eligibility (from paper and NICE guidelines)
    SURGERY_FEV1_THRESHOLDS = {
        "Lobectomy": 40.0,      # Minimum FEV1% for lobectomy
        "Pneumonectomy": 55.0,  # Minimum FEV1% for pneumonectomy
    }

    def __init__(self, lucada_ontology):
        """
        Initialize guideline engine.

        Args:
            lucada_ontology: LUCADAOntology instance
        """
        self.onto = lucada_ontology.onto
        if not self.onto:
            raise ValueError("Ontology not created. Call lucada_ontology.create() first.")

        self.rules: Dict[str, GuidelineRule] = {}
        self._load_default_rules()

    def _load_default_rules(self):
        """Load default NICE guidelines"""
        logger.info("Loading clinical guideline rules...")
        for rule in self.NICE_GUIDELINES:
            self.add_rule(rule)
        logger.info(f"✓ Loaded {len(self.rules)} guideline rules")

    def add_rule(self, rule: GuidelineRule):
        """
        Add a guideline rule to the ontology as a Patient Scenario.

        Creates:
        1. PatientScenario subclass with OWL criteria
        2. Decision individual
        3. TreatmentPlan individual
        4. Links: Scenario → Decision → TreatmentPlan
        """
        self.rules[rule.rule_id] = rule

        with self.onto:
            try:
                # Create Patient Scenario subclass
                scenario_class = types.new_class(
                    rule.name,
                    (self.onto.PatientScenario,)
                )

                # Create reference individuals for argumentation chain
                scenario_ref = scenario_class(f"Reference_{rule.name}")

                # Create Decision individual
                decision_name = f"{rule.name}_Decision"
                decision_ref = self.onto.Decision(f"Reference_{decision_name}")

                # Create TreatmentPlan individual
                treatment_class = getattr(self.onto, rule.recommended_treatment, None)
                if treatment_class:
                    treatment_ref = treatment_class(f"Reference_{rule.recommended_treatment}_For_{rule.name}")
                else:
                    # Fallback to generic treatment plan
                    treatment_ref = self.onto.TreatmentPlan(f"Reference_{rule.recommended_treatment}_Plan")
                    treatment_ref.treatment_plan_type = rule.recommended_treatment

                # Set intent
                if rule.treatment_intent == "Curative":
                    treatment_ref.has_intent = [self.onto.Reference_Curative_Intent]
                else:
                    treatment_ref.has_intent = [self.onto.Reference_Palliative_Intent]

                # Create argumentation links
                scenario_ref.supports_decision = [decision_ref]
                decision_ref.entails = [treatment_ref]

                logger.debug(f"  Added rule: {rule.rule_id} - {rule.name}")

            except Exception as e:
                logger.error(f"Failed to add rule {rule.rule_id}: {e}")

    def classify_patient(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Classify a patient and return applicable treatment recommendations.

        This uses pattern matching instead of OWL reasoner for better performance.
        Production system would use HermiT/Pellet reasoner for full ontological inference.

        Args:
            patient_data: Dictionary with patient clinical data

        Returns:
            List of applicable guideline rules with recommendations
        """
        recommendations = []

        tnm_stage = patient_data.get("tnm_stage", "").upper()
        histology = patient_data.get("histology_type", "")
        performance_status = patient_data.get("performance_status", 0)
        age = patient_data.get("age", 0)
        fev1_percent = patient_data.get("fev1_percent", 100.0)

        # Biomarker data
        egfr_status = patient_data.get("egfr_status", "unknown")
        alk_status = patient_data.get("alk_status", "unknown")
        pdl1_score = patient_data.get("pdl1_score", None)

        for rule_id, rule in self.rules.items():
            match_result = self._matches_rule(
                rule, tnm_stage, histology, performance_status, age,
                fev1_percent, egfr_status, alk_status, pdl1_score
            )
            if match_result["matches"]:
                rec = {
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "source": rule.source,
                    "description": rule.description,
                    "recommended_treatment": rule.recommended_treatment,
                    "treatment_intent": rule.treatment_intent,
                    "evidence_level": rule.evidence_level,
                    "contraindications": rule.contraindications,
                    "survival_benefit": rule.survival_benefit,
                    "priority": self._calculate_priority(rule, tnm_stage, performance_status)
                }
                # Add any warnings (e.g., FEV1 concerns)
                if match_result.get("warnings"):
                    rec["warnings"] = match_result["warnings"]
                recommendations.append(rec)

        # Sort by priority (Grade A evidence + good PS = higher priority)
        recommendations.sort(key=lambda x: x["priority"], reverse=True)

        return recommendations

    def _matches_rule(
        self,
        rule: GuidelineRule,
        tnm_stage: str,
        histology: str,
        performance_status: int,
        age: int,
        fev1_percent: float = 100.0,
        egfr_status: str = "unknown",
        alk_status: str = "unknown",
        pdl1_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Check if patient matches a rule's criteria.

        Returns:
            Dictionary with 'matches' (bool) and optional 'warnings' (list)
        """
        result = {"matches": False, "warnings": []}

        # Rule-specific matching logic
        if rule.rule_id == "R1":  # Chemo for Stage III-IV NSCLC
            stage_match = tnm_stage in ["III", "IIIA", "IIIB", "IV"]
            histology_match = self._is_nsclc(histology)
            ps_match = performance_status <= 1
            result["matches"] = stage_match and histology_match and ps_match

        elif rule.rule_id == "R2":  # Surgery for Stage I-II NSCLC
            stage_match = tnm_stage in ["IA", "IB", "IIA", "IIB"]
            histology_match = self._is_nsclc(histology)
            ps_match = performance_status <= 1
            fev1_adequate = fev1_percent >= self.SURGERY_FEV1_THRESHOLDS.get("Lobectomy", 40.0)

            result["matches"] = stage_match and histology_match and ps_match

            # Add FEV1 warnings if applicable
            if result["matches"]:
                if fev1_percent < self.SURGERY_FEV1_THRESHOLDS["Pneumonectomy"]:
                    result["warnings"].append(
                        f"FEV1 {fev1_percent}% may limit surgical options to lobectomy (pneumonectomy requires ≥55%)"
                    )
                if not fev1_adequate:
                    result["warnings"].append(
                        f"FEV1 {fev1_percent}% below threshold for lobectomy (≥40% required) - consider radiotherapy"
                    )
                    # Still match but with strong warning

        elif rule.rule_id == "R3":  # Radiotherapy for Stage I-IIIA
            stage_match = tnm_stage in ["IA", "IB", "IIA", "IIB", "IIIA"]
            histology_match = self._is_nsclc(histology)
            ps_match = performance_status <= 2
            result["matches"] = stage_match and histology_match and ps_match

        elif rule.rule_id == "R4":  # Palliative for advanced + poor PS
            stage_match = tnm_stage in ["IIIB", "IV"]
            ps_match = performance_status >= 3
            result["matches"] = stage_match and ps_match

        elif rule.rule_id == "R5":  # Chemo for SCLC
            histology_match = self._is_sclc(histology)
            ps_match = performance_status <= 2
            result["matches"] = histology_match and ps_match

        elif rule.rule_id == "R6":  # Chemoradio for Stage IIIA/B
            stage_match = tnm_stage in ["IIIA", "IIIB"]
            histology_match = self._is_nsclc(histology)
            ps_match = performance_status <= 1
            result["matches"] = stage_match and histology_match and ps_match

            if result["matches"] and fev1_percent < 50.0:
                result["warnings"].append(
                    f"FEV1 {fev1_percent}% may increase toxicity risk with concurrent chemoradiation"
                )

        elif rule.rule_id == "R7":  # Immunotherapy for advanced NSCLC with high PD-L1
            stage_match = tnm_stage in ["IIIB", "IV"]
            histology_match = self._is_nsclc(histology)
            ps_match = performance_status <= 1
            # PD-L1 ≥50% for monotherapy
            pdl1_high = pdl1_score is not None and pdl1_score >= 50.0

            result["matches"] = stage_match and histology_match and ps_match and pdl1_high

        elif rule.rule_id == "R8":  # EGFR TKI for EGFR+ NSCLC
            stage_match = tnm_stage in ["IIIB", "IV"]
            histology_match = self._is_nsclc(histology)
            ps_match = performance_status <= 2
            egfr_positive = egfr_status.lower() == "positive"

            result["matches"] = stage_match and histology_match and ps_match and egfr_positive

        elif rule.rule_id == "R9":  # ALK inhibitor for ALK+ NSCLC
            stage_match = tnm_stage in ["IIIB", "IV"]
            histology_match = self._is_nsclc(histology)
            ps_match = performance_status <= 2
            alk_positive = alk_status.lower() == "positive"

            result["matches"] = stage_match and histology_match and ps_match and alk_positive

        elif rule.rule_id == "R10":  # Chemo-immunotherapy for PD-L1 1-49%
            stage_match = tnm_stage == "IV"
            histology_match = self._is_nsclc(histology)
            ps_match = performance_status <= 1
            pdl1_low = pdl1_score is not None and 1.0 <= pdl1_score < 50.0

            result["matches"] = stage_match and histology_match and ps_match and pdl1_low

        return result

    def _is_nsclc(self, histology: str) -> bool:
        """Check if histology is NSCLC type"""
        nsclc_types = [
            "NonSmallCellCarcinoma",
            "Adenocarcinoma",
            "SquamousCellCarcinoma",
            "LargeCellCarcinoma"
        ]
        return any(h in histology for h in nsclc_types)

    def _is_sclc(self, histology: str) -> bool:
        """Check if histology is SCLC type"""
        return "SmallCell" in histology

    def _calculate_priority(self, rule: GuidelineRule, stage: str, ps: int) -> int:
        """
        Calculate recommendation priority.

        Higher priority for:
        - Grade A evidence
        - Curative intent
        - Early stage disease
        - Good performance status
        """
        priority = 0

        # Evidence level (40 points max)
        if rule.evidence_level == "Grade A":
            priority += 40
        elif rule.evidence_level == "Grade B":
            priority += 25
        else:
            priority += 10

        # Treatment intent (30 points max)
        if rule.treatment_intent == "Curative":
            priority += 30

        # Stage (20 points max - earlier is better)
        stage_priority = {
            "IA": 20, "IB": 18,
            "IIA": 15, "IIB": 13,
            "IIIA": 10, "IIIB": 7,
            "IV": 3
        }
        priority += stage_priority.get(stage, 0)

        # Performance status (10 points max)
        priority += (4 - ps) * 2.5

        return int(priority)

    def get_rule_by_id(self, rule_id: str) -> Optional[GuidelineRule]:
        """Get a specific rule by ID"""
        return self.rules.get(rule_id)

    def get_all_rules(self) -> List[GuidelineRule]:
        """Get all registered rules"""
        return list(self.rules.values())

    def get_rules_by_treatment(self, treatment: str) -> List[GuidelineRule]:
        """Get all rules recommending a specific treatment"""
        return [r for r in self.rules.values() if r.recommended_treatment == treatment]


if __name__ == "__main__":
    from lucada_ontology import LUCADAOntology

    # Create ontology and rule engine
    lucada = LUCADAOntology()
    onto = lucada.create()

    engine = GuidelineRuleEngine(lucada)

    # Test patient
    test_patient = {
        "patient_id": "TEST001",
        "name": "Test Patient",
        "age": 68,
        "sex": "M",
        "tnm_stage": "IIIA",
        "histology_type": "Adenocarcinoma",
        "performance_status": 1
    }

    print("\n" + "=" * 80)
    print("GUIDELINE RULE ENGINE TEST")
    print("=" * 80)
    print(f"\nTest Patient:")
    print(f"  Stage: {test_patient['tnm_stage']}")
    print(f"  Histology: {test_patient['histology_type']}")
    print(f"  Performance Status: WHO {test_patient['performance_status']}")

    recommendations = engine.classify_patient(test_patient)

    print(f"\n✓ Found {len(recommendations)} applicable treatment recommendations:\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['recommended_treatment']} (Priority: {rec['priority']})")
        print(f"   Rule: {rec['rule_id']} - {rec['source']}")
        print(f"   Evidence: {rec['evidence_level']} | Intent: {rec['treatment_intent']}")
        print(f"   Survival: {rec['survival_benefit']}")
        print()
