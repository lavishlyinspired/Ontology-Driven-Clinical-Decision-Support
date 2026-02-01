"""
Explanation Agent (Agent 6 of 6)
Generates MDT summaries and clinician-friendly explanations.

Responsibilities:
- Generate MDT (Multi-Disciplinary Team) summaries
- Format reasoning for clinicians
- Create audit-ready documentation
- Support different output formats (clinical, technical, patient)

Tools: format_mdt_summary(), generate_explanation(), format_for_audit()
Data Sources: Classification results, inference records
NEVER: Direct Neo4j writes
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

# Centralized logging
from ..logging_config import get_logger, log_agent_action

logger = get_logger(__name__)

from ..db.models import (
    PatientFactWithCodes,
    ClassificationResult,
    InferenceRecord,
    MDTSummary,
    TreatmentRecommendation,
    EvidenceLevel
)


class ExplanationAgent:
    """
    Agent 6: Explanation Agent
    Generates MDT summaries and clinician-friendly explanations.
    READ-ONLY: Never writes to Neo4j.
    """

    # Evidence level descriptions for clinicians
    EVIDENCE_DESCRIPTIONS = {
        EvidenceLevel.GRADE_A: "Strong evidence from randomized controlled trials",
        EvidenceLevel.GRADE_B: "Moderate evidence from well-designed studies",
        EvidenceLevel.GRADE_C: "Limited evidence; expert consensus",
        EvidenceLevel.EXPERT_OPINION: "Based on clinical expertise",
    }

    def __init__(self):
        self.name = "ExplanationAgent"
        self.version = "1.0.0"

    def execute(
        self,
        patient: PatientFactWithCodes,
        classification: ClassificationResult,
        inference_id: Optional[str] = None,
        lab_results: Optional[List[Dict]] = None,
        lab_interpretations: Optional[List[Dict]] = None,
        medications: Optional[List[Dict]] = None,
        drug_interactions: Optional[List[Dict]] = None,
        monitoring_protocol: Optional[Dict] = None,
        eligible_trials: Optional[List[Dict]] = None
    ) -> MDTSummary:
        """
        Execute explanation generation: create MDT-ready summary.

        Args:
            patient: Patient data with SNOMED codes
            classification: Classification results with recommendations
            inference_id: Optional ID of saved inference record
            lab_results: Optional lab results data
            lab_interpretations: Optional lab interpretations
            medications: Optional medications list
            drug_interactions: Optional drug interactions list
            monitoring_protocol: Optional monitoring protocol
            eligible_trials: Optional clinical trials list

        Returns:
            MDTSummary ready for clinical review
        """
        logger.info(f"[{self.name}] Generating MDT summary for patient {patient.patient_id}...")

        # Generate clinical summary
        clinical_summary = self._generate_clinical_summary(patient, classification)

        # Format recommendations for MDT
        formatted_recommendations = self._format_recommendations(classification.recommendations)

        # Generate reasoning explanation
        reasoning_explanation = self._explain_reasoning(classification.reasoning_chain)

        # Create key considerations
        key_considerations = self._identify_key_considerations(patient, classification)

        # Generate discussion points for MDT
        discussion_points = self._generate_discussion_points(patient, classification)

        # Generate new sections for lab/medication/monitoring/trials
        lab_section = self._generate_lab_section(lab_interpretations) if lab_interpretations else ""
        medication_section = self._generate_medication_section(medications, drug_interactions) if medications else ""
        monitoring_section = self._generate_monitoring_section(monitoring_protocol) if monitoring_protocol else ""
        trials_section = self._generate_trials_section(eligible_trials) if eligible_trials else ""

        # Combine all sections into extended summary
        extended_summary = clinical_summary
        if medication_section:
            extended_summary += f"\n\n{medication_section}"
        if lab_section:
            extended_summary += f"\n\n{lab_section}"
        if monitoring_section:
            extended_summary += f"\n\n{monitoring_section}"
        if trials_section:
            extended_summary += f"\n\n{trials_section}"

        mdt_summary = MDTSummary(
            patient_id=patient.patient_id,
            generated_at=datetime.utcnow(),
            inference_id=inference_id,
            clinical_summary=extended_summary,
            classification_scenario=classification.scenario,
            scenario_confidence=classification.scenario_confidence,
            formatted_recommendations=formatted_recommendations,
            reasoning_explanation=reasoning_explanation,
            key_considerations=key_considerations,
            discussion_points=discussion_points,
            guideline_references=classification.guideline_refs,
            snomed_mappings=self._format_snomed_mappings(patient),
            disclaimer=self._get_disclaimer()
        )

        logger.info(f"[{self.name}] ✓ Generated MDT summary for {patient.patient_id}")
        return mdt_summary

    def _generate_clinical_summary(
        self, 
        patient: PatientFactWithCodes,
        classification: ClassificationResult
    ) -> str:
        """Generate a clinical summary paragraph."""
        age_str = f"{patient.age_at_diagnosis} year old" if patient.age_at_diagnosis else "Patient"
        
        # Handle sex - could be string or enum
        sex_str = ""
        if patient.sex:
            sex_str = patient.sex.value if hasattr(patient.sex, 'value') else str(patient.sex)
        
        summary_parts = [
            f"{age_str} {sex_str} patient".strip(),
            f"diagnosed with {patient.histology_type} of the lung",
        ]
        
        if patient.laterality:
            summary_parts.append(f"({patient.laterality} side)")
        
        summary_parts.append(f"at TNM Stage {patient.tnm_stage}.")
        
        if patient.performance_status is not None:
            summary_parts.append(f"ECOG Performance Status is {patient.performance_status}.")
        
        summary_parts.append(
            f"Based on NICE guidelines, this case is classified as '{classification.scenario}' "
            f"with {classification.scenario_confidence:.0%} confidence."
        )
        
        return " ".join(summary_parts)

    def _format_recommendations(
        self, 
        recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Format recommendations for clinical presentation."""
        formatted = []
        
        for rec in recommendations:
            # Handle evidence level - could be string or enum
            evidence_level = rec.get("evidence_level", "Grade C")
            if isinstance(evidence_level, str):
                try:
                    evidence_enum = EvidenceLevel(evidence_level)
                except ValueError:
                    evidence_enum = EvidenceLevel.GRADE_C
            else:
                evidence_enum = evidence_level
                
            evidence_desc = self.EVIDENCE_DESCRIPTIONS.get(
                evidence_enum, 
                "Evidence level not specified"
            )
            
            # Handle intent - could be string or enum
            intent = rec.get("intent", "Unknown")
            intent_str = intent.value if hasattr(intent, 'value') else str(intent)
            
            formatted_rec = {
                "rank": str(rec.get("rank", 0)),
                "treatment": rec.get("treatment", "Not specified"),
                "intent": intent_str,
                "evidence": f"{evidence_enum.value} - {evidence_desc}",
                "guideline": rec.get("guideline_reference") or "Not specified",
                "rationale": rec.get("rationale") or "",
            }
            
            contraindications = rec.get("contraindications")
            if contraindications:
                formatted_rec["cautions"] = "; ".join(contraindications)
            
            biomarker = rec.get("requires_biomarker")
            if biomarker:
                formatted_rec["biomarker_required"] = biomarker
            
            formatted.append(formatted_rec)
        
        return formatted

    def _explain_reasoning(self, reasoning_chain: List[str]) -> str:
        """Convert reasoning chain to readable explanation."""
        if not reasoning_chain:
            return "No reasoning chain available."
        
        explanation_parts = ["The classification was determined as follows:"]
        
        for i, step in enumerate(reasoning_chain, 1):
            explanation_parts.append(f"  {i}. {step}")
        
        return "\n".join(explanation_parts)

    def _identify_key_considerations(
        self, 
        patient: PatientFactWithCodes,
        classification: ClassificationResult
    ) -> List[str]:
        """Identify key clinical considerations for MDT discussion."""
        considerations = []
        
        # Performance status considerations
        if patient.performance_status is not None:
            ps = int(patient.performance_status)
            if ps >= 2:
                considerations.append(
                    f"⚠️ Performance Status {ps}: May affect treatment tolerance. "
                    "Consider dose modifications or alternative approaches."
                )
        
        # Stage-specific considerations
        if patient.tnm_stage in ["IIIA", "IIIB"]:
            considerations.append(
                "🔄 Stage III disease: Multimodality approach may be indicated. "
                "Consider resectability assessment."
            )
        
        if patient.tnm_stage in ["IV", "IVA", "IVB"]:
            considerations.append(
                "🧬 Stage IV disease: Recommend comprehensive biomarker testing "
                "(PD-L1, EGFR, ALK, ROS1, BRAF, KRAS, MET, RET, NTRK)."
            )
        
        # Histology considerations
        if str(patient.histology_type) == "Adenocarcinoma":
            considerations.append(
                "🔬 Adenocarcinoma: Higher likelihood of actionable driver mutations. "
                "Ensure molecular profiling is complete."
            )
        
        if str(patient.histology_type) == "SmallCellCarcinoma":
            considerations.append(
                "⚡ SCLC: Rapidly progressive disease. "
                "Expedite staging and treatment initiation."
            )
        
        # Confidence considerations
        if classification.scenario_confidence < 0.8:
            considerations.append(
                f"⚠️ Classification confidence is {classification.scenario_confidence:.0%}. "
                "Additional clinical review recommended."
            )
        
        return considerations

    def _generate_discussion_points(
        self, 
        patient: PatientFactWithCodes,
        classification: ClassificationResult
    ) -> List[str]:
        """Generate suggested discussion points for MDT meeting."""
        points = []
        
        # Always include these
        points.append("Confirm TNM staging with radiology review")
        points.append("Review pathology for complete histological classification")
        
        # Stage-specific
        if patient.tnm_stage in ["I", "IA", "IB", "II", "IIA", "IIB"]:
            points.append("Assess surgical fitness and patient preference")
            points.append("Consider SABR if surgery declined or not suitable")
        
        if patient.tnm_stage in ["IIIA", "IIIB", "IIIC"]:
            points.append("Evaluate for concurrent vs sequential chemoradiotherapy")
            points.append("Discuss durvalumab consolidation eligibility")
        
        if patient.tnm_stage in ["IV", "IVA", "IVB"]:
            points.append("Review molecular profiling results")
            points.append("Discuss systemic therapy options based on biomarkers")
            points.append("Consider palliative care integration")
        
        # Add recommendation-specific points
        for rec in classification.recommendations[:2]:  # Top 2 recommendations
            # Handle dict or object
            biomarker = rec.get("requires_biomarker") if isinstance(rec, dict) else rec.requires_biomarker
            treatment = rec.get("treatment") if isinstance(rec, dict) else rec.treatment
            
            if biomarker:
                points.append(f"Verify {biomarker} status before {treatment}")
        
        return points

    def _format_snomed_mappings(self, patient: PatientFactWithCodes) -> Dict[str, str]:
        """Format SNOMED-CT mappings for audit purposes."""
        mappings = {}
        
        if patient.snomed_diagnosis_code:
            mappings["Diagnosis"] = f"SCTID:{patient.snomed_diagnosis_code}"
        if patient.snomed_histology_code:
            mappings["Histology"] = f"SCTID:{patient.snomed_histology_code}"
        if patient.snomed_stage_code:
            mappings["Stage"] = f"SCTID:{patient.snomed_stage_code}"
        if patient.snomed_ps_code:
            mappings["Performance Status"] = f"SCTID:{patient.snomed_ps_code}"
        if patient.snomed_laterality_code:
            mappings["Laterality"] = f"SCTID:{patient.snomed_laterality_code}"
        
        return mappings

    def _get_disclaimer(self) -> str:
        """Return standard disclaimer for clinical decision support."""
        return (
            "DISCLAIMER: This summary is generated by an AI clinical decision support system "
            "and is intended to assist, not replace, clinical judgment. All recommendations "
            "should be reviewed by the multi-disciplinary team and discussed with the patient. "
            "The treating clinician retains full responsibility for treatment decisions."
        )

    def _generate_lab_section(self, lab_interpretations: List[Dict]) -> str:
        """Generate laboratory results section for MDT summary."""
        if not lab_interpretations:
            return ""

        lines = ["## Laboratory Results"]

        # Identify critical values
        critical_labs = [
            lab for lab in lab_interpretations
            if lab.get('severity') in ['grade3', 'grade4', 'critical']
        ]

        if critical_labs:
            lines.append("\n### Critical Values")
            for lab in critical_labs:
                loinc_name = lab.get('loinc_name', lab.get('test_name', 'Unknown'))
                value = lab.get('value', 'N/A')
                units = lab.get('units', lab.get('unit', ''))
                interpretation = lab.get('interpretation', '')
                lines.append(f"- 🔴 **{loinc_name}**: {value} {units} ({interpretation})")

        # List all abnormal values
        abnormal_labs = [
            lab for lab in lab_interpretations
            if lab.get('interpretation') in ['high', 'low', 'abnormal']
            and lab.get('severity') not in ['grade3', 'grade4', 'critical']
        ]

        if abnormal_labs:
            lines.append("\n### Abnormal Values")
            for lab in abnormal_labs:
                loinc_name = lab.get('loinc_name', lab.get('test_name', 'Unknown'))
                value = lab.get('value', 'N/A')
                units = lab.get('units', lab.get('unit', ''))
                interpretation = lab.get('interpretation', '')
                severity = lab.get('severity', '')
                lines.append(f"- ⚠️ {loinc_name}: {value} {units} ({interpretation}, {severity})")

        return "\n".join(lines)

    def _generate_medication_section(self, medications: List[Dict], drug_interactions: List[Dict] = None) -> str:
        """Generate medications section for MDT summary."""
        if not medications:
            return ""

        lines = ["## Current Medications"]

        for med in medications:
            drug_name = med.get('drug_name', med.get('name', 'Unknown'))
            dose = med.get('dose', '')
            route = med.get('route', '')
            frequency = med.get('frequency', '')
            lines.append(f"- {drug_name} {dose} {route} {frequency}".strip())

        # Add drug interaction warnings
        if drug_interactions:
            severe_interactions = [
                di for di in drug_interactions
                if di.get('severity', '').upper() == 'SEVERE'
            ]

            if severe_interactions:
                lines.append("\n### Drug Interaction Warnings")
                for interaction in severe_interactions:
                    drug1 = interaction.get('drug1', 'Drug 1')
                    drug2 = interaction.get('drug2', 'Drug 2')
                    clinical_effect = interaction.get('clinical_effect', 'Unknown effect')
                    recommendation = interaction.get('recommendation', 'Review with pharmacist')
                    lines.append(f"- ⚠️ **{drug1} + {drug2}**: {clinical_effect}")
                    lines.append(f"  Recommendation: {recommendation}")

        return "\n".join(lines)

    def _generate_monitoring_section(self, monitoring_protocol: Dict) -> str:
        """Generate monitoring protocol section for MDT summary."""
        if not monitoring_protocol:
            return ""

        lines = ["## Monitoring Protocol"]

        regimen = monitoring_protocol.get('regimen', 'Unknown')
        frequency = monitoring_protocol.get('frequency', 'As indicated')
        lines.append(f"\n**Regimen**: {regimen}")
        lines.append(f"**Frequency**: {frequency}")

        # List tests to monitor
        tests = monitoring_protocol.get('tests_to_monitor', [])
        if tests:
            lines.append("\n**Labs to Monitor**:")
            for test in tests:
                if isinstance(test, str):
                    lines.append(f"- {test}")
                else:
                    test_name = test.get('loinc_name', test.get('name', 'Unknown'))
                    test_freq = test.get('frequency', frequency)
                    lines.append(f"- {test_name} ({test_freq})")

        # List dose adjustments if present
        dose_adjustments = monitoring_protocol.get('dose_adjustments', [])
        if dose_adjustments:
            lines.append("\n**Dose Adjustment Criteria**:")
            for adjustment in dose_adjustments:
                if isinstance(adjustment, str):
                    lines.append(f"- {adjustment}")
                else:
                    parameter = adjustment.get('parameter', 'Unknown')
                    threshold = adjustment.get('threshold', '')
                    action = adjustment.get('action', 'Review dosing')
                    lines.append(f"- {parameter} {threshold}: {action}")

        return "\n".join(lines)

    def _generate_trials_section(self, eligible_trials: List[Dict]) -> str:
        """Generate clinical trials section for MDT summary."""
        if not eligible_trials:
            return ""

        lines = ["## Eligible Clinical Trials"]

        # Sort by match score (descending)
        sorted_trials = sorted(
            eligible_trials,
            key=lambda t: t.get('match_score', t.get('eligibility_score', 0)),
            reverse=True
        )

        # Show top 3 trials
        for trial in sorted_trials[:3]:
            nct_id = trial.get('nct_id', 'Unknown')
            title = trial.get('title', trial.get('brief_title', 'Unknown trial'))
            phase = trial.get('phase', 'Unknown')
            score = trial.get('match_score', trial.get('eligibility_score', 0))
            lines.append(f"\n- **{title}**")
            lines.append(f"  NCT ID: {nct_id} | Phase: {phase} | Match Score: {score:.0f}/100")

        if len(sorted_trials) > 3:
            lines.append(f"\n{len(sorted_trials) - 3} additional trials available. Review full trial matching results for details.")

        return "\n".join(lines)

    def format_for_patient(
        self, 
        mdt_summary: MDTSummary
    ) -> str:
        """
        Format summary in patient-friendly language.
        Simplified explanation without technical terms.
        """
        lines = [
            f"Summary for Patient {mdt_summary.patient_id}",
            "=" * 40,
            "",
            mdt_summary.clinical_summary,
            "",
            "Recommended Treatment Options:",
            ""
        ]
        
        for rec in mdt_summary.formatted_recommendations:
            lines.append(f"  Option {rec['rank']}: {rec['treatment']}")
            lines.append(f"    Purpose: {rec['intent']}")
            lines.append("")
        
        lines.extend([
            "Important:",
            "  - These options will be discussed with your care team",
            "  - You can ask questions about any option",
            "  - Your preferences matter in choosing treatment",
            "",
            "Please discuss these options with your doctor."
        ])
        
        return "\n".join(lines)

    def format_for_audit(
        self, 
        mdt_summary: MDTSummary,
        inference_record: Optional[InferenceRecord] = None
    ) -> Dict[str, Any]:
        """
        Format for audit and compliance documentation.
        Includes full provenance and version information.
        """
        audit_record = {
            "document_type": "MDT_AUDIT_RECORD",
            "generated_by": self.name,
            "agent_version": self.version,
            "timestamp": mdt_summary.generated_at.isoformat(),
            "patient_id": mdt_summary.patient_id,
            "inference_id": mdt_summary.inference_id,
            "classification": {
                "scenario": mdt_summary.classification_scenario,
                "confidence": mdt_summary.scenario_confidence,
            },
            "recommendations_count": len(mdt_summary.formatted_recommendations),
            "guideline_references": mdt_summary.guideline_references,
            "snomed_mappings": mdt_summary.snomed_mappings,
        }
        
        if inference_record:
            audit_record["provenance"] = {
                "ontology_version": inference_record.ontology_version,
                "llm_model": inference_record.llm_model,
                "agent_chain": inference_record.agent_chain,
                "status": inference_record.status.value,
            }
        
        return audit_record
