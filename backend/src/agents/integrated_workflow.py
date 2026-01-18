"""
Integrated Multi-Agent Workflow with All 2025-2026 Enhancements

This module provides the complete integrated workflow combining:
- Dynamic Orchestrator (2026)
- Specialized NSCLC/SCLC agents
- Biomarker and Comorbidity agents
- Negotiation Protocol
- Advanced Analytics Suite
- Context Graph tracking
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .dynamic_orchestrator import (
    DynamicWorkflowOrchestrator,
    WorkflowComplexity,
    AgentExecution
)
from .ingestion_agent import IngestionAgent
from .semantic_mapping_agent import SemanticMappingAgent
from .classification_agent import ClassificationAgent
from .biomarker_agent import BiomarkerAgent
from .nsclc_agent import NSCLCAgent
from .sclc_agent import SCLCAgent
from .comorbidity_agent import ComorbidityAgent
from .negotiation_protocol import NegotiationProtocol, NegotiationStrategy, AgentProposal
from .conflict_resolution_agent import ConflictResolutionAgent
from .explanation_agent import ExplanationAgent

try:
    from ..analytics.uncertainty_quantifier import UncertaintyQuantifier
    from ..analytics.survival_analyzer import SurvivalAnalyzer
    from ..analytics.counterfactual_engine import CounterfactualEngine
    from ..analytics.clinical_trial_matcher import ClinicalTrialMatcher
except ImportError:
    # Analytics modules are optional
    UncertaintyQuantifier = None
    SurvivalAnalyzer = None
    CounterfactualEngine = None
    ClinicalTrialMatcher = None

logger = logging.getLogger(__name__)


class IntegratedLCAWorkflow:
    """
    Complete Integrated Workflow for Lung Cancer Assistant

    Features:
    - Adaptive complexity-based routing
    - Specialized NSCLC/SCLC agents
    - Multi-agent negotiation
    - Self-corrective reasoning
    - Advanced analytics suite
    - Dynamic context graphs
    - Full audit trails
    """

    def __init__(self, neo4j_tools=None, enable_analytics=True, enable_negotiation=True):
        """
        Initialize integrated workflow

        Args:
            neo4j_tools: Neo4j database connection (optional)
            enable_analytics: Enable advanced analytics modules
            enable_negotiation: Enable multi-agent negotiation
        """
        self.neo4j_tools = neo4j_tools
        self.enable_analytics = enable_analytics
        self.enable_negotiation = enable_negotiation

        # Core agents
        self.ingestion = IngestionAgent()
        self.semantic_mapping = SemanticMappingAgent()
        self.classification = ClassificationAgent()
        self.conflict_resolution = ConflictResolutionAgent()
        self.explanation = ExplanationAgent()

        # Specialized agents (2025)
        self.biomarker = BiomarkerAgent()
        self.nsclc = NSCLCAgent()
        self.sclc = SCLCAgent()
        self.comorbidity = ComorbidityAgent() if neo4j_tools else None

        # Negotiation protocol (2025)
        self.negotiation = NegotiationProtocol(
            strategy=NegotiationStrategy.HYBRID
        ) if enable_negotiation else None

        # Dynamic orchestrator (2026)
        self.orchestrator = DynamicWorkflowOrchestrator()

        # Analytics suite (2025) - optional
        if enable_analytics:
            self.uncertainty_quantifier = UncertaintyQuantifier(neo4j_tools) if UncertaintyQuantifier and neo4j_tools else None
            self.survival_analyzer = SurvivalAnalyzer(neo4j_tools) if SurvivalAnalyzer and neo4j_tools else None
            self.counterfactual_engine = CounterfactualEngine(self) if CounterfactualEngine else None
            self.clinical_trial_matcher = ClinicalTrialMatcher() if ClinicalTrialMatcher else None
        else:
            self.uncertainty_quantifier = None
            self.survival_analyzer = None
            self.counterfactual_engine = None
            self.clinical_trial_matcher = None

    async def analyze_patient_comprehensive(
        self,
        patient_data: Dict[str, Any],
        persist: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive patient analysis with all enhancements

        Workflow:
        1. Assess complexity
        2. Ingestion and semantic mapping
        3. Parallel specialized agent execution
        4. Multi-agent negotiation (if enabled)
        5. Advanced analytics (if enabled)
        6. Generate explanation with uncertainty metrics
        7. Optionally persist to Neo4j

        Args:
            patient_data: Patient clinical data
            persist: Whether to save results to Neo4j

        Returns:
            Complete analysis results with recommendations, confidence, analytics
        """

        start_time = datetime.now()
        workflow_results = {
            "patient_id": patient_data.get("patient_id", "unknown"),
            "workflow_version": "3.0_integrated",
            "timestamps": {"start": start_time.isoformat()},
            "agent_chain": [],
            "errors": []
        }

        try:
            # Step 1: Assess complexity
            complexity = self.orchestrator.assess_complexity(patient_data)
            workflow_results["complexity"] = complexity.value
            logger.info(f"Patient complexity assessed: {complexity.value}")

            # Step 2: Ingestion
            patient_fact, ingestion_errors = self.ingestion.execute(patient_data)
            workflow_results["agent_chain"].append("IngestionAgent")

            if not patient_fact:
                workflow_results["errors"].extend(ingestion_errors)
                workflow_results["status"] = "failed_ingestion"
                return workflow_results

            # Step 3: Semantic Mapping
            patient_with_codes, mapping_confidence = self.semantic_mapping.execute(patient_fact)
            workflow_results["agent_chain"].append("SemanticMappingAgent")
            workflow_results["mapping_confidence"] = mapping_confidence

            # Step 4: Determine if SCLC or NSCLC
            is_sclc = "small cell" in patient_with_codes.histology_type.lower()

            # Step 5: Parallel specialized agent execution
            proposals = []

            if is_sclc:
                # SCLC pathway
                sclc_proposal = self.sclc.execute(patient_with_codes)
                proposals.append(self._convert_sclc_proposal(sclc_proposal))
                workflow_results["agent_chain"].append("SCLCAgent")
            else:
                # NSCLC pathway
                biomarker_profile = patient_data.get("biomarker_profile", {})

                # Execute NSCLC agent
                nsclc_proposal = self.nsclc.execute(patient_with_codes, biomarker_profile)
                proposals.append(self._convert_nsclc_proposal(nsclc_proposal))
                workflow_results["agent_chain"].append("NSCLCAgent")

                # Execute Biomarker agent if profile available
                if biomarker_profile:
                    biomarker_proposal = self.biomarker.execute(
                        patient_with_codes,
                        biomarker_profile
                    )
                    proposals.append(self._convert_biomarker_proposal(biomarker_proposal))
                    workflow_results["agent_chain"].append("BiomarkerAgent")

            # Execute Comorbidity agent if available
            if self.comorbidity:
                comorbidity_profile = patient_data.get("comorbidities", [])
                if proposals:  # Assess safety of proposed treatments
                    for proposal in proposals:
                        comorbidity_assessment = self.comorbidity.execute(
                            patient_with_codes,
                            proposal.treatment,
                            comorbidity_profile
                        )
                        proposal.risk_score = comorbidity_assessment.risk_score
                workflow_results["agent_chain"].append("ComorbidityAgent")

            # Step 6: Negotiation (if multiple proposals and enabled)
            if self.negotiation and len(proposals) > 1:
                negotiation_result = self.negotiation.negotiate(proposals)
                final_recommendation = negotiation_result.selected_treatment
                workflow_results["negotiation"] = {
                    "strategy": negotiation_result.negotiation_strategy,
                    "consensus_score": negotiation_result.consensus_score,
                    "proposals_considered": len(proposals)
                }
                workflow_results["agent_chain"].append("NegotiationProtocol")
            else:
                final_recommendation = proposals[0] if proposals else None

            # Step 7: Advanced analytics (if enabled and complexity warrants it)
            analytics_results = {}

            if self.enable_analytics and complexity in [
                WorkflowComplexity.COMPLEX,
                WorkflowComplexity.CRITICAL
            ]:
                # Uncertainty quantification
                if self.uncertainty_quantifier and final_recommendation:
                    uncertainty = await self._run_uncertainty_analysis(
                        final_recommendation,
                        patient_with_codes
                    )
                    analytics_results["uncertainty"] = uncertainty
                    workflow_results["agent_chain"].append("UncertaintyQuantifier")

                # Survival analysis
                if self.survival_analyzer:
                    survival = await self._run_survival_analysis(
                        patient_with_codes,
                        final_recommendation
                    )
                    analytics_results["survival"] = survival
                    workflow_results["agent_chain"].append("SurvivalAnalyzer")

                # Clinical trial matching
                if self.clinical_trial_matcher:
                    trials = await self._match_clinical_trials(patient_data)
                    analytics_results["clinical_trials"] = trials
                    workflow_results["agent_chain"].append("ClinicalTrialMatcher")

            workflow_results["analytics"] = analytics_results

            # Step 8: Generate explanation
            mdt_summary = self.explanation.execute(
                patient_with_codes,
                final_recommendation
            )
            workflow_results["agent_chain"].append("ExplanationAgent")
            workflow_results["mdt_summary"] = mdt_summary

            # Step 9: Compile final results
            end_time = datetime.now()
            workflow_results.update({
                "status": "success",
                "recommendations": [final_recommendation] if final_recommendation else [],
                "proposals_evaluated": len(proposals),
                "timestamps": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "processing_time_ms": int((end_time - start_time).total_seconds() * 1000)
            })

            # Step 10: Context graph (from orchestrator)
            workflow_results["context_graph"] = self.orchestrator.context_graph.to_dict()

            return workflow_results

        except Exception as e:
            logger.error(f"Workflow error: {e}", exc_info=True)
            workflow_results["status"] = "error"
            workflow_results["errors"].append(str(e))
            return workflow_results

    def _convert_nsclc_proposal(self, nsclc_proposal) -> AgentProposal:
        """Convert NSCLC proposal to standard AgentProposal"""
        return AgentProposal(
            agent_id="nsclc_agent",
            agent_type="NSCLCAgent",
            treatment=nsclc_proposal.treatment,
            confidence=nsclc_proposal.confidence,
            evidence_level=nsclc_proposal.evidence_level,
            treatment_intent=nsclc_proposal.treatment_intent,
            rationale=nsclc_proposal.rationale,
            guideline_reference=getattr(nsclc_proposal, 'guideline_reference', 'NCCN NSCLC 2025'),
            contraindications=getattr(nsclc_proposal, 'contraindications', []),
            risk_score=nsclc_proposal.risk_score
        )

    def _convert_sclc_proposal(self, sclc_proposal) -> AgentProposal:
        """Convert SCLC proposal to standard AgentProposal"""
        return AgentProposal(
            agent_id="sclc_agent",
            agent_type="SCLCAgent",
            treatment=sclc_proposal.treatment,
            confidence=sclc_proposal.confidence,
            evidence_level=sclc_proposal.evidence_level,
            treatment_intent=sclc_proposal.treatment_intent,
            rationale=sclc_proposal.rationale,
            guideline_reference=getattr(sclc_proposal, 'guideline_reference', 'NCCN SCLC 2025'),
            contraindications=getattr(sclc_proposal, 'contraindications', []),
            risk_score=sclc_proposal.risk_score
        )

    def _convert_biomarker_proposal(self, biomarker_result) -> AgentProposal:
        """Convert Biomarker agent result to AgentProposal"""
        # BiomarkerAgent already returns AgentProposal, but verify structure
        if isinstance(biomarker_result, AgentProposal):
            return biomarker_result
        
        # Fallback for legacy format
        return AgentProposal(
            agent_id="biomarker_agent",
            agent_type="BiomarkerAgent",
            treatment=getattr(biomarker_result, 'treatment', str(biomarker_result)),
            confidence=getattr(biomarker_result, 'confidence', 0.95),
            evidence_level=getattr(biomarker_result, 'evidence_level', "Grade A"),
            treatment_intent=getattr(biomarker_result, 'treatment_intent', "targeted"),
            rationale=getattr(biomarker_result, 'rationale', "Biomarker-driven precision medicine"),
            guideline_reference=getattr(biomarker_result, 'guideline_reference', 'NCCN Biomarker Guidelines 2025'),
            contraindications=getattr(biomarker_result, 'contraindications', []),
            risk_score=getattr(biomarker_result, 'risk_score', 0.2),
            expected_benefit=getattr(biomarker_result, 'expected_benefit', None)
        )
    
    def _extract_biomarkers(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract biomarker information from patient data"""
        biomarkers = {}
        
        # Check for biomarker_profile in patient data
        if 'biomarker_profile' in patient_data:
            biomarkers = patient_data['biomarker_profile']
        
        # Also extract individual biomarker fields
        biomarker_fields = [
            'egfr_mutation', 'egfr_mutation_type', 'alk_rearrangement',
            'ros1_rearrangement', 'braf_mutation', 'met_exon14',
            'ret_rearrangement', 'kras_mutation', 'pdl1_tps', 
            'tmb_score', 'her2_mutation', 'ntrk_fusion'
        ]
        
        for field in biomarker_fields:
            if field in patient_data and patient_data[field]:
                biomarkers[field] = patient_data[field]
        
        return biomarkers

    async def _run_uncertainty_analysis(self, recommendation, patient):
        """Run uncertainty quantification"""
        try:
            metrics = self.uncertainty_quantifier.quantify_recommendation_uncertainty(
                recommendation,
                patient,
                similar_patients=[]
            )
            return {
                "confidence": metrics.confidence_score,
                "epistemic_uncertainty": metrics.epistemic_uncertainty,
                "aleatoric_uncertainty": metrics.aleatoric_uncertainty
            }
        except Exception as e:
            logger.warning(f"Uncertainty analysis failed: {e}")
            return None

    async def _run_survival_analysis(self, patient, recommendation):
        """Run survival analysis"""
        try:
            km_result = self.survival_analyzer.kaplan_meier_analysis(
                treatment=recommendation.treatment if hasattr(recommendation, 'treatment') else "Chemotherapy",
                stage=patient.tnm_stage,
                histology=patient.histology_type
            )
            return {
                "median_survival_days": km_result.get("median_survival_days"),
                "1_year_survival": km_result.get("survival_probabilities", {}).get("1_year")
            }
        except Exception as e:
            logger.warning(f"Survival analysis failed: {e}")
            return None

    async def _match_clinical_trials(self, patient_data):
        """Match clinical trials"""
        try:
            matches = self.clinical_trial_matcher.find_eligible_trials(
                patient=patient_data,
                max_results=5
            )
            return [
                {
                    "nct_id": m.trial.nct_id,
                    "title": m.trial.title,
                    "match_score": m.match_score
                }
                for m in matches[:5]
            ]
        except Exception as e:
            logger.warning(f"Clinical trial matching failed: {e}")
            return []


# Convenience function for backward compatibility
async def analyze_patient_integrated(patient_data: Dict[str, Any], persist: bool = False):
    """
    Convenience function for integrated workflow

    Args:
        patient_data: Patient clinical data
        persist: Whether to persist to Neo4j

    Returns:
        Complete analysis results
    """
    workflow = IntegratedLCAWorkflow()
    return await workflow.analyze_patient_comprehensive(patient_data, persist)
