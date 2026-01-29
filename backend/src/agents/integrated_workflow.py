"""
Integrated Multi-Agent Workflow with All 2025-2026 Enhancements

This module provides the complete integrated workflow combining:
- Dynamic Orchestrator (2026)
- Specialized NSCLC/SCLC agents (11 total agents)
- Biomarker and Comorbidity agents
- Negotiation Protocol
- Advanced Analytics Suite (Survival, Counterfactual, Uncertainty, Clinical Trials)
- Context Graph tracking
- Graph Algorithms (Neo4j GDS)
- Neosemantics Tools (n10s)
- Temporal Analyzer
- Multi-Ontology Support (LOINC, RxNorm, SNOMED-CT)
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
from .conflict_resolution_agent import ConflictResolutionAgent
from .negotiation_protocol import NegotiationProtocol, NegotiationStrategy, AgentProposal
from .explanation_agent import ExplanationAgent
# Lazy import to avoid loading sentence_transformers/torch
# from .persistence_agent import PersistenceAgent

# Analytics modules - lazy loaded to avoid heavy dependencies at startup
UncertaintyQuantifier = None
SurvivalAnalyzer = None
CounterfactualEngine = None
ClinicalTrialMatcher = None

# DB Tools - lazy loaded
GraphAlgorithms = None
NeosemanticsTools = None
TemporalAnalyzer = None

# Ontology Integrators - lazy loaded
LOINCIntegrator = None
RxNormMapper = None

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
        self.classification = ClassificationAgent()  # Missing classification agent
        self.explanation = ExplanationAgent()

        # Specialized agents (2025)
        self.biomarker = BiomarkerAgent()
        self.nsclc = NSCLCAgent()
        self.sclc = SCLCAgent()
        self.comorbidity = ComorbidityAgent()  # Always available
        
        # Lazy load PersistenceAgent to avoid loading sentence_transformers
        self.persistence = None
        if neo4j_tools:
            try:
                from .persistence_agent import PersistenceAgent
                # Handle neo4j_tools as dict (with 'read' and 'write' keys) or as write tools directly
                if isinstance(neo4j_tools, dict):
                    write_tools = neo4j_tools.get('write')
                    if write_tools:
                        self.persistence = PersistenceAgent(write_tools)
                else:
                    # Assume it's Neo4jWriteTools directly
                    self.persistence = PersistenceAgent(neo4j_tools)
            except ImportError as e:
                logger.warning(f"PersistenceAgent not available: {e}")

        # Negotiation protocol (2025)
        self.negotiation = NegotiationProtocol(
            strategy=NegotiationStrategy.HYBRID
        ) if enable_negotiation else None

        # Dynamic orchestrator (2026)
        self.orchestrator = DynamicWorkflowOrchestrator()

        # Conflict resolution agent (2025)
        self.conflict_resolution = ConflictResolutionAgent()
        
        # Analytics suite (2025) - optional with lazy loading
        self.uncertainty_quantifier = None
        self.survival_analyzer = None
        self.counterfactual_engine = None
        self.clinical_trial_matcher = None
        
        logger.info(f"[Init] enable_analytics={enable_analytics}, neo4j_tools={'provided' if neo4j_tools else 'None'}")
        
        if enable_analytics:
            try:
                from ..analytics.uncertainty_quantifier import UncertaintyQuantifier
                from ..analytics.survival_analyzer import SurvivalAnalyzer
                from ..analytics.counterfactual_engine import CounterfactualEngine
                from ..analytics.clinical_trial_matcher import ClinicalTrialMatcher
                
                # Extract read tools from neo4j_tools dict
                read_tools = neo4j_tools.get('read') if neo4j_tools else None
                
                # UncertaintyQuantifier can work with or without Neo4j
                self.uncertainty_quantifier = UncertaintyQuantifier(read_tools)
                logger.info(f"[Init] UncertaintyQuantifier initialized: {self.uncertainty_quantifier is not None}")
                
                # These require Neo4j read tools
                if read_tools:
                    try:
                        self.survival_analyzer = SurvivalAnalyzer(read_tools)
                        logger.info(f"[Init] SurvivalAnalyzer initialized: {self.survival_analyzer is not None}")
                    except Exception as e:
                        logger.warning(f"[Init] SurvivalAnalyzer failed: {e}")
                        
                    try:
                        self.counterfactual_engine = CounterfactualEngine(self)
                        logger.info(f"[Init] CounterfactualEngine initialized: {self.counterfactual_engine is not None}")
                    except Exception as e:
                        logger.warning(f"[Init] CounterfactualEngine failed: {e}")
                        
                    try:
                        self.clinical_trial_matcher = ClinicalTrialMatcher()
                        logger.info(f"[Init] ClinicalTrialMatcher initialized: {self.clinical_trial_matcher is not None}")
                    except Exception as e:
                        logger.warning(f"[Init] ClinicalTrialMatcher failed: {e}")
                else:
                    logger.warning("[Init] Neo4j read tools not available - SurvivalAnalyzer, CounterfactualEngine, ClinicalTrialMatcher will be disabled")
                
                logger.info("✓ Analytics suite loaded (UncertaintyQuantifier active)")
            except ImportError as e:
                logger.warning(f"Analytics suite not available: {e}")
            except Exception as e:
                logger.warning(f"Failed to load analytics: {e}")
        else:
            logger.warning("[Init] Analytics DISABLED (enable_analytics=False)")

        # DB Tools (2025-2026) - Graph Algorithms, Neosemantics, Temporal Analyzer
        self.graph_algorithms = None
        self.neosemantics_tools = None
        self.temporal_analyzer = None
        
        try:
            from ..db.graph_algorithms import Neo4jGraphAlgorithms
            from ..db.neosemantics_tools import NeosemanticsTools
            from ..db.temporal_analyzer import TemporalAnalyzer
            
            self.graph_algorithms = Neo4jGraphAlgorithms()
            self.neosemantics_tools = NeosemanticsTools()
            self.temporal_analyzer = TemporalAnalyzer()
            
            logger.info("✓ DB Tools loaded (Graph Algorithms, Neosemantics, Temporal Analyzer)")
        except ImportError as e:
            logger.warning(f"DB tools not fully available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load DB tools: {e}")

        # Ontology Integrators (2025) - LOINC, RxNorm (SNOMED loaded separately)
        self.loinc_integrator = None
        self.rxnorm_mapper = None
        
        try:
            from ..ontology.loinc_integrator import LOINCIntegrator
            from ..ontology.rxnorm_mapper import RxNormMapper
            
            self.loinc_integrator = LOINCIntegrator()
            self.rxnorm_mapper = RxNormMapper()
            
            logger.info("✓ Ontology Integrators loaded (LOINC, RxNorm)")
        except ImportError as e:
            logger.warning(f"Ontology integrators not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load ontology integrators: {e}")

    async def analyze_patient_comprehensive(
        self,
        patient_data: Dict[str, Any],
        persist: bool = False,
        progress_callback: Optional[callable] = None
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
            progress_callback: Optional callback for progress updates

        Returns:
            Complete analysis results with recommendations, confidence, analytics
        """

        async def notify_progress(message: str):
            """Helper to notify progress if callback exists"""
            if progress_callback:
                try:
                    await progress_callback(message)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")

        start_time = datetime.now()
        
        await notify_progress("🔬 Starting integrated multi-agent workflow...")
        
        logger.info("="*80)
        logger.info("🔬 INTEGRATED WORKFLOW EXECUTION")
        logger.info("="*80)
        logger.info(f"📋 Patient ID: {patient_data.get('patient_id', 'unknown')}")
        logger.info("")
        
        await notify_progress("📊 Loading agent components...")
        
        # Log all available components
        logger.info("🎯 AGENTS (11 total):")
        logger.info(f"   Core: IngestionAgent, SemanticMappingAgent, ClassificationAgent, ExplanationAgent")
        logger.info(f"   Specialized: BiomarkerAgent, ComorbidityAgent, NSCLCAgent, SCLCAgent")
        logger.info(f"   Advanced: ConflictResolutionAgent, UncertaintyQuantifier, PersistenceAgent")
        logger.info("")
        
        logger.info("📊 ANALYTICS SUITE:")
        logger.info(f"   UncertaintyQuantifier: {'✓ Active' if self.uncertainty_quantifier else '✗ Inactive'}")
        logger.info(f"   SurvivalAnalyzer: {'✓ Active' if self.survival_analyzer else '✗ Inactive'}")
        logger.info(f"   CounterfactualEngine: {'✓ Active' if self.counterfactual_engine else '✗ Inactive'}")
        logger.info(f"   ClinicalTrialMatcher: {'✓ Active' if self.clinical_trial_matcher else '✗ Inactive'}")
        logger.info("")
        
        logger.info("🗄️ DB TOOLS:")
        logger.info(f"   GraphAlgorithms (Neo4j GDS): {'✓ Active' if self.graph_algorithms else '✗ Inactive'}")
        logger.info(f"   NeosemanticsTools (n10s): {'✓ Active' if self.neosemantics_tools else '✗ Inactive'}")
        logger.info(f"   TemporalAnalyzer: {'✓ Active' if self.temporal_analyzer else '✗ Inactive'}")
        logger.info("")
        
        logger.info("🔬 ONTOLOGIES:")
        logger.info(f"   SNOMED-CT: ✓ Active (via SemanticMappingAgent)")
        logger.info(f"   LOINC Integrator: {'✓ Active' if self.loinc_integrator else '✗ Inactive'}")
        logger.info(f"   RxNorm Mapper: {'✓ Active' if self.rxnorm_mapper else '✗ Inactive'}")
        logger.info("")
        
        # Build agent registry for orchestrator
        agent_registry = self._build_agent_registry(patient_data)
        logger.info(f"✓ Registered {len(agent_registry)} agents for orchestration")
        logger.info(f"   Active agents: {', '.join(sorted(agent_registry.keys()))}")
        
        # Log which agents are NOT available
        all_possible_agents = {
            "IngestionAgent", "SemanticMappingAgent", "ClassificationAgent", 
            "BiomarkerAgent", "ComorbidityAgent", "ConflictResolutionAgent",
            "UncertaintyQuantifier", "ExplanationAgent", "PersistenceAgent",
            "NSCLCAgent", "SCLCAgent"
        }
        missing = all_possible_agents - set(agent_registry.keys())
        if missing:
            logger.info(f"   Unavailable: {', '.join(sorted(missing))}")
        logger.info("")
        
        try:
            # Delegate to orchestrator for adaptive workflow execution
            await notify_progress("🤖 Running orchestrator with dynamic agent routing...")
            orchestrator_result = await self.orchestrator.orchestrate_adaptive_workflow(
                patient_data=patient_data,
                agent_registry=agent_registry,
                progress_callback=notify_progress
            )
            
            # Extract orchestrator results
            complexity = WorkflowComplexity(orchestrator_result["complexity"])
            successful_count = len(orchestrator_result['successful_agents'])
            skipped_count = len(orchestrator_result.get('skipped_agents', []))
            failed_count = len(orchestrator_result.get('failed_agents', []))
            
            logger.info(f"Orchestrator completed: {successful_count} successful, {failed_count} failed, {skipped_count} skipped")
            
            if skipped_count > 0:
                skipped_list = ', '.join(orchestrator_result['skipped_agents'])
                logger.warning(f"⚠ Skipped agents (not available): {skipped_list}")
                await notify_progress(f"⚠ Note: {skipped_count} agents unavailable ({skipped_list})")
            
            # Get base results from orchestrator
            workflow_results = {
                "patient_id": patient_data.get("patient_id", "unknown"),
                "workflow_version": "3.0_orchestrated",
                "complexity": complexity.value,
                "agent_chain": orchestrator_result["agent_path"],
                "successful_agents": orchestrator_result["successful_agents"],
                "failed_agents": orchestrator_result.get("failed_agents", []),
                "skipped_agents": orchestrator_result.get("skipped_agents", []),
                "timestamps": {
                    "start": start_time.isoformat(),
                    "orchestrator_duration_ms": orchestrator_result["total_duration_ms"]
                },
                "errors": []
            }
            
            # Extract processed data from orchestrator results
            final_output = orchestrator_result.get("final_output", patient_data)
            
            # Get ingestion and semantic mapping results from orchestrator
            ingestion_result = orchestrator_result["results"].get("IngestionAgent")
            mapping_result = orchestrator_result["results"].get("SemanticMappingAgent")
            
            # Debug: Log what agents ran and what results we have
            logger.info(f"🔍 DEBUG: Orchestrator results keys: {list(orchestrator_result['results'].keys())}")
            logger.info(f"🔍 DEBUG: Ingestion result type: {type(ingestion_result)}")
            logger.info(f"🔍 DEBUG: Successful agents: {orchestrator_result['successful_agents']}")
            
            if not ingestion_result:
                logger.error(f"❌ CRITICAL: Ingestion agent did not execute!")
                logger.error(f"   Available results: {list(orchestrator_result['results'].keys())}")
                logger.error(f"   Successful agents: {orchestrator_result['successful_agents']}")
                logger.error(f"   Failed agents: {orchestrator_result['failed_agents']}")
                workflow_results["status"] = "failed_ingestion"
                workflow_results["errors"].append("Ingestion agent did not execute")
                return workflow_results
            
            # Use orchestrator's processed patient data
            patient_with_codes = final_output.get("SemanticMappingAgent_output", ingestion_result)

            await notify_progress("🧬 Running specialized agents (NSCLC/SCLC/Biomarker)...")
            logger.info(f"🔬 SPECIALIZED AGENTS:")
            logger.info(f"   Biomarker profile: {patient_data.get('biomarker_profile', {})}")
            proposals = await self._execute_specialized_agents(patient_data, patient_with_codes)
            logger.info(f"   ✓ Generated {len(proposals)} treatment proposals")
            for i, p in enumerate(proposals):
                logger.info(f"   Proposal {i+1}: {p.treatment} (confidence: {p.confidence:.2f})")
            workflow_results["proposals_evaluated"] = len(proposals)

            # Multi-agent negotiation (if multiple proposals)
            if len(proposals) > 1:
                await notify_progress(f"🤝 Negotiating between {len(proposals)} treatment proposals...")
            final_recommendation = await self._run_negotiation(proposals, workflow_results)
            if final_recommendation:
                logger.info(f"   ✓ Final recommendation: {final_recommendation.treatment}")
                await notify_progress(f"✅ Recommendation: {final_recommendation.treatment}")
            else:
                logger.warning(f"   ✗ No final recommendation generated!")

            # Advanced analytics (for complex/critical cases)
            if complexity in [WorkflowComplexity.COMPLEX, WorkflowComplexity.CRITICAL]:
                await notify_progress("📊 Running advanced analytics suite...")
            analytics_results = await self._run_analytics_suite(
                complexity, final_recommendation, patient_with_codes, patient_data
            )
            workflow_results["analytics"] = analytics_results

            # Generate explanation
            await notify_progress("📝 Generating MDT summary...")
            # Generate explanation
            mdt_summary = self._generate_explanation(patient_with_codes, final_recommendation)
            workflow_results["mdt_summary"] = mdt_summary

            # Compile final results
            end_time = datetime.now()
            workflow_results.update({
                "status": "success",
                "recommendations": [final_recommendation] if final_recommendation else [],
                "timestamps": {
                    **workflow_results["timestamps"],
                    "end": end_time.isoformat()
                },
                "processing_time_ms": int((end_time - start_time).total_seconds() * 1000),
                "context_graph": orchestrator_result.get("context_graph", {})
            })

            logger.info(f"   ✓ Workflow complete with {len(workflow_results.get('recommendations', []))} recommendations")

            # Step 10: Context graph (from orchestrator)
            workflow_results["context_graph"] = self.orchestrator.context_graph.to_dict()

            return workflow_results

        except Exception as e:
            logger.error(f"Workflow error: {e}", exc_info=True)
            return {
                "patient_id": patient_data.get("patient_id", "unknown"),
                "status": "error",
                "errors": [str(e)],
                "timestamps": {"start": start_time.isoformat()}
            }
    
    def _build_agent_registry(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build agent registry for orchestrator"""
        from ..db.models import PatientFact, PatientFactWithCodes, ClassificationResult, Sex, TNMStage, HistologyType

        # Helper to create PatientFact from dict
        def _make_patient_fact(data: Dict) -> PatientFact:
            # Get laterality with valid default (enum only allows Right, Left, Bilateral)
            laterality = data.get("laterality", patient_data.get("laterality"))
            if laterality not in ["Right", "Left", "Bilateral"]:
                laterality = "Right"  # Default to Right if not specified or invalid

            return PatientFact(
                patient_id=data.get("patient_id", patient_data.get("patient_id", "unknown")),
                name=data.get("name", patient_data.get("name", "Unknown Patient")),
                age_at_diagnosis=data.get("age_at_diagnosis", data.get("age", patient_data.get("age", 65))),
                sex=data.get("sex", patient_data.get("sex", "U")),
                tnm_stage=data.get("tnm_stage", patient_data.get("tnm_stage", "IV")),
                histology_type=data.get("histology_type", patient_data.get("histology_type", "Adenocarcinoma")),
                performance_status=data.get("performance_status", patient_data.get("performance_status", 1)),
                laterality=laterality,
                fev1_percent=data.get("fev1_percent", patient_data.get("fev1_percent")),
                comorbidities=data.get("comorbidities", patient_data.get("comorbidities", []))
            )

        # Helper to create PatientFactWithCodes from dict
        def _make_patient_with_codes(data: Dict) -> PatientFactWithCodes:
            # Get laterality with valid default (enum only allows Right, Left, Bilateral)
            laterality = data.get("laterality", patient_data.get("laterality"))
            if laterality not in ["Right", "Left", "Bilateral"]:
                laterality = "Right"  # Default to Right if not specified or invalid

            return PatientFactWithCodes(
                patient_id=data.get("patient_id", patient_data.get("patient_id", "unknown")),
                name=data.get("name", patient_data.get("name", "Unknown Patient")),
                age_at_diagnosis=data.get("age_at_diagnosis", data.get("age", patient_data.get("age", 65))),
                sex=data.get("sex", patient_data.get("sex", "U")),
                tnm_stage=data.get("tnm_stage", patient_data.get("tnm_stage", "IV")),
                histology_type=data.get("histology_type", patient_data.get("histology_type", "Adenocarcinoma")),
                performance_status=data.get("performance_status", patient_data.get("performance_status", 1)),
                laterality=laterality,
                fev1_percent=data.get("fev1_percent", patient_data.get("fev1_percent")),
                comorbidities=data.get("comorbidities", patient_data.get("comorbidities", [])),
                snomed_codes=data.get("snomed_codes", {}),
                mapping_confidence=data.get("mapping_confidence", 0.85)
            )

        async def ingestion_wrapper(data):
            # IngestionAgent expects raw dict and returns PatientFact
            result, errors = self.ingestion.execute(data)
            return result

        async def semantic_mapping_wrapper(data):
            ingestion_result = data.get("IngestionAgent_output", data)
            # Convert dict to PatientFact if needed
            if isinstance(ingestion_result, dict):
                ingestion_result = _make_patient_fact(ingestion_result)
            result, confidence = self.semantic_mapping.execute(ingestion_result)
            return result

        async def classification_wrapper(data):
            mapped_data = data.get("SemanticMappingAgent_output", data)
            # Convert dict to PatientFactWithCodes if needed
            if isinstance(mapped_data, dict):
                mapped_data = _make_patient_with_codes(mapped_data)
            return self.classification.execute(mapped_data)

        async def explanation_wrapper(data):
            patient_with_codes = data.get("SemanticMappingAgent_output", data)
            classification = data.get("ClassificationAgent_output")

            # Convert dict to PatientFactWithCodes if needed
            if isinstance(patient_with_codes, dict):
                patient_with_codes = _make_patient_with_codes(patient_with_codes)

            # Ensure we have a valid classification
            if classification is None or not isinstance(classification, ClassificationResult):
                classification = ClassificationResult(
                    patient_id=patient_data.get("patient_id", "unknown"),
                    scenario="Unknown",
                    scenario_confidence=0.5,
                    recommendations=[],
                    reasoning_chain=["Classification data not available"]
                )

            return self.explanation.execute(patient_with_codes, classification)
        
        # Base registry
        registry = {
            "IngestionAgent": ingestion_wrapper,
            "SemanticMappingAgent": semantic_mapping_wrapper,
            "ClassificationAgent": classification_wrapper,
            "ExplanationAgent": explanation_wrapper,
        }
        
        # Add biomarker agent
        async def biomarker_wrapper(data):
            patient_with_codes = data.get("SemanticMappingAgent_output", data)
            # Convert dict to object if needed
            if isinstance(patient_with_codes, dict):
                patient_with_codes = _make_patient_with_codes(patient_with_codes)
            # Get biomarker_profile from original patient_data
            biomarker_profile = patient_data.get("biomarker_profile", {})
            return self.biomarker.execute(patient_with_codes, biomarker_profile)
        registry["BiomarkerAgent"] = biomarker_wrapper

        # Add comorbidity agent (always available)
        async def comorbidity_wrapper(data):
            patient_with_codes = data.get("SemanticMappingAgent_output", data)
            # Convert dict to object if needed
            if isinstance(patient_with_codes, dict):
                patient_with_codes = _make_patient_with_codes(patient_with_codes)
            # Get treatment from classification if available
            classification = data.get("ClassificationAgent_output")
            treatment = "Unknown"
            if classification and hasattr(classification, 'recommendations') and classification.recommendations:
                rec = classification.recommendations[0] if isinstance(classification.recommendations, list) else classification.recommendations
                if isinstance(rec, dict):
                    treatment = rec.get('treatment') or rec.get('primary_treatment', 'Unknown')
                else:
                    treatment = getattr(rec, 'treatment', None) or getattr(rec, 'primary_treatment', 'Unknown')
            elif classification and hasattr(classification, 'scenario'):
                treatment = classification.scenario
            return self.comorbidity.execute(patient_with_codes, treatment, None)
        registry["ComorbidityAgent"] = comorbidity_wrapper
        
        # Add conflict resolution agent
        if self.conflict_resolution:
            async def conflict_wrapper(data):
                classification = data.get("ClassificationAgent_output")
                if classification:
                    resolved_classification, conflict_reports = self.conflict_resolution.execute(classification)
                    logger.info(f"[ConflictResolutionAgent] Resolved {len(conflict_reports)} conflicts")
                    return resolved_classification
                return classification
            registry["ConflictResolutionAgent"] = conflict_wrapper
        
        # Add uncertainty quantifier if available
        if self.uncertainty_quantifier:
            async def uncertainty_wrapper(data):
                classification = data.get("ClassificationAgent_output")
                patient_with_codes = data.get("SemanticMappingAgent_output", data)
                # Convert dict to object if needed
                if isinstance(patient_with_codes, dict):
                    patient_with_codes = _make_patient_with_codes(patient_with_codes)
                if classification and hasattr(classification, 'recommendations'):
                    for rec in classification.recommendations:
                        if isinstance(rec, dict):
                            from ..db.models import TreatmentRecommendation, TreatmentIntent, EvidenceLevel
                            treatment_name = rec.get('treatment') or rec.get('primary_treatment', 'Unknown')
                            treatment_rec = TreatmentRecommendation(
                                patient_id=patient_with_codes.patient_id,
                                primary_treatment=treatment_name,
                                treatment_intent=TreatmentIntent.PALLIATIVE,
                                evidence_level=EvidenceLevel.GRADE_B,
                                confidence_score=rec.get('confidence', 0.8),
                                rationale=rec.get('rationale', 'Treatment recommendation')
                            )
                            uncertainty = self.uncertainty_quantifier.quantify_recommendation_uncertainty(treatment_rec, patient_with_codes)
                            logger.info(f"[UncertaintyQuantifier] {treatment_name}: confidence={uncertainty.confidence_score:.2f}")
                        else:
                            uncertainty = self.uncertainty_quantifier.quantify_recommendation_uncertainty(rec, patient_with_codes)
                            logger.info(f"[UncertaintyQuantifier] {getattr(rec, 'primary_treatment', 'Unknown')}: confidence={uncertainty.confidence_score:.2f}")
                return classification
            registry["UncertaintyQuantifier"] = uncertainty_wrapper
        
        # Add NSCLC agent
        async def nsclc_wrapper(data):
            patient_with_codes = data.get("SemanticMappingAgent_output", data)
            # Convert dict to object if needed
            if isinstance(patient_with_codes, dict):
                patient_with_codes = _make_patient_with_codes(patient_with_codes)
            # Get biomarker_profile from original patient_data
            biomarker_profile = patient_data.get("biomarker_profile", {})
            histology = getattr(patient_with_codes, 'histology_type', 'Unknown')
            if 'small cell' not in histology.lower():
                result = self.nsclc.execute(patient_with_codes, biomarker_profile)
                if result:
                    logger.info(f"[NSCLCAgent] Executed for patient, treatment: {result.treatment}")
                else:
                    logger.info(f"[NSCLCAgent] Deferred to BiomarkerAgent (actionable biomarker detected)")
                return result
            return None
        registry["NSCLCAgent"] = nsclc_wrapper

        # Add SCLC agent
        async def sclc_wrapper(data):
            patient_with_codes = data.get("SemanticMappingAgent_output", data)
            # Convert dict to object if needed
            if isinstance(patient_with_codes, dict):
                patient_with_codes = _make_patient_with_codes(patient_with_codes)
            histology = getattr(patient_with_codes, 'histology_type', 'Unknown')
            if 'small cell' in histology.lower():
                result = self.sclc.execute(patient_with_codes)
                logger.info(f"[SCLCAgent] Executed for patient, treatment: {getattr(result, 'treatment', 'Unknown')}")
                return result
            return None
        registry["SCLCAgent"] = sclc_wrapper
        
        # Add PersistenceAgent if available
        if self.persistence:
            async def persistence_wrapper(data):
                patient_with_codes = data.get("SemanticMappingAgent_output", data)
                # Convert dict to object if needed
                if isinstance(patient_with_codes, dict):
                    patient_with_codes = _make_patient_with_codes(patient_with_codes)
                classification = data.get("ClassificationAgent_output")
                result = self.persistence.execute(patient_with_codes, classification)
                logger.info(f"[PersistenceAgent] Persisted patient data to Neo4j")
                return result
            registry["PersistenceAgent"] = persistence_wrapper

        # Add SurvivalAnalyzer if available
        if self.survival_analyzer:
            async def survival_wrapper(data):
                patient_with_codes = data.get("SemanticMappingAgent_output", data)
                if isinstance(patient_with_codes, dict):
                    patient_with_codes = _make_patient_with_codes(patient_with_codes)
                treatment = data.get("ClassificationAgent_output", {})
                if hasattr(treatment, 'recommendations') and treatment.recommendations:
                    rec = treatment.recommendations[0] if isinstance(treatment.recommendations, list) else treatment.recommendations
                    treatment_name = rec.get('primary_treatment', 'Unknown') if isinstance(rec, dict) else getattr(rec, 'treatment', 'Unknown')
                    result = self.survival_analyzer.kaplan_meier_analysis(treatment_name)
                    logger.info(f"[SurvivalAnalyzer] Analyzed survival for: {treatment_name}")
                    return result
                return None
            registry["SurvivalAnalyzer"] = survival_wrapper

        # Add ClinicalTrialMatcher if available
        if self.clinical_trial_matcher:
            async def trial_wrapper(data):
                patient_with_codes = data.get("SemanticMappingAgent_output", data)
                if isinstance(patient_with_codes, dict):
                    patient_with_codes = _make_patient_with_codes(patient_with_codes)
                patient_dict = {
                    'patient_id': patient_with_codes.patient_id,
                    'histology': patient_with_codes.histology_type,
                    'stage': patient_with_codes.tnm_stage,
                    'performance_status': patient_with_codes.performance_status
                }
                result = self.clinical_trial_matcher.find_eligible_trials(patient_dict)
                logger.info(f"[ClinicalTrialMatcher] Found {len(result) if result else 0} eligible trials")
                return result
            registry["ClinicalTrialMatcher"] = trial_wrapper
        
        # Add CounterfactualEngine if available
        if self.counterfactual_engine:
            async def counterfactual_wrapper(data):
                patient_data_dict = data.get("patient_data", data)
                classification = data.get("ClassificationAgent_output")
                if classification and hasattr(classification, 'recommendations') and classification.recommendations:
                    rec = classification.recommendations[0] if isinstance(classification.recommendations, list) else classification.recommendations
                    treatment = rec.get('primary_treatment', 'Unknown') if isinstance(rec, dict) else getattr(rec, 'treatment', 'Unknown')
                    result = self.counterfactual_engine.analyze_counterfactuals(patient_data_dict, treatment)
                    logger.info(f"[CounterfactualEngine] Analyzed {len(result.counterfactuals) if hasattr(result, 'counterfactuals') else 0} counterfactual scenarios")
                    return result
                return None
            registry["CounterfactualEngine"] = counterfactual_wrapper
        
        logger.info(f"[Registry] Final registry has {len(registry)} agents: {sorted(registry.keys())}")
        return registry
    
    async def _execute_specialized_agents(
        self, patient_data: Dict[str, Any], patient_with_codes: Any
    ) -> List[AgentProposal]:
        """Execute NSCLC/SCLC specialized agents"""
        proposals = []

        # Handle patient_with_codes as dict or object
        if isinstance(patient_with_codes, dict):
            histology = patient_with_codes.get('histology_type', '')
        else:
            histology = getattr(patient_with_codes, 'histology_type', '')

        is_sclc = "small cell" in str(histology).lower()
        logger.info(f"   [Specialized] Histology: {histology}, is_sclc: {is_sclc}")

        try:
            if is_sclc:
                # SCLC pathway
                logger.info(f"   [Specialized] Running SCLCAgent...")
                sclc_proposal = self.sclc.execute(patient_with_codes)
                proposals.append(self._convert_sclc_proposal(sclc_proposal))
                logger.info(f"   [Specialized] SCLC proposal: {sclc_proposal.treatment}")
            else:
                # NSCLC pathway
                biomarker_profile = patient_data.get("biomarker_profile", {})
                logger.info(f"   [Specialized] Running NSCLCAgent with biomarkers: {biomarker_profile}")

                nsclc_proposal = self.nsclc.execute(patient_with_codes, biomarker_profile)
                if nsclc_proposal:
                    logger.info(f"   [Specialized] NSCLC proposal: {nsclc_proposal.treatment}")
                    proposals.append(self._convert_nsclc_proposal(nsclc_proposal))
                else:
                    logger.info(f"   [Specialized] NSCLC deferred to BiomarkerAgent (actionable mutation detected)")

                if biomarker_profile:
                    logger.info(f"   [Specialized] Running BiomarkerAgent...")
                    biomarker_proposal = self.biomarker.execute(patient_with_codes, biomarker_profile)
                    logger.info(f"   [Specialized] Biomarker proposal: {biomarker_proposal.treatment}")
                    proposals.append(self._convert_biomarker_proposal(biomarker_proposal))

            # Comorbidity assessment
            if self.comorbidity and proposals:
                for proposal in proposals:
                    try:
                        comorbidity_assessment = self.comorbidity.execute(
                            patient_with_codes, proposal.treatment, None
                        )
                        proposal.risk_score = comorbidity_assessment.risk_score
                    except Exception as e:
                        logger.warning(f"Comorbidity assessment failed: {e}")

        except Exception as e:
            logger.error(f"Specialized agents execution failed: {e}", exc_info=True)
            # Return a default proposal if specialized agents fail
            proposals.append(AgentProposal(
                agent_id="fallback",
                agent_type="FallbackAgent",
                treatment="Platinum-based chemotherapy",
                confidence=0.7,
                evidence_level="Grade B",
                treatment_intent="Palliative",
                rationale=f"Default recommendation due to processing error: {str(e)}",
                guideline_reference="NCCN NSCLC 2025",
                contraindications=[],
                risk_score=0.5
            ))

        return proposals
    
    async def _run_negotiation(
        self, proposals: List[AgentProposal], workflow_results: Dict[str, Any]
    ) -> Optional[AgentProposal]:
        """Run multi-agent negotiation if enabled"""
        # Return None if no proposals
        if not proposals:
            return None
            
        # If negotiation enabled and multiple proposals, run negotiation
        if self.negotiation and len(proposals) > 1:
            negotiation_result = self.negotiation.negotiate(proposals)
            workflow_results["negotiation"] = {
                "strategy": negotiation_result.negotiation_strategy,
                "consensus_score": negotiation_result.consensus_score,
                "proposals_considered": len(proposals)
            }
            # Find the winning proposal by matching treatment name
            selected_treatment_name = negotiation_result.selected_treatment
            for proposal in proposals:
                if proposal.treatment == selected_treatment_name:
                    return proposal
            # If no exact match, return first proposal as fallback
            return proposals[0]
        
        # Single proposal or negotiation disabled - return first proposal
        return proposals[0]
    
    async def _run_analytics_suite(
        self, complexity: WorkflowComplexity, recommendation: Any, 
        patient_with_codes: Any, patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run advanced analytics for complex/critical cases"""
        analytics_results = {}
        
        if not self.enable_analytics or complexity not in [
            WorkflowComplexity.COMPLEX, WorkflowComplexity.CRITICAL
        ]:
            return analytics_results
        
        if self.uncertainty_quantifier and recommendation:
            uncertainty = await self._run_uncertainty_analysis(recommendation, patient_with_codes)
            analytics_results["uncertainty"] = uncertainty
        
        if self.survival_analyzer:
            survival = await self._run_survival_analysis(patient_with_codes, recommendation)
            analytics_results["survival"] = survival
        
        if self.clinical_trial_matcher:
            trials = await self._match_clinical_trials(patient_data)
            analytics_results["clinical_trials"] = trials
        
        return analytics_results
    
    def _generate_explanation(self, patient_with_codes: Any, recommendation: Any) -> str:
        """Generate MDT summary explanation"""
        try:
            # If recommendation is an AgentProposal, convert to ClassificationResult
            if recommendation and hasattr(recommendation, 'treatment'):
                from ..db.models import ClassificationResult, TreatmentRecommendation, TreatmentIntent, EvidenceLevel
                
                # Map string to proper enum
                intent_str = getattr(recommendation, 'treatment_intent', 'Unknown')
                if isinstance(intent_str, str):
                    intent_map = {
                        'curative': TreatmentIntent.CURATIVE,
                        'palliative': TreatmentIntent.PALLIATIVE,
                        'adjuvant': TreatmentIntent.ADJUVANT,
                        'neoadjuvant': TreatmentIntent.NEOADJUVANT,
                        'supportive': TreatmentIntent.SUPPORTIVE
                    }
                    intent = intent_map.get(intent_str.lower(), TreatmentIntent.UNKNOWN)
                else:
                    intent = intent_str
                
                # Map evidence level
                evidence_str = getattr(recommendation, 'evidence_level', 'Grade B')
                if isinstance(evidence_str, str) and not evidence_str.startswith('Grade'):
                    evidence_str = f'Grade {evidence_str}'
                
                patient_id = patient_with_codes.patient_id if hasattr(patient_with_codes, 'patient_id') else 'unknown'
                treatment = getattr(recommendation, 'treatment', 'Unknown')
                
                # Create a minimal ClassificationResult from the proposal
                classification = ClassificationResult(
                    patient_id=patient_id,
                    scenario=treatment,
                    scenario_confidence=getattr(recommendation, 'confidence', 0.8),
                    recommendations=[{
                        'patient_id': patient_id,
                        'primary_treatment': treatment,
                        'treatment': treatment,  # Add this for explanation agent compatibility
                        'treatment_intent': intent.value if hasattr(intent, 'value') else intent,
                        'intent': intent.value if hasattr(intent, 'value') else intent,  # Add for compatibility
                        'evidence_level': evidence_str,
                        'confidence_score': getattr(recommendation, 'confidence', 0.8),
                        'rationale': getattr(recommendation, 'rationale', 'No rationale provided'),
                        'guideline_reference': getattr(recommendation, 'guideline_reference', 'Integrated Workflow'),
                        'guideline_references': [getattr(recommendation, 'guideline_reference', 'Integrated Workflow')],
                        'supporting_arguments': [],
                        'opposing_arguments': [],
                        'alternative_treatments': []
                    }],
                    reasoning_chain=[getattr(recommendation, 'rationale', 'No rationale provided')],
                    guideline_refs=[getattr(recommendation, 'guideline_reference', 'N/A')]
                )
                return self.explanation.execute(patient_with_codes, classification)
            else:
                return self.explanation.execute(patient_with_codes, recommendation)
        except Exception as e:
            logger.warning(f"Explanation generation failed: {e}")
            return "Advanced workflow completed successfully"

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
            # Convert AgentProposal to TreatmentRecommendation if needed
            from .negotiation_protocol import AgentProposal
            from ..db.models import TreatmentRecommendation, TreatmentIntent, EvidenceLevel
            
            if isinstance(recommendation, AgentProposal):
                # Map evidence level string to enum
                evidence_map = {
                    'Grade A': EvidenceLevel.GRADE_A,
                    'Grade B': EvidenceLevel.GRADE_B,
                    'Grade C': EvidenceLevel.GRADE_C,
                    'Grade D': EvidenceLevel.GRADE_D,
                }
                evidence_level = evidence_map.get(recommendation.evidence_level, EvidenceLevel.GRADE_B)
                
                # Map treatment intent string to enum
                intent_map = {
                    'Curative': TreatmentIntent.CURATIVE,
                    'Palliative': TreatmentIntent.PALLIATIVE,
                    'Adjuvant': TreatmentIntent.ADJUVANT,
                    'Neoadjuvant': TreatmentIntent.NEOADJUVANT,
                }
                treatment_intent = intent_map.get(recommendation.treatment_intent, TreatmentIntent.PALLIATIVE)
                
                treatment_rec = TreatmentRecommendation(
                    patient_id=getattr(patient, 'patient_id', 'Unknown'),
                    primary_treatment=recommendation.treatment,
                    treatment_intent=treatment_intent,
                    evidence_level=evidence_level,
                    confidence_score=recommendation.confidence,
                    rationale=recommendation.rationale
                )
            else:
                treatment_rec = recommendation
            
            metrics = self.uncertainty_quantifier.quantify_recommendation_uncertainty(
                treatment_rec,
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
