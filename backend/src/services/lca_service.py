"""
Main Service Orchestration for Lung Cancer Assistant
Coordinates all system components: Ontology, Agents, Database, Vector Store
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
import uuid
import logging
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.lca_agents import create_lca_workflow, PatientState
from src.agents.integrated_workflow import IntegratedLCAWorkflow
from src.agents.dynamic_orchestrator import DynamicWorkflowOrchestrator, WorkflowComplexity
from src.ontology.lucada_ontology import LUCADAOntology
from src.ontology.guideline_rules import GuidelineRuleEngine
from src.db.neo4j_schema import LUCADAGraphDB, Neo4jConfig
# Lazy import for vector store to avoid loading PyTorch unless needed
# from src.db.vector_store import LUCADAVectorStore
from src.db.provenance_tracker import ProvenanceTracker, get_provenance_tracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TreatmentRecommendation:
    """Treatment recommendation with full details"""
    treatment_type: str
    rule_id: str
    rule_source: str
    evidence_level: str
    treatment_intent: str
    survival_benefit: Optional[str]
    contraindications: List[str]
    priority: int
    confidence_score: float = 0.8


@dataclass
class PatientDecisionSupport:
    """Complete decision support output"""
    patient_id: str
    timestamp: datetime
    patient_scenarios: List[str]
    recommendations: List[TreatmentRecommendation]
    mdt_summary: str
    similar_patients: List[Dict[str, Any]]
    semantic_guidelines: List[Dict[str, Any]]
    
    # Enhanced metadata
    workflow_type: str = "basic"  # "basic", "6-agent", "integrated"
    complexity_level: Optional[str] = None
    provenance_record_id: Optional[str] = None
    execution_time_ms: Optional[int] = None


class LungCancerAssistantService:
    """
    Main service coordinating all LCA components.

    Integrates:
    - LUCADA OWL Ontology
    - Guideline Rule Engine
    - LangGraph Agent Workflow
    - Neo4j Graph Database
    - Neo4j Vector Store (embeddings)
    """

    def __init__(
        self,
        ontology_path: str = "lucada_ontology.owl",
        use_neo4j: bool = False,
        use_vector_store: bool = True,
        neo4j_config: Optional[Neo4jConfig] = None,
        enable_advanced_workflow: bool = True,
        enable_provenance: bool = True
    ):
        """
        Initialize the LCA service.

        Args:
            ontology_path: Path to LUCADA ontology file
            use_neo4j: Enable Neo4j graph database
            use_vector_store: Enable vector store for semantic search
            neo4j_config: Optional Neo4j configuration
            enable_advanced_workflow: Enable complexity-based routing to advanced workflow
            enable_provenance: Enable comprehensive provenance tracking
        """
        logger.info("Initializing Lung Cancer Assistant Service...")

        # Initialize ontology
        logger.info("Creating LUCADA ontology...")
        self.ontology = LUCADAOntology()
        self.onto = self.ontology.create()
        logger.info(f"✓ Ontology created with {len(list(self.onto.classes()))} classes")

        # Initialize guideline engine
        logger.info("Loading clinical guidelines...")
        self.rule_engine = GuidelineRuleEngine(self.ontology)
        logger.info(f"✓ Loaded {len(self.rule_engine.rules)} guideline rules")

        # Initialize basic LangGraph workflow
        logger.info("Creating AI agent workflow...")
        self.workflow = create_lca_workflow()
        logger.info("✓ LangGraph workflow ready")
        
        # Initialize advanced workflow components
        self.enable_advanced_workflow = enable_advanced_workflow
        self.orchestrator = None
        self.integrated_workflow = None
        
        if enable_advanced_workflow:
            logger.info("Initializing advanced workflow components...")
            # Initialize Neo4j tools for integrated workflow
            neo4j_tools = None
            
            # Try to create Neo4j tools from config or from existing graph_db
            if use_neo4j:
                try:
                    from src.db.neo4j_tools import Neo4jReadTools, Neo4jWriteTools
                    import os
                    # Use neo4j_config if provided, otherwise use environment variables
                    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
                    user = os.getenv('NEO4J_USER', 'neo4j')
                    password = os.getenv('NEO4J_PASSWORD', '123456789')
                    database = os.getenv('NEO4J_DATABASE', 'neo4j')

                    if neo4j_config:
                        uri = neo4j_config.get('uri', uri)
                        user = neo4j_config.get('user', user)
                        password = neo4j_config.get('password', password)
                        database = neo4j_config.get('database', database)
                    
                    neo4j_tools = {
                        'read': Neo4jReadTools(uri=uri, user=user, password=password, database=database),
                        'write': Neo4jWriteTools(uri=uri, user=user, password=password, database=database)
                    }
                    logger.info("✓ Neo4j tools initialized for integrated workflow")
                except Exception as e:
                    logger.warning(f"Neo4j tools initialization failed: {e}")
            
            self.orchestrator = DynamicWorkflowOrchestrator()
            self.integrated_workflow = IntegratedLCAWorkflow(
                neo4j_tools=neo4j_tools,
                enable_analytics=True,
                enable_negotiation=True
            )
            logger.info("✓ Advanced workflow ready")
        
        # Initialize provenance tracking
        self.enable_provenance = enable_provenance
        self.provenance_tracker = None
        if enable_provenance:
            logger.info("Initializing provenance tracking...")
            self.provenance_tracker = get_provenance_tracker()
            logger.info("✓ Provenance tracker ready")

        # Initialize Neo4j (optional)
        self.graph_db = None
        if use_neo4j:
            try:
                logger.info("Connecting to Neo4j...")
                self.graph_db = LUCADAGraphDB(neo4j_config)
                if self.graph_db.driver:
                    self.graph_db.setup_schema()
                    logger.info("✓ Neo4j graph database connected")
                else:
                    logger.warning("Neo4j not available")
            except Exception as e:
                logger.warning(f"Neo4j initialization failed: {e}")

        # Initialize vector store (optional)
        self.vector_store = None
        if use_vector_store:
            try:
                logger.info("Initializing vector store...")
                # Lazy import to avoid loading PyTorch unless needed
                from src.db.vector_store import LUCADAVectorStore
                self.vector_store = LUCADAVectorStore()

                # Add guidelines to vector store if empty
                if self.vector_store._get_document_count() == 0:
                    guidelines = self.rule_engine.get_all_rules()
                    guidelines_data = [
                        {
                            "rule_id": g.rule_id,
                            "name": g.name,
                            "source": g.source,
                            "description": g.description,
                            "recommended_treatment": g.recommended_treatment,
                            "evidence_level": g.evidence_level,
                            "treatment_intent": g.treatment_intent
                        }
                        for g in guidelines
                    ]
                    self.vector_store.add_guidelines(guidelines_data)

                logger.info("✓ Vector store initialized")
            except Exception as e:
                logger.warning(f"Vector store initialization failed: {e}")

        logger.info("=" * 80)
        logger.info("✓ LCA Service Ready")
        logger.info("=" * 80)

    async def process_patient(
        self,
        patient_data: Dict[str, Any],
        use_ai_workflow: bool = True,
        force_advanced: bool = True,
        progress_callback: Optional[callable] = None
    ) -> PatientDecisionSupport:
        """
        Process a patient through the full decision support pipeline.
        
        Automatically routes to advanced workflow for complex cases.

        Args:
            patient_data: Patient clinical data
            use_ai_workflow: Whether to run AI agent workflow (takes ~20s)
            force_advanced: Force use of advanced integrated workflow

        Returns:
            Complete decision support with recommendations and provenance
        """
        start_time = datetime.utcnow()
        patient_id = patient_data.get("patient_id") or str(uuid.uuid4())[:8].upper()
        patient_data["patient_id"] = patient_id

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing Patient: {patient_id}")
        logger.info(f"{'=' * 80}")
        
        # Start provenance tracking
        provenance_session_id = None
        if self.enable_provenance and self.provenance_tracker:
            provenance_session_id = self.provenance_tracker.start_session(
                patient_id=patient_id,
                workflow_type="auto",  # Will be determined by orchestrator
                complexity_level=None
            )
            self.provenance_tracker.track_ontology_version("LUCADA", "1.0.0")
            self.provenance_tracker.track_ontology_version("SNOMED-CT", "2025-01-17")
        
        # Route: use advanced workflow ONLY if explicitly requested, otherwise use basic
        if force_advanced and self.enable_advanced_workflow and self.integrated_workflow:
            return await self._execute_integrated_workflow(
                patient_data, 
                start_time, 
                provenance_session_id
            )
        else:
            return await self._execute_basic_workflow(
                patient_data, 
                use_ai_workflow, 
                start_time,
                provenance_session_id,
                progress_callback
            )
    
    async def _execute_basic_workflow(
        self,
        patient_data: Dict[str, Any],
        use_ai_workflow: bool,
        start_time: datetime,
        provenance_session_id: Optional[str],
        progress_callback: Optional[callable] = None
    ) -> PatientDecisionSupport:
        """Execute patient processing with basic LangGraph workflow"""
        patient_id = patient_data["patient_id"]
        complexity_level = "simple"  # Basic workflow handles simple cases
        
        logger.info("→ Using BASIC workflow")
        if progress_callback:
            await progress_callback("🔧 Initializing basic workflow...")
        
        # Track data ingestion
        if self.provenance_tracker and provenance_session_id:
            self.provenance_tracker.track_data_ingestion("user_input", patient_data)


        # Step 1: Store in Neo4j (if available)
        if self.graph_db and self.graph_db.driver:
            if progress_callback:
                await progress_callback("💾 Storing patient in database...")
            logger.info("Storing in Neo4j...")
            self.graph_db.create_patient(patient_data)
            logger.info("✓ Patient stored in graph database")

        # Normalize biomarker data format for rule engine
        if progress_callback:
            await progress_callback("🧬 Normalizing biomarker data...")
        normalized_patient_data = self._normalize_biomarker_data(patient_data)
        
        logger.info(f"Original patient data: {patient_data}")
        logger.info(f"Normalized patient data: {normalized_patient_data}")

        # Step 2: Run rule-based classification
        if progress_callback:
            await progress_callback(f"📋 Classifying with {len(self.rule_engine.rules)} guideline rules...")
        logger.info("Classifying with guideline rules...")
        activity_id = None
        if self.provenance_tracker:
            activity_id = self.provenance_tracker.track_agent_execution(
                agent_name="GuidelineRuleEngine",
                agent_version="1.0.0",
                input_entity_ids=[],
                parameters={"rules_count": len(self.rule_engine.rules)}
            )
        
        ontology_recommendations = self.rule_engine.classify_patient(normalized_patient_data)
        logger.info(f"✓ Found {len(ontology_recommendations)} applicable guidelines")
        if ontology_recommendations:
            for rec in ontology_recommendations:
                logger.info(f"  - {rec.get('rule_id')}: {rec.get('recommended_treatment')}")
        
        if progress_callback:
            await progress_callback(f"✅ Matched {len(ontology_recommendations)} treatment guidelines")
        
        if self.provenance_tracker and activity_id:
            self.provenance_tracker.track_agent_completion(activity_id, "classification_result")

        # Step 3: Find similar patients (if Neo4j available)
        similar_patients = []
        if self.graph_db and self.graph_db.driver:
            if progress_callback:
                await progress_callback("🔍 Searching for similar patient cases...")
            logger.info("Finding similar patients...")
            similar_patients = self.graph_db.find_similar_patients(patient_id, limit=5)
            logger.info(f"✓ Found {len(similar_patients)} similar patients")
            if progress_callback:
                await progress_callback(f"✅ Found {len(similar_patients)} similar cases")

        # Step 4: Semantic guideline search (if vector store available)
        semantic_guidelines = []
        if self.vector_store:
            if progress_callback:
                await progress_callback("📚 Searching semantic guidelines...")
            logger.info("Searching semantic guidelines...")
            patient_description = f"""
            Patient with {patient_data.get('tnm_stage')}
            {patient_data.get('histology_type')}
            lung cancer, performance status {patient_data.get('performance_status')}
            """
            semantic_guidelines = self.vector_store.search_guidelines(
                patient_description.strip(),
                n_results=3
            )
            logger.info(f"✓ Found {len(semantic_guidelines)} semantic matches")
            if progress_callback:
                await progress_callback(f"✅ Retrieved {len(semantic_guidelines)} guidelines")

        # Step 5: Run AI agent workflow (optional, takes time)
        explanation = ""
        arguments = []

        if use_ai_workflow:
            if progress_callback:
                await progress_callback("🤖 Running AI argumentation (20-30s)...")
            logger.info("Running AI agent workflow (this takes ~20 seconds)...")
            
            ai_activity_id = None
            if self.provenance_tracker:
                ai_activity_id = self.provenance_tracker.track_agent_execution(
                    agent_name="LangGraphWorkflow",
                    agent_version="1.0.0",
                    input_entity_ids=["classification_result"],
                    parameters={"workflow_type": "basic_langgraph"}
                )

            initial_state: PatientState = {
                "patient_id": patient_id,
                "patient_data": patient_data,
                "applicable_rules": ontology_recommendations,
                "treatment_recommendations": ontology_recommendations,
                "arguments": [],
                "explanation": "",
                "messages": []
            }

            try:
                final_state = await asyncio.to_thread(
                    self.workflow.invoke,
                    initial_state
                )

                explanation = final_state.get("explanation", "")
                arguments = final_state.get("arguments", [])
                logger.info("✓ AI workflow completed")
                
                if progress_callback:
                    await progress_callback(f"✅ Generated arguments for {len(arguments)} treatment options")
                
                if self.provenance_tracker and ai_activity_id:
                    self.provenance_tracker.track_agent_completion(ai_activity_id, "mdt_summary")

            except Exception as e:
                logger.error(f"AI workflow failed: {e}")
                explanation = f"AI workflow error: {str(e)}"
                if self.provenance_tracker and ai_activity_id:
                    self.provenance_tracker.track_agent_completion(ai_activity_id, "error", "failed", str(e))
        else:
            logger.info("Skipping AI workflow (use_ai_workflow=False)")

        # Step 6: Store recommendations in Neo4j
        if self.graph_db and self.graph_db.driver:
            if progress_callback:
                await progress_callback("💾 Saving recommendations to database...")
            self.graph_db.store_recommendation(patient_id, ontology_recommendations)

        # Step 7: Compile results
        if progress_callback:
            await progress_callback("📋 Compiling treatment recommendations...")
        
        recommendations = [
            TreatmentRecommendation(
                treatment_type=r.get("recommended_treatment", "Unknown"),
                rule_id=r.get("rule_id", ""),
                rule_source=r.get("source", "NICE Guidelines"),
                evidence_level=r.get("evidence_level", "Grade C"),
                treatment_intent=r.get("treatment_intent", "Unknown"),
                survival_benefit=r.get("survival_benefit"),
                contraindications=r.get("contraindications", []),
                priority=r.get("priority", 0),
                confidence_score=0.8
            )
            for r in ontology_recommendations
        ]

        # Generate MDT summary if we didn't run the full AI workflow
        if not explanation and recommendations:
            # Simple summary from rule engine
            top_rec = recommendations[0] if recommendations else None
            if top_rec:
                explanation = f"""
                **MDT Summary for Patient {patient_id}**

                **Classification**: {patient_data.get('tnm_stage')} {patient_data.get('histology_type')}

                **Primary Recommendation**: {top_rec.treatment_type}
                - Evidence Level: {top_rec.evidence_level}
                - Source: {top_rec.rule_source}
                - Intent: {top_rec.treatment_intent}

                **Total Guidelines Matched**: {len(recommendations)}

                **Note**: This is a rule-based summary. For detailed clinical argumentation,
                enable the AI workflow.
                """

        # Compile patient scenarios
        scenarios = [
            f"{r.get('rule_id')}: {r.get('recommended_treatment')}"
            for r in ontology_recommendations
        ]
        
        # End provenance session
        provenance_record_id = None
        if self.provenance_tracker and provenance_session_id:
            record = self.provenance_tracker.end_session()
            provenance_record_id = record.record_id
        
        # Calculate execution time
        execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        logger.info(f"{'=' * 80}")
        logger.info(f"✓ Patient {patient_id} processing complete ({execution_time_ms}ms)")
        logger.info(f"{'=' * 80}\n")

        return PatientDecisionSupport(
            patient_id=patient_id,
            timestamp=datetime.utcnow(),
            patient_scenarios=scenarios,
            recommendations=recommendations,
            mdt_summary=explanation.strip(),
            similar_patients=similar_patients,
            semantic_guidelines=semantic_guidelines,
            workflow_type="basic",
            complexity_level=complexity_level,
            provenance_record_id=provenance_record_id,
            execution_time_ms=execution_time_ms
        )
    
    async def _execute_integrated_workflow(
        self,
        patient_data: Dict[str, Any],
        start_time: datetime,
        provenance_session_id: Optional[str]
    ) -> PatientDecisionSupport:
        """Execute patient processing with integrated advanced workflow (orchestrator handles routing)"""
        patient_id = patient_data["patient_id"]
        
        logger.info("=" * 80)
        logger.info("ADVANCED INTEGRATED WORKFLOW (Orchestrated)")
        logger.info("=" * 80)
        
        logger.info("")
        logger.info("📊 SYSTEM COMPONENTS ACTIVE:")
        logger.info(f"   🔬 Ontology: LUCADA v1.0.0 + SNOMED-CT 2025-01-17")
        logger.info(f"   📝 Provenance Tracker: {'✓ Active' if self.provenance_tracker else '✗ Inactive'}")
        logger.info(f"   🗄️  Neo4j Graph DB: {'✓ Connected' if self.graph_db else '✗ Disconnected'}")
        logger.info(f"   🔍 Vector Store: {'✓ Active' if self.vector_store else '✗ Inactive'}")
        logger.info(f"   🧬 Biomarker Analysis: ✓ Active")
        logger.info(f"   📊 Analytics Suite: ✓ Active")
        logger.info("")
        
        logger.info("🛠️  TOOLS & ALGORITHMS:")
        logger.info(f"   • Guideline Matching: Semantic similarity (all-MiniLM-L6-v2)")
        logger.info(f"   • Patient Similarity: Graph-based (TNM + PS + Histology)")
        logger.info(f"   • Conflict Resolution: Evidence hierarchy (Grade A > B > C)")
        logger.info(f"   • Uncertainty Quantification: Bayesian + Historical outcomes")
        logger.info("")
        
        # Track data ingestion
        if self.provenance_tracker and provenance_session_id:
            logger.info("📋 PROVENANCE TRACKING:")
            logger.info(f"   Session ID: {provenance_session_id}")
            self.provenance_tracker.track_data_ingestion("user_input", patient_data)
            logger.info(f"   ✓ Data ingestion tracked")
            logger.info("")
        
        try:
            # Run integrated workflow (orchestrator handles complexity assessment internally)
            result = await self.integrated_workflow.analyze_patient_comprehensive(
                patient_data=patient_data,
                persist=bool(self.graph_db and self.graph_db.driver)
            )
            
            # Extract complexity from orchestrator's result
            complexity_level = result.get("complexity", "unknown")
            logger.info(f"✓ Advanced workflow completed: {result.get('status')} (complexity: {complexity_level})")
            
            # Update provenance with actual complexity
            if self.provenance_tracker and provenance_session_id:
                self.provenance_tracker.track_agent_execution(
                    agent_name="IntegratedLCAWorkflow",
                    agent_version="3.0_orchestrated",
                    input_entity_ids=[],
                    parameters={
                        "complexity_level": complexity_level,
                        "agent_chain": result.get("agent_chain", [])
                    }
                )
            
            # Convert integrated workflow result to PatientDecisionSupport
            recommendations_data = result.get("recommendations", [])
            recommendations = []
            
            for rec in recommendations_data:
                if isinstance(rec, dict):
                    recommendations.append(TreatmentRecommendation(
                        treatment_type=rec.get("treatment", "Unknown"),
                        rule_id=rec.get("agent_id", ""),
                        rule_source=rec.get("guideline_reference", "NCCN 2025"),
                        evidence_level=rec.get("evidence_level", "Grade A"),
                        treatment_intent=rec.get("treatment_intent", "Unknown"),
                        survival_benefit=rec.get("expected_benefit"),
                        contraindications=rec.get("contraindications", []),
                        priority=90,
                        confidence_score=rec.get("confidence", 0.85)
                    ))
                else:
                    # Handle AgentProposal objects
                    recommendations.append(TreatmentRecommendation(
                        treatment_type=getattr(rec, 'treatment', 'Unknown'),
                        rule_id=getattr(rec, 'agent_id', ''),
                        rule_source=getattr(rec, 'guideline_reference', 'NCCN 2025'),
                        evidence_level=getattr(rec, 'evidence_level', 'Grade A'),
                        treatment_intent=getattr(rec, 'treatment_intent', 'Unknown'),
                        survival_benefit=getattr(rec, 'expected_benefit', None),
                        contraindications=getattr(rec, 'contraindications', []),
                        priority=90,
                        confidence_score=getattr(rec, 'confidence', 0.85)
                    ))
            
            # Get MDT summary
            mdt_summary = result.get("mdt_summary", "Advanced workflow completed successfully")
            
            # Compile patient scenarios from agent chain
            scenarios = result.get("agent_chain", [])
            
            # Find similar patients using graph database instead of vector store
            similar_patients = []
            if self.graph_db and self.graph_db.driver:
                try:
                    logger.info("📊 GRAPH ALGORITHM: Finding similar patients...")
                    logger.info(f"   Algorithm: Neo4j pattern matching (TNM stage + Performance status ±1 + Histology match)")
                    similar_patients = self.graph_db.find_similar_patients(patient_id, limit=3)
                    logger.info(f"   ✓ Found {len(similar_patients)} similar patients")
                except Exception as e:
                    logger.warning(f"Similar patient search failed: {e}")
            
            # Get semantic guidelines
            semantic_guidelines = self.rule_engine.get_semantic_guidelines(patient_data)
            
        except Exception as e:
            logger.error(f"Advanced workflow failed: {e}", exc_info=True)
            
            # Fallback to basic workflow
            logger.info("Falling back to basic workflow...")
            return await self._execute_basic_workflow(
                patient_data, True, start_time, provenance_session_id
            )
        
        # End provenance session
        provenance_record_id = None
        if self.provenance_tracker and provenance_session_id:
            record = self.provenance_tracker.end_session()
            provenance_record_id = record.record_id
        
        # Calculate execution time
        execution_time_ms = result.get("processing_time_ms", int((datetime.utcnow() - start_time).total_seconds() * 1000))
        
        logger.info(f"{'=' * 80}")
        logger.info(f"✓ Patient {patient_id} advanced workflow complete ({execution_time_ms}ms)")
        logger.info(f"{'=' * 80}\n")
        
        return PatientDecisionSupport(
            patient_id=patient_id,
            timestamp=datetime.utcnow(),
            patient_scenarios=scenarios,
            recommendations=recommendations,
            mdt_summary=mdt_summary,
            similar_patients=[],  # Integrated workflow handles this internally
            semantic_guidelines=[],
            workflow_type="integrated",
            complexity_level=complexity_level,
            provenance_record_id=provenance_record_id,
            execution_time_ms=execution_time_ms
        )
    
    def get_provenance_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve provenance record for audit/compliance"""
        if not self.provenance_tracker:
            return None
        return self.provenance_tracker.export_record(record_id)
    
    def query_patient_provenance(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get all provenance records for a patient"""
        if not self.provenance_tracker:
            return []
        records = self.provenance_tracker.query_by_patient(patient_id)
        return [r.to_dict() for r in records]
    
    async def assess_complexity(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess patient complexity and return routing recommendation"""
        if not self.orchestrator:
            return {
                "complexity": "unknown",
                "recommended_workflow": "basic",
                "reason": "Advanced workflow not enabled"
            }
        
        complexity = self.orchestrator.assess_complexity(patient_data)
        use_advanced = complexity in [WorkflowComplexity.COMPLEX, WorkflowComplexity.CRITICAL]
        
        return {
            "complexity": complexity.value,
            "recommended_workflow": "integrated" if use_advanced else "basic",
            "reason": f"Complexity level: {complexity.value}",
            "factors": {
                "stage": patient_data.get("tnm_stage"),
                "performance_status": patient_data.get("performance_status"),
                "comorbidities_count": len(patient_data.get("comorbidities", [])),
                "biomarkers_available": bool(patient_data.get("biomarker_profile"))
            }
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            "ontology": {
                "classes": len(list(self.onto.classes())),
                "object_properties": len(list(self.onto.object_properties())),
                "data_properties": len(list(self.onto.data_properties())),
                "individuals": len(list(self.onto.individuals()))
            },
            "guidelines": {
                "total_rules": len(self.rule_engine.rules),
                "rules": [
                    {
                        "rule_id": r.rule_id,
                        "treatment": r.recommended_treatment,
                        "evidence": r.evidence_level
                    }
                    for r in self.rule_engine.get_all_rules()
                ]
            },
            "workflows": {
                "basic": "enabled",
                "advanced_integrated": "enabled" if self.enable_advanced_workflow else "disabled",
                "provenance_tracking": "enabled" if self.enable_provenance else "disabled"
            },
            "neo4j": {
                "enabled": self.graph_db is not None and self.graph_db.driver is not None,
                "status": "connected" if (self.graph_db and self.graph_db.driver) else "disabled"
            },
            "vector_store": {
                "enabled": self.vector_store is not None,
                "documents": self.vector_store.collection.count() if self.vector_store else 0
            }
        }


        return stats

    def _normalize_biomarker_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize nested biomarker_profile to flat fields for rule engine"""
        normalized = patient_data.copy()
        
        biomarker_profile = patient_data.get("biomarker_profile", {})
        
        # Map biomarker_profile fields to expected flat fields
        if "egfr_mutation" in biomarker_profile:
            normalized["egfr_status"] = "positive" if biomarker_profile["egfr_mutation"] else "negative"
            if biomarker_profile.get("egfr_mutation_type"):
                normalized["egfr_mutation_type"] = biomarker_profile["egfr_mutation_type"]
        
        if "alk_rearrangement" in biomarker_profile:
            normalized["alk_status"] = "positive" if biomarker_profile["alk_rearrangement"] else "negative"
        
        if "pdl1_tps" in biomarker_profile:
            normalized["pdl1_score"] = biomarker_profile["pdl1_tps"]
        
        return normalized

    def close(self):
        """Cleanup resources"""
        if self.graph_db:
            self.graph_db.close()
        logger.info("✓ LCA Service shut down")


# Command-line interface
async def main():
    """Main CLI for testing"""
    import json

    print("\n" + "=" * 80)
    print("LUNG CANCER ASSISTANT SERVICE - TEST")
    print("=" * 80)

    # Initialize service
    service = LungCancerAssistantService(
        use_neo4j=False,  # Set to True if Neo4j is running
        use_vector_store=True
    )

    # Test patient
    test_patient = {
        "patient_id": "SERVICE_TEST_001",
        "name": "John Doe",
        "age": 68,
        "sex": "M",
        "tnm_stage": "IIIA",
        "histology_type": "Adenocarcinoma",
        "performance_status": 1,
        "fev1_percent": 65.0,
        "comorbidities": ["COPD", "Hypertension"],
        "laterality": "Right"
    }

    print(f"\nTest Patient: {test_patient['name']}")
    print(f"  Stage: {test_patient['tnm_stage']}")
    print(f"  Histology: {test_patient['histology_type']}")
    print(f"  PS: WHO {test_patient['performance_status']}")

    # Process patient
    result = await service.process_patient(
        test_patient,
        use_ai_workflow=False  # Set to True to run AI agents (takes ~20s)
    )

    # Display results
    print(f"\n{'=' * 80}")
    print("DECISION SUPPORT RESULTS")
    print("=" * 80)

    print(f"\nPatient ID: {result.patient_id}")
    print(f"Timestamp: {result.timestamp}")

    print(f"\nApplicable Scenarios: {len(result.patient_scenarios)}")
    for scenario in result.patient_scenarios:
        print(f"  - {scenario}")

    print(f"\nTreatment Recommendations: {len(result.recommendations)}")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"\n{i}. {rec.treatment_type} (Priority: {rec.priority})")
        print(f"   Rule: {rec.rule_id} - {rec.evidence_level}")
        print(f"   Intent: {rec.treatment_intent}")
        print(f"   Survival: {rec.survival_benefit}")

    if result.mdt_summary:
        print(f"\n{'=' * 80}")
        print("MDT SUMMARY")
        print("=" * 80)
        print(result.mdt_summary)

    # System stats
    stats = service.get_system_stats()
    print(f"\n{'=' * 80}")
    print("SYSTEM STATISTICS")
    print("=" * 80)
    print(json.dumps(stats, indent=2))

    service.close()


if __name__ == "__main__":
    asyncio.run(main())
