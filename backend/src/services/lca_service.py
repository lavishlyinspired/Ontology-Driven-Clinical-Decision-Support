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
from src.ontology.lucada_ontology import LUCADAOntology
from src.ontology.guideline_rules import GuidelineRuleEngine
from src.db.neo4j_schema import LUCADAGraphDB, Neo4jConfig
from src.db.vector_store import LUCADAVectorStore

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
        neo4j_config: Optional[Neo4jConfig] = None
    ):
        """
        Initialize the LCA service.

        Args:
            ontology_path: Path to LUCADA ontology file
            use_neo4j: Enable Neo4j graph database
            use_vector_store: Enable vector store for semantic search
            neo4j_config: Optional Neo4j configuration
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

        # Initialize LangGraph workflow
        logger.info("Creating AI agent workflow...")
        self.workflow = create_lca_workflow()
        logger.info("✓ LangGraph workflow ready")

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
        use_ai_workflow: bool = True
    ) -> PatientDecisionSupport:
        """
        Process a patient through the full decision support pipeline.

        Args:
            patient_data: Patient clinical data
            use_ai_workflow: Whether to run AI agent workflow (takes ~20s)

        Returns:
            Complete decision support with recommendations
        """
        patient_id = patient_data.get("patient_id") or str(uuid.uuid4())[:8].upper()
        patient_data["patient_id"] = patient_id

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing Patient: {patient_id}")
        logger.info(f"{'=' * 80}")

        # Step 1: Store in Neo4j (if available)
        if self.graph_db and self.graph_db.driver:
            logger.info("Storing in Neo4j...")
            self.graph_db.create_patient(patient_data)
            logger.info("✓ Patient stored in graph database")

        # Step 2: Run rule-based classification
        logger.info("Classifying with guideline rules...")
        ontology_recommendations = self.rule_engine.classify_patient(patient_data)
        logger.info(f"✓ Found {len(ontology_recommendations)} applicable guidelines")

        # Step 3: Find similar patients (if Neo4j available)
        similar_patients = []
        if self.graph_db and self.graph_db.driver:
            logger.info("Finding similar patients...")
            similar_patients = self.graph_db.find_similar_patients(patient_id, limit=5)
            logger.info(f"✓ Found {len(similar_patients)} similar patients")

        # Step 4: Semantic guideline search (if vector store available)
        semantic_guidelines = []
        if self.vector_store:
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

        # Step 5: Run AI agent workflow (optional, takes time)
        explanation = ""
        arguments = []

        if use_ai_workflow:
            logger.info("Running AI agent workflow (this takes ~20 seconds)...")

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

            except Exception as e:
                logger.error(f"AI workflow failed: {e}")
                explanation = f"AI workflow error: {str(e)}"
        else:
            logger.info("Skipping AI workflow (use_ai_workflow=False)")

        # Step 6: Store recommendations in Neo4j
        if self.graph_db and self.graph_db.driver:
            self.graph_db.store_recommendation(patient_id, ontology_recommendations)

        # Step 7: Compile results
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

        result = PatientDecisionSupport(
            patient_id=patient_id,
            timestamp=datetime.now(),
            patient_scenarios=[r.get("rule_name", r.get("rule_id", "")) for r in ontology_recommendations],
            recommendations=recommendations,
            mdt_summary=explanation,
            similar_patients=similar_patients,
            semantic_guidelines=semantic_guidelines
        )

        logger.info(f"{'=' * 80}")
        logger.info(f"✓ Patient {patient_id} processing complete")
        logger.info(f"{'=' * 80}\n")

        return result

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
