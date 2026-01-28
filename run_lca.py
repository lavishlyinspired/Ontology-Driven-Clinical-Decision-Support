"""
Quick Run Script for Lung Cancer Assistant
Comprehensive interface integrating all system components:
- 14 Agents (Core + Specialized + Analytics)
- Ontologies (LUCADA, SNOMED-CT, LOINC, RxNorm)
- Digital Twin Engine
- MCP Server
- Advanced Analytics (Survival, Uncertainty, Counterfactual, Clinical Trials)
"""

import asyncio
import sys
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")

# Configure logging BEFORE importing anything else
def setup_logging():
    """Configure logging to suppress verbose Neo4j messages"""
    # Suppress Neo4j INFO messages about constraints/indexes
    logging.getLogger('neo4j.notifications').setLevel(logging.ERROR)
    logging.getLogger('neo4j').setLevel(logging.WARNING)

    # Keep application logs at INFO
    logging.getLogger('src').setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )

setup_logging()

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from src.services.lca_service import LungCancerAssistantService


async def display_ontology_info():
    """Display ontology integration information"""
    print("\n" + "="*80)
    print("ONTOLOGY INTEGRATION STATUS")
    print("="*80)

    try:
        from src.ontology.loinc_integrator import LOINCIntegrator
        from src.ontology.rxnorm_mapper import RxNormMapper
        from src.ontology.snomed_loader import SNOMEDLoader

        loinc = LOINCIntegrator()
        rxnorm = RxNormMapper()
        snomed = SNOMEDLoader()

        print(f"\n  SNOMED-CT:")
        print(f"    - Status: Loaded")
        print(f"    - Lung cancer codes: {len(snomed.LUNG_CANCER_CODES)} concepts")
        print(f"    - Histology mappings: {len(snomed.HISTOLOGY_CODES)} types")
        print(f"    - Stage mappings: {len(snomed.TNM_STAGE_CODES)} stages")

        print(f"\n  LOINC (Laboratory Tests):")
        print(f"    - Status: {'Available' if loinc else 'Not loaded'}")
        print(f"    - Local mappings: {len(getattr(loinc, 'local_mappings', {}))} tests")
        print(f"    - Example: EGFR Mutation → LOINC:21639-1")

        print(f"\n  RxNorm (Medications):")
        print(f"    - Status: {'Available' if rxnorm else 'Not loaded'}")
        print(f"    - Drug mappings: {len(getattr(rxnorm, 'drug_mappings', {}))} drugs")
        print(f"    - Example: Pembrolizumab → RxNorm:1547220")

    except ImportError as e:
        print(f"  Warning: Some ontology modules not available: {e}")


async def display_analytics_demo(service, patient_data):
    """Display comprehensive analytics for a patient"""
    print("\n" + "="*80)
    print("ADVANCED ANALYTICS DEMONSTRATION")
    print("="*80)

    # Try to get analytics results
    try:
        if service.integrated_workflow:
            wf = service.integrated_workflow

            # 1. Uncertainty Quantification
            if wf.uncertainty_quantifier:
                print("\n  [1] UNCERTAINTY QUANTIFICATION")
                print("      Method: Bayesian + Historical Outcomes")
                print("      Factors: Similar patient outcomes, data completeness, evidence strength")
                print("      Output: Confidence intervals, epistemic/aleatoric uncertainty")

            # 2. Survival Analysis
            if wf.survival_analyzer:
                print("\n  [2] SURVIVAL ANALYSIS")
                print("      Method: Kaplan-Meier + Cox Proportional Hazards")
                print("      Output: Median survival, 1-year/5-year survival probabilities")
                # Demo analysis
                result = wf.survival_analyzer.kaplan_meier_analysis(
                    treatment="Platinum-based chemotherapy",
                    stage=patient_data.get('tnm_stage', 'IV'),
                    histology=patient_data.get('histology_type', 'Adenocarcinoma')
                )
                if result:
                    print(f"      Demo: Median survival = {result.get('median_survival_days', 'N/A')} days")

            # 3. Clinical Trial Matching
            if wf.clinical_trial_matcher:
                print("\n  [3] CLINICAL TRIAL MATCHING")
                print("      Source: ClinicalTrials.gov API")
                print("      Matching: Histology, Stage, Biomarkers, Performance Status")

            # 4. Counterfactual Analysis
            if wf.counterfactual_engine:
                print("\n  [4] COUNTERFACTUAL ANALYSIS ('What-If' Scenarios)")
                print("      Scenarios: Biomarker changes, Earlier detection, Different treatments")
                print("      Output: Treatment recommendations changes, outcome predictions")

            # 5. Graph Algorithms
            if wf.graph_algorithms:
                print("\n  [5] GRAPH ALGORITHMS (Neo4j GDS)")
                print("      Algorithms: Node Similarity, Pathfinding, Centrality")
                print("      Use: Finding clinically similar patients")

            # 6. Temporal Analysis
            if wf.temporal_analyzer:
                print("\n  [6] TEMPORAL ANALYSIS")
                print("      Method: Time-series analysis of disease progression")
                print("      Output: Progression velocity, intervention windows")

    except Exception as e:
        print(f"  Analytics demo error: {e}")


async def run_digital_twin_demo(patient_data):
    """Run Digital Twin demonstration for the patient"""
    print("\n" + "="*80)
    print("DIGITAL TWIN ENGINE")
    print("="*80)

    try:
        from src.digital_twin import DigitalTwinEngine, UpdateType

        patient_id = patient_data.get('patient_id', 'DEMO_001')

        print(f"\n  Initializing Digital Twin for patient {patient_id}...")

        # Create and initialize twin
        twin = DigitalTwinEngine(
            patient_id=patient_id,
            neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
            neo4j_password=os.getenv('NEO4J_PASSWORD', '123456789')
        )

        # Prepare patient data for twin
        twin_patient_data = {
            **patient_data,
            "cancer_type": "SCLC" if "SmallCell" in patient_data.get('histology_type', '') else "NSCLC"
        }

        init_result = await twin.initialize(twin_patient_data)

        print(f"\n  Twin ID: {init_result['twin_id']}")
        print(f"  State: {init_result['state']}")
        print(f"  Context Graph: {init_result['context_graph_nodes']} nodes, {init_result['context_graph_edges']} edges")

        # Show current state
        state = twin.get_current_state()
        print(f"\n  DIGITAL TWIN STATUS:")
        print(f"    - Lifecycle State: {state['state']}")
        print(f"    - Active Alerts: {len(state['active_alerts'])}")
        print(f"    - Graph Layers: {state['context_graph']['layers']}")

        # Get predictions
        print(f"\n  Generating trajectory predictions...")
        predictions = await twin.predict_trajectories()

        print(f"  PREDICTED PATHWAYS:")
        for pathway in predictions.get('pathways', []):
            print(f"    - {pathway['description']}")
            print(f"      Probability: {pathway['probability']:.0%}, PFS: {pathway['median_pfs_months']} months")

        print(f"\n  Prediction Confidence: {predictions.get('confidence', 0):.0%}")

        return twin

    except ImportError as e:
        print(f"  Digital Twin not available: {e}")
        return None
    except Exception as e:
        print(f"  Digital Twin error: {e}")
        return None


async def offer_mcp_server():
    """Offer to start MCP server"""
    print("\n" + "="*80)
    print("MCP SERVER (Model Context Protocol)")
    print("="*80)
    print("\n  The MCP Server exposes 60+ tools for Claude AI integration:")
    print("    - Patient CRUD operations")
    print("    - 14 specialized agents")
    print("    - Analytics suite")
    print("    - Neo4j/Vector Store queries")
    print("    - Digital Twin management")
    print("\n  To start MCP Server separately, run:")
    print("    python start_mcp_server.py")
    print("\n  Or configure in Claude Desktop claude_desktop_config.json")


async def main():
    """Main interface"""
    print("\n" + "="*80)
    print("LUNG CANCER ASSISTANT - COMPREHENSIVE CLINICAL DECISION SUPPORT")
    print("="*80)

    # Configuration
    print("\nConfiguration:")
    print(f"  Neo4j: Enabled (URI: {os.getenv('NEO4J_URI', 'bolt://localhost:7687')})")
    print(f"  Vector Store: Enabled (Model: all-MiniLM-L6-v2)")
    print(f"  AI Workflow: Enabled (14 agents)")
    print(f"  Digital Twin: Available")
    print(f"  MCP Server: Available on port {os.getenv('MCP_SERVER_PORT', '3000')}")

    # Initialize service
    print("\nInitializing service...")
    service = LungCancerAssistantService(
        use_neo4j=True,  # Enabled by default for full persistence
        use_vector_store=True,
        enable_advanced_workflow=True,
        enable_provenance=True
    )

    # Example patients with more comprehensive data
    example_patients = {
        "1": {
            "patient_id": "EARLY_STAGE_001",
            "name": "Early Stage Patient",
            "age": 62,
            "sex": "F",
            "tnm_stage": "IA",
            "histology_type": "Adenocarcinoma",
            "performance_status": 0,
            "fev1_percent": 85.0,
            "laterality": "Right",
            "comorbidities": ["Hypertension"],
            "biomarker_profile": {
                "egfr_mutation": "Negative",
                "alk_rearrangement": "Negative",
                "pdl1_tps": 10
            }
        },
        "2": {
            "patient_id": "ADVANCED_001",
            "name": "Advanced Stage Patient",
            "age": 71,
            "sex": "M",
            "tnm_stage": "IV",
            "histology_type": "SquamousCellCarcinoma",
            "performance_status": 1,
            "fev1_percent": 55.0,
            "laterality": "Left",
            "comorbidities": ["COPD", "Type 2 Diabetes"],
            "biomarker_profile": {
                "egfr_mutation": "Negative",
                "pdl1_tps": 80,
                "kras_mutation": "G12C"
            }
        },
        "3": {
            "patient_id": "LOCALLY_ADVANCED_001",
            "name": "Locally Advanced Patient",
            "age": 68,
            "sex": "M",
            "tnm_stage": "IIIA",
            "histology_type": "Adenocarcinoma",
            "performance_status": 1,
            "fev1_percent": 65.0,
            "laterality": "Right",
            "comorbidities": ["Atrial Fibrillation"],
            "biomarker_profile": {
                "egfr_mutation": "Exon 19 deletion",
                "alk_rearrangement": "Negative",
                "pdl1_tps": 30
            }
        }
    }

    # Menu
    print("\n" + "="*80)
    print("SELECT PATIENT TO PROCESS")
    print("="*80)
    print("\n1. Early Stage (IA Adenocarcinoma, PS 0)")
    print("2. Advanced Stage (IV Squamous, PS 1)")
    print("3. Locally Advanced (IIIA Adenocarcinoma, PS 1, EGFR+)")
    print("\nOptions:")
    print("  o = Show Ontology Integration")
    print("  m = Show MCP Server Info")
    print("  q = Quit")
    print("\nEnter choice (1-3, o, m, q): ", end="")

    choice = input().strip().lower()

    if choice == 'q':
        print("\nExiting...")
        service.close()
        return

    if choice == 'o':
        await display_ontology_info()
        service.close()
        return

    if choice == 'm':
        await offer_mcp_server()
        service.close()
        return

    if choice not in example_patients:
        print(f"\n Invalid choice: {choice}")
        service.close()
        return

    # Get patient
    patient = example_patients[choice]

    print(f"\n{'='*80}")
    print(f"PROCESSING PATIENT: {patient['name']}")
    print("="*80)
    print(f"\nPatient Details:")
    print(f"  ID: {patient['patient_id']}")
    print(f"  Age: {patient['age']}, Sex: {patient['sex']}")
    print(f"  Stage: {patient['tnm_stage']}")
    print(f"  Histology: {patient['histology_type']}")
    print(f"  Performance Status: WHO {patient['performance_status']}")
    print(f"  FEV1: {patient['fev1_percent']}%")
    if patient.get('comorbidities'):
        print(f"  Comorbidities: {', '.join(patient['comorbidities'])}")
    if patient.get('biomarker_profile'):
        bp = patient['biomarker_profile']
        print(f"  Biomarkers:")
        for k, v in bp.items():
            print(f"    - {k}: {v}")

    print("\nSelect analysis type:")
    print("  1 = Quick (rule-based only)")
    print("  2 = Full AI Workflow (14 agents)")
    print("  3 = Full + Digital Twin")
    print("  4 = Full + Analytics Demo")
    print("  5 = Comprehensive (All features)")
    print("\nChoice (1-5): ", end="")

    analysis_choice = input().strip()

    use_ai = analysis_choice in ['2', '3', '4', '5']
    use_twin = analysis_choice in ['3', '5']
    use_analytics_demo = analysis_choice in ['4', '5']

    # Process patient
    result = await service.process_patient(patient, use_ai_workflow=use_ai)

    # Display results
    print(f"\n{'='*80}")
    print("DECISION SUPPORT RESULTS")
    print("="*80)

    print(f"\nApplicable Guidelines: {len(result.recommendations)}")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"\n{i}. {rec.treatment_type} (Priority: {rec.priority})")
        print(f"   Rule: {rec.rule_id} ({rec.evidence_level})")
        print(f"   Intent: {rec.treatment_intent}")
        print(f"   Confidence: {rec.confidence_score:.0%}")
        print(f"   Survival Benefit: {rec.survival_benefit}")
        if rec.contraindications:
            print(f"   Contraindications: {', '.join(rec.contraindications[:2])}")

    if result.mdt_summary:
        print(f"\n{'='*80}")
        print("MDT SUMMARY (AI-Generated)")
        print("="*80)

        # Format MDT summary nicely
        summary = result.mdt_summary
        if hasattr(summary, 'clinical_summary'):
            print(f"\n{summary.clinical_summary}")

            if hasattr(summary, 'formatted_recommendations') and summary.formatted_recommendations:
                print(f"\n{'-'*80}")
                print("RECOMMENDATIONS:")
                for i, rec in enumerate(summary.formatted_recommendations, 1):
                    print(f"\n  {i}. {rec.get('treatment', 'Unknown')}")
                    print(f"     Intent: {rec.get('intent', 'Unknown')}")
                    print(f"     Evidence: {rec.get('evidence', 'Unknown')}")
                    if rec.get('rationale'):
                        print(f"     Rationale: {rec.get('rationale')}")

            if hasattr(summary, 'key_considerations') and summary.key_considerations:
                print(f"\n{'-'*80}")
                print("KEY CONSIDERATIONS:")
                for consideration in summary.key_considerations:
                    print(f"  * {consideration}")

            if hasattr(summary, 'discussion_points') and summary.discussion_points:
                print(f"\n{'-'*80}")
                print("MDT DISCUSSION POINTS:")
                for point in summary.discussion_points:
                    print(f"  * {point}")

            if hasattr(summary, 'disclaimer'):
                print(f"\n{'-'*80}")
                print(f"\n{summary.disclaimer}")
        else:
            print(str(summary))

    # Show analytics demo if requested
    if use_analytics_demo:
        await display_analytics_demo(service, patient)

    # Run Digital Twin if requested
    if use_twin:
        await run_digital_twin_demo(patient)

    # Show workflow metadata
    print(f"\n{'='*80}")
    print("WORKFLOW METADATA")
    print("="*80)
    print(f"  Workflow Type: {result.workflow_type}")
    print(f"  Complexity Level: {result.complexity_level}")
    print(f"  Execution Time: {result.execution_time_ms}ms")
    if result.provenance_record_id:
        print(f"  Provenance Record: {result.provenance_record_id}")

    if result.semantic_guidelines:
        print(f"\n{'='*80}")
        print(f"SEMANTIC GUIDELINE MATCHES: {len(result.semantic_guidelines)}")
        print("="*80)
        for sg in result.semantic_guidelines[:3]:
            print(f"\n  Rule: {sg['rule_id']}")
            print(f"  Similarity: {sg['similarity_score']:.3f}")

    # Save results
    output_file = Path("output") / f"{result.patient_id}_results.json"
    output_file.parent.mkdir(exist_ok=True)

    output_data = {
        "patient_id": result.patient_id,
        "timestamp": result.timestamp.isoformat(),
        "workflow_type": result.workflow_type,
        "complexity_level": result.complexity_level,
        "execution_time_ms": result.execution_time_ms,
        "provenance_record_id": result.provenance_record_id,
        "recommendations": [
            {
                "treatment": rec.treatment_type,
                "rule_id": rec.rule_id,
                "priority": rec.priority,
                "evidence_level": rec.evidence_level,
                "intent": rec.treatment_intent,
                "confidence": rec.confidence_score
            }
            for rec in result.recommendations
        ],
        "mdt_summary": result.mdt_summary.model_dump() if hasattr(result.mdt_summary, 'model_dump') else str(result.mdt_summary)
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n Results saved to: {output_file}")

    # Cleanup
    service.close()

    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print("\nAdditional Commands:")
    print("  - Run MCP Server: python start_mcp_server.py")
    print("  - Digital Twin Demo: python demo_digital_twin.py")
    print("  - API Server: uvicorn backend.src.api.main:app --reload")


if __name__ == "__main__":
    asyncio.run(main())
