"""
Quick Run Script for Lung Cancer Assistant
Simplified interface for processing patients
"""

import asyncio
import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Centralized logging
from src.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

from src.services.lca_service import LungCancerAssistantService


async def main():
    """Main interface"""
    print("\n" + "="*80)
    print("LUNG CANCER ASSISTANT - QUICK RUN")
    print("="*80)

    # Configuration
    print("\nConfiguration:")
    print("  Neo4j: Enabled (persistence active)")
    print("  Vector Store: Enabled")
    print("  AI Workflow: Enabled (takes ~20 seconds)")

    # Initialize service
    print("\nInitializing service...")
    service = LungCancerAssistantService(
        use_neo4j=True,  # Enabled by default for full persistence
        use_vector_store=True
    )

    # Example patients
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
            "laterality": "Right"
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
            "laterality": "Left"
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
            "laterality": "Right"
        }
    }

    # Menu
    print("\n" + "="*80)
    print("SELECT PATIENT TO PROCESS")
    print("="*80)
    print("\n1. Early Stage (IA Adenocarcinoma, PS 0)")
    print("2. Advanced Stage (IV Squamous, PS 1)")
    print("3. Locally Advanced (IIIA Adenocarcinoma, PS 1)")
    print("\nEnter choice (1-3), or 'q' to quit: ", end="")

    choice = input().strip()

    if choice.lower() == 'q':
        print("\nExiting...")
        service.close()
        return

    if choice not in example_patients:
        print(f"\n⚠ Invalid choice: {choice}")
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

    print("\nRun AI workflow? (takes ~20 seconds)")
    print("  y = Yes (full analysis with LLM)")
    print("  n = No (guideline matching only)")
    print("\nChoice (y/n): ", end="")

    ai_choice = input().strip().lower()
    use_ai = ai_choice == 'y'

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
        print(f"   Survival Benefit: {rec.survival_benefit}")
        if rec.contraindications:
            print(f"   Contraindications: {', '.join(rec.contraindications[:2])}")

    if result.mdt_summary:
        print(f"\n{'='*80}")
        print("MDT SUMMARY (AI-Generated)")
        print("="*80)
        print(result.mdt_summary)

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
        "recommendations": [
            {
                "treatment": rec.treatment_type,
                "rule_id": rec.rule_id,
                "priority": rec.priority,
                "evidence_level": rec.evidence_level,
                "intent": rec.treatment_intent
            }
            for rec in result.recommendations
        ],
        "mdt_summary": result.mdt_summary
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Cleanup
    service.close()

    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
