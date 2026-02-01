"""
Test script to debug the LCA assistant issue with "68M, stage IIIA adenocarcinoma, EGFR Ex19del+"
"""

import asyncio
import sys
import os
from pathlib import Path

# Set up paths
project_root = Path(__file__).parent
backend_root = project_root / "backend"
os.chdir(str(backend_root))
sys.path.insert(0, str(backend_root))

from src.services.lca_service import LungCancerAssistantService
from src.services.conversation_service import ConversationService


async def test_patient_extraction():
    """Test if patient data extraction works for the problematic query"""
    # Initialize services
    lca_service = LungCancerAssistantService(
        use_neo4j=False,
        use_vector_store=False,
        enable_advanced_workflow=False,
        enable_provenance=False
    )

    conversation_service = ConversationService(lca_service, enable_enhanced_features=False)

    # Test message
    message = "68M, stage IIIA adenocarcinoma, EGFR Ex19del+"

    print("=" * 80)
    print("Testing patient data extraction")
    print("=" * 80)
    print(f"Input message: {message}")
    print()

    # Extract patient data
    patient_data = conversation_service._extract_patient_data(message)

    print("Extracted patient data:")
    for key, value in patient_data.items():
        if not key.startswith("_"):
            print(f"  {key}: {value}")

    extraction_meta = patient_data.get("_extraction_meta", {})
    print(f"\nExtraction metadata:")
    print(f"  Extracted fields: {extraction_meta.get('extracted_fields', [])}")
    print(f"  Missing fields: {extraction_meta.get('missing_fields', [])}")
    print(f"  Complete: {extraction_meta.get('extraction_complete', False)}")

    # Test rule matching
    if extraction_meta.get('extraction_complete', False):
        print("\n" + "=" * 80)
        print("Testing rule matching")
        print("=" * 80)

        # Normalize biomarker data
        normalized_data = lca_service._normalize_biomarker_data(patient_data)

        print("Normalized patient data:")
        for key, value in normalized_data.items():
            if not key.startswith("_"):
                print(f"  {key}: {value}")

        # Get matching rules
        recommendations = lca_service.rule_engine.classify_patient(normalized_data)

        print(f"\nMatched {len(recommendations)} rules:")
        for rec in recommendations:
            print(f"  {rec['rule_id']}: {rec['recommended_treatment']} ({rec['evidence_level']}, priority={rec['priority']})")

        if not recommendations:
            print("  WARNING: No rules matched!")
            print("  This would trigger fallback recommendations")

            # Test fallback
            fallback = lca_service._create_fallback_recommendations(normalized_data)
            print(f"\nFallback created {len(fallback)} recommendations:")
            for rec in fallback:
                print(f"  {rec['rule_id']}: {rec['recommended_treatment']}")

        # Test full processing
        print("\n" + "=" * 80)
        print("Testing full LCA processing")
        print("=" * 80)

        try:
            result = await lca_service.process_patient(
                patient_data=patient_data,
                use_ai_workflow=False,
                force_advanced=False
            )

            print(f"Processing successful!")
            print(f"  Patient ID: {result.patient_id}")
            print(f"  Recommendations: {len(result.recommendations)}")
            print(f"  Workflow type: {result.workflow_type}")
            print(f"  Execution time: {result.execution_time_ms}ms")

            if result.recommendations:
                print(f"\nRecommendations:")
                for i, rec in enumerate(result.recommendations[:3], 1):
                    print(f"  {i}. {rec.treatment_type} ({rec.evidence_level}, {rec.treatment_intent})")
            else:
                print("\n  WARNING: No recommendations in result!")

        except Exception as e:
            print(f"ERROR during processing: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("\nERROR: Patient data extraction incomplete!")
        print("Cannot proceed with rule matching")

    lca_service.close()


if __name__ == "__main__":
    asyncio.run(test_patient_extraction())
