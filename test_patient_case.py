"""
Direct test script for patient case processing.
Tests the exact case that's failing: "68M, stage IIIA adenocarcinoma, EGFR Ex19del+"
"""
import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.src.services.lca_service import LungCancerAssistantService
from backend.src.logging_config import get_logger

logger = get_logger(__name__)

async def test_patient_case():
    """Test processing of EGFR+ Stage IIIA patient"""
    
    print("\n" + "=" * 80)
    print("TESTING PATIENT CASE: 68M, Stage IIIA Adenocarcinoma, EGFR Ex19del+")
    print("=" * 80 + "\n")
    
    # Initialize LCA Service
    print("Initializing LCA Service...")
    lca_service = LungCancerAssistantService(
        use_neo4j=False,  # Disable Neo4j for this test
        enable_advanced_workflow=False  # Use basic workflow only
    )
    print(f"✓ LCA Service initialized with {len(lca_service.rule_engine.rules)} rules\n")
    
    # Print loaded rules
    print("Loaded Rules:")
    for rule_id in list(lca_service.rule_engine.rules.keys())[:10]:
        rule = lca_service.rule_engine.rules[rule_id]
        print(f"  {rule_id}: {rule.name}")
    print()
    
    # Check if R8A is loaded
    if "R8A" in lca_service.rule_engine.rules:
        print("✓ R8A rule is loaded")
        r8a = lca_service.rule_engine.rules["R8A"]
        print(f"   Description: {r8a.description}")
        print(f"   Treatment: {r8a.recommended_treatment}")
    else:
        print("❌ R8A rule NOT loaded!")
    print()
    
    # Create patient data
    patient_data = {
        "patient_id": "TEST_001",
        "age": 68,
        "gender": "male",
        "tnm_stage": "IIIA",
        "histology_type": "adenocarcinoma",
        "performance_status": 1,
        "biomarker_profile": {
            "egfr_mutation": True,
            "egfr_mutation_type": "Ex19del",
            "alk_rearrangement": False,
            "pdl1_tps": 5
        }
    }
    
    print("Patient Data:")
    print(f"  Age: {patient_data['age']}")
    print(f"  Gender: {patient_data['gender']}")
    print(f"  Stage: {patient_data['tnm_stage']}")
    print(f"  Histology: {patient_data['histology_type']}")
    print(f"  PS: {patient_data['performance_status']}")
    print(f"  EGFR: {patient_data['biomarker_profile']['egfr_mutation']} ({patient_data['biomarker_profile']['egfr_mutation_type']})")
    print()
    
    # Process patient
    print("Processing patient...")
    print("-" * 80)
    
    result = await lca_service.process_patient(
        patient_data=patient_data,
        use_ai_workflow=False,  # Skip AI workflow for faster testing
        force_advanced=False
    )
    
    print("-" * 80)
    print()
    
    # Display results
    print("RESULTS:")
    print(f"  Recommendations: {len(result.recommendations)}")
    
    if result.recommendations:
        print("\n  Matched Guidelines:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"\n  {i}. {rec.rule_id}: {rec.treatment}")
            print(f"     Source: {rec.source}")
            print(f"     Evidence: {rec.evidence_level}")
            print(f"     Intent: {rec.intent}")
            if rec.survival_benefit:
                print(f"     Benefit: {rec.survival_benefit}")
    else:
        print("  ❌ NO RECOMMENDATIONS GENERATED!")
        print("\n  This indicates a problem with rule matching.")
        
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(test_patient_case())
