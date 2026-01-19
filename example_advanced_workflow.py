"""
Example: Advanced Workflow & Provenance Integration

Demonstrates the new unified workflow system with:
- Automatic complexity-based routing
- Advanced integrated workflow
- Comprehensive provenance tracking
"""

import asyncio
import json
from datetime import datetime


async def example_1_basic_routing():
    """Example 1: Automatic routing for a simple case"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Automatic Routing (Simple Case)")
    print("="*80)
    
    from backend.src.services.lca_service import LungCancerAssistantService
    
    # Initialize service with all features
    service = LungCancerAssistantService(
        enable_advanced_workflow=True,
        enable_provenance=True
    )
    
    # Simple case: Early stage, good PS
    patient_data = {
        "patient_id": "SIMPLE_001",
        "age": 62,
        "sex": "F",
        "tnm_stage": "IA",
        "histology_type": "Adenocarcinoma",
        "performance_status": 0,
        "fev1_percent": 85.0,
        "comorbidities": []
    }
    
    print(f"\nPatient: {patient_data['patient_id']}")
    print(f"Stage: {patient_data['tnm_stage']} (early stage)")
    print(f"PS: {patient_data['performance_status']} (excellent)")
    print(f"Comorbidities: None")
    
    # Process patient - will automatically use BASIC workflow
    result = await service.process_patient(patient_data)
    
    print(f"\n✓ Processing complete:")
    print(f"  Workflow used: {result.workflow_type.upper()}")
    print(f"  Complexity: {result.complexity_level}")
    print(f"  Execution time: {result.execution_time_ms}ms")
    print(f"  Recommendations: {len(result.recommendations)}")
    
    if result.recommendations:
        print(f"\n  Top recommendation: {result.recommendations[0].treatment_type}")
        print(f"  Evidence level: {result.recommendations[0].evidence_level}")
    
    if result.provenance_record_id:
        print(f"\n  Provenance record: {result.provenance_record_id}")


async def example_2_complex_routing():
    """Example 2: Automatic routing for a complex case"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Automatic Routing (Complex Case)")
    print("="*80)
    
    from backend.src.services.lca_service import LungCancerAssistantService
    
    service = LungCancerAssistantService(
        enable_advanced_workflow=True,
        enable_provenance=True
    )
    
    # Complex case: Advanced stage, poor PS, multiple comorbidities
    patient_data = {
        "patient_id": "COMPLEX_001",
        "age": 72,
        "sex": "M",
        "tnm_stage": "IIIB",
        "histology_type": "Adenocarcinoma",
        "performance_status": 2,
        "fev1_percent": 45.0,
        "comorbidities": ["COPD", "Diabetes", "Hypertension"],
        "biomarker_profile": {
            "pdl1_tps": 65,
            "egfr_mutation": False,
            "alk_rearrangement": False
        }
    }
    
    print(f"\nPatient: {patient_data['patient_id']}")
    print(f"Stage: {patient_data['tnm_stage']} (advanced)")
    print(f"PS: {patient_data['performance_status']} (limited activity)")
    print(f"Comorbidities: {len(patient_data['comorbidities'])}")
    print(f"Biomarkers: PD-L1 {patient_data['biomarker_profile']['pdl1_tps']}%")
    
    # Process patient - will automatically use ADVANCED workflow
    result = await service.process_patient(patient_data)
    
    print(f"\n✓ Processing complete:")
    print(f"  Workflow used: {result.workflow_type.upper()}")
    print(f"  Complexity: {result.complexity_level}")
    print(f"  Execution time: {result.execution_time_ms}ms")
    print(f"  Recommendations: {len(result.recommendations)}")
    
    if result.patient_scenarios:
        print(f"\n  Agent chain:")
        for i, agent in enumerate(result.patient_scenarios[:5], 1):
            print(f"    {i}. {agent}")
        if len(result.patient_scenarios) > 5:
            print(f"    ... and {len(result.patient_scenarios) - 5} more")
    
    if result.provenance_record_id:
        print(f"\n  Provenance record: {result.provenance_record_id}")


async def example_3_complexity_assessment():
    """Example 3: Pre-assessment of patient complexity"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Complexity Assessment")
    print("="*80)
    
    from backend.src.services.lca_service import LungCancerAssistantService
    
    service = LungCancerAssistantService(
        enable_advanced_workflow=True
    )
    
    # Various patient scenarios
    patients = [
        {
            "patient_id": "P001",
            "tnm_stage": "IA",
            "performance_status": 0,
            "comorbidities": []
        },
        {
            "patient_id": "P002",
            "tnm_stage": "IIIA",
            "performance_status": 1,
            "comorbidities": ["COPD"]
        },
        {
            "patient_id": "P003",
            "tnm_stage": "IV",
            "performance_status": 2,
            "comorbidities": ["COPD", "Diabetes", "CAD"]
        }
    ]
    
    print("\nAssessing multiple patients:\n")
    
    for patient in patients:
        assessment = await service.assess_complexity(patient)
        
        print(f"{patient['patient_id']}: Stage {patient['tnm_stage']}, PS {patient['performance_status']}")
        print(f"  → Complexity: {assessment['complexity']}")
        print(f"  → Workflow: {assessment['recommended_workflow']}")
        print()


async def example_4_provenance_tracking():
    """Example 4: Complete provenance tracking"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Provenance Tracking")
    print("="*80)
    
    from backend.src.services.lca_service import LungCancerAssistantService
    
    service = LungCancerAssistantService(
        enable_provenance=True
    )
    
    patient_data = {
        "patient_id": "PROV_001",
        "age": 65,
        "sex": "M",
        "tnm_stage": "IIIA",
        "histology_type": "Adenocarcinoma",
        "performance_status": 1
    }
    
    print(f"\nProcessing patient with provenance tracking...")
    
    # Process patient
    result = await service.process_patient(patient_data)
    
    print(f"\n✓ Patient processed")
    print(f"  Provenance record: {result.provenance_record_id}")
    
    # Retrieve provenance record
    if result.provenance_record_id:
        record = service.get_provenance_record(result.provenance_record_id)
        
        print(f"\nProvenance Details:")
        print(f"  Patient ID: {record['patient_id']}")
        print(f"  Workflow: {record['workflow_type']}")
        print(f"  Created: {record['created_at']}")
        
        print(f"\n  Execution Chain:")
        for i, agent in enumerate(record['execution_chain'], 1):
            print(f"    {i}. {agent}")
        
        print(f"\n  Data Sources:")
        for source in record.get('data_sources', []):
            print(f"    • {source['source']}")
        
        print(f"\n  Ontology Versions:")
        for onto, version in record.get('ontology_versions', {}).items():
            print(f"    • {onto}: {version}")
        
        # Save to file
        filename = f"provenance_{result.patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(record, f, indent=2)
        print(f"\n  ✓ Full record exported to: {filename}")


async def example_5_workflow_comparison():
    """Example 5: Compare basic vs advanced workflow"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Workflow Comparison")
    print("="*80)
    
    from backend.src.services.lca_service import LungCancerAssistantService
    
    service = LungCancerAssistantService(
        enable_advanced_workflow=True,
        enable_provenance=True
    )
    
    # Moderate complexity patient
    patient_data = {
        "patient_id": "COMPARE_001",
        "age": 68,
        "sex": "M",
        "tnm_stage": "IIIA",
        "histology_type": "Adenocarcinoma",
        "performance_status": 1,
        "comorbidities": ["COPD"]
    }
    
    print(f"\nComparing workflows for patient: {patient_data['patient_id']}")
    print(f"Stage: {patient_data['tnm_stage']}, PS: {patient_data['performance_status']}\n")
    
    # Run basic workflow
    print("Running BASIC workflow...")
    basic_result = await service.process_patient(
        patient_data=patient_data,
        force_advanced=False
    )
    
    # Run advanced workflow
    print("Running ADVANCED workflow...")
    advanced_result = await service.process_patient(
        patient_data=patient_data,
        force_advanced=True
    )
    
    # Compare results
    print("\n" + "-"*80)
    print("COMPARISON")
    print("-"*80)
    
    print(f"\nBasic Workflow:")
    print(f"  Execution time: {basic_result.execution_time_ms}ms")
    print(f"  Recommendations: {len(basic_result.recommendations)}")
    if basic_result.recommendations:
        print(f"  Top rec: {basic_result.recommendations[0].treatment_type}")
    
    print(f"\nAdvanced Workflow:")
    print(f"  Execution time: {advanced_result.execution_time_ms}ms")
    print(f"  Recommendations: {len(advanced_result.recommendations)}")
    if advanced_result.recommendations:
        print(f"  Top rec: {advanced_result.recommendations[0].treatment_type}")
    print(f"  Agents involved: {len(advanced_result.patient_scenarios)}")
    
    print(f"\nTime difference: {advanced_result.execution_time_ms - basic_result.execution_time_ms}ms")
    print(f"Additional agents: {len(advanced_result.patient_scenarios) - 4}")


async def example_6_force_advanced():
    """Example 6: Force advanced workflow for any case"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Force Advanced Workflow")
    print("="*80)
    
    from backend.src.services.lca_service import LungCancerAssistantService
    
    service = LungCancerAssistantService(
        enable_advanced_workflow=True,
        enable_provenance=True
    )
    
    # Even a simple case
    patient_data = {
        "patient_id": "FORCE_ADVANCED_001",
        "age": 60,
        "sex": "F",
        "tnm_stage": "IA",  # Early stage
        "histology_type": "Adenocarcinoma",
        "performance_status": 0,  # Excellent PS
        "comorbidities": []  # No complications
    }
    
    print(f"\nPatient: {patient_data['patient_id']}")
    print(f"Stage: {patient_data['tnm_stage']} (would normally use basic workflow)")
    print(f"\nForcing advanced workflow for comprehensive analysis...")
    
    # Force advanced workflow
    result = await service.process_patient(
        patient_data=patient_data,
        force_advanced=True  # Override automatic routing
    )
    
    print(f"\n✓ Processing complete:")
    print(f"  Workflow used: {result.workflow_type.upper()}")
    print(f"  Execution time: {result.execution_time_ms}ms")
    
    print(f"\n  Benefits of forcing advanced workflow:")
    print(f"    • Comprehensive multi-agent analysis")
    print(f"    • Advanced analytics suite")
    print(f"    • Full provenance tracking")
    print(f"    • Uncertainty quantification")


async def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("ADVANCED WORKFLOW & PROVENANCE EXAMPLES")
    print("="*80)
    print("\nDemonstrating the new unified workflow system with:")
    print("  ✓ Automatic complexity-based routing")
    print("  ✓ Advanced integrated workflow")
    print("  ✓ Comprehensive provenance tracking")
    print("\n" + "="*80)
    
    # Run examples
    await example_1_basic_routing()
    await example_2_complex_routing()
    await example_3_complexity_assessment()
    await example_4_provenance_tracking()
    await example_5_workflow_comparison()
    await example_6_force_advanced()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)
    print("\nNext steps:")
    print("  1. Try CLI commands: python cli.py run-advanced-workflow")
    print("  2. Explore MCP tools in Claude Desktop")
    print("  3. Read ADVANCED_WORKFLOW_GUIDE.md for more details")
    print()


if __name__ == "__main__":
    asyncio.run(main())
