"""
Ingest Comprehensive Test Patients

This script ingests all comprehensive test patients into the LCA system
for testing all workflows, ontologies, and functionalities.

Usage:
    python scripts/ingest_test_patients.py
    python scripts/ingest_test_patients.py --patient TEST-NSCLC-001
    python scripts/ingest_test_patients.py --skip-duplicates
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.src.agents.lca_workflow import LCAWorkflow, analyze_patient
from backend.src.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def ingest_patient(patient_data: Dict[str, Any], workflow: LCAWorkflow) -> Dict[str, Any]:
    """
    Ingest a single patient through the full LCA workflow.
    
    Args:
        patient_data: Patient clinical data
        workflow: LCAWorkflow instance
        
    Returns:
        Analysis result dictionary
    """
    try:
        logger.info(f"Ingesting patient: {patient_data.get('patient_id', 'Unknown')}")
        
        # Run through integrated workflow
        result = await analyze_patient(
            patient_data=patient_data,
            persist=True,  # Save to Neo4j
            enable_analytics=True
        )
        
        if result.get("status") == "success":
            logger.info(f"✓ Successfully ingested {patient_data.get('patient_id')}")
            logger.info(f"  Primary recommendation: {result.get('primary_recommendation', 'N/A')}")
            logger.info(f"  Evidence level: {result.get('evidence_level', 'N/A')}")
        else:
            logger.error(f"✗ Failed to ingest {patient_data.get('patient_id')}: {result.get('message')}")
            
        return result
        
    except Exception as e:
        logger.error(f"Error ingesting patient {patient_data.get('patient_id')}: {e}", exc_info=True)
        return {
            "status": "error",
            "patient_id": patient_data.get('patient_id'),
            "message": str(e)
        }


async def ingest_all_test_patients(
    test_file: Path,
    specific_patient: str = None,
    skip_duplicates: bool = False
):
    """
    Ingest all test patients from JSON file.
    
    Args:
        test_file: Path to test patients JSON
        specific_patient: If provided, only ingest this patient ID
        skip_duplicates: Skip patients that already exist in Neo4j
    """
    # Load test patients
    logger.info(f"Loading test patients from: {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_patients = data.get('test_patients', [])
    logger.info(f"Found {len(test_patients)} test patients")
    
    # Filter if specific patient requested
    if specific_patient:
        test_patients = [
            p for p in test_patients 
            if p['patient_data'].get('patient_id') == specific_patient
        ]
        if not test_patients:
            logger.error(f"Patient {specific_patient} not found in test data")
            return
        logger.info(f"Filtering to specific patient: {specific_patient}")
    
    # Initialize workflow
    workflow = LCAWorkflow()
    
    # Track results
    results = {
        "success": [],
        "failed": [],
        "skipped": []
    }
    
    # Ingest each patient
    for idx, test_case in enumerate(test_patients, 1):
        test_id = test_case.get('test_id')
        description = test_case.get('description')
        patient_data = test_case.get('patient_data')
        patient_id = patient_data.get('patient_id')
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[{idx}/{len(test_patients)}] {test_id}: {description}")
        logger.info(f"{'='*80}")
        
        # Check for duplicates if requested
        if skip_duplicates:
            # TODO: Check if patient exists in Neo4j
            # For now, just log
            logger.info(f"Duplicate check not implemented yet")
        
        # Ingest patient
        result = await ingest_patient(patient_data, workflow)
        
        # Track result
        if result.get('status') == 'success':
            results['success'].append({
                'test_id': test_id,
                'patient_id': patient_id,
                'description': description
            })
        else:
            results['failed'].append({
                'test_id': test_id,
                'patient_id': patient_id,
                'description': description,
                'error': result.get('message')
            })
        
        # Small delay between patients
        await asyncio.sleep(1)
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("INGESTION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total: {len(test_patients)}")
    logger.info(f"✓ Success: {len(results['success'])}")
    logger.info(f"✗ Failed: {len(results['failed'])}")
    logger.info(f"⊘ Skipped: {len(results['skipped'])}")
    
    if results['failed']:
        logger.info(f"\nFailed patients:")
        for failed in results['failed']:
            logger.info(f"  - {failed['patient_id']}: {failed['error']}")
    
    logger.info(f"\n{'='*80}")
    
    return results


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ingest comprehensive test patients')
    parser.add_argument(
        '--patient',
        type=str,
        help='Specific patient ID to ingest (e.g., TEST-NSCLC-001)'
    )
    parser.add_argument(
        '--skip-duplicates',
        action='store_true',
        help='Skip patients that already exist in Neo4j'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        default='data/comprehensive_test_patients.json',
        help='Path to test patients JSON file'
    )
    
    args = parser.parse_args()
    
    # Resolve test file path
    project_root = Path(__file__).parent.parent
    test_file = project_root / args.test_file
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        sys.exit(1)
    
    # Run ingestion
    await ingest_all_test_patients(
        test_file=test_file,
        specific_patient=args.patient,
        skip_duplicates=args.skip_duplicates
    )


if __name__ == "__main__":
    asyncio.run(main())
