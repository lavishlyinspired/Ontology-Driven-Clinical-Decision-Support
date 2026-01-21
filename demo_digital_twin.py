"""
Digital Twin Demo - Complete Example

Shows how to use the Digital Twin Engine for a patient journey:
1. Initialize twin
2. Update with new data
3. Get predictions
4. Handle alerts
5. Track progression
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from src.digital_twin import (
    DigitalTwinEngine,
    UpdateType,
    create_digital_twin
)


async def demo_patient_journey():
    """
    Demonstrate a complete patient journey with Digital Twin Engine
    """
    print("="*80)
    print("🏥 DIGITAL TWIN ENGINE - PATIENT JOURNEY DEMO")
    print("="*80)
    print()
    
    # ========================================
    # STEP 1: PATIENT PRESENTATION
    # ========================================
    print("📋 STEP 1: New Patient Presentation\n")
    
    patient_data = {
        "patient_id": "P12345",
        "age": 68,
        "gender": "Male",
        "smoking_history": "Former smoker (quit 5 years ago)",
        
        # Clinical presentation
        "stage": "IIIA",
        "histology": "Adenocarcinoma",
        "cancer_type": "NSCLC",
        "performance_status": 1,
        
        # Biomarkers
        "biomarkers": {
            "EGFR": "Exon 19 deletion positive",
            "ALK": "Negative",
            "PD-L1": "50% TPS"
        },
        
        # Initial findings
        "tumor_size_mm": 42,
        "lymph_nodes": "N2 (mediastinal)",
        
        # Comorbidities
        "comorbidities": ["Hypertension", "Type 2 Diabetes"],
        "creatinine": 1.1,
        "egfr_renal": 65
    }
    
    print(f"   Patient: {patient_data['patient_id']}")
    print(f"   Age: {patient_data['age']}, {patient_data['gender']}")
    print(f"   Diagnosis: {patient_data['stage']} {patient_data['histology']}")
    print(f"   EGFR: {patient_data['biomarkers']['EGFR']}")
    print(f"   Performance Status: {patient_data['performance_status']}")
    print()
    
    # ========================================
    # STEP 2: INITIALIZE DIGITAL TWIN
    # ========================================
    print("🤖 STEP 2: Initializing Digital Twin\n")
    
    twin = await create_digital_twin("P12345", patient_data)
    
    print(f"   ✅ Twin Created: {twin.twin_id}")
    print(f"   Context Graph: {len(twin.context_graph.nodes)} nodes, {len(twin.context_graph.edges)} edges")
    print(f"   State: {twin.state.value}")
    print()
    
    # ========================================
    # STEP 3: BASELINE ANALYSIS
    # ========================================
    print("🔬 STEP 3: Baseline Analysis Complete\n")
    
    state = twin.get_current_state()
    print(f"   Initial Assessment:")
    print(f"   - Context graph layers: {state['context_graph']['layers']}")
    print(f"   - Active alerts: {len(state['active_alerts'])}")
    print()
    
    # ========================================
    # STEP 4: MONTH 3 - FIRST FOLLOW-UP
    # ========================================
    print("📅 STEP 4: Month 3 - First Follow-up (Imaging)\n")
    
    imaging_update = await twin.update({
        "type": UpdateType.IMAGING.value,
        "data": {
            "scan_type": "CT Chest",
            "scan_date": "2026-04-15",
            "findings": "Partial response - tumor reduced to 28mm (33% decrease)",
            "recist_status": "PR",
            "tumor_size_mm": 28,
            "new_lesions": 0
        }
    })
    
    print(f"   📊 Imaging Results:")
    print(f"      Tumor size: 42mm → 28mm (33% reduction)")
    print(f"      RECIST: {imaging_update.get('update_data', {}).get('data', {}).get('recist_status', 'N/A')}")
    print()
    
    print(f"   🤖 Twin Response:")
    print(f"      Agents re-run: {', '.join(imaging_update.get('affected_agents', []))}")
    print(f"      New alerts: {len(imaging_update.get('new_alerts', []))}")
    
    for alert in imaging_update.get('new_alerts', []):
        print(f"      ✓ {alert['message']}")
    print()
    
    # ========================================
    # STEP 5: MONTH 8 - PROGRESSION DETECTED
    # ========================================
    print("📅 STEP 5: Month 8 - Disease Progression Detected\n")
    
    progression_update = await twin.update({
        "type": UpdateType.IMAGING.value,
        "data": {
            "scan_type": "CT Chest",
            "scan_date": "2026-09-15",
            "findings": "Disease progression - tumor increased to 38mm (36% increase from nadir)",
            "recist_status": "PD",
            "tumor_size_mm": 38,
            "new_lesions": 1,
            "new_lesion_location": "Left hilar lymph node"
        }
    })
    
    print(f"   📊 Imaging Results:")
    print(f"      Tumor size: 28mm → 38mm (36% increase)")
    print(f"      New lesions: 1 (left hilar)")
    print(f"      RECIST: Progressive Disease")
    print()
    
    print(f"   ⚠️  ALERTS GENERATED:")
    for alert in progression_update.get('new_alerts', []):
        print(f"      {alert['severity'].upper()}: {alert['message']}")
        print(f"         Confidence: {alert['confidence']:.0%}")
    print()
    
    # ========================================
    # STEP 6: RESISTANCE MUTATION TESTING
    # ========================================
    print("📅 STEP 6: Resistance Mutation Analysis\n")
    
    resistance_update = await twin.update({
        "type": UpdateType.LAB_RESULT.value,
        "data": {
            "test": "EGFR Resistance Mutation Panel",
            "test_date": "2026-09-20",
            "results": {
                "T790M": "Detected (allele frequency 15%)",
                "C797S": "Not detected",
                "MET_amplification": "Not detected"
            },
            "resistance_mechanism": "T790M-mediated resistance"
        }
    })
    
    print(f"   🧬 Molecular Testing:")
    print(f"      T790M mutation: DETECTED (15% AF)")
    print(f"      Resistance mechanism identified")
    print()
    
    print(f"   🤖 Twin Analysis:")
    print(f"      Agents activated: {', '.join(resistance_update.get('affected_agents', []))}")
    print()
    
    print(f"   💊 TREATMENT RECOMMENDATIONS:")
    for alert in resistance_update.get('new_alerts', []):
        print(f"      • {alert['message']}")
        if alert.get('recommended_actions'):
            for action in alert['recommended_actions']:
                print(f"        - {action}")
    print()
    
    # ========================================
    # STEP 7: TRAJECTORY PREDICTIONS
    # ========================================
    print("🔮 STEP 7: Trajectory Predictions\n")
    
    predictions = await twin.predict_trajectories()
    
    print(f"   Predicted Pathways:")
    for i, pathway in enumerate(predictions.get('pathways', []), 1):
        print(f"      {i}. {pathway['description']}")
        print(f"         Probability: {pathway['probability']:.0%}")
        print(f"         Median PFS: {pathway['median_pfs_months']} months")
        print(f"         Based on {pathway['supporting_cases']} similar cases")
        print()
    
    print(f"   Analysis confidence: {predictions.get('confidence', 0):.0%}")
    print()
    
    # ========================================
    # STEP 8: TREATMENT CHANGE
    # ========================================
    print("📅 STEP 8: Treatment Modification\n")
    
    treatment_update = await twin.update({
        "type": UpdateType.TREATMENT_CHANGE.value,
        "data": {
            "action": "switch_therapy",
            "previous_treatment": "Osimertinib 80mg daily",
            "new_treatment": "Osimertinib 80mg daily (third-generation EGFR TKI)",
            "reason": "T790M resistance mutation detected",
            "start_date": "2026-09-25"
        }
    })
    
    print(f"   💊 Treatment Plan Updated:")
    print(f"      New therapy: Osimertinib (targets T790M)")
    print(f"      Rationale: Second-line for T790M+ resistance")
    print()
    
    print(f"   🤖 Twin Status:")
    print(f"      Total updates processed: {len(twin.snapshots)}")
    print(f"      Active alerts: {len(twin.active_alerts)}")
    print()
    
    # ========================================
    # STEP 9: FINAL STATE & REASONING CHAIN
    # ========================================
    print("📊 STEP 9: Current Twin State\n")
    
    final_state = twin.get_current_state()
    
    print(f"   Twin ID: {final_state['twin_id']}")
    print(f"   Created: {final_state['created_at']}")
    print(f"   Last Updated: {final_state['last_updated']}")
    print(f"   State: {final_state['state']}")
    print()
    
    print(f"   Context Graph Structure:")
    for layer, count in final_state['context_graph']['layers'].items():
        print(f"      {layer}: {count} nodes")
    print()
    
    print(f"   Total Snapshots: {final_state['snapshots_count']}")
    print(f"   Active Alerts: {len(final_state['active_alerts'])}")
    print()
    
    # ========================================
    # STEP 10: EXPORT TWIN
    # ========================================
    print("💾 STEP 10: Export Digital Twin\n")
    
    export_data = twin.export_twin()
    
    print(f"   Exported Components:")
    print(f"      Patient data: ✓")
    print(f"      Context graph: {len(export_data['context_graph']['nodes'])} nodes")
    print(f"      Snapshots: {len(export_data['snapshots'])}")
    print(f"      Active alerts: {len(export_data['active_alerts'])}")
    print(f"      Predictions: ✓")
    print()
    
    print("="*80)
    print("✅ DIGITAL TWIN DEMO COMPLETE")
    print("="*80)
    print()
    print("Key Achievements:")
    print("  ✓ Patient digital twin created and maintained")
    print("  ✓ Real-time updates processed (imaging, labs, treatment)")
    print("  ✓ Progression detected and resistance mechanism identified")
    print("  ✓ Treatment recommendations generated")
    print("  ✓ Trajectory predictions calculated")
    print("  ✓ Complete audit trail maintained in context graph")
    print()


if __name__ == "__main__":
    asyncio.run(demo_patient_journey())
