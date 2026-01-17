"""
Sample Patient Data Generator
Creates sample patients following the LUCADA pattern from the original paper

This includes the canonical Jenny_Sesen example from Figure 2 of the paper.
"""

import json
import os
from typing import List, Dict, Any
from datetime import datetime
import random

# Sample patients based on the original LCA paper and clinical scenarios
SAMPLE_PATIENTS: List[Dict[str, Any]] = [
    # Jenny_Sesen - The canonical example from Figure 2 of the original paper
    {
        "patient_id": "Jenny_Sesen_200312",
        "name": "Jenny_Sesen",
        "sex": "F",
        "age": 72,
        "diagnosis": "Malignant Neoplasm of Lung",
        "tnm_stage": "IIA",
        "histology_type": "Carcinosarcoma",
        "laterality": "Right",
        "performance_status": 1,
        "fev1_percent": 75.0,
        "comorbidities": [],
        "notes": "Canonical example patient from original LCA paper (Sesen et al., University of Oxford)"
    },
    
    # Early stage NSCLC - Surgery candidate (R2)
    {
        "patient_id": "LC-2024-001",
        "name": "John_Smith",
        "sex": "M",
        "age": 65,
        "diagnosis": "Malignant Neoplasm of Lung",
        "tnm_stage": "IB",
        "histology_type": "Adenocarcinoma",
        "laterality": "Left",
        "performance_status": 0,
        "fev1_percent": 85.0,
        "comorbidities": [],
        "notes": "Early stage NSCLC, ideal surgery candidate per NICE R2"
    },
    
    # Locally advanced NSCLC - Chemoradiotherapy candidate (R6)
    {
        "patient_id": "LC-2024-002",
        "name": "Mary_Williams",
        "sex": "F",
        "age": 58,
        "diagnosis": "Malignant Neoplasm of Lung",
        "tnm_stage": "IIIA",
        "histology_type": "SquamousCellCarcinoma",
        "laterality": "Right",
        "performance_status": 1,
        "fev1_percent": 70.0,
        "comorbidities": ["COPD"],
        "notes": "Locally advanced NSCLC, chemoradiotherapy candidate per NICE R6"
    },
    
    # Advanced NSCLC - Chemotherapy candidate (R1)
    {
        "patient_id": "LC-2024-003",
        "name": "Robert_Johnson",
        "sex": "M",
        "age": 68,
        "diagnosis": "Malignant Neoplasm of Lung",
        "tnm_stage": "IV",
        "histology_type": "Adenocarcinoma",
        "laterality": "Right",
        "performance_status": 1,
        "fev1_percent": None,
        "comorbidities": ["Hypertension"],
        "notes": "Metastatic NSCLC, chemotherapy candidate per NICE R1"
    },
    
    # SCLC - Chemotherapy candidate (R5)
    {
        "patient_id": "LC-2024-004",
        "name": "Susan_Davis",
        "sex": "F",
        "age": 62,
        "diagnosis": "Malignant Neoplasm of Lung",
        "tnm_stage": "IIIA",
        "histology_type": "SmallCellCarcinoma",
        "laterality": "Left",
        "performance_status": 1,
        "fev1_percent": 65.0,
        "comorbidities": [],
        "notes": "Limited stage SCLC, chemotherapy candidate per NICE R5"
    },
    
    # Poor PS - Palliative care candidate (R4)
    {
        "patient_id": "LC-2024-005",
        "name": "William_Brown",
        "sex": "M",
        "age": 78,
        "diagnosis": "Malignant Neoplasm of Lung",
        "tnm_stage": "IV",
        "histology_type": "LargeCellCarcinoma",
        "laterality": "Bilateral",
        "performance_status": 3,
        "fev1_percent": 40.0,
        "comorbidities": ["CardiovascularDisease", "Diabetes"],
        "notes": "Advanced disease with poor PS, palliative care candidate per NICE R4"
    },
    
    # Immunotherapy candidate with high PD-L1 (R7)
    {
        "patient_id": "LC-2024-006",
        "name": "Elizabeth_Miller",
        "sex": "F",
        "age": 55,
        "diagnosis": "Malignant Neoplasm of Lung",
        "tnm_stage": "IV",
        "histology_type": "Adenocarcinoma",
        "laterality": "Right",
        "performance_status": 0,
        "fev1_percent": 80.0,
        "comorbidities": [],
        "biomarkers": ["PDL1High"],
        "notes": "Advanced NSCLC with PD-L1 ≥50%, immunotherapy candidate per R7"
    },
    
    # EGFR-positive - Targeted therapy candidate (R8)
    {
        "patient_id": "LC-2024-007",
        "name": "James_Wilson",
        "sex": "M",
        "age": 48,
        "diagnosis": "Malignant Neoplasm of Lung",
        "tnm_stage": "IIIB",
        "histology_type": "Adenocarcinoma",
        "laterality": "Left",
        "performance_status": 1,
        "fev1_percent": 90.0,
        "comorbidities": [],
        "biomarkers": ["EGFRPositive"],
        "notes": "EGFR-mutant advanced NSCLC, osimertinib candidate per R8"
    },
    
    # ALK-positive - Targeted therapy candidate (R9)
    {
        "patient_id": "LC-2024-008",
        "name": "Patricia_Anderson",
        "sex": "F",
        "age": 42,
        "diagnosis": "Malignant Neoplasm of Lung",
        "tnm_stage": "IV",
        "histology_type": "Adenocarcinoma",
        "laterality": "Right",
        "performance_status": 1,
        "fev1_percent": 88.0,
        "comorbidities": [],
        "biomarkers": ["ALKPositive"],
        "notes": "ALK-rearranged advanced NSCLC, alectinib candidate per R9"
    },
    
    # Borderline surgery candidate (R2/R3)
    {
        "patient_id": "LC-2024-009",
        "name": "Thomas_Taylor",
        "sex": "M",
        "age": 70,
        "diagnosis": "Malignant Neoplasm of Lung",
        "tnm_stage": "IIB",
        "histology_type": "SquamousCellCarcinoma",
        "laterality": "Left",
        "performance_status": 2,
        "fev1_percent": 55.0,
        "comorbidities": ["COPD", "CardiovascularDisease"],
        "notes": "Borderline surgical candidate due to comorbidities, may benefit from R3 radiotherapy"
    },
    
    # Young patient with extensive SCLC
    {
        "patient_id": "LC-2024-010",
        "name": "Jennifer_Martinez",
        "sex": "F",
        "age": 45,
        "diagnosis": "Malignant Neoplasm of Lung",
        "tnm_stage": "IV",
        "histology_type": "SmallCellCarcinoma",
        "laterality": "Right",
        "performance_status": 1,
        "fev1_percent": 75.0,
        "comorbidities": [],
        "notes": "Extensive stage SCLC, aggressive treatment indicated per R5"
    }
]


def get_jenny_sesen() -> Dict[str, Any]:
    """
    Get the canonical Jenny_Sesen patient from the original paper.
    
    This patient is used as the primary example throughout the LCA paper:
    - DB Identifier: 200312
    - Female, 72 years old
    - Stage IIA Carcinosarcoma
    - Right-sided tumor
    - Performance Status 1
    
    Expected recommendations per NICE guidelines:
    - R3: Radiotherapy (Stage I-IIIA, PS 0-2)
    """
    return SAMPLE_PATIENTS[0]


def get_sample_patients() -> List[Dict[str, Any]]:
    """Get all sample patients."""
    return SAMPLE_PATIENTS


def get_patient_by_id(patient_id: str) -> Dict[str, Any]:
    """Get a specific sample patient by ID."""
    for patient in SAMPLE_PATIENTS:
        if patient["patient_id"] == patient_id:
            return patient
    return None


def generate_random_patient() -> Dict[str, Any]:
    """Generate a random patient for testing."""
    import uuid
    
    stages = ["IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IV"]
    histologies = ["Adenocarcinoma", "SquamousCellCarcinoma", "LargeCellCarcinoma", 
                   "SmallCellCarcinoma", "Carcinosarcoma"]
    sexes = ["M", "F"]
    lateralities = ["Left", "Right", "Bilateral"]
    
    return {
        "patient_id": f"LC-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}",
        "name": f"Patient_{random.randint(1000, 9999)}",
        "sex": random.choice(sexes),
        "age": random.randint(40, 85),
        "diagnosis": "Malignant Neoplasm of Lung",
        "tnm_stage": random.choice(stages),
        "histology_type": random.choice(histologies),
        "laterality": random.choice(lateralities),
        "performance_status": random.randint(0, 4),
        "fev1_percent": random.uniform(40.0, 100.0) if random.random() > 0.3 else None,
        "comorbidities": random.sample(["COPD", "Diabetes", "CardiovascularDisease", "Dementia"], 
                                        k=random.randint(0, 2)),
        "notes": "Randomly generated test patient"
    }


def save_sample_patients(filepath: str = "data/sample_patients.json"):
    """Save sample patients to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    data = {
        "generated_at": datetime.now().isoformat(),
        "source": "LCA Sample Patient Generator",
        "description": "Sample patients including Jenny_Sesen from original paper",
        "patients": SAMPLE_PATIENTS
    }
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved {len(SAMPLE_PATIENTS)} sample patients to {filepath}")
    return filepath


def load_sample_patients(filepath: str = "data/sample_patients.json") -> List[Dict[str, Any]]:
    """Load sample patients from JSON file."""
    if not os.path.exists(filepath):
        return SAMPLE_PATIENTS
    
    with open(filepath, "r") as f:
        data = json.load(f)
    
    return data.get("patients", [])


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LCA Sample Patient Generator")
    print("=" * 60)
    
    # Show Jenny_Sesen
    jenny = get_jenny_sesen()
    print("\nCanonical Patient (from paper Figure 2):")
    print(f"  Name: {jenny['name']}")
    print(f"  Age: {jenny['age']}y {jenny['sex']}")
    print(f"  Stage: {jenny['tnm_stage']}")
    print(f"  Histology: {jenny['histology_type']}")
    print(f"  PS: {jenny['performance_status']}")
    
    # Show all patients
    print(f"\nTotal sample patients: {len(SAMPLE_PATIENTS)}")
    for p in SAMPLE_PATIENTS:
        print(f"  - {p['patient_id']}: {p['tnm_stage']} {p['histology_type']}, PS {p['performance_status']}")
    
    # Save to file
    save_sample_patients()
    
    print("\n" + "=" * 60)
