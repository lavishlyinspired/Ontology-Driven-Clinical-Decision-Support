"""
Synthetic Patient Data Generator
Creates realistic lung cancer patient records for testing LUCADA system
"""

import random
import uuid
import json
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime, timedelta


@dataclass
class SyntheticPatient:
    """Synthetic lung cancer patient"""
    patient_id: str
    name: str
    sex: str
    age_at_diagnosis: int
    tnm_stage: str
    histology_type: str
    performance_status: int
    laterality: str
    diagnosis_date: str
    fev1_percent: Optional[float] = None
    comorbidities: List[str] = None

    def to_dict(self):
        data = asdict(self)
        if data['comorbidities'] is None:
            data['comorbidities'] = []
        # Add 'age' field for IngestionAgent compatibility
        data['age'] = data['age_at_diagnosis']
        return data


class SyntheticPatientGenerator:
    """Generate realistic synthetic lung cancer patients"""

    TNM_STAGES = ["IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IV"]
    TNM_WEIGHTS = [0.10, 0.08, 0.12, 0.10, 0.15, 0.15, 0.30]

    HISTOLOGY_TYPES = [
        ("Adenocarcinoma", 0.40),
        ("SquamousCellCarcinoma", 0.25),
        ("SmallCellCarcinoma", 0.15),
        ("LargeCellCarcinoma", 0.10),
        ("Carcinosarcoma", 0.05),
        ("NonSmallCellCarcinoma", 0.05)
    ]

    PERFORMANCE_STATUS_BY_STAGE = {
        "IA": [0, 0, 0, 1, 1],
        "IB": [0, 0, 1, 1, 1],
        "IIA": [0, 1, 1, 1, 2],
        "IIB": [0, 1, 1, 2, 2],
        "IIIA": [1, 1, 2, 2, 2],
        "IIIB": [1, 2, 2, 2, 3],
        "IV": [1, 2, 2, 3, 3, 4]
    }

    COMORBIDITIES = [
        "COPD", "CardiovascularDisease", "Diabetes",
        "Hypertension", "Dementia", "RenalDisease"
    ]

    FIRST_NAMES_M = ["James", "John", "Robert", "Michael", "William", "David",
                     "Richard", "Joseph", "Thomas", "Charles"]
    FIRST_NAMES_F = ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth",
                     "Barbara", "Susan", "Jessica", "Sarah", "Karen"]
    LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                  "Miller", "Davis", "Rodriguez", "Martinez"]

    def generate_patient(self) -> SyntheticPatient:
        """Generate a single synthetic patient"""

        # Demographics
        sex = random.choice(["M", "F"])
        first_name = random.choice(
            self.FIRST_NAMES_M if sex == "M" else self.FIRST_NAMES_F
        )
        last_name = random.choice(self.LAST_NAMES)

        # Age: lung cancer peaks 65-74
        age = int(random.gauss(70, 10))
        age = max(40, min(95, age))

        # Stage (weighted random)
        stage = random.choices(self.TNM_STAGES, weights=self.TNM_WEIGHTS)[0]

        # Histology (weighted random)
        histologies, weights = zip(*self.HISTOLOGY_TYPES)
        histology = random.choices(histologies, weights=weights)[0]

        # Performance status (correlated with stage)
        ps = random.choice(self.PERFORMANCE_STATUS_BY_STAGE[stage])

        # Laterality
        laterality = random.choices(
            ["Right", "Left", "Bilateral"],
            weights=[0.55, 0.40, 0.05]
        )[0]

        # FEV1 (lung function) - worse with advanced disease
        base_fev1 = 85 - (self.TNM_STAGES.index(stage) * 5)
        fev1 = max(30, min(120, base_fev1 + random.gauss(0, 15)))

        # Comorbidities (more likely with age and poor PS)
        num_comorbidities = random.choices(
            [0, 1, 2, 3],
            weights=[0.3, 0.35, 0.25, 0.10]
        )[0]
        comorbidities = random.sample(self.COMORBIDITIES, num_comorbidities)

        # Diagnosis date
        days_ago = random.randint(30, 365)
        diagnosis_date = datetime.now() - timedelta(days=days_ago)

        return SyntheticPatient(
            patient_id=str(uuid.uuid4())[:8].upper(),
            name=f"{first_name} {last_name}",
            sex=sex,
            age_at_diagnosis=age,
            tnm_stage=stage,
            histology_type=histology,
            performance_status=ps,
            laterality=laterality,
            diagnosis_date=diagnosis_date.isoformat(),
            fev1_percent=round(fev1, 1),
            comorbidities=comorbidities
        )

    def generate_cohort(self, n: int = 100) -> List[SyntheticPatient]:
        """Generate a cohort of synthetic patients"""
        return [self.generate_patient() for _ in range(n)]


if __name__ == "__main__":
    generator = SyntheticPatientGenerator()

    # Generate 100 synthetic patients
    patients = generator.generate_cohort(100)

    # Export to JSON
    patient_data = [p.to_dict() for p in patients]

    with open("synthetic_patients.json", "w") as f:
        json.dump(patient_data, f, indent=2)

    print(f"✓ Generated {len(patients)} synthetic patients")
    print(f"\nStage distribution:")
    for stage in ["IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IV"]:
        count = sum(1 for p in patients if p.tnm_stage == stage)
        print(f"  {stage}: {count}")

    print(f"\nHistology distribution:")
    hist_types = set(p.histology_type for p in patients)
    for hist in sorted(hist_types):
        count = sum(1 for p in patients if p.histology_type == hist)
        print(f"  {hist}: {count}")
