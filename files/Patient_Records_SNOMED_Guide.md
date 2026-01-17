# Patient Records Sources & SNOMED-CT Integration Guide

## 1. Patient Records Sources

### 1.1 Original LUCADA Database (Not Publicly Available)

The paper used the **LUCADA (Lung Cancer Data)** database maintained by the NHS England National Lung Cancer Audit. This contains ~115,000+ patient records with:
- Demographics (age, sex)
- TNM staging
- Histology type
- Performance status
- Treatment received
- Outcomes/survival data

**Access:** Requires formal application through NHS England Data Access Request Service (DARS)
- URL: https://digital.nhs.uk/services/data-access-request-service-dars
- Typically restricted to UK-based researchers with ethical approval

---

### 1.2 Publicly Available Alternatives

#### A. TCGA (The Cancer Genome Atlas) - **RECOMMENDED**

| Dataset | Description | Size | Access |
|---------|-------------|------|--------|
| **TCGA-LUAD** | Lung Adenocarcinoma | ~500 patients | Free |
| **TCGA-LUSC** | Lung Squamous Cell Carcinoma | ~500 patients | Free |

**Contains:**
- Clinical data (age, sex, TNM staging, survival)
- Histopathology images
- Genomic data
- Treatment information

**How to Access:**
```
1. Visit: https://portal.gdc.cancer.gov/
2. Filter by Project: TCGA-LUAD or TCGA-LUSC
3. Download clinical data (JSON/TSV format)
4. Access imaging data via: https://www.cancerimagingarchive.net/
```

**Sample TCGA Clinical Fields:**
```json
{
  "case_id": "TCGA-05-4244",
  "demographic": {
    "gender": "male",
    "age_at_diagnosis": 68,
    "race": "white"
  },
  "diagnoses": [{
    "primary_diagnosis": "Adenocarcinoma, NOS",
    "tumor_stage": "Stage IIA",
    "ajcc_staging_system_edition": "7th",
    "ajcc_pathologic_t": "T2a",
    "ajcc_pathologic_n": "N0",
    "ajcc_pathologic_m": "M0"
  }],
  "treatments": [{
    "treatment_type": "Pharmaceutical Therapy",
    "treatment_or_therapy": "yes"
  }]
}
```

#### B. SEER (Surveillance, Epidemiology, and End Results)

**Size:** 344,797+ lung cancer cases (2004-2010 data alone)

**Contains:**
- Population-based cancer statistics
- TNM staging data
- Survival outcomes
- Demographic information

**Access:**
```
1. Visit: https://seer.cancer.gov/
2. Apply for SEER*Stat software access
3. Submit research agreement
4. Download customized datasets
```

**Limitations:** No individual-level treatment details

#### C. NLST (National Lung Screening Trial)

**Size:** 53,454 participants (26,722 in LDCT arm)

**Contains:**
- Low-dose CT images
- Clinical outcomes
- Pixel-level tumor annotations (NLSTseg subset)

**Access:** https://cdas.cancer.gov/nlst/

#### D. cBioPortal Datasets

**URL:** https://www.cbioportal.org/datasets

**Lung Cancer Collections:**
- MSK-IMPACT Clinical Sequencing Cohort
- Lung Cancer (MSKCC, 2020)
- Pan-Lung Cancer (TCGA, Nature 2014)

#### E. Kaggle Lung Cancer Datasets

**URL:** https://www.kaggle.com/search?q=lung+cancer

Various synthetic and real datasets for research/learning

---

### 1.3 Creating Synthetic Patient Data

For development/testing, generate synthetic patients that match LUCADA schema:

```python
# synthetic_patients.py

import random
import uuid
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta

@dataclass
class SyntheticPatient:
    patient_id: str
    name: str
    sex: str
    age_at_diagnosis: int
    tnm_stage: str
    histology_type: str
    performance_status: int
    laterality: str
    diagnosis_date: datetime
    fev1_percent: Optional[float] = None
    comorbidities: List[str] = None
    
    def to_dict(self):
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "sex": self.sex,
            "age": self.age_at_diagnosis,
            "tnm_stage": self.tnm_stage,
            "histology_type": self.histology_type,
            "performance_status": self.performance_status,
            "laterality": self.laterality,
            "diagnosis_date": self.diagnosis_date.isoformat(),
            "fev1_percent": self.fev1_percent,
            "comorbidities": self.comorbidities or []
        }

class SyntheticPatientGenerator:
    """Generate realistic synthetic lung cancer patients."""
    
    TNM_STAGES = ["IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IV"]
    TNM_WEIGHTS = [0.10, 0.08, 0.12, 0.10, 0.15, 0.15, 0.30]  # Stage IV most common
    
    HISTOLOGY_TYPES = [
        ("Adenocarcinoma", 0.40),          # Most common NSCLC
        ("SquamousCellCarcinoma", 0.25),   # Second most common
        ("SmallCellCarcinoma", 0.15),      # SCLC
        ("LargeCellCarcinoma", 0.10),      
        ("Carcinosarcoma", 0.05),
        ("NonSmallCellCarcinoma_NOS", 0.05)
    ]
    
    PERFORMANCE_STATUS_BY_STAGE = {
        "IA": [0, 0, 0, 1, 1],      # Mostly good PS for early stage
        "IB": [0, 0, 1, 1, 1],
        "IIA": [0, 1, 1, 1, 2],
        "IIB": [0, 1, 1, 2, 2],
        "IIIA": [1, 1, 2, 2, 2],
        "IIIB": [1, 2, 2, 2, 3],
        "IV": [1, 2, 2, 3, 3, 4]    # Worse PS for advanced stage
    }
    
    COMORBIDITIES = [
        "COPD", "Cardiovascular_Disease", "Diabetes", 
        "Hypertension", "Renal_Disease", "Previous_Cancer"
    ]
    
    FIRST_NAMES_M = ["James", "John", "Robert", "Michael", "William", "David"]
    FIRST_NAMES_F = ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Susan"]
    LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"]
    
    def generate_patient(self) -> SyntheticPatient:
        """Generate a single synthetic patient."""
        
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
            name=f"{first_name}_{last_name}",
            sex=sex,
            age_at_diagnosis=age,
            tnm_stage=stage,
            histology_type=histology,
            performance_status=ps,
            laterality=laterality,
            diagnosis_date=diagnosis_date,
            fev1_percent=round(fev1, 1),
            comorbidities=comorbidities
        )
    
    def generate_cohort(self, n: int = 100) -> List[SyntheticPatient]:
        """Generate a cohort of synthetic patients."""
        return [self.generate_patient() for _ in range(n)]


# Usage example
if __name__ == "__main__":
    generator = SyntheticPatientGenerator()
    
    # Generate 100 synthetic patients
    patients = generator.generate_cohort(100)
    
    # Export to JSON
    import json
    patient_data = [p.to_dict() for p in patients]
    
    with open("synthetic_patients.json", "w") as f:
        json.dump(patient_data, f, indent=2)
    
    print(f"Generated {len(patients)} synthetic patients")
    print(f"Stage distribution:")
    for stage in ["IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IV"]:
        count = sum(1 for p in patients if p.tnm_stage == stage)
        print(f"  {stage}: {count}")
```

---

## 2. SNOMED-CT Integration

### 2.1 What is SNOMED-CT?

**SNOMED CT** (Systematized Nomenclature of Medicine Clinical Terms) is the most comprehensive clinical terminology in the world, containing:
- 350,000+ concepts
- 1.5M+ relationships
- Hierarchical structure with formal definitions
- Available as OWL 2 ontology

### 2.2 How SNOMED-CT is Used in LUCADA Ontology

The paper's LUCADA ontology imports a **SNOMED-CT module** for:

1. **Clinical Finding concepts** (diagnoses)
2. **Body Structure concepts** (anatomy)
3. **Procedure concepts** (treatments)
4. **Morphology concepts** (histology)

### 2.3 Key SNOMED-CT Codes for Lung Cancer

Based on my OLS4 search, here are the relevant SNOMED codes:

```
LUNG CANCER DIAGNOSIS CODES
├── 254637007 - Non-small cell lung cancer (NSCLC)
│   ├── 424132000 - NSCLC, TNM stage 1
│   ├── 425048006 - NSCLC, TNM stage 2
│   ├── 422968005 - NSCLC, TNM stage 3
│   ├── 423121009 - NSCLC, TNM stage 4
│   ├── 723301009 - Squamous NSCLC
│   ├── 1255725002 - Non-small cell adenocarcinoma
│   ├── 426964009 - EGFR positive NSCLC
│   ├── 427038005 - EGFR negative NSCLC
│   └── 830151004 - ALK fusion positive NSCLC
│
├── 254632001 - Small cell lung cancer (SCLC)
│
└── 363358000 - Malignant neoplasm of lung (parent class)

HISTOLOGY CODES
├── 35917007 - Adenocarcinoma
├── 59367005 - Squamous cell carcinoma
├── 254632001 - Small cell carcinoma
└── 67101007 - Large cell carcinoma

PROCEDURE CODES
├── 387713003 - Surgical procedure
│   ├── 173171007 - Lobectomy of lung
│   └── 49795001 - Pneumonectomy
├── 367336001 - Chemotherapy
├── 108290001 - Radiation oncology
└── 103735009 - Palliative care
```

### 2.4 Accessing SNOMED-CT

#### Option A: SNOMED International (Official)

**URL:** https://www.snomed.org/get-snomed

- Requires membership (free for many countries)
- Full OWL 2 release available
- Updated twice yearly

#### Option B: NCBO BioPortal (Free Access)

**URL:** https://bioportal.bioontology.org/ontologies/SNOMEDCT

```python
# Query BioPortal API
import requests

API_KEY = "your-api-key"  # Get free key at bioportal.bioontology.org/accounts/new

def search_snomed(term):
    url = "https://data.bioontology.org/search"
    params = {
        "q": term,
        "ontologies": "SNOMEDCT",
        "apikey": API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

# Example: Find lung cancer codes
results = search_snomed("non-small cell lung cancer")
```

#### Option C: OLS4 API (EBI - Used in Your Setup)

You already have access via your MCP tools!

```python
# Using OLS4 (what you have connected)
from your_mcp_client import ols4_search, ols4_fetch, ols4_get_descendants

# Search for lung cancer concepts
results = ols4_search(query="lung cancer SNOMED")

# Get descendants of NSCLC
descendants = ols4_get_descendants(
    class_iri="http://snomed.info/id/254637007",
    ontology_id="snomed"
)
```

### 2.5 Integrating SNOMED-CT in Your Implementation

```python
# snomed_integration.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import requests

@dataclass
class SNOMEDConcept:
    """Represents a SNOMED-CT concept."""
    sctid: str          # SNOMED CT Identifier
    fsn: str            # Fully Specified Name
    preferred_term: str
    semantic_tag: str   # e.g., "disorder", "procedure", "finding"
    parents: List[str]
    
class SNOMEDLungCancerModule:
    """
    SNOMED-CT module for lung cancer decision support.
    Maps clinical data to SNOMED codes for ontological reasoning.
    """
    
    # Core concept mappings
    HISTOLOGY_MAP = {
        "Adenocarcinoma": "35917007",
        "SquamousCellCarcinoma": "59367005", 
        "SmallCellCarcinoma": "254632001",
        "LargeCellCarcinoma": "67101007",
        "NonSmallCellCarcinoma": "254637007",
        "Carcinosarcoma": "128885008"
    }
    
    STAGE_MAP = {
        "IA": "424132000",  # NSCLC Stage 1
        "IB": "424132000",
        "IIA": "425048006", # NSCLC Stage 2
        "IIB": "425048006",
        "IIIA": "422968005", # NSCLC Stage 3
        "IIIB": "422968005",
        "IV": "423121009"    # NSCLC Stage 4
    }
    
    TREATMENT_MAP = {
        "Surgery": "387713003",
        "Chemotherapy": "367336001",
        "Radiotherapy": "108290001",
        "Chemoradiotherapy": "703423002",
        "PalliativeCare": "103735009",
        "Immunotherapy": "76334006"
    }
    
    PERFORMANCE_STATUS_MAP = {
        0: "373803006",  # WHO PS Grade 0
        1: "373804000",  # WHO PS Grade 1
        2: "373805004",  # WHO PS Grade 2
        3: "373806003",  # WHO PS Grade 3
        4: "373807007"   # WHO PS Grade 4
    }
    
    def __init__(self, ols_api_base: str = "https://www.ebi.ac.uk/ols4/api"):
        self.ols_api = ols_api_base
    
    def get_snomed_code(self, category: str, value: str) -> Optional[str]:
        """Get SNOMED code for a clinical value."""
        maps = {
            "histology": self.HISTOLOGY_MAP,
            "stage": self.STAGE_MAP,
            "treatment": self.TREATMENT_MAP
        }
        return maps.get(category, {}).get(value)
    
    def get_snomed_iri(self, sctid: str) -> str:
        """Convert SCTID to full IRI."""
        return f"http://snomed.info/id/{sctid}"
    
    def search_concepts(self, term: str) -> List[Dict]:
        """Search SNOMED-CT via OLS4."""
        url = f"{self.ols_api}/search"
        params = {
            "q": term,
            "ontology": "snomed",
            "rows": 20
        }
        response = requests.get(url, params=params)
        if response.ok:
            data = response.json()
            return data.get("response", {}).get("docs", [])
        return []
    
    def get_ancestors(self, sctid: str) -> List[str]:
        """Get ancestor concepts (for subsumption reasoning)."""
        iri = self.get_snomed_iri(sctid)
        url = f"{self.ols_api}/ontologies/snomed/terms/{iri}/ancestors"
        response = requests.get(url)
        if response.ok:
            data = response.json()
            return [
                item.get("obo_id", "").replace("SNOMED:", "")
                for item in data.get("_embedded", {}).get("terms", [])
            ]
        return []
    
    def is_subtype_of(self, child_sctid: str, parent_sctid: str) -> bool:
        """Check if one concept is a subtype of another."""
        ancestors = self.get_ancestors(child_sctid)
        return parent_sctid in ancestors
    
    def map_patient_to_snomed(self, patient_data: Dict) -> Dict[str, str]:
        """Map patient clinical data to SNOMED codes."""
        return {
            "diagnosis": self.get_snomed_code(
                "histology", patient_data.get("histology_type")
            ),
            "stage": self.get_snomed_code(
                "stage", patient_data.get("tnm_stage")
            ),
            "performance_status": self.PERFORMANCE_STATUS_MAP.get(
                patient_data.get("performance_status")
            )
        }
    
    def generate_owl_expression(self, patient_data: Dict) -> str:
        """
        Generate OWL 2 class expression for patient.
        This enables ontological classification.
        """
        histology_code = self.get_snomed_code("histology", patient_data.get("histology_type"))
        stage = patient_data.get("tnm_stage", "")
        ps = patient_data.get("performance_status", 0)
        
        # Build OWL expression (matches paper's format)
        expression = f"""
        (hasClinicalFinding some 
            (NeoplasticDisease and 
                (hasPreTNMStaging value "{stage}") and 
                (hasHistology some <{self.get_snomed_iri(histology_code)}>)
            )
        ) 
        and 
        (hasPerformanceStatus some <{self.get_snomed_iri(self.PERFORMANCE_STATUS_MAP[ps])}>)
        """
        return expression.strip()


# Usage example
if __name__ == "__main__":
    snomed = SNOMEDLungCancerModule()
    
    # Example patient
    patient = {
        "histology_type": "Adenocarcinoma",
        "tnm_stage": "IIIA",
        "performance_status": 1
    }
    
    # Map to SNOMED codes
    codes = snomed.map_patient_to_snomed(patient)
    print(f"SNOMED Codes: {codes}")
    
    # Generate OWL expression
    owl_expr = snomed.generate_owl_expression(patient)
    print(f"OWL Expression:\n{owl_expr}")
    
    # Check if adenocarcinoma is subtype of NSCLC
    is_nsclc = snomed.is_subtype_of("35917007", "254637007")
    print(f"Adenocarcinoma is NSCLC: {is_nsclc}")
```

### 2.6 Using SNOMED in Neo4j

```cypher
// Create SNOMED concept nodes
CREATE (c:SNOMEDConcept {
    sctid: "254637007",
    fsn: "Non-small cell lung cancer (disorder)",
    preferred_term: "NSCLC"
})

// Link patient diagnosis to SNOMED
MATCH (p:Patient {patient_id: "ABC123"})
MATCH (s:SNOMEDConcept {sctid: "254637007"})
CREATE (p)-[:HAS_DIAGNOSIS {snomed_code: "254637007"}]->(s)

// Query: Find all NSCLC patients
MATCH (p:Patient)-[:HAS_DIAGNOSIS]->(s:SNOMEDConcept)
WHERE s.sctid IN ["254637007", "35917007", "59367005"]
RETURN p.patient_id, p.name, s.preferred_term
```

---

## 3. Summary: Recommended Approach

### For Development/Testing:
1. Generate **synthetic patients** using the provided generator
2. Use **TCGA-LUAD** clinical data for realistic distribution patterns
3. Access SNOMED via **OLS4 API** (already in your MCP setup)

### For Production:
1. Apply for **SEER** or **TCGA** data access
2. Obtain **SNOMED-CT license** via SNOMED International
3. Consider **FHIR integration** for EHR connectivity

### Data Pipeline:
```
Patient EHR → Map to SNOMED codes → Store in Neo4j → 
Classify with OWL → Generate recommendations → Display in UI
```
