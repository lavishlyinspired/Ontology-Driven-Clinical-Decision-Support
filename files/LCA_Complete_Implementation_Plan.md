# Lung Cancer Assistant (LCA) - Modern Implementation Plan

## Part 1: Complete Paper Analysis

---

## Executive Summary

This document provides a complete analysis of the "Lung Cancer Assistant" paper (Sesen et al., University of Oxford) and a detailed implementation plan using modern agentic AI frameworks, GenAI technologies, and contemporary software architecture patterns.

---

## 1. Paper Overview

**Title:** Lung Cancer Assistant: An Ontology-Driven, Online Decision Support Prototype for Lung Cancer Treatment Selection

**Authors:** M. Berkan Sesen, Rene Banares-Alcantara, John Fox, Timor Kadir, J. Michael Brady (University of Oxford)

**Core Contribution:** An OWL 2-based clinical decision support system that:
1. Models the LUCADA lung cancer database as an ontology
2. Formalizes NICE clinical guidelines as OWL 2 class expressions
3. Uses ontological inference for patient classification and treatment recommendations
4. Provides argumentation-based decision support

---

## 2. Problem Domain

### 2.1 Clinical Context
- Lung cancer: most common cancer type, 21% of cancer-related deaths
- Multidisciplinary Teams (MDTs) face complex treatment decisions
- Need for guideline-compliant, evidence-based recommendations
- ~115,000 patient records in LUCADA database

### 2.2 Technical Challenges
- No existing CIG formalism supported probabilistic inference integration
- Many CIG formalisms discontinued or lacking execution engines
- Need for standardized vocabulary (SNOMED-CT integration)
- Scalability issues with 115K+ patient records

---

## 3. LUCADA Ontology Architecture

### 3.1 Ontology Statistics

| Component | Count |
|-----------|-------|
| Classes | 396 |
| Object Properties | 35 |
| Datatype Properties | 60 |
| Patient Records | ~115,000 |

### 3.2 Class Hierarchy (from Figure 1)

```
SNOMED-CT Domain (Orange nodes in paper)
├── Patient
│   └── Data Properties:
│       - Sex
│       - Age_Diagnosis
│       - MDT_Discussion_Indicator
│       - MDT_Discussion_Date
│       - FEV1AbsoluteAmount
│       - Clinical_Trial_Status
│       - Valid_Diagnosis_Date_Present
│       - Valid_Death_Date_Present
│       - Survival_Cohort
│       - Survival[days]
│
├── Patient Referral
│   └── Data Properties:
│       - Referral_Decision_Date
│       - First_Seen_Date
│       - Place_First_Seen_Site_Code
│       - Referral_Date
│
├── Treatment Plan
│   └── Data Properties:
│       - Treatment_Plan_Type
│
├── Clinical Finding
│   └── Data Properties:
│       - TNM_Staging_Version
│       - Diagnosis_Site_Code
│       - Basis_of_Diagnosis
│       - Has_Pre_TNM_Staging
│       - Pre_Site_Specific_Staging_Class
│       - Investigation_Result_Date
│       - Has_Post_TNM_Staging
│       - Post_Site_Specific_Staging_Class
│
├── Body Structure
│   └── Neoplasm
│
├── Procedure
│   ├── Evaluation Procedure
│   │   ├── CT Scan
│   │   ├── PET Scan
│   │   ├── Bronchoscopy
│   │   └── CT Guided Biopsy
│   │
│   └── Therapeutic Procedure
│       ├── Surgery (Surgery_Site_Code, Surgery_Decision_Date, etc.)
│       ├── Chemotherapy (Chemo_Site_Code, Decision_to_Treat_Date, etc.)
│       ├── Brachytherapy
│       ├── Teletherapy/Radiotherapy
│       ├── Palliative Care
│       └── Active Monitoring
│
├── Outcome
│   └── Data Properties:
│       - Was_Death_Related_to_Treatment
│       - Cancer_Morbidity_Type
│       - PCI_Status
│       - Original_Treatment_Plan_Carried_Out
│       - Treatment_Failure_Reason
│
└── Histology (Carcinosarcoma, NonSmallCellCarcinoma, etc.)

Argumentation Domain (Black nodes in paper)
├── Patient Scenario (subclass of Patient AND Argumentation)
├── Argument
├── Decision
└── Intent
```

### 3.3 Key Object Properties

| Property | Domain | Range | Purpose |
|----------|--------|-------|---------|
| hasClinicalFinding | Patient | Clinical Finding | Links patient to diagnoses |
| hasCancerReferral | Patient | Patient Referral | Links to referral info |
| hasTreatmentPlan | Patient | Treatment Plan | Links to treatment plans |
| includesTreatment | Treatment Plan | Procedure | Links plan to procedures |
| hasProcedureSite | Procedure | Body Structure | Anatomical location |
| hasHistology | Clinical Finding | Histology | Tumor type classification |
| laterality | Body Structure | Side | Left/Right tumor location |
| hasOutcome | Treatment Plan | Outcome | Treatment results |
| resultsInArgument | Patient Scenario | Argument | Guideline reasoning |
| supports/opposesDecision | Argument | Decision | Pro/con arguments |
| entails | Decision | Treatment Plan | Recommended treatment |

---

## 4. Patient Representation Model (Figure 2)

### 4.1 Example Patient: Jenny_Sesen

**Database Record:**
```
DB Identifier: 200312
Name: Jenny_Sesen
Sex: Female (F)
Age: 72
Primary Diagnosis: Malignant Neoplasm of Lung
TNM Staging: Stage II A
Histology: Carcinosarcoma
Tumour Laterality: Right
```

### 4.2 Ontological Representation

```
Patient Individual: Jenny_Sesen
├── Data Properties:
│   ├── DB_Identifier: 200312
│   ├── Sex: Female
│   └── Age_Diagnosis: 72
│
├── hasClinicalFinding → Cancer Individual: Cancer_Jenny_Sesen
│   ├── Type: Malignant Neoplasm of Lung
│   ├── Data Properties:
│   │   ├── Diagnosis_Site_Code: R13OH
│   │   ├── Basis_of_Diagnosis: Clinical
│   │   └── Has_TNM_Staging: Stage II A
│   │
│   └── hasHistology → Histology Individual: Tumour_Jenny_Sesen
│       └── Type: Carcinosarcoma
│
└── laterality → Reference Individual: Reference_Right
    └── Type: Side (Right)
```

### 4.3 Key Design Patterns
1. **Patient-specific individuals** for unique patient data
2. **Reference individuals** for standard values (Side, Severity, etc.)
3. **Compound concept definitions** for complex data items

---

## 5. Guideline Rule Formalization

### 5.1 Rule Structure

Each guideline rule has two components:
- **Head (Antecedent):** Specifies patient eligibility criteria
- **Body (Consequent):** Specifies recommended action(s)

### 5.2 Example Rule R1 from NICE Guidelines

**Natural Language:**
> "Offer chemotherapy to patients with stage III or IV NSCLC and good performance status (WHO 0, 1 or a Karnofsky score of 80–100)."

**OWL 2 Expression (E1):**
```owl
(hasClinicalFinding some 
    (NeoplasticDisease and 
        ((hasPreTNMStaging value "III") or (hasPreTNMStaging value "IV")) and 
        (hasPreHistology some NonSmallCellCarcinoma)
    )
) 
and 
(hasPerformanceStatus some 
    (WHOPerfStatusGrade0 or WHOPerfStatusGrade1)
)
```

**Plain English Translation:**
"Patients whose performance status is either 0 or 1, AND who have Neoplastic Disease with:
- TNM staging of either III or IV, AND
- Histology finding type of Non-Small Cell Carcinoma"

### 5.3 Rule Execution Flow (Figure 3)

```
┌─────────────────────┐     SUPPORTSDECISION     ┌─────────────────────┐
│   Patient Scenario  │ ─────────────────────────►│      Decision       │
│   ┌───────────────┐ │                           │ ┌─────────────────┐ │
│   │ ChemoRule001  │ │                           │ │ChemoPlanDecision│ │
│   │ ◇ Reference_  │ │                           │ │001              │ │
│   │   ChemoRule001│ │                           │ │ ◇ Reference_    │ │
│   └───────────────┘ │                           │ │   ChemoPlan...  │ │
└─────────────────────┘                           │ └─────────────────┘ │
                                                  └──────────┬──────────┘
                                                             │
                                                             │ ENTAILS
                                                             ▼
                                                  ┌─────────────────────┐
                                                  │    Treatment Plan   │
                                                  │ ┌─────────────────┐ │
                                                  │ │ ChemotherapyPlan│ │
                                                  │ │ ◇ Reference_    │ │
                                                  │ │   ChemoTherapy  │ │
                                                  │ │   Plan          │ │
                                                  │ └─────────────────┘ │
                                                  └─────────────────────┘
```

**Process:**
1. Patient Scenario class defined with equivalent class expression (the rule criteria)
2. Ontology reasoner classifies all matching patients into the Patient Scenario
3. Reference individual links Patient Scenario → Decision → Treatment Plan
4. System retrieves applicable treatment recommendations for each patient

---

## 6. System Architecture (Original)

### 6.1 Technical Stack
- **Language:** Java v.6
- **Ontology API:** OWL API v.3.2.3
- **Web Framework:** GWT SDK
- **Reasoners Tested:** HermiT, Fact++, Pellet
- **Database:** LUCADA (~115,000 records)

### 6.2 Scalability Solution

**Problem:** Full ontology (1 GB, 700K+ individuals) couldn't be classified by any reasoner.

**Solution:** Temporary Patient Classification
1. Create new patient → Add to ontology temporarily
2. Run reasoner classification
3. Store inferred Patient Scenario memberships in database
4. Remove patient from ontology
5. Repeat for each patient

### 6.3 User Interface (Figure 4)

**LCA Treatment Tab Features:**
- Patient summary header (ID, Age, Sex, Primary Diagnosis, Pre-op Staging, Histology, Performance Status)
- Tab navigation (Patient, Care Plan/MDT, Key Investigations, Treatment, Outcome, Decision Support, Inference)
- Treatment sub-tabs (Surgery, Chemotherapy, Radiotherapy, Brachytherapy, Palliative Care, Active Monitoring)
- Treatment Options panel with:
  - Expandable treatment recommendations
  - Rule-based arguments with survival rates
  - Supporting/opposing arguments per treatment
# LCA Implementation Plan - Part 2: Modern Architecture

---

## 7. Modern Technology Stack

### 7.1 Technology Mapping

| Component | Original (2011) | Modern (2025) |
|-----------|-----------------|---------------|
| Language | Java 6 | Python 3.11+ / TypeScript |
| Ontology API | OWL API 3.2.3 | Owlready2, RDFLib, py-horned-owl |
| Web Framework | GWT | FastAPI + Next.js/React |
| Reasoner | HermiT/Pellet | ELK, HermiT (via py4j), LLM-based |
| Database | Relational | Neo4j + PostgreSQL (hybrid) |
| AI/ML | None | LangChain, LangGraph, Claude/GPT-4 |
| Vector Store | None | Chroma, Pinecone, or Neo4j Vector |
| Deployment | On-premise | Docker + Kubernetes |

### 7.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Modern LCA Architecture                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐  │
│  │   Frontend (Next.js)  │    │       API Gateway (FastAPI)      │  │
│  │   - Patient Dashboard │◄──►│   - REST/GraphQL endpoints       │  │
│  │   - Decision Support  │    │   - WebSocket for real-time      │  │
│  │   - Treatment Browser │    │   - Authentication/Authorization │  │
│  └──────────────────────┘    └──────────────┬───────────────────┘  │
│                                              │                       │
│  ┌───────────────────────────────────────────┼───────────────────┐  │
│  │                 Agentic Layer (LangGraph)                      │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │  │
│  │  │ Guideline   │  │ Treatment   │  │ Argumentation       │   │  │
│  │  │ Reasoning   │  │ Selection   │  │ Agent               │   │  │
│  │  │ Agent       │  │ Agent       │  │                     │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │  │
│  │  │ Patient     │  │ Ontology    │  │ Explanation         │   │  │
│  │  │ Classifier  │  │ Query       │  │ Generator           │   │  │
│  │  │ Agent       │  │ Agent       │  │ Agent               │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │  │
│  └───────────────────────────────────────────┼───────────────────┘  │
│                                              │                       │
│  ┌───────────────────────────────────────────┼───────────────────┐  │
│  │                   Knowledge Layer                              │  │
│  │  ┌─────────────────┐  ┌────────────────┐  ┌───────────────┐  │  │
│  │  │  Neo4j Graph    │  │  Vector Store  │  │  OWL Ontology │  │  │
│  │  │  - Patient Data │  │  - Guidelines  │  │  - SNOMED-CT  │  │  │
│  │  │  - Relationships│  │  - Literature  │  │  - LUCADA     │  │  │
│  │  │  - Temporal     │  │  - Embeddings  │  │  - Rules      │  │  │
│  │  └─────────────────┘  └────────────────┘  └───────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Implementation Phases

### Phase 1: Foundation & Ontology Layer (Weeks 1-4)
- Project setup and infrastructure
- LUCADA ontology implementation in Owlready2
- Neo4j schema design and implementation
- Guideline rules formalization

### Phase 2: Agentic AI Layer (Weeks 5-8)
- LangGraph workflow design
- Agent implementation (Classification, Recommendation, Argumentation, Explanation)
- Tool definitions and integration
- Vector embeddings for semantic search

### Phase 3: API & Frontend (Weeks 9-12)
- FastAPI backend implementation
- Next.js frontend development
- Real-time updates via WebSocket
- Authentication and authorization

### Phase 4: Testing & Deployment (Weeks 13-16)
- Unit and integration testing
- Clinical validation
- Docker containerization
- Kubernetes deployment

---

## 9. Project Structure

```
lung-cancer-assistant/
├── backend/
│   ├── src/
│   │   ├── ontology/
│   │   │   ├── __init__.py
│   │   │   ├── lucada_ontology.py      # OWL ontology management
│   │   │   ├── guideline_rules.py      # Clinical guidelines
│   │   │   └── snomed_module.py        # SNOMED-CT integration
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── lca_agents.py           # LangGraph agents
│   │   │   ├── classification.py       # Patient classification
│   │   │   ├── recommendation.py       # Treatment recommendations
│   │   │   ├── argumentation.py        # Clinical arguments
│   │   │   └── explanation.py          # MDT summaries
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── main.py                 # FastAPI app
│   │   │   ├── routes/
│   │   │   │   ├── patients.py
│   │   │   │   ├── treatments.py
│   │   │   │   └── guidelines.py
│   │   │   └── middleware/
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── neo4j_schema.py         # Neo4j operations
│   │   │   ├── postgres_models.py      # PostgreSQL models
│   │   │   └── vector_store.py         # Embeddings store
│   │   └── services/
│   │       ├── __init__.py
│   │       └── lca_service.py          # Main orchestration
│   ├── tests/
│   │   ├── test_ontology.py
│   │   ├── test_agents.py
│   │   └── test_api.py
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx
│   │   │   ├── patients/
│   │   │   └── guidelines/
│   │   ├── components/
│   │   │   ├── PatientForm.tsx
│   │   │   ├── TreatmentRecommendations.tsx
│   │   │   ├── ArgumentPanel.tsx
│   │   │   └── MDTSummary.tsx
│   │   └── lib/
│   │       ├── api.ts
│   │       └── types.ts
│   ├── package.json
│   └── Dockerfile
├── ontology/
│   ├── lucada.owl                      # Main ontology file
│   ├── snomed_module.owl               # SNOMED-CT extract
│   └── guidelines/
│       ├── nice_2011.owl
│       └── bts_guidelines.owl
├── data/
│   ├── sample_patients.json
│   └── guideline_rules.json
├── docker-compose.yml
├── kubernetes/
│   ├── deployment.yaml
│   └── services.yaml
└── README.md
```

---

## 10. Dependencies

### 10.1 Python Backend (requirements.txt)

```
# Core
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
python-dotenv==1.0.0

# Ontology
owlready2==0.45
rdflib==7.0.0

# AI/ML
langchain==0.1.0
langchain-anthropic==0.1.0
langchain-openai==0.0.5
langgraph==0.0.20
anthropic==0.18.0
openai==1.10.0

# Database
neo4j==5.16.0
psycopg2-binary==2.9.9
sqlalchemy==2.0.25

# Vector Store
chromadb==0.4.22

# Utilities
httpx==0.26.0
pyyaml==6.0.1
numpy==1.26.3
pandas==2.1.4
```

### 10.2 Node.js Frontend (package.json)

```json
{
  "name": "lca-frontend",
  "version": "2.0.0",
  "dependencies": {
    "next": "14.1.0",
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "typescript": "5.3.3",
    "@tanstack/react-query": "5.17.0",
    "axios": "1.6.5",
    "tailwindcss": "3.4.1",
    "lucide-react": "0.312.0",
    "recharts": "2.10.3",
    "zod": "3.22.4",
    "react-hook-form": "7.49.3"
  }
}
```
# LCA Implementation Plan - Part 3: Core Implementation

---

## 11. LUCADA Ontology Implementation

```python
# backend/src/ontology/lucada_ontology.py

from owlready2 import *
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
import datetime

class TNMStage(Enum):
    STAGE_IA = "IA"
    STAGE_IB = "IB"
    STAGE_IIA = "IIA"
    STAGE_IIB = "IIB"
    STAGE_IIIA = "IIIA"
    STAGE_IIIB = "IIIB"
    STAGE_IV = "IV"

class WHOPerformanceStatus(Enum):
    GRADE_0 = 0  # Fully active
    GRADE_1 = 1  # Restricted but ambulatory
    GRADE_2 = 2  # Ambulatory, capable of self-care
    GRADE_3 = 3  # Limited self-care
    GRADE_4 = 4  # Completely disabled

class LUCADAOntology:
    """
    Modern implementation of the LUCADA Ontology using Owlready2.
    Based on SNOMED-CT module with domain-specific extensions.
    
    This implements the ontology structure shown in Figure 1 of the paper.
    """
    
    def __init__(self, ontology_path: str = "lucada.owl"):
        self.ontology_path = ontology_path
        self.onto = get_ontology(f"file://{ontology_path}")
        
        if not self.onto.loaded:
            self._create_ontology()
        else:
            self.onto.load()
    
    def _create_ontology(self):
        """Create the LUCADA ontology structure from scratch."""
        with self.onto:
            # ========================================
            # SNOMED-CT Clinical Domain Classes
            # ========================================
            
            # Base Patient Class
            class Patient(Thing):
                """SNOMED-CT Patient class with LUCADA extensions."""
                pass
            
            # Patient Data Properties (from Figure 1)
            class sex(DataProperty, FunctionalProperty):
                domain = [Patient]
                range = [str]
            
            class age_at_diagnosis(DataProperty, FunctionalProperty):
                domain = [Patient]
                range = [int]
            
            class mdt_discussion_indicator(DataProperty, FunctionalProperty):
                domain = [Patient]
                range = [bool]
            
            class fev1_absolute_amount(DataProperty, FunctionalProperty):
                domain = [Patient]
                range = [float]
            
            class clinical_trial_status(DataProperty, FunctionalProperty):
                domain = [Patient]
                range = [str]
            
            class survival_days(DataProperty, FunctionalProperty):
                domain = [Patient]
                range = [int]
            
            class survival_cohort(DataProperty, FunctionalProperty):
                domain = [Patient]
                range = [str]
            
            # Clinical Finding Classes
            class ClinicalFinding(Thing):
                """SNOMED-CT Clinical Finding."""
                pass
            
            class NeoplasticDisease(ClinicalFinding):
                """Malignant neoplasms including lung cancer."""
                pass
            
            class MalignantNeoplasmOfLung(NeoplasticDisease):
                """Primary lung cancer diagnosis."""
                pass
            
            class Comorbidity(ClinicalFinding):
                """Patient comorbidities."""
                pass
            
            class Dementia(Comorbidity):
                pass
            
            class CardiovascularDisease(Comorbidity):
                pass
            
            class SevereWeightLoss(Comorbidity):
                """Compound concept: Weight Loss Finding 'Severity' Severe"""
                pass
            
            # Clinical Finding Data Properties
            class tnm_staging_version(DataProperty):
                domain = [ClinicalFinding]
                range = [str]
            
            class diagnosis_site_code(DataProperty):
                domain = [ClinicalFinding]
                range = [str]
            
            class basis_of_diagnosis(DataProperty):
                domain = [ClinicalFinding]
                range = [str]
            
            class has_pre_tnm_staging(DataProperty):
                domain = [ClinicalFinding]
                range = [str]
            
            class has_post_tnm_staging(DataProperty):
                domain = [ClinicalFinding]
                range = [str]
            
            class pre_site_specific_staging_class(DataProperty):
                domain = [ClinicalFinding]
                range = [str]
            
            class investigation_result_date(DataProperty):
                domain = [ClinicalFinding]
                range = [datetime.datetime]
            
            # Histology Classes
            class Histology(Thing):
                """Tumor histology classification."""
                pass
            
            class NonSmallCellCarcinoma(Histology):
                """NSCLC histology type."""
                pass
            
            class SmallCellCarcinoma(Histology):
                """SCLC histology type."""
                pass
            
            class Carcinosarcoma(Histology):
                """Carcinosarcoma histology type."""
                pass
            
            class Adenocarcinoma(NonSmallCellCarcinoma):
                """Adenocarcinoma subtype of NSCLC."""
                pass
            
            class SquamousCellCarcinoma(NonSmallCellCarcinoma):
                """Squamous cell carcinoma subtype."""
                pass
            
            class LargeCellCarcinoma(NonSmallCellCarcinoma):
                """Large cell carcinoma subtype."""
                pass
            
            # Body Structure Classes
            class BodyStructure(Thing):
                """SNOMED-CT Body Structure."""
                pass
            
            class Neoplasm(BodyStructure):
                """Tumor body structure."""
                pass
            
            class Side(Thing):
                """Laterality reference class."""
                pass
            
            # Patient Referral Class
            class PatientReferral(Thing):
                """Patient referral information."""
                pass
            
            class referral_decision_date(DataProperty):
                domain = [PatientReferral]
                range = [datetime.datetime]
            
            class first_seen_date(DataProperty):
                domain = [PatientReferral]
                range = [datetime.datetime]
            
            class place_first_seen_site_code(DataProperty):
                domain = [PatientReferral]
                range = [str]
            
            # Treatment Plan Classes
            class TreatmentPlan(Thing):
                """Treatment plan container."""
                pass
            
            class treatment_plan_type(DataProperty):
                domain = [TreatmentPlan]
                range = [str]
            
            # Procedure Classes (from Figure 1)
            class Procedure(Thing):
                """SNOMED-CT Procedure."""
                pass
            
            class EvaluationProcedure(Procedure):
                """Diagnostic procedures."""
                pass
            
            class CTScan(EvaluationProcedure):
                pass
            
            class PETScan(EvaluationProcedure):
                pass
            
            class Bronchoscopy(EvaluationProcedure):
                pass
            
            class CTGuidedBiopsy(EvaluationProcedure):
                pass
            
            class TherapeuticProcedure(Procedure):
                """Treatment procedures."""
                pass
            
            class Surgery(TherapeuticProcedure):
                pass
            
            class Chemotherapy(TherapeuticProcedure):
                pass
            
            class Radiotherapy(TherapeuticProcedure):
                """Also known as Teletherapy."""
                pass
            
            class Brachytherapy(TherapeuticProcedure):
                pass
            
            class PalliativeCare(TherapeuticProcedure):
                pass
            
            class ActiveMonitoring(TherapeuticProcedure):
                pass
            
            # Procedure Data Properties
            class surgery_site_code(DataProperty):
                domain = [Surgery]
                range = [str]
            
            class surgery_decision_date(DataProperty):
                domain = [Surgery]
                range = [datetime.datetime]
            
            class excision_margin(DataProperty):
                domain = [Surgery]
                range = [str]
            
            class decision_to_treat_date(DataProperty):
                domain = [TherapeuticProcedure]
                range = [datetime.datetime]
            
            class treatment_start_date(DataProperty):
                domain = [TherapeuticProcedure]
                range = [datetime.datetime]
            
            # Outcome Class
            class Outcome(Thing):
                """Treatment outcome."""
                pass
            
            class was_death_related_to_treatment(DataProperty):
                domain = [Outcome]
                range = [bool]
            
            class cancer_morbidity_type(DataProperty):
                domain = [Outcome]
                range = [str]
            
            class pci_status(DataProperty):
                domain = [Outcome]
                range = [str]
            
            class original_treatment_plan_carried_out(DataProperty):
                domain = [Outcome]
                range = [bool]
            
            class treatment_failure_reason(DataProperty):
                domain = [Outcome]
                range = [str]
            
            # Performance Status Classes
            class PerformanceStatus(Thing):
                """WHO/ECOG Performance Status."""
                pass
            
            class WHOPerfStatusGrade0(PerformanceStatus):
                """Fully active."""
                pass
            
            class WHOPerfStatusGrade1(PerformanceStatus):
                """Restricted but ambulatory."""
                pass
            
            class WHOPerfStatusGrade2(PerformanceStatus):
                """Ambulatory, capable of self-care."""
                pass
            
            class WHOPerfStatusGrade3(PerformanceStatus):
                """Limited self-care."""
                pass
            
            class WHOPerfStatusGrade4(PerformanceStatus):
                """Completely disabled."""
                pass
            
            # ========================================
            # Object Properties
            # ========================================
            
            class has_clinical_finding(ObjectProperty):
                domain = [Patient]
                range = [ClinicalFinding]
            
            class has_cancer_referral(ObjectProperty):
                domain = [Patient]
                range = [PatientReferral]
            
            class has_treatment_plan(ObjectProperty):
                domain = [Patient]
                range = [TreatmentPlan]
            
            class includes_treatment(ObjectProperty):
                domain = [TreatmentPlan]
                range = [Procedure]
            
            class has_procedure_site(ObjectProperty):
                domain = [Procedure]
                range = [BodyStructure]
            
            class has_histology(ObjectProperty):
                domain = [ClinicalFinding]
                range = [Histology]
            
            class laterality(ObjectProperty):
                domain = [BodyStructure]
                range = [Side]
            
            class has_outcome(ObjectProperty):
                domain = [TreatmentPlan]
                range = [Outcome]
            
            class has_performance_status(ObjectProperty):
                domain = [Patient]
                range = [PerformanceStatus]
            
            # ========================================
            # Argumentation Domain Classes
            # ========================================
            
            class Argumentation(Thing):
                """Base argumentation class."""
                pass
            
            class PatientScenario(Patient, Argumentation):
                """
                Hybrid class representing a hypothetical patient cohort
                that fulfills guideline rule criteria.
                This is the key innovation from the paper.
                """
                pass
            
            class Argument(Argumentation):
                """Treatment argument (pro or con)."""
                pass
            
            class Decision(Argumentation):
                """Treatment decision."""
                pass
            
            class Intent(Argumentation):
                """Treatment intent."""
                pass
            
            # Argumentation Object Properties
            class results_in_argument(ObjectProperty):
                domain = [PatientScenario]
                range = [Argument]
            
            class supports_decision(ObjectProperty):
                domain = [Argument]
                range = [Decision]
            
            class opposes_decision(ObjectProperty):
                domain = [Argument]
                range = [Decision]
            
            class entails(ObjectProperty):
                domain = [Decision]
                range = [TreatmentPlan]
            
            class has_intent(ObjectProperty):
                domain = [TreatmentPlan]
                range = [Intent]
            
            # ========================================
            # Reference Individuals
            # ========================================
            
            # Side references
            Reference_Right = Side("Reference_Right")
            Reference_Left = Side("Reference_Left")
            Reference_Bilateral = Side("Reference_Bilateral")
            
            # Performance status references
            Reference_WHO_0 = WHOPerfStatusGrade0("Reference_WHO_0")
            Reference_WHO_1 = WHOPerfStatusGrade1("Reference_WHO_1")
            Reference_WHO_2 = WHOPerfStatusGrade2("Reference_WHO_2")
            Reference_WHO_3 = WHOPerfStatusGrade3("Reference_WHO_3")
            Reference_WHO_4 = WHOPerfStatusGrade4("Reference_WHO_4")
        
        self.onto.save()
    
    def create_patient_individual(
        self,
        patient_id: str,
        sex: str,
        age: int,
        diagnosis: str,
        tnm_stage: str,
        histology_type: str,
        laterality: str,
        performance_status: int
    ) -> Thing:
        """
        Create a patient individual in the ontology.
        Follows the pattern from Figure 2 in the paper.
        """
        with self.onto:
            # Create patient individual
            patient = self.onto.Patient(patient_id)
            patient.sex = sex
            patient.age_at_diagnosis = age
            
            # Create cancer finding individual
            cancer = self.onto.MalignantNeoplasmOfLung(f"Cancer_{patient_id}")
            cancer.has_pre_tnm_staging = tnm_stage
            cancer.basis_of_diagnosis = "Clinical"
            
            # Create histology individual - map string to class
            histology_mapping = {
                "NonSmallCellCarcinoma": self.onto.NonSmallCellCarcinoma,
                "SmallCellCarcinoma": self.onto.SmallCellCarcinoma,
                "Carcinosarcoma": self.onto.Carcinosarcoma,
                "Adenocarcinoma": self.onto.Adenocarcinoma,
                "SquamousCellCarcinoma": self.onto.SquamousCellCarcinoma,
                "LargeCellCarcinoma": self.onto.LargeCellCarcinoma,
            }
            histology_class = histology_mapping.get(histology_type, self.onto.Histology)
            tumor = histology_class(f"Tumour_{patient_id}")
            
            # Create body structure for laterality
            neoplasm = self.onto.Neoplasm(f"Neoplasm_{patient_id}")
            
            # Link histology to cancer finding
            cancer.has_histology.append(tumor)
            
            # Link cancer finding to patient
            patient.has_clinical_finding.append(cancer)
            
            # Set laterality using reference individual
            laterality_mapping = {
                "right": self.onto.Reference_Right,
                "left": self.onto.Reference_Left,
                "bilateral": self.onto.Reference_Bilateral,
            }
            neoplasm.laterality = laterality_mapping.get(
                laterality.lower(), 
                self.onto.Reference_Right
            )
            
            # Set performance status
            ps_mapping = {
                0: self.onto.Reference_WHO_0,
                1: self.onto.Reference_WHO_1,
                2: self.onto.Reference_WHO_2,
                3: self.onto.Reference_WHO_3,
                4: self.onto.Reference_WHO_4,
            }
            if performance_status in ps_mapping:
                patient.has_performance_status.append(ps_mapping[performance_status])
            
            return patient
    
    def run_reasoner(self):
        """Run OWL reasoner for classification."""
        with self.onto:
            sync_reasoner_pellet(infer_property_values=True)
    
    def get_patient_scenarios(self, patient: Thing) -> List[Thing]:
        """Get all Patient Scenario classes a patient belongs to."""
        return [
            cls for cls in patient.is_a 
            if isinstance(cls, type) and issubclass(cls, self.onto.PatientScenario)
        ]
    
    def remove_patient(self, patient_id: str):
        """Remove a patient from the ontology (for scalability)."""
        with self.onto:
            patient = self.onto.search_one(iri=f"*{patient_id}")
            if patient:
                destroy_entity(patient)
    
    def save(self, path: str = None):
        """Save ontology to file."""
        self.onto.save(file=path or self.ontology_path)
```

---

## 12. Guideline Rules Implementation

```python
# backend/src/ontology/guideline_rules.py

from owlready2 import *
from dataclasses import dataclass
from typing import List, Dict, Any
import types

@dataclass
class GuidelineRule:
    """
    Represents a clinical guideline rule with head (criteria) and body (action).
    Based on the rule structure from Section 4 of the paper.
    """
    rule_id: str
    name: str
    source: str  # e.g., "NICE 2011", "BTS Guidelines"
    description: str
    owl_expression: str  # OWL 2 class expression
    recommended_treatment: str
    evidence_level: str  # e.g., "Grade A", "Grade B", "Grade C"

class GuidelineRuleEngine:
    """
    Engine for creating and executing guideline rules as OWL 2 expressions.
    Implements the Patient Scenario pattern from Figure 3.
    """
    
    # NICE Lung Cancer 2011 Guideline Rules (extracted from paper)
    NICE_GUIDELINES = [
        GuidelineRule(
            rule_id="R1",
            name="ChemoRule001",
            source="NICE Lung Cancer 2011",
            description="Offer chemotherapy to patients with stage III or IV NSCLC and good performance status (WHO 0, 1)",
            owl_expression="""
                (has_clinical_finding some 
                    (NeoplasticDisease and 
                        ((has_pre_tnm_staging value "III") or 
                         (has_pre_tnm_staging value "IIIA") or 
                         (has_pre_tnm_staging value "IIIB") or 
                         (has_pre_tnm_staging value "IV")) and 
                        (has_histology some NonSmallCellCarcinoma)
                    )
                ) 
                and 
                (has_performance_status some (WHOPerfStatusGrade0 or WHOPerfStatusGrade1))
            """,
            recommended_treatment="Chemotherapy",
            evidence_level="Grade A"
        ),
        GuidelineRule(
            rule_id="R2",
            name="SurgeryRule001",
            source="NICE Lung Cancer 2011",
            description="Consider surgery for patients with stage I-II NSCLC and good performance status",
            owl_expression="""
                (has_clinical_finding some 
                    (NeoplasticDisease and 
                        ((has_pre_tnm_staging value "IA") or 
                         (has_pre_tnm_staging value "IB") or
                         (has_pre_tnm_staging value "IIA") or 
                         (has_pre_tnm_staging value "IIB")) and 
                        (has_histology some NonSmallCellCarcinoma)
                    )
                ) 
                and 
                (has_performance_status some (WHOPerfStatusGrade0 or WHOPerfStatusGrade1))
            """,
            recommended_treatment="Surgery",
            evidence_level="Grade A"
        ),
        GuidelineRule(
            rule_id="R3",
            name="RadioRule001",
            source="NICE Lung Cancer 2011",
            description="Offer radical radiotherapy for stage I-III NSCLC unsuitable for surgery",
            owl_expression="""
                (has_clinical_finding some 
                    (NeoplasticDisease and 
                        ((has_pre_tnm_staging value "IA") or 
                         (has_pre_tnm_staging value "IB") or
                         (has_pre_tnm_staging value "IIA") or 
                         (has_pre_tnm_staging value "IIB") or
                         (has_pre_tnm_staging value "IIIA")) and 
                        (has_histology some NonSmallCellCarcinoma)
                    )
                ) 
                and 
                (has_performance_status some 
                    (WHOPerfStatusGrade0 or WHOPerfStatusGrade1 or WHOPerfStatusGrade2))
            """,
            recommended_treatment="Radiotherapy",
            evidence_level="Grade B"
        ),
        GuidelineRule(
            rule_id="R4",
            name="PalliativeRule001",
            source="NICE Lung Cancer 2011",
            description="Consider palliative care for patients with advanced disease and poor performance status",
            owl_expression="""
                (has_clinical_finding some 
                    (NeoplasticDisease and 
                        (has_pre_tnm_staging value "IV")
                    )
                ) 
                and 
                (has_performance_status some (WHOPerfStatusGrade3 or WHOPerfStatusGrade4))
            """,
            recommended_treatment="Palliative Care",
            evidence_level="Grade C"
        ),
        GuidelineRule(
            rule_id="R5",
            name="SCLCChemoRule001",
            source="NICE Lung Cancer 2011",
            description="Offer chemotherapy for SCLC with good performance status",
            owl_expression="""
                (has_clinical_finding some 
                    (NeoplasticDisease and 
                        (has_histology some SmallCellCarcinoma)
                    )
                ) 
                and 
                (has_performance_status some (WHOPerfStatusGrade0 or WHOPerfStatusGrade1))
            """,
            recommended_treatment="Chemotherapy",
            evidence_level="Grade A"
        ),
        GuidelineRule(
            rule_id="R6",
            name="ChemoRadioRule001",
            source="NICE Lung Cancer 2011",
            description="Consider chemoradiotherapy for stage IIIA/IIIB NSCLC with good PS",
            owl_expression="""
                (has_clinical_finding some 
                    (NeoplasticDisease and 
                        ((has_pre_tnm_staging value "IIIA") or 
                         (has_pre_tnm_staging value "IIIB")) and 
                        (has_histology some NonSmallCellCarcinoma)
                    )
                ) 
                and 
                (has_performance_status some (WHOPerfStatusGrade0 or WHOPerfStatusGrade1))
            """,
            recommended_treatment="Chemoradiotherapy",
            evidence_level="Grade B"
        ),
    ]
    
    def __init__(self, ontology):
        self.onto = ontology
        self.rules: Dict[str, GuidelineRule] = {}
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default NICE guidelines."""
        for rule in self.NICE_GUIDELINES:
            self.add_rule(rule)
    
    def add_rule(self, rule: GuidelineRule):
        """
        Add a guideline rule to the ontology as a Patient Scenario.
        Creates the equivalent class expression and argumentation links.
        """
        self.rules[rule.rule_id] = rule
        
        with self.onto.onto:
            # Create Patient Scenario subclass for the rule
            patient_scenario_cls = types.new_class(
                rule.name,
                (self.onto.onto.PatientScenario,)
            )
            
            # Create corresponding Decision
            decision_name = f"{rule.name}Decision"
            decision_ref = self.onto.onto.Decision(f"Reference_{decision_name}")
            
            # Create Treatment Plan reference
            treatment_plan_name = f"{rule.recommended_treatment}Plan"
            treatment_ref = self.onto.onto.TreatmentPlan(f"Reference_{treatment_plan_name}")
            treatment_ref.treatment_plan_type = rule.recommended_treatment
            
            # Create scenario reference individual
            scenario_ref = patient_scenario_cls(f"Reference_{rule.name}")
            
            # Link: PatientScenario → supports_decision → Decision
            scenario_ref.supports_decision = [decision_ref]
            
            # Link: Decision → entails → TreatmentPlan
            decision_ref.entails = [treatment_ref]
    
    def classify_patient(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Classify a patient and return applicable treatment recommendations.
        Uses pattern matching instead of OWL reasoner for better performance.
        """
        recommendations = []
        
        tnm_stage = patient_data.get("tnm_stage", "")
        histology = patient_data.get("histology_type", "")
        performance_status = patient_data.get("performance_status", 0)
        
        # Check each rule
        for rule_id, rule in self.rules.items():
            if self._matches_rule(rule, tnm_stage, histology, performance_status):
                recommendations.append({
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "source": rule.source,
                    "description": rule.description,
                    "recommended_treatment": rule.recommended_treatment,
                    "evidence_level": rule.evidence_level
                })
        
        return recommendations
    
    def _matches_rule(
        self, 
        rule: GuidelineRule, 
        tnm_stage: str, 
        histology: str, 
        performance_status: int
    ) -> bool:
        """Check if patient matches a rule's criteria."""
        
        # Rule-specific matching logic
        if rule.rule_id == "R1":  # Chemo for Stage III-IV NSCLC
            stage_match = tnm_stage in ["III", "IIIA", "IIIB", "IV"]
            histology_match = "NonSmallCell" in histology or histology in [
                "Adenocarcinoma", "SquamousCellCarcinoma", "LargeCellCarcinoma"
            ]
            ps_match = performance_status <= 1
            return stage_match and histology_match and ps_match
        
        elif rule.rule_id == "R2":  # Surgery for Stage I-II NSCLC
            stage_match = tnm_stage in ["IA", "IB", "IIA", "IIB"]
            histology_match = "NonSmallCell" in histology or histology in [
                "Adenocarcinoma", "SquamousCellCarcinoma", "LargeCellCarcinoma"
            ]
            ps_match = performance_status <= 1
            return stage_match and histology_match and ps_match
        
        elif rule.rule_id == "R3":  # Radiotherapy for Stage I-IIIA
            stage_match = tnm_stage in ["IA", "IB", "IIA", "IIB", "IIIA"]
            histology_match = "NonSmallCell" in histology or histology in [
                "Adenocarcinoma", "SquamousCellCarcinoma", "LargeCellCarcinoma"
            ]
            ps_match = performance_status <= 2
            return stage_match and histology_match and ps_match
        
        elif rule.rule_id == "R4":  # Palliative for Stage IV poor PS
            stage_match = tnm_stage == "IV"
            ps_match = performance_status >= 3
            return stage_match and ps_match
        
        elif rule.rule_id == "R5":  # Chemo for SCLC
            histology_match = "SmallCell" in histology
            ps_match = performance_status <= 1
            return histology_match and ps_match
        
        elif rule.rule_id == "R6":  # Chemoradio for Stage IIIA/B
            stage_match = tnm_stage in ["IIIA", "IIIB"]
            histology_match = "NonSmallCell" in histology or histology in [
                "Adenocarcinoma", "SquamousCellCarcinoma", "LargeCellCarcinoma"
            ]
            ps_match = performance_status <= 1
            return stage_match and histology_match and ps_match
        
        return False
    
    def get_rule_by_id(self, rule_id: str) -> GuidelineRule:
        """Get a specific rule by ID."""
        return self.rules.get(rule_id)
    
    def get_all_rules(self) -> List[GuidelineRule]:
        """Get all registered rules."""
        return list(self.rules.values())
```
# LCA Implementation Plan - Part 4: Agentic AI & Database Layer

---

## 13. Neo4j Knowledge Graph Integration

```python
# backend/src/db/neo4j_schema.py

from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "lucada"

class LUCADAGraphDB:
    """
    Neo4j implementation for LUCADA patient data and relationships.
    Complements OWL ontology for efficient querying and storage.
    """
    
    SCHEMA_QUERIES = [
        # Constraints
        "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE",
        "CREATE CONSTRAINT rule_id IF NOT EXISTS FOR (r:GuidelineRule) REQUIRE r.rule_id IS UNIQUE",
        
        # Indexes
        "CREATE INDEX patient_stage IF NOT EXISTS FOR (p:Patient) ON (p.tnm_stage)",
        "CREATE INDEX histology_type IF NOT EXISTS FOR (h:Histology) ON (h.type)",
        
        # Full-text search
        """CREATE FULLTEXT INDEX patient_search IF NOT EXISTS
           FOR (p:Patient) ON EACH [p.name, p.notes]""",
        
        # Vector index for embeddings
        """CREATE VECTOR INDEX guideline_embeddings IF NOT EXISTS
           FOR (g:GuidelineRule) ON (g.embedding)
           OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}"""
    ]
    
    def __init__(self, config: Neo4jConfig = None):
        self.config = config or Neo4jConfig()
        self.driver = GraphDatabase.driver(
            self.config.uri,
            auth=(self.config.user, self.config.password)
        )
    
    def setup_schema(self):
        """Initialize database schema."""
        with self.driver.session(database=self.config.database) as session:
            for query in self.SCHEMA_QUERIES:
                try:
                    session.run(query)
                except Exception as e:
                    print(f"Schema query skipped: {e}")
    
    def create_patient(self, patient_data: Dict[str, Any]) -> str:
        """Create a patient node with all relationships (mirrors Figure 2)."""
        query = """
        MERGE (p:Patient {patient_id: $patient_id})
        SET p.name = $name,
            p.sex = $sex,
            p.age_at_diagnosis = $age,
            p.tnm_stage = $tnm_stage,
            p.performance_status = $performance_status,
            p.created_at = datetime()
        
        MERGE (cf:ClinicalFinding {id: $patient_id + '_finding'})
        SET cf.diagnosis = $diagnosis,
            cf.tnm_stage = $tnm_stage,
            cf.basis_of_diagnosis = 'Clinical'
        
        MERGE (h:Histology {type: $histology_type})
        
        MERGE (bs:BodyStructure:Neoplasm {id: $patient_id + '_tumor'})
        SET bs.laterality = $laterality
        
        MERGE (ps:PerformanceStatus {grade: $performance_status})
        
        MERGE (p)-[:HAS_CLINICAL_FINDING]->(cf)
        MERGE (cf)-[:HAS_HISTOLOGY]->(h)
        MERGE (cf)-[:AFFECTS]->(bs)
        MERGE (p)-[:HAS_PERFORMANCE_STATUS]->(ps)
        
        RETURN p.patient_id as patient_id
        """
        
        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, **patient_data)
            return result.single()["patient_id"]
    
    def find_similar_patients(self, patient_id: str, limit: int = 10) -> List[Dict]:
        """Find patients with similar clinical profiles."""
        query = """
        MATCH (p:Patient {patient_id: $patient_id})
        
        WITH p.tnm_stage as stage, p.performance_status as ps
        
        MATCH (similar:Patient)
        WHERE similar.patient_id <> $patient_id
          AND similar.tnm_stage = stage
          AND abs(similar.performance_status - ps) <= 1
        
        MATCH (similar)-[:HAS_CLINICAL_FINDING]->(cf)-[:HAS_HISTOLOGY]->(h)
        
        OPTIONAL MATCH (similar)-[:HAS_TREATMENT_PLAN]->(tp)-[:HAS_OUTCOME]->(o)
        
        RETURN similar.patient_id as patient_id,
               similar.name as name,
               similar.age_at_diagnosis as age,
               similar.tnm_stage as stage,
               h.type as histology,
               collect(DISTINCT tp.type) as treatments,
               avg(o.survival_days) as avg_survival
        ORDER BY similar.age_at_diagnosis
        LIMIT $limit
        """
        
        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, patient_id=patient_id, limit=limit)
            return [dict(record) for record in result]
    
    def get_treatment_statistics(self, treatment_type: str) -> Dict:
        """Get outcome statistics for a treatment type."""
        query = """
        MATCH (tp:TreatmentPlan {type: $treatment_type})-[:HAS_OUTCOME]->(o:Outcome)
        MATCH (p:Patient)-[:HAS_TREATMENT_PLAN]->(tp)
        
        RETURN tp.type as treatment,
               count(DISTINCT p) as patient_count,
               avg(o.survival_days) as avg_survival_days,
               percentileDisc(o.survival_days, 0.5) as median_survival_days
        """
        
        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, treatment_type=treatment_type)
            record = result.single()
            return dict(record) if record else {}
    
    def close(self):
        self.driver.close()
```

---

## 14. LangGraph Agent Implementation

```python
# backend/src/agents/lca_agents.py

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Annotated, Sequence, Dict, Any, List
from pydantic import BaseModel, Field
import operator

# ========================================
# State Definitions
# ========================================

class PatientState(TypedDict):
    """State for patient classification workflow."""
    patient_id: str
    patient_data: Dict[str, Any]
    applicable_rules: List[Dict[str, Any]]
    treatment_recommendations: List[Dict[str, Any]]
    arguments: List[Dict[str, Any]]
    explanation: str
    messages: Annotated[Sequence[Any], operator.add]

# ========================================
# Agent Classes
# ========================================

class PatientClassificationAgent:
    """
    Agent responsible for classifying patients into Patient Scenarios.
    Implements the ontological inference from Section 4 of the paper.
    """
    
    def __init__(self, llm=None):
        self.llm = llm or ChatAnthropic(model="claude-sonnet-4-20250514")
        self.system_prompt = """You are a clinical decision support agent for lung cancer treatment.

Your role is to classify patients according to NICE and BTS guidelines based on:
- TNM Stage (IA, IB, IIA, IIB, IIIA, IIIB, IV)
- Histology (NSCLC subtypes: Adenocarcinoma, Squamous Cell, Large Cell; SCLC)
- WHO Performance Status (0-4)

For each patient, identify which guideline rules apply:
- R1: Chemotherapy for Stage III-IV NSCLC with PS 0-1
- R2: Surgery for Stage I-II NSCLC with PS 0-1
- R3: Radiotherapy for Stage I-IIIA NSCLC with PS 0-2
- R4: Palliative Care for Stage IV with PS 3-4
- R5: Chemotherapy for SCLC with PS 0-1
- R6: Chemoradiotherapy for Stage IIIA/B NSCLC with PS 0-1

Always cite the specific guideline rule IDs in your classifications."""
    
    def classify(self, state: PatientState) -> PatientState:
        """Classify patient and identify applicable rules."""
        patient_data = state["patient_data"]
        
        prompt = f"""Classify this lung cancer patient:

Patient Data:
- TNM Stage: {patient_data.get('tnm_stage')}
- Histology: {patient_data.get('histology_type')}
- Performance Status: WHO {patient_data.get('performance_status')}
- Age: {patient_data.get('age')}

List all applicable guideline rules with their IDs."""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        state["messages"] = state.get("messages", []) + [response]
        
        return state


class TreatmentRecommendationAgent:
    """
    Agent for generating treatment recommendations.
    Implements the Decision → Treatment Plan pattern from Figure 3.
    """
    
    def __init__(self, llm=None):
        self.llm = llm or ChatAnthropic(model="claude-sonnet-4-20250514")
        self.system_prompt = """You are a lung cancer treatment recommendation agent.

Generate ranked treatment recommendations following evidence hierarchy:
1. Grade A evidence (randomized controlled trials)
2. Grade B evidence (well-designed clinical studies)
3. Grade C evidence (expert opinion)

For each recommendation provide:
- Treatment type
- Supporting guideline rule(s)
- Evidence level
- Expected outcomes"""
    
    def recommend(self, state: PatientState) -> PatientState:
        """Generate treatment recommendations."""
        applicable_rules = state.get("applicable_rules", [])
        patient_data = state["patient_data"]
        
        prompt = f"""Generate treatment recommendations:

Applicable Rules: {applicable_rules}
Patient: Stage {patient_data.get('tnm_stage')}, {patient_data.get('histology_type')}, PS {patient_data.get('performance_status')}

Provide ranked recommendations with evidence levels."""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        state["messages"] = state.get("messages", []) + [response]
        
        return state


class ArgumentationAgent:
    """
    Agent for generating clinical arguments.
    Implements the Argument → supports/opposes → Decision pattern.
    """
    
    def __init__(self, llm=None):
        self.llm = llm or ChatAnthropic(model="claude-sonnet-4-20250514")
        self.system_prompt = """You are a clinical argumentation agent.

For each treatment option, generate:
1. Supporting arguments (reasons to recommend)
2. Opposing arguments (contraindications, risks)

Structure each argument as:
- Claim: The main assertion
- Evidence: Supporting data or guideline reference
- Strength: Strong/Moderate/Weak"""
    
    def generate_arguments(self, state: PatientState) -> PatientState:
        """Generate supporting and opposing arguments."""
        recommendations = state.get("treatment_recommendations", [])
        patient_data = state["patient_data"]
        
        arguments = []
        for rec in recommendations:
            prompt = f"""Generate clinical arguments for {rec.get('treatment', 'this treatment')}:

Patient: Stage {patient_data.get('tnm_stage')}, PS {patient_data.get('performance_status')}, Age {patient_data.get('age')}

Provide 2-3 supporting and 1-2 opposing arguments."""
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            arguments.append({
                "treatment": rec.get("treatment"),
                "arguments": response.content
            })
        
        state["arguments"] = arguments
        return state


class ExplanationAgent:
    """Agent for generating MDT-ready explanations."""
    
    def __init__(self, llm=None):
        self.llm = llm or ChatAnthropic(model="claude-sonnet-4-20250514")
        self.system_prompt = """Generate clear, clinician-friendly explanations for MDT meetings.

Include:
- Patient classification summary
- Ranked treatment options with rationale
- Key arguments for/against each option
- Suggested discussion points"""
    
    def explain(self, state: PatientState) -> PatientState:
        """Generate final MDT explanation."""
        prompt = f"""Generate MDT summary:

Patient ID: {state['patient_id']}
Rules: {state.get('applicable_rules', [])}
Recommendations: {state.get('treatment_recommendations', [])}
Arguments: {state.get('arguments', [])}"""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        state["explanation"] = response.content
        
        return state


# ========================================
# Workflow Definition
# ========================================

def create_lca_workflow():
    """Create the main LCA decision support workflow."""
    
    classifier = PatientClassificationAgent()
    recommender = TreatmentRecommendationAgent()
    argumenter = ArgumentationAgent()
    explainer = ExplanationAgent()
    
    workflow = StateGraph(PatientState)
    
    workflow.add_node("classify_patient", classifier.classify)
    workflow.add_node("recommend_treatment", recommender.recommend)
    workflow.add_node("generate_arguments", argumenter.generate_arguments)
    workflow.add_node("explain", explainer.explain)
    
    workflow.set_entry_point("classify_patient")
    workflow.add_edge("classify_patient", "recommend_treatment")
    workflow.add_edge("recommend_treatment", "generate_arguments")
    workflow.add_edge("generate_arguments", "explain")
    workflow.add_edge("explain", END)
    
    return workflow.compile()
```

---

## 15. Main Service Orchestration

```python
# backend/src/services/lca_service.py

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
import uuid

from ..agents.lca_agents import create_lca_workflow, PatientState
from ..ontology.lucada_ontology import LUCADAOntology
from ..ontology.guideline_rules import GuidelineRuleEngine
from ..db.neo4j_schema import LUCADAGraphDB

@dataclass
class TreatmentRecommendation:
    treatment_type: str
    rule_id: str
    rule_source: str
    evidence_level: str
    survival_rate: Optional[float]
    supporting_arguments: List[str]
    opposing_arguments: List[str]
    confidence_score: float

@dataclass
class PatientDecisionSupport:
    patient_id: str
    timestamp: datetime
    patient_scenarios: List[str]
    recommendations: List[TreatmentRecommendation]
    mdt_summary: str
    similar_patients: List[Dict[str, Any]]

class LungCancerAssistantService:
    """Main service coordinating all LCA components."""
    
    def __init__(
        self,
        ontology_path: str = "lucada.owl",
        neo4j_config: Optional[Dict] = None
    ):
        self.ontology = LUCADAOntology(ontology_path)
        self.rule_engine = GuidelineRuleEngine(self.ontology)
        self.graph_db = LUCADAGraphDB(neo4j_config)
        self.workflow = create_lca_workflow()
        
        self.graph_db.setup_schema()
    
    async def process_patient(
        self,
        patient_data: Dict[str, Any]
    ) -> PatientDecisionSupport:
        """Process a patient through the full decision support pipeline."""
        patient_id = patient_data.get("patient_id") or str(uuid.uuid4())
        
        # Step 1: Create patient in ontology
        patient_individual = self.ontology.create_patient_individual(
            patient_id=patient_id,
            sex=patient_data.get("sex", "Unknown"),
            age=patient_data.get("age", 0),
            diagnosis=patient_data.get("diagnosis", "Malignant Neoplasm of Lung"),
            tnm_stage=patient_data.get("tnm_stage", "Unknown"),
            histology_type=patient_data.get("histology_type", "NonSmallCellCarcinoma"),
            laterality=patient_data.get("laterality", "Unknown"),
            performance_status=patient_data.get("performance_status", 0)
        )
        
        # Step 2: Store in Neo4j
        self.graph_db.create_patient(patient_data)
        
        # Step 3: Run rule-based classification
        ontology_recommendations = self.rule_engine.classify_patient(patient_data)
        
        # Step 4: Run AI agent workflow
        initial_state: PatientState = {
            "patient_id": patient_id,
            "patient_data": patient_data,
            "applicable_rules": ontology_recommendations,
            "treatment_recommendations": ontology_recommendations,
            "arguments": [],
            "explanation": "",
            "messages": []
        }
        
        final_state = await asyncio.to_thread(
            self.workflow.invoke,
            initial_state
        )
        
        # Step 5: Find similar patients
        similar = self.graph_db.find_similar_patients(patient_id)
        
        # Step 6: Compile results
        recommendations = [
            TreatmentRecommendation(
                treatment_type=r.get("recommended_treatment", "Unknown"),
                rule_id=r.get("rule_id", ""),
                rule_source=r.get("source", "NICE Guidelines"),
                evidence_level=r.get("evidence_level", "Grade C"),
                survival_rate=None,
                supporting_arguments=["Good performance status", "Stage appropriate"],
                opposing_arguments=["Consider age and comorbidities"],
                confidence_score=0.8
            )
            for r in ontology_recommendations
        ]
        
        return PatientDecisionSupport(
            patient_id=patient_id,
            timestamp=datetime.now(),
            patient_scenarios=[r.get("rule_name") for r in ontology_recommendations],
            recommendations=recommendations,
            mdt_summary=final_state.get("explanation", ""),
            similar_patients=similar
        )
    
    def close(self):
        self.graph_db.close()
```
# LCA Implementation Plan - Part 5: API, Frontend & Deployment

---

## 16. FastAPI Backend

```python
# backend/src/api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

app = FastAPI(
    title="Lung Cancer Assistant API",
    description="Ontology-driven clinical decision support for lung cancer",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientInput(BaseModel):
    patient_id: Optional[str] = None
    name: str
    sex: str = Field(..., pattern="^[MF]$")
    age: int = Field(..., ge=0, le=120)
    tnm_stage: str
    histology_type: str
    performance_status: int = Field(..., ge=0, le=4)
    laterality: str = "Unknown"
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Jenny Sesen",
                "sex": "F",
                "age": 72,
                "tnm_stage": "IIA",
                "histology_type": "Carcinosarcoma",
                "performance_status": 1,
                "laterality": "Right"
            }
        }

class TreatmentRecommendationResponse(BaseModel):
    treatment_type: str
    rule_id: str
    rule_source: str
    evidence_level: str
    survival_rate: Optional[float]
    supporting_arguments: List[str]
    opposing_arguments: List[str]
    confidence_score: float

class DecisionSupportResponse(BaseModel):
    patient_id: str
    timestamp: datetime
    patient_scenarios: List[str]
    recommendations: List[TreatmentRecommendationResponse]
    mdt_summary: str
    similar_patients_count: int

lca_service = None

@app.on_event("startup")
async def startup():
    global lca_service
    from ..services.lca_service import LungCancerAssistantService
    lca_service = LungCancerAssistantService()

@app.post("/api/v1/patients/analyze", response_model=DecisionSupportResponse)
async def analyze_patient(patient: PatientInput):
    """Analyze a patient and generate treatment recommendations."""
    result = await lca_service.process_patient(patient.model_dump())
    return DecisionSupportResponse(
        patient_id=result.patient_id,
        timestamp=result.timestamp,
        patient_scenarios=result.patient_scenarios,
        recommendations=[...],
        mdt_summary=result.mdt_summary,
        similar_patients_count=len(result.similar_patients)
    )

@app.get("/api/v1/guidelines")
async def list_guidelines():
    """List all available clinical guideline rules."""
    return list(lca_service.rule_engine.rules.values())
```

---

## 17. Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      - neo4j

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000

  neo4j:
    image: neo4j:5.15-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
    volumes:
      - neo4j_data:/data

volumes:
  neo4j_data:
```

---

## 18. Summary & Implementation Checklist

### Phase 1: Foundation (Weeks 1-4)
- [ ] Set up project structure
- [ ] Implement LUCADA ontology in Owlready2
- [ ] Create Neo4j schema and database layer
- [ ] Formalize 6 NICE guideline rules

### Phase 2: Agentic AI (Weeks 5-8)
- [ ] Implement LangGraph workflow
- [ ] Create Classification Agent
- [ ] Create Recommendation Agent
- [ ] Create Argumentation Agent
- [ ] Create Explanation Agent

### Phase 3: API & Frontend (Weeks 9-12)
- [ ] Build FastAPI backend
- [ ] Create Next.js frontend
- [ ] Implement patient form
- [ ] Build treatment recommendations UI

### Phase 4: Deployment (Weeks 13-16)
- [ ] Docker containerization
- [ ] Integration testing
- [ ] Clinical validation
- [ ] Production deployment

---

## Key Innovations Over Original Paper

| Aspect | Original (2011) | Modern (2025) |
|--------|-----------------|---------------|
| Inference | OWL reasoner only | Hybrid: OWL + LLM agents |
| Scalability | Temporary patient add/remove | Neo4j graph database |
| Explanations | Rule IDs only | Natural language MDT summaries |
| Similar Patients | Not implemented | Vector similarity search |
| User Interface | GWT | Modern React/Next.js |
| Deployment | On-premise | Cloud-native containers |
