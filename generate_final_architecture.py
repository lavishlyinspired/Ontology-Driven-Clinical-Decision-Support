"""
Script to generate the complete final LCA_Architecture.md document
Combines content from all architecture documents into one comprehensive guide
"""

import os

def append_to_architecture(content):
    """Append content to the LCA_Architecture.md file"""
    with open("LCA_Architecture.md", "a", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Added {len(content)} characters to LCA_Architecture.md")

# Part I Remaining Sections (3-6)
part1_remaining = """
## 3. The LUCADA Ontology

### Understanding OWL Ontologies in Healthcare

OWL (Web Ontology Language) is a formal language for representing complex knowledge structures. Unlike simple databases that store facts, ontologies define **what concepts exist** and **how they relate to each other**. This distinction is crucial in healthcare, where a "Stage IIIA adenocarcinoma" isn't just a text string—it's a concept with defined relationships to treatment protocols, prognosis expectations, and eligibility criteria for clinical trials.

In the LCA system, the LUCADA ontology serves as the "schema" for all clinical reasoning. When the system encounters a patient with "Adenocarcinoma" histology, it doesn't just pattern-match against rules—it understands that Adenocarcinoma is a subclass of Non-Small Cell Lung Cancer (NSCLC), which in turn determines which treatment guidelines apply.

### Ontology Class Hierarchy

```mermaid
classDiagram
    class Thing {
        <<OWL>>
    }

    class Patient {
        +patientId: string
        +name: string
        +sex: Sex 
        +ageAtDiagnosis: int
        +performanceStatus: int
    }

    class ClinicalFinding {
        +diagnosis: string
        +tnmStage: TNMStage
        +basisOfDiagnosis: string
    }

    class Histology {
        +type: HistologyType
        +snomedCode: string
    }

    class TreatmentPlan {
        +type: string
        +intent: TreatmentIntent
        +snomedCode: string
    }

    class PatientScenario {
        +scenarioId: string
        +description: string
    }

    class Argument {
        +claim: string
        +evidence: string
        +strength: string
    }

    class Decision {
        +recommendation: string
        +confidence: float
    }

    class BiomarkerProfile {
        +egfrMutation: string
        +alkRearrangement: string
        +ros1Rearrangement: string
        +pdl1TPS: int
    }

    Thing <|-- Patient
    Thing <|-- ClinicalFinding
    Thing <|-- Histology
    Thing <|-- TreatmentPlan
    Thing <|-- PatientScenario
    Thing <|-- Argument
    Thing <|-- Decision
    Thing <|-- BiomarkerProfile

    Patient "1" --> "*" ClinicalFinding : hasClinicalFinding
    ClinicalFinding "1" --> "1" Histology : hasHistology
    Patient "1" --> "*" TreatmentPlan : receivedTreatment
    Patient "1" --> "0..1" BiomarkerProfile : hasBiomarkerProfile
    PatientScenario "1" --> "*" Argument : hasArgument
    Argument "*" --> "1" Decision : supportsDecision
```

### Core Data Properties

The LUCADA ontology defines 60+ data properties for comprehensive patient representation:

**Patient Demographics:**
- `patientId` (string): Unique identifier
- `name` (string): Patient name
- `sex` (Sex enum): Male/Female
- `ageAtDiagnosis` (integer): Age when diagnosed
- `dateOfBirth` (datetime): Birth date

**Clinical Characteristics:**
- `tnmStage` (string): TNM staging (IA, IB, IIA, IIB, IIIA, IIIB, IV)
- `histologyType` (string): Adenocarcinoma, Squamous Cell, Small Cell, Large Cell
- `performanceStatus` (integer): WHO/ECOG 0-4
- `fev1Percent` (float): Forced Expiratory Volume (lung function)
- `laterality` (string): Left/Right/Bilateral

**Biomarker Properties (NEW 2025):**
- `egfrMutation` (string): Positive/Negative
- `egfrMutationType` (string): Ex19del, L858R, T790M
- `alkRearrangement` (string): Positive/Negative
- `ros1Rearrangement` (string): Positive/Negative
- `brafMutation` (string): V600E, other
- `metExon14` (string): Present/Absent
- `retRearrangement` (string): Positive/Negative
- `ntrkFusion` (string): Positive/Negative
- `pdl1TPS` (integer): 0-100 (Tumor Proportion Score)
- `krasMutation` (string): G12C, other
- `her2Mutation` (string): Positive/Negative

**Lab Results (LOINC-mapped):**
- `hemoglobin` (float): g/dL
- `creatinine` (float): mg/dL
- `eGFR` (float): mL/min/1.73m²
- `alt` (float): U/L (liver function)
- `ast` (float): U/L (liver function)

### Working with the Ontology in Python

The LCA system uses the **owlready2** library to load and manipulate OWL ontologies:

```python
from owlready2 import get_ontology
from backend.src.ontology.lucada_ontology import LUCADAOntology

# Initialize the LUCADA ontology
lucada = LUCADAOntology()
lucada.create()  # Creates all classes, properties, and individuals

# Access the ontology
onto = lucada.onto

# Create a new patient instance
with onto:
    patient = onto.Patient("P12345")
    patient.ageAtDiagnosis = 68
    patient.tnmStage = "IIIA"
    patient.histologyType = "Adenocarcinoma"
    patient.performanceStatus = 1
    patient.egfrMutation = "Positive"
    patient.egfrMutationType = "Ex19del"

# Query the ontology for all EGFR-positive patients
egfr_positive = [p for p in onto.Patient.instances()
                 if hasattr(p, 'egfrMutation') and p.egfrMutation == "Positive"]

# Save the ontology to OWL file
lucada.save("data/lucada_with_patients.owl")
```

The ontology file is serialized in OWL Functional Syntax format:

```owl
Declaration(Class(:Patient))
Declaration(Class(:BiomarkerProfile))
Declaration(DataProperty(:ageAtDiagnosis))
Declaration(DataProperty(:egfrMutation))
Declaration(DataProperty(:egfrMutationType))

DataPropertyDomain(:ageAtDiagnosis :Patient)
DataPropertyRange(:ageAtDiagnosis xsd:integer)

DataPropertyDomain(:egfrMutation :BiomarkerProfile)
DataPropertyRange(:egfrMutation xsd:string)

Declaration(NamedIndividual(:P12345))
ClassAssertion(:Patient :P12345)
DataPropertyAssertion(:ageAtDiagnosis :P12345 "68"^^xsd:integer)
DataPropertyAssertion(:egfrMutation :P12345 "Positive"^^xsd:string)
```

This formal representation enables external OWL reasoners (like HermiT or Pellet) to perform automated reasoning over the knowledge base.

---

## 4. SNOMED-CT Integration

### Why SNOMED-CT Matters for Healthcare Interoperability

Healthcare data suffers from a fundamental problem: the same clinical concept can be expressed in dozens of different ways. A "heart attack" might be recorded as "myocardial infarction," "MI," "cardiac infarction," or "coronary thrombosis" depending on the institution, clinician, or EMR system. This variability makes it nearly impossible to aggregate data across systems, compare outcomes, or apply consistent decision rules.

SNOMED-CT solves this by providing a **canonical identifier** (SCTID) for each clinical concept. When the LCA system encounters "Adenocarcinoma" in a patient record, it maps it to SNOMED code `35917007`. Now, whether the original record said "adenocarcinoma," "Adenocarcinoma of lung," or "ADENO CA," the system can consistently apply the same treatment guidelines.

Moreover, SNOMED-CT isn't just a flat list of codes—it's a full ontology with hierarchical relationships. This means the system can reason that `35917007` (Adenocarcinoma) is a subtype of `254637007` (Non-small cell lung cancer), which enables generalized rules to apply correctly.

### SNOMED Mapping Architecture

```mermaid
flowchart LR
    subgraph "Clinical Input"
        H[Histology: Adenocarcinoma]
        S[Stage: IIIA]
        PS[Performance Status: 1]
        L[Laterality: Right]
    end

    subgraph "SNOMED Mapper"
        M[SemanticMappingAgent]
        HM[Histology Map]
        SM[Stage Map]
        PSM[PS Map]
        LM[Laterality Map]
    end

    subgraph "SNOMED-CT Codes"
        HC[SCTID: 35917007]
        SC[SCTID: 422968005]
        PSC[SCTID: 373804000]
        LC[SCTID: 39607008]
    end

    H --> M
    S --> M
    PS --> M
    L --> M

    M --> HM --> HC
    M --> SM --> SC
    M --> PSM --> PSC
    M --> LM --> LC

    style M fill:#e76f51,stroke:#333,color:#fff
```

### Key SNOMED Concept Categories

| Category | Examples | SNOMED Codes |
|----------|----------|--------------|
| **Diagnoses** | Adenocarcinoma, NSCLC, SCLC | 35917007, 254637007, 60573004 |
| **Stages** | Stage IA, Stage IIIA, Stage IV | 422968005, 422968005, 423032009 |
| **Treatments** | Chemotherapy, Surgery, Radiotherapy | 367336001, 387713003, 108290001 |
| **Performance Status** | WHO Grade 0-4 | 373803006 - 373807007 |
| **Outcomes** | Complete Response, Partial Response | 268910001, 268911002 |
| **Biomarkers** | EGFR mutation, ALK rearrangement | 416939005, 445352005 |

### SNOMED-CT Loader: Parsing 378,000+ Concepts

The LCA system includes a custom SNOMED-CT loader that parses the official SNOMED OWL distribution files:

```python
from backend.src.ontology.snomed_loader import SNOMEDLoader

# Load SNOMED-CT from OWL file
loader = SNOMEDLoader()
loader.load(load_full=False)  # Use pre-defined mappings for efficiency

# Map a clinical term to SNOMED
code = loader.get_snomed_code("Adenocarcinoma", category="histology")
print(f"SNOMED Code: {code.code}")  # 35917007
print(f"Display Name: {code.display}")  # Adenocarcinoma

# Reverse lookup
concept = loader.get_concept_by_code("35917007")
print(f"Concept: {concept.preferred_term}")  # Adenocarcinoma

# Find parent concepts (subsumption)
parents = loader.get_parent_concepts("35917007")
# Returns: [254637007 (Non-small cell lung cancer), ...]
```

The loader handles OWL Functional Syntax parsing, which differs from the more common RDF/XML format. This is important because SNOMED International distributes their terminology in this format for efficiency and precision.

**Statistics:**
- **Total Concepts**: 378,416 classes
- **Object Properties**: 126
- **Annotation Labels**: 378,553
- **File Size**: ~800MB uncompressed

---

## 5. Clinical Guidelines as Rules

Clinical guidelines like NICE CG121 are traditionally written in natural language for human clinicians. The challenge for a clinical decision support system is **formalization**—converting prose like "Consider surgical resection for patients with Stage I-II NSCLC who have adequate lung function and performance status 0-1" into machine-executable rules.

The LCA system addresses this through a structured rule representation that captures:

1. **Eligibility Criteria**: Which patients qualify for this recommendation?
2. **Treatment Modality**: What intervention is being recommended?
3. **Evidence Level**: How strong is the supporting evidence?
4. **Intent**: Curative vs. palliative?
5. **Priority**: When multiple rules match, which takes precedence?

### The 7 NICE CG121 Guidelines

```mermaid
graph TB
    subgraph "NICE CG121 Guidelines"
        R1[R1: Chemotherapy for Stage III-IV NSCLC]
        R2[R2: Surgery for Stage I-II NSCLC]
        R3[R3: Radiotherapy for I-IIIA NSCLC]
        R4[R4: Palliative Care for Poor PS]
        R5[R5: Chemotherapy for SCLC]
        R6[R6: Chemoradiotherapy for IIIA/IIIB]
        R7[R7: Immunotherapy for Advanced NSCLC]
    end

    subgraph "Evidence Levels"
        A[Grade A - Strong]
        B[Grade B - Moderate]
        C[Grade C - Limited]
        D[Grade D - Weak]
    end

    R1 --> A
    R2 --> A
    R3 --> B
    R4 --> C
    R5 --> A
    R6 --> A
    R7 --> A

    style A fill:#2d6a4f,stroke:#333,color:#fff
    style B fill:#40916c,stroke:#333,color:#fff
    style C fill:#74c69d,stroke:#333,color:#fff
    style D fill:#b7e4c7,stroke:#333,color:#000
```

### Rule Structure

Each guideline rule contains:

```mermaid
graph LR
    subgraph "GuidelineRule Structure"
        R[Rule R2]
        R --> ID[rule_id: R2]
        R --> N[name: Surgery for Stage I-II]
        R --> S[source: NICE CG121]
        R --> D[description: Surgical resection...]
        R --> T[treatment: Lobectomy/Pneumonectomy]
        R --> I[intent: Curative]
        R --> E[evidence_level: Grade A]
        R --> P[priority: 90]
        R --> C[criteria: Lambda Function]
    end

    style R fill:#264653,stroke:#333,color:#fff
```

### Implementing Guidelines in Python

Each guideline rule is represented as a Python dataclass:

```python
from dataclasses import dataclass
from typing import Callable
from backend.src.models.patient_models import PatientFactWithCodes

@dataclass
class GuidelineRule:
    rule_id: str
    name: str
    source: str
    description: str
    treatment: str
    intent: str  # "curative" or "palliative"
    evidence_level: str  # "Grade A", "Grade B", etc.
    priority: int  # Higher = more important
    criteria: Callable[[PatientFactWithCodes], bool]

# Rule R2: Surgery for Stage I-II NSCLC
R2 = GuidelineRule(
    rule_id="R2",
    name="Surgery for Stage I-II NSCLC",
    source="NICE CG121",
    description="Offer surgical resection to patients with Stage I-II NSCLC "
                "who are fit for surgery with adequate lung function",
    treatment="Lobectomy or Pneumonectomy",
    intent="curative",
    evidence_level="Grade A",
    priority=90,
    criteria=lambda p: (
        p.tnm_stage in ["IA", "IB", "IIA", "IIB"] and
        p.histology_type != "Small Cell Carcinoma" and
        p.performance_status <= 1 and
        (p.fev1_percent is None or p.fev1_percent >= 50)
    )
)
```

When a patient is processed, the ClassificationAgent iterates through all rules and collects those whose criteria match:

```python
def match_guidelines(patient: PatientFactWithCodes) -> List[GuidelineRule]:
    matching_rules = []
    for rule in ALL_GUIDELINE_RULES:
        try:
            if rule.criteria(patient):
                matching_rules.append(rule)
        except Exception as e:
            logger.warning(f"Rule {rule.rule_id} evaluation failed: {e}")

    return sorted(matching_rules, key=lambda r: -r.priority)
```

### Patient Classification Logic

```mermaid
flowchart TD
    START[Patient Data] --> CHK1{SCLC?}

    CHK1 -->|Yes| SCLC[SCLC Pathway]
    CHK1 -->|No| CHK2{Stage I-II?}

    CHK2 -->|Yes| CHK3{Good PS 0-1?}
    CHK2 -->|No| CHK4{Stage III?}

    CHK3 -->|Yes| EARLY_OP[Early Stage Operable]
    CHK3 -->|No| EARLY_INOP[Early Stage Inoperable]

    CHK4 -->|Yes| CHK5{Resectable?}
    CHK4 -->|No| CHK6{Stage IV}

    CHK5 -->|Yes| LA_RES[Locally Advanced Resectable]
    CHK5 -->|No| LA_UNRES[Locally Advanced Unresectable]

    CHK6 --> CHK7{Good PS 0-2?}
    CHK7 -->|Yes| META_GOOD[Metastatic Good PS]
    CHK7 -->|No| META_POOR[Metastatic Poor PS]

    EARLY_OP --> REC1[Surgery + Adjuvant Chemo]
    EARLY_INOP --> REC2[Radiotherapy]
    LA_RES --> REC3[Chemoradiotherapy + Surgery]
    LA_UNRES --> REC4[Chemoradiotherapy]
    META_GOOD --> REC5[Chemotherapy/Immunotherapy]
    META_POOR --> REC6[Palliative Care]
    SCLC --> REC7[Chemotherapy + PCI]

    style START fill:#f4a261,stroke:#333,color:#000
    style EARLY_OP fill:#2a9d8f,stroke:#333,color:#fff
    style LA_RES fill:#2a9d8f,stroke:#333,color:#fff
    style META_GOOD fill:#2a9d8f,stroke:#333,color:#fff
    style META_POOR fill:#e76f51,stroke:#333,color:#fff
```

### Example: End-to-End Classification

**Patient**: 68-year-old female, Stage IIIA Adenocarcinoma, PS 1, FEV1 65%

1. **SCLC Check**: Is histology "Small Cell Carcinoma"? → **No** (Adenocarcinoma is NSCLC)
2. **Stage I-II Check**: Is stage in [IA, IB, IIA, IIB]? → **No** (Stage IIIA)
3. **Stage III Check**: Is stage in [IIIA, IIIB]? → **Yes**
4. **Resectable Check**: Is FEV1 ≥ 50% AND PS ≤ 2? → **Yes** (FEV1=65%, PS=1)
5. **Result**: Patient classified as **"Locally Advanced Resectable"**

Matching Rules:
- **R6**: Chemoradiotherapy for Stage IIIA/IIIB (Grade A, Priority 85)
- **R1**: Chemotherapy for Stage III-IV NSCLC (Grade A, Priority 80)

The ConflictResolutionAgent will prioritize R6 based on priority score and specificity.

---

## 6. The 6-Agent Workflow

### Why a Multi-Agent Architecture?

Traditional monolithic applications bundle all logic into a single codebase, making it difficult to test, maintain, or modify individual components. The LCA system takes a different approach inspired by microservices architecture: each agent has a **single responsibility** and communicates through well-defined interfaces.

This design provides several advantages:

1. **Testability**: Each agent can be unit-tested in isolation
2. **Maintainability**: Updating the classification logic doesn't affect persistence code
3. **Auditability**: The agent chain provides a clear audit trail of what happened at each step
4. **Extensibility**: New agents can be added without modifying existing ones
5. **Parallelization**: Independent agents can run concurrently (2026 enhancement)

The 6-agent workflow is orchestrated by LangGraph, a framework for building stateful, multi-step AI workflows. Each agent receives the current workflow state, performs its specific task, and updates the state for the next agent.

### Agent Pipeline Overview

```mermaid
sequenceDiagram
    participant Input as Raw Patient Data
    participant A1 as IngestionAgent
    participant A2 as SemanticMappingAgent
    participant A3 as ClassificationAgent
    participant A4 as ConflictResolutionAgent
    participant A5 as PersistenceAgent
    participant A6 as ExplanationAgent
    participant Output as Decision Support Response

    Input->>A1: Raw JSON data
    Note over A1: Validate & Normalize
    A1->>A2: PatientFact
    Note over A2: Map to SNOMED-CT
    A2->>A3: PatientFactWithCodes
    Note over A3: Apply NICE Guidelines
    A3->>A4: ClassificationResult
    Note over A4: Resolve Conflicts
    A4->>A5: Resolved Classification
    Note over A5: Save to Neo4j
    A5->>A6: Write Receipt
    Note over A6: Generate MDT Summary
    A6->>Output: Complete Response
```

### Agent Responsibilities

```mermaid
graph TB
    subgraph "Agent 1: IngestionAgent"
        A1[IngestionAgent]
        A1_1[Validate required fields]
        A1_2[Normalize TNM staging]
        A1_3[Normalize histology types]
        A1_4[Calculate age groups]
        A1 --> A1_1
        A1 --> A1_2
        A1 --> A1_3
        A1 --> A1_4
    end

    subgraph "Agent 2: SemanticMappingAgent"
        A2[SemanticMappingAgent]
        A2_1[Map histology to SNOMED]
        A2_2[Map stage to SNOMED]
        A2_3[Map PS to SNOMED]
        A2_4[Calculate confidence]
        A2 --> A2_1
        A2 --> A2_2
        A2 --> A2_3
        A2 --> A2_4
    end

    subgraph "Agent 3: ClassificationAgent"
        A3[ClassificationAgent]
        A3_1[Determine patient scenario]
        A3_2[Match guideline rules]
        A3_3[Generate recommendations]
        A3_4[Build reasoning chain]
        A3 --> A3_1
        A3 --> A3_2
        A3 --> A3_3
        A3 --> A3_4
    end

    subgraph "Agent 4: ConflictResolutionAgent"
        A4[ConflictResolutionAgent]
        A4_1[Detect conflicts]
        A4_2[Apply evidence hierarchy]
        A4_3[Rank recommendations]
        A4_4[Deduplicate]
        A4 --> A4_1
        A4 --> A4_2
        A4 --> A4_3
        A4 --> A4_4
    end

    subgraph "Agent 5: PersistenceAgent"
        A5[PersistenceAgent]
        A5_1[Save patient facts]
        A5_2[Save inference result]
        A5_3[Create audit trail]
        A5_4[Generate embeddings]
        A5 --> A5_1
        A5 --> A5_2
        A5 --> A5_3
        A5 --> A5_4
    end

    subgraph "Agent 6: ExplanationAgent"
        A6[ExplanationAgent]
        A6_1[Generate clinical summary]
        A6_2[Format recommendations]
        A6_3[List key considerations]
        A6_4[Create MDT discussion points]
        A6 --> A6_1
        A6 --> A6_2
        A6 --> A6_3
        A6 --> A6_4
    end

    style A1 fill:#e63946,stroke:#333,color:#fff
    style A2 fill:#f4a261,stroke:#333,color:#000
    style A3 fill:#2a9d8f,stroke:#333,color:#fff
    style A4 fill:#264653,stroke:#333,color:#fff
    style A5 fill:#9b2226,stroke:#333,color:#fff
    style A6 fill:#457b9d,stroke:#333,color:#fff
```

### Deep Dive: What Each Agent Does

**Agent 1: IngestionAgent** ([ingestion_agent.py:267](backend/src/agents/ingestion_agent.py))

The IngestionAgent is the gateway to the system. It receives raw patient data and performs critical validation and normalization:

```python
def execute(self, patient_data: Dict[str, Any]) -> PatientFact:
    # Validate required fields
    required = ["patient_id", "age_at_diagnosis", "tnm_stage", "histology_type"]
    for field in required:
        if field not in patient_data:
            raise ValidationError(f"Missing required field: {field}")

    # Normalize TNM staging (handle variations like "3A" vs "IIIA")
    stage = self._normalize_stage(patient_data["tnm_stage"])

    # Normalize histology (handle abbreviations, synonyms)
    histology = self._normalize_histology(patient_data["histology_type"])

    # Calculate derived fields
    age_group = self._calculate_age_group(patient_data["age_at_diagnosis"])

    return PatientFact(
        patient_id=patient_data["patient_id"],
        age_at_diagnosis=patient_data["age_at_diagnosis"],
        age_group=age_group,
        tnm_stage=stage,
        histology_type=histology,
        # ... other fields
    )
```

**Agent 2: SemanticMappingAgent** ([semantic_mapping_agent.py:215](backend/src/agents/semantic_mapping_agent.py))

Takes the normalized PatientFact and enriches it with SNOMED-CT codes:

```python
def execute(self, patient_fact: PatientFact) -> PatientFactWithCodes:
    snomed_codes = {}

    # Map histology
    histology_code = self.snomed_loader.get_snomed_code(
        patient_fact.histology_type,
        category="histology"
    )
    snomed_codes["histology"] = {
        "code": histology_code.code,
        "display": histology_code.display
    }

    # Map stage, performance status, etc.
    # ...

    return PatientFactWithCodes(
        **patient_fact.dict(),
        snomed_codes=snomed_codes,
        mapping_confidence=self._calculate_confidence(snomed_codes)
    )
```

**Agent 3: ClassificationAgent** ([classification_agent.py:396](backend/src/agents/classification_agent.py))

The clinical brain of the system. Determines the patient's scenario and matches applicable guidelines:

```python
def execute(self, patient: PatientFactWithCodes) -> ClassificationResult:
    # Determine patient scenario
    scenario = self._determine_scenario(patient)

    # Match guideline rules
    matching_rules = self._match_guidelines(patient)

    # Generate recommendations
    recommendations = [
        Recommendation(
            treatment=rule.treatment,
            rule_id=rule.rule_id,
            evidence_level=rule.evidence_level,
            confidence=self._calculate_confidence(patient, rule)
        )
        for rule in matching_rules
    ]

    # Build reasoning chain
    reasoning = self._build_reasoning_chain(patient, scenario, matching_rules)

    return ClassificationResult(
        patient_id=patient.patient_id,
        scenario=scenario,
        recommendations=recommendations,
        reasoning_chain=reasoning
    )
```

**Agent 4: ConflictResolutionAgent** ([conflict_resolution_agent.py:271](backend/src/agents/conflict_resolution_agent.py))

When multiple guidelines match, this agent resolves conflicts based on evidence hierarchy, specificity, and clinical appropriateness:

```python
def execute(self, classification: ClassificationResult) -> ClassificationResult:
    # Detect conflicts
    conflicts = self._detect_conflicts(classification.recommendations)

    if not conflicts:
        return classification  # No conflicts

    # Apply resolution strategies
    resolved = self._apply_evidence_hierarchy(classification.recommendations)
    resolved = self._apply_specificity_ranking(resolved)
    resolved = self._deduplicate(resolved)

    return ClassificationResult(
        **classification.dict(exclude={"recommendations"}),
        recommendations=resolved,
        conflicts_detected=len(conflicts)
    )
```

**Agent 5: PersistenceAgent** ([persistence_agent.py:317](backend/src/agents/persistence_agent.py))

**THE ONLY AGENT THAT WRITES TO NEO4J.** This strict separation ensures data integrity:

```python
def execute(self, classification: ClassificationResult) -> WriteReceipt:
    # Generate patient embedding for similarity search
    embedding = self._generate_embedding(classification)

    # Save to Neo4j
    write_result = self.neo4j_write_tools.save_patient_and_inference(
        patient=classification.patient,
        classification=classification,
        embedding=embedding
    )

    return WriteReceipt(
        patient_id=classification.patient_id,
        inference_id=write_result.inference_id,
        timestamp=datetime.now(),
        nodes_created=write_result.nodes_created,
        relationships_created=write_result.relationships_created
    )
```

**Agent 6: ExplanationAgent** ([explanation_agent.py:380](backend/src/agents/explanation_agent.py))

Generates human-readable clinical summaries suitable for MDT (Multi-Disciplinary Team) discussions:

```python
def execute(self, classification: ClassificationResult) -> MDTSummary:
    # Generate clinical summary
    summary = self._generate_clinical_summary(classification)

    # Format recommendations for clinical review
    formatted_recs = self._format_recommendations(classification.recommendations)

    # List key considerations
    considerations = self._extract_key_considerations(classification)

    # Create MDT discussion points
    discussion_points = self._generate_discussion_points(classification)

    return MDTSummary(
        patient_id=classification.patient_id,
        clinical_summary=summary,
        recommendations=formatted_recs,
        key_considerations=considerations,
        mdt_discussion_points=discussion_points
    )
```

### Neo4j Access Pattern

**CRITICAL PRINCIPLE: "Neo4j as a tool, not a brain"**

```mermaid
graph LR
    subgraph "READ-ONLY Agents"
        A1[IngestionAgent]
        A2[SemanticMappingAgent]
        A3[ClassificationAgent]
        A4[ConflictResolutionAgent]
        A6[ExplanationAgent]
    end

    subgraph "WRITE Agent"
        A5[PersistenceAgent]
    end

    subgraph "Neo4j Operations"
        R[Neo4jReadTools]
        W[Neo4jWriteTools]
    end

    A1 -.->|Optional Read| R
    A2 -.->|Optional Read| R
    A3 -.->|Optional Read| R
    A4 -.->|Optional Read| R
    A6 -.->|Optional Read| R

    A5 ==>|EXCLUSIVE WRITE| W

    style A5 fill:#9b2226,stroke:#333,color:#fff
    style W fill:#9b2226,stroke:#333,color:#fff
```

### Workflow Code Example

Here's how the 6-agent workflow is orchestrated using LangGraph:

```python
from langgraph.graph import StateGraph, END
from backend.src.agents import (
    IngestionAgent, SemanticMappingAgent, ClassificationAgent,
    ConflictResolutionAgent, PersistenceAgent, ExplanationAgent
)

class LCAWorkflow:
    def __init__(self, neo4j_read_tools, neo4j_write_tools, lucada, snomed):
        # Initialize all agents
        self.ingestion = IngestionAgent(lucada)
        self.semantic_mapping = SemanticMappingAgent(snomed)
        self.classification = ClassificationAgent(lucada)
        self.conflict_resolution = ConflictResolutionAgent()
        self.persistence = PersistenceAgent(neo4j_write_tools)
        self.explanation = ExplanationAgent()

        # Build the workflow graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)

        # Add nodes for each agent
        workflow.add_node("ingestion", self._run_ingestion)
        workflow.add_node("semantic_mapping", self._run_semantic_mapping)
        workflow.add_node("classification", self._run_classification)
        workflow.add_node("conflict_resolution", self._run_conflict_resolution)
        workflow.add_node("persistence", self._run_persistence)
        workflow.add_node("explanation", self._run_explanation)

        # Define the linear flow
        workflow.set_entry_point("ingestion")
        workflow.add_edge("ingestion", "semantic_mapping")
        workflow.add_edge("semantic_mapping", "classification")
        workflow.add_edge("classification", "conflict_resolution")
        workflow.add_edge("conflict_resolution", "persistence")
        workflow.add_edge("persistence", "explanation")
        workflow.add_edge("explanation", END)

        return workflow.compile()

    def run(self, patient_data: Dict) -> DecisionSupportResponse:
        initial_state = WorkflowState(raw_patient_data=patient_data)
        final_state = self.graph.invoke(initial_state)
        return final_state.decision_support_response
```

---

"""

print("Generating Part I (sections 3-6)...")
append_to_architecture(part1_remaining)
print("\n✅ Part I Complete!")
print("\nRun this script to continue building the document.")
print("Next: Part II (2025 Enhancements)")
