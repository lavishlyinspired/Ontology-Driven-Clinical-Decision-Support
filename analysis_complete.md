# Ontology-Driven Clinical Decision Support - Complete Analysis

## Project Overview

**Project Name:** Lung Cancer Assistant (LCA)
**Type:** Ontology-driven clinical decision support system with AI agents
**Based On:** Sesen et al., University of Oxford (2013)
**Tech Stack:** Python 3.11+, OWL 2 (owlready2), LangChain/LangGraph, Neo4j, Ollama LLM, React/Next.js, FastAPI

**Core Purpose:** Provides intelligent, evidence-based clinical decision support for lung cancer treatment using OWL 2 ontological reasoning, SNOMED-CT integration, LangGraph agentic workflows, local LLM inference (Ollama), and Neo4j graph database.

---

## Architecture Summary

### Layer Architecture

```
┌────────────────────────────────────────────────────────┐
│  Frontend (React/Next.js)                              │
│  - Dashboard, Chat, Patient Mgmt, Analytics, Twin Viz  │
├────────────────────────────────────────────────────────┤
│  REST API (FastAPI) - 35+ routes                       │
├────────────────────────────────────────────────────────┤
│  Service Layer (50+ services)                          │
│  - Orchestration, RAG, FHIR, Auth, Cache, Batch, Audit │
├────────────────────────────────────────────────────────┤
│  Agent Layer (14 agents via LangGraph)                 │
│  - 6 Core + 7 Specialized + 1 Orchestrator            │
├────────────────────────────────────────────────────────┤
│  Ontology Layer (OWL 2 + SNOMED-CT)                   │
│  - LUCADA Ontology, Guideline Rules, LOINC, RxNorm    │
├────────────────────────────────────────────────────────┤
│  Analytics Layer                                       │
│  - Survival, Counterfactual, Uncertainty, Trials       │
├────────────────────────────────────────────────────────┤
│  Data Layer (Neo4j, Redis, PostgreSQL, Vector Store)   │
└────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **"Neo4j as a tool, not a brain"** - All reasoning in Python/OWL; Neo4j is storage only
2. **Single Writer Pattern** - Only PersistenceAgent writes to Neo4j
3. **Lazy Loading** - Heavy modules (torch, sentence-transformers) loaded on demand
4. **Evidence-Based** - All recommendations backed by clinical guidelines (NICE, NCCN, ASCO, ESMO)
5. **Privacy-First** - All inference local; no external API calls for sensitive data

---

## Core Modules

### Ontology Layer (6 modules in `backend/src/ontology/`)

| Module | Purpose | Key Class |
|--------|---------|-----------|
| `lucada_ontology.py` | Core LUCADA ontology (60+ classes, 35+ object props, 60+ data props) | `LUCADAOntology` |
| `guideline_rules.py` | NICE CG121 guideline rules engine (10+ rules: R1-R10) | `GuidelineRuleEngine` |
| `snomed_loader.py` | SNOMED-CT OWL loader (60+ lung cancer codes) | `SNOMEDLoader` |
| `clinical_mappings.py` | Clinical concept mapping utilities | mapping functions |
| `loinc_integrator.py` | Laboratory test code mapping | `LOINCIntegrator` |
| `rxnorm_mapper.py` | Medication terminology mapping | `RxNormMapper` |

### Agent Layer (14 agents in `backend/src/agents/`)

**Core 6-Agent Pipeline** (LangGraph StateGraph):
1. `IngestionAgent` - Validates/normalizes raw patient data → `PatientFact`
2. `SemanticMappingAgent` - Maps to SNOMED-CT codes → `PatientFactWithCodes`
3. `ClassificationAgent` - Applies LUCADA ontology + NICE guidelines → `ClassificationResult`
4. `ConflictResolutionAgent` - Resolves conflicting recommendations
5. `PersistenceAgent` - Writes to Neo4j (ONLY writer) → `WriteReceipt`
6. `ExplanationAgent` - Generates MDT summaries → `MDTSummary`

**Specialized Agents:**
- `BiomarkerAgent` - Molecular profiling (EGFR, ALK, ROS1, BRAF, PD-L1)
- `NSCLCAgent` - Non-small cell lung cancer pathways
- `SCLCAgent` - Small cell lung cancer pathways
- `ComorbidityAgent` - Comorbidity risk assessment
- `LabInterpretationAgent` - Lab result analysis
- `MedicationManagementAgent` - Drug interaction checking
- `MonitoringCoordinatorAgent` - Treatment monitoring protocols
- `NegotiationProtocol` - Multi-agent consensus

**Workflow Orchestrators:**
- `lca_workflow.py` - Main 6-agent LangGraph workflow (`LCAWorkflow`)
- `lca_agents.py` - LangGraph workflow creation helpers
- `integrated_workflow.py` - Full 11-14 agent pipeline
- `dynamic_orchestrator.py` - Adaptive routing

### Analytics Layer (4 modules in `backend/src/analytics/`)

| Module | Purpose | Key Class |
|--------|---------|-----------|
| `survival_analyzer.py` | Kaplan-Meier curves, Cox models, log-rank tests | `SurvivalAnalyzer` |
| `uncertainty_quantifier.py` | Bayesian epistemic/aleatoric uncertainty | `UncertaintyQuantifier` |
| `counterfactual_engine.py` | What-if treatment analysis | `CounterfactualEngine` |
| `clinical_trial_matcher.py` | Trial eligibility matching | `ClinicalTrialMatcher` |

### Data Models (`backend/src/db/models.py`)

Key Pydantic models and their flow:

```
PatientFact (10 LUCADA properties)
  → PatientFactWithCodes (+ SNOMED-CT codes)
    → ClassificationResult (scenario, recommendations, reasoning)
      → TreatmentRecommendation (with TreatmentArguments pro/con)
        → MDTSummary (natural language report)
          → DecisionSupportResponse (final API output)
```

Important Enums: `TNMStage` (IA-IV), `HistologyType` (6 types), `PerformanceStatus` (WHO 0-4), `EvidenceLevel` (Grade A-D), `TreatmentIntent` (Curative/Palliative/Adjuvant/Neoadjuvant/Supportive)

### Service Layer (50+ in `backend/src/services/`)

Key services: `lca_service.py` (main orchestrator), `conversation_service.py` (chatbot), `clustering_service.py` (patient cohorts), `embedding_service.py` (vectors), `rag_service.py` (RAG), `analytics_service.py`, `audit_service.py`, `auth_service.py`, `cache_service.py`, `batch_service.py`, `clinical_trials_service.py`, `fhir_service.py`, `transparency_service.py`

### Database Layer (`backend/src/db/`)

| Module | Purpose |
|--------|---------|
| `neo4j_schema.py` | Neo4j schema definition (`LUCADAGraphDB`) |
| `neo4j_tools.py` | CRUD operations (`Neo4jReadTools`, `Neo4jWriteTools`) |
| `graph_algorithms.py` | GDS algorithms |
| `vector_store.py` | Vector embeddings store |
| `provenance_tracker.py` | Audit trail system |
| `temporal_analyzer.py` | Temporal trend analysis |

---

## End-to-End Data Flow

```
Raw Patient Data
  ↓ IngestionAgent (validate, normalize TNM/histology/PS)
PatientFact
  ↓ SemanticMappingAgent (SNOMED-CT, LOINC, RxNorm codes)
PatientFactWithCodes
  ↓ ClassificationAgent (LUCADA ontology + NICE guidelines)
  ↓   ├─ BiomarkerAgent (targeted therapies)
  ↓   ├─ NSCLCAgent / SCLCAgent (cancer-type pathways)
  ↓   ├─ ComorbidityAgent (risk factors)
  ↓   └─ NegotiationProtocol (consensus)
ClassificationResult
  ↓ ConflictResolutionAgent (resolve conflicts, rank)
ResolvedClassificationResult
  ↓ PersistenceAgent (write to Neo4j)
WriteReceipt
  ↓ ExplanationAgent (generate MDT summary via Ollama)
MDTSummary → DecisionSupportResponse
```

---

## Neo4j Graph Schema

```
(Patient)-[:HAS_CLINICAL_FINDING]->(ClinicalFinding)
(Patient)-[:HAS_TREATMENT_PLAN]->(TreatmentPlan)
(Patient)-[:HAS_RECOMMENDATION]->(RecommendationRecord)
(Patient)-[:SIMILAR_TO]->(Patient)
(ClinicalFinding)-[:CLASSIFIED_AS]->(PatientScenario)
(TreatmentPlan)-[:SUPPORTED_BY]->(Guideline)
(RecommendationRecord)-[:PROPOSED_BY]->(Agent)
(RecommendationRecord)-[:JUSTIFIED_BY]->(Evidence)
```

---

## External Dependencies

**Python:** owlready2, rdflib, langchain, langgraph, langchain-ollama, mcp, neo4j, sentence-transformers, numpy, pandas, lifelines, fastapi, rich
**Infrastructure:** Neo4j 5.15, Redis 7, Ollama (llama3.2/mistral), PostgreSQL 15, HAPI FHIR Server

---

## Test Suite (16 files in `tests/`)

Key test files: `test_ontology.py`, `test_agents.py`, `test_neo4j_integration.py`, `test_integrated_workflow.py`, `test_6agent_workflow.py`, `test_advanced_agents.py`, `test_digital_twin_engine.py`, `test_analytics.py`, `test_patient_crud.py`

---

## Entry Points

| Script | Purpose |
|--------|---------|
| `cli.py` | Main CLI (`python cli.py setup/run/test`) |
| `start_backend.py` | FastAPI server |
| `run_lca.py` | Interactive demo |
| `demo_full_system.py` | Complete feature demo |

---

## Data Generation

- `data/synthetic_patient_generator.py` - `SyntheticPatientGenerator` class generates realistic cohorts
- `data/sample_patients.py` - Pre-defined test patients (Jenny_Sesen canonical example, plus stage-specific cases)

---

## Existing Notebooks (in `notebooks/`)

1. `LCA_Experiments.ipynb` - Full system demo
2. `LCA_Complete_Implementation_Demo.ipynb` - Complete workflow walkthrough
3. `LCA_2025_Improvements_Demo.ipynb` - Latest features
4. `00_SETUP_AND_CONFIG.ipynb` - Setup guide

---

## Key API Import Paths

```python
# Models
from backend.src.db.models import PatientFact, ClassificationResult, MDTSummary, DecisionSupportResponse

# Agents
from backend.src.agents.ingestion_agent import IngestionAgent
from backend.src.agents.semantic_mapping_agent import SemanticMappingAgent
from backend.src.agents.classification_agent import ClassificationAgent
from backend.src.agents.conflict_resolution_agent import ConflictResolutionAgent
from backend.src.agents.explanation_agent import ExplanationAgent
from backend.src.agents.biomarker_agent import BiomarkerAgent, BiomarkerProfile
from backend.src.agents.nsclc_agent import NSCLCAgent
from backend.src.agents.comorbidity_agent import ComorbidityAgent

# Workflow
from backend.src.agents.lca_workflow import LCAWorkflow

# Ontology
from backend.src.ontology.lucada_ontology import LUCADAOntology
from backend.src.ontology.guideline_rules import GuidelineRuleEngine
from backend.src.ontology.snomed_loader import SNOMEDLoader

# Analytics
from backend.src.analytics.survival_analyzer import SurvivalAnalyzer
from backend.src.analytics.uncertainty_quantifier import UncertaintyQuantifier
from backend.src.analytics.counterfactual_engine import CounterfactualEngine
from backend.src.analytics.clinical_trial_matcher import ClinicalTrialMatcher

# Data generation
from data.synthetic_patient_generator import SyntheticPatientGenerator
from data.sample_patients import SAMPLE_PATIENTS
```
