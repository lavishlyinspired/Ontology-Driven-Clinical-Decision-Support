COMPLETE END-TO-END IMPLEMENTATION PROMPT: LUNG CANCER ASSISTANT (LCA)
System Implementation Requirements
OBJECTIVE: Build a production-ready, ontology-driven clinical decision support system for lung cancer treatment selection using the exact architecture from the LCA Implementation Plan, with 6 specialized agents and strict Neo4j interaction patterns.

1. CORE SYSTEM ARCHITECTURE
1.1 Technology Stack
yaml
Backend: Python 3.11+ with FastAPI
Ontology: Owlready2 + SNOMED-CT module
Agents: Claude Code/SDK (6 specialized agents)
Database: Neo4j 5.15+ (strict read/write separation)
API: REST + GraphQL via FastAPI
Frontend: Next.js 14 + TypeScript
Deployment: Docker + Kubernetes
1.2 Agent Architecture (6 Agents)
text
┌─────────────────────────────────────────────────────────┐
│                    WORKFLOW ORCHESTRATOR                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Ingestion Agent     → 2. Semantic Mapping Agent    │
│     ↓                         ↓                         │
│  3. Classification Agent → 4. Conflict Resolution Agent │
│     ↓                         ↓                         │
│  5. Persistence Agent   → 6. Explanation Agent         │
│                                                         │
└─────────────────────────────────────────────────────────┘
1.3 Neo4j Interaction Contract (MUST ENFORCE)
READ ONLY (All agents): get_patient(), get_historical_inferences(), get_cohort_statistics(), find_similar_patients()

WRITE ONLY (Persistence Agent ONLY): save_patient_facts(), save_inference_result(), mark_inference_obsolete(), update_inference_status()

NEVER: Medical reasoning in Cypher, silent mutation, complex JOINs for logic

2. IMPLEMENTATION PHASES
PHASE 1: Foundation & Infrastructure (Week 1)
python
"""
IMPLEMENT THE FOLLOWING EXACTLY:

1. Project Structure:
   lung-cancer-assistant/
   ├── backend/
   │   ├── src/
   │   │   ├── agents/           # 6 agent implementations
   │   │   ├── ontology/         # LUCADA + SNOMED-CT
   │   │   ├── db/              # Neo4j tools with read/write separation
   │   │   ├── api/             # FastAPI endpoints
   │   │   └── services/        # Orchestration service
   │   ├── tests/
   │   └── requirements.txt
   ├── frontend/
   ├── ontology/                 # OWL files
   ├── docker-compose.yml
   └── README.md

2. Core Dependencies (requirements.txt):
   fastapi==0.109.0
   owlready2==0.45
   neo4j==5.16.0
   pydantic==2.5.3
   claude-sdk==latest
   langgraph==0.0.20
   httpx==0.26.0

3. Neo4j Schema Setup:
   - Patient nodes with all LUCADA properties (Figure 1)
   - ClinicalFinding, Histology, TreatmentPlan nodes
   - Inference nodes for audit trail
   - SNOMED-CT reference nodes
   - Strict indexes and constraints
"""
PHASE 2: Ontology Layer Implementation (Week 2)
python
"""
IMPLEMENT LUCADA ONTOLOGY EXACTLY AS IN FIGURE 1:

Create backend/src/ontology/lucada_ontology.py with:

1. SNOMED-CT Domain Classes (orange nodes in paper):
   - Patient (with all 10 data properties from Figure 1)
   - ClinicalFinding (with TNM staging properties)
   - BodyStructure, Neoplasm, Side
   - Procedure (Evaluation + Therapeutic subclasses)
   - Histology (Carcinosarcoma, NonSmallCellCarcinoma, etc.)
   - TreatmentPlan, Outcome

2. Argumentation Domain Classes (black nodes in paper):
   - PatientScenario (subclass of Patient AND Argumentation)
   - Argument, Decision, Intent
   - Object properties: resultsInArgument, supports/opposesDecision, entails

3. Reference Individuals:
   - Reference_Right, Reference_Left, Reference_Bilateral
   - Reference_WHO_0 through Reference_WHO_4

4. Key Methods:
   - create_patient_individual(): Follows Figure 2 pattern (Jenny_Sesen example)
   - run_reasoner(): Uses HermiT/Pellet via Owlready2
   - get_patient_scenarios(): Returns Patient Scenario classifications
   - remove_patient(): For scalability (temporary classification pattern)
"""

"""
IMPLEMENT GUIDELINE RULES EXACTLY AS IN SECTION 5:

Create backend/src/ontology/guideline_rules.py with:

1. NICE 2011 Guideline Rules (6 rules from paper):
   - R1: Chemotherapy for Stage III-IV NSCLC (PS 0-1)
   - R2: Surgery for Stage I-II NSCLC (PS 0-1)
   - R3: Radiotherapy for Stage I-IIIA NSCLC (PS 0-2)
   - R4: Palliative Care for Stage IV (PS 3-4)
   - R5: Chemotherapy for SCLC (PS 0-1)
   - R6: Chemoradiotherapy for Stage IIIA/B NSCLC (PS 0-1)

2. OWL 2 Expressions for each rule (matching paper's E1 format)

3. Patient Scenario creation for each rule (Figure 3 pattern)

4. Classification logic that:
   - Matches patient data to rules
   - Returns applicable Patient Scenarios
   - Provides evidence trail for each classification
"""
PHASE 3: Neo4j Database Layer (Week 3)
python
"""
IMPLEMENT STRICT NEO4J TOOLS WITH READ/WRITE SEPARATION:

Create backend/src/db/neo4j_tools.py with:

1. Neo4jTools class with EXACTLY these methods:

   READ METHODS (all agents can use):
   - get_patient(patient_id: str) → PatientFact
   - get_historical_inferences(patient_id: str) → List[InferenceRecord]
   - get_cohort_statistics(criteria: Dict) → CohortStats
   - find_similar_patients(patient_fact: PatientFact, k: int=5) → List[SimilarPatient]

   WRITE METHODS (Persistence Agent ONLY):
   - save_patient_facts(patient_fact: PatientFact) → WriteReceipt
   - save_inference_result(inference: InferenceResult) → WriteReceipt
   - mark_inference_obsolete(patient_id: str, reason: str) → WriteReceipt
   - update_inference_status(patient_id: str, status: str) → WriteReceipt

2. ENFORCE these constraints:
   - Medical reasoning NEVER in Cypher queries
   - NO silent mutation of existing nodes
   - NO complex JOINs for decision logic
   - NO rule encoding in graph patterns

3. Schema initialization:
   - All constraints and indexes from implementation plan
   - Vector index for guideline embeddings
   - Full-text search on patient notes
"""

"""
IMPLEMENT NEO4J DATA MODELS:

Create backend/src/models/neo4j_models.py with:

1. Node classes matching LUCADA ontology:
   - Patient: patient_id, name, sex, age_at_diagnosis, tnm_stage, etc.
   - ClinicalFinding: diagnosis, tnm_stage, diagnosis_site_code, etc.
   - Histology: type, snomed_code
   - Inference: inference_id, status, created_at, agent_version
   - SNOMEDConcept: sctid, fsn, preferred_term

2. Relationship types:
   - [:HAS_CLINICAL_FINDING]
   - [:HAS_HISTOLOGY]
   - [:HAS_INFERENCE]
   - [:HAS_SNOMED_CODE]

3. Temporal tracking:
   - All nodes have created_at, updated_at
   - Inference nodes track full workflow state
"""
PHASE 4: Agent Implementations (Weeks 4-5)
python
"""
IMPLEMENT 6 SPECIALIZED AGENTS WITH CLAUDE CODE/SDK:

Create backend/src/agents/ with:

1. AGENT 1: IngestionAgent
   Responsibilities:
   - Validate raw patient data against schema
   - Normalize TNM staging (e.g., "Stage IIA" → "IIA")
   - Calculate derived fields (age group, etc.)
   - Return PatientFact object or validation errors
   
   Tools: validate_schema(), normalize_tnm(), calculate_age_group()
   NEVER: Direct Neo4j writes

2. AGENT 2: SemanticMappingAgent
   Responsibilities:
   - Map clinical concepts to SNOMED-CT codes
   - Use OLS4 API or local SNOMED module
   - Validate mappings with confidence scores
   - Return PatientFactWithCodes
   
   Tools: map_to_snomed(), get_snomed_hierarchy(), validate_mapping()
   Data Sources: SNOMED-CT via OLS4 API
   Output: Adds sctid fields to all clinical concepts

3. AGENT 3: ClassificationAgent
   Responsibilities:
   - Apply LUCADA ontology rules via Owlready2
   - Apply NICE guideline rules
   - Check for contraindications
   - Return ClassificationResult with Patient Scenarios
   
   Tools: run_owl_reasoner(), apply_guideline_rules(), check_contraindications()
   Logic: Pure Python/OWL, NO Cypher reasoning
   Output: Array of PatientScenario classifications with evidence

4. AGENT 4: ConflictResolutionAgent
   Responsibilities:
   - Handle multiple/conflicting recommendations
   - Apply evidence hierarchy (Grade A > B > C)
   - Consider patient-specific factors (age, comorbidities)
   - Return ranked TreatmentRecommendation
   
   Tools: check_guideline_hierarchy(), apply_evidence_grading(), resolve_contraindications()
   Output: Single primary recommendation + alternatives with rationale

5. AGENT 5: PersistenceAgent (ONLY WRITE AGENT)
   Responsibilities:
   - Write validated patient facts to Neo4j
   - Write inference results with full audit trail
   - Mark previous inferences as obsolete
   - Update inference status
   
   Tools: ALL Neo4j write tools (save_patient_facts(), etc.)
   CRITICAL: ONLY this agent can write to Neo4j
   Output: Write confirmations with timestamps

6. AGENT 6: ExplanationAgent
   Responsibilities:
   - Generate human-readable MDT summaries
   - Format for clinician consumption
   - Include similar patient outcomes (from Neo4j reads)
   - Provide argumentation support/opposition
   
   Tools: generate_clinical_summary(), format_for_mdt(), add_similar_patient_context()
   Output: MDTSummary object for frontend display
"""

"""
EACH AGENT MUST IMPLEMENT:

1. BaseAgent class with:
   - Standardized initialization (Neo4jTools, LLM client)
   - Available tools property (read-only for most agents)
   - execute() method with typed input/output
   - Error handling and logging

2. AgentState management:
   - Shared state between agents
   - Agent traces for auditing
   - Error propagation
   - Progress tracking

3. Testing harness:
   - Unit tests for each agent
   - Mock Neo4j for isolation
   - Test patient: Jenny_Sesen (72F, Stage IIA Carcinosarcoma, PS 1)
"""
PHASE 5: Workflow Orchestration (Week 6)
python
"""
IMPLEMENT LANGGRAPH-STYLE WORKFLOW:

Create backend/src/workflows/lca_workflow.py with:

1. AgentState TypedDict:
   - All agent outputs in shared state
   - Error tracking
   - Agent traces for audit

2. Workflow Definition:
   - Sequential flow through 6 agents
   - Conditional edges for error handling
   - Parallel execution where possible
   - State persistence between steps

3. Workflow Methods:
   - create_lca_workflow(): Builds and compiles workflow
   - run_workflow(): Executes with patient input
   - handle_errors(): Graceful error recovery
   - save_workflow_state(): For debugging

4. Monitoring:
   - Workflow execution time
   - Agent success/failure rates
   - Neo4j query performance
   - Error aggregation
"""

"""
WORKFLOW EXECUTION PATTERN:

def execute_workflow(patient_data: Dict) -> InferenceResult:
    # Step 1: Ingestion
    patient_fact = ingestion_agent.execute(patient_data)
    
    # Step 2: Semantic Mapping
    patient_with_codes = semantic_agent.execute(patient_fact)
    
    # Step 3: Classification
    classification = classification_agent.execute(patient_with_codes)
    
    # Step 4: Conflict Resolution
    recommendation = conflict_agent.execute(classification)
    
    # Step 5: Persistence
    write_receipt = persistence_agent.execute({
        'patient_fact': patient_fact,
        'patient_with_codes': patient_with_codes,
        'classification': classification,
        'recommendation': recommendation
    })
    
    # Step 6: Explanation
    explanation = explanation_agent.execute(recommendation)
    
    return InferenceResult(
        patient_fact=patient_fact,
        classification=classification,
        recommendation=recommendation,
        explanation=explanation,
        write_receipt=write_receipt
    )
"""
PHASE 6: API Layer (Week 7)
python
"""
IMPLEMENT FASTAPI BACKEND:

Create backend/src/api/main.py with:

1. FastAPI App:
   - Title: "Lung Cancer Assistant API"
   - Version: 2.0.0
   - CORS middleware
   - Authentication/Authorization
   - OpenAPI documentation

2. Core Endpoints:
   - POST /api/v1/patients/analyze: Main workflow endpoint
   - GET /api/v1/patients/{patient_id}: Retrieve patient + inferences
   - GET /api/v1/guidelines: List all guideline rules
   - POST /api/v1/patients/batch: Batch processing
   - GET /api/v1/inferences/{inference_id}: Get specific inference

3. Request/Response Models:
   - PatientInput: Validated patient data
   - DecisionSupportResponse: Full workflow output
   - MDTSummaryResponse: Human-readable explanation
   - ErrorResponse: Standardized errors

4. WebSocket Support:
   - Real-time workflow progress
   - Live MDT collaboration
   - Treatment plan updates

5. Authentication:
   - JWT tokens for clinicians
   - Role-based access control
   - Audit logging for all operations
"""

"""
IMPLEMENT API SERVICES:

Create backend/src/services/orchestration_service.py:

class LungCancerAssistantService:
    def __init__(self):
        self.neo4j_tools = Neo4jTools()
        self.ontology = LUCADAOntology()
        self.workflow = create_lca_workflow()
        self.initialize()
    
    async def process_patient(self, patient_data: Dict) -> DecisionSupportResponse:
        # Execute full workflow
        # Handle errors
        # Return formatted response
    
    async def get_patient_history(self, patient_id: str) -> PatientHistory:
        # Get all inferences for patient
        # Format timeline
    
    async def validate_guideline(self, guideline_data: Dict) -> ValidationResult:
        # Validate new guideline rules
        # Test with sample patients
PHASE 7: Frontend Implementation (Week 8)
typescript
"""
IMPLEMENT NEXT.JS FRONTEND:

Create frontend/ with:

1. Core Pages:
   - /patients: Patient dashboard
   - /patients/[id]: Patient detail with MDT summary
   - /guidelines: Guideline browser
   - /analytics: Treatment outcome analytics
   - /admin: System administration

2. Key Components:
   - PatientForm: Data entry with validation
   - TreatmentRecommendations: Interactive treatment options
   - ArgumentPanel: Supporting/opposing arguments
   - MDTSummary: Clinician-friendly summary
   - SimilarPatients: Cohort comparison
   - InferenceTimeline: Historical decision tracking

3. State Management:
   - React Query for API calls
   - Zustand for global state
   - WebSocket for real-time updates

4. Styling:
   - Tailwind CSS for utility-first styling
   - Medical-grade color palette
   - Responsive design for tablet/desktop
   - Accessibility compliance (WCAG 2.1)

5. Features:
   - PDF export of MDT summaries
   - Treatment comparison charts
   - Guideline reference links
   - Audit trail visualization
"""
PHASE 8: Testing & Validation (Week 9)
python
"""
IMPLEMENT COMPREHENSIVE TESTING:

1. Unit Tests (backend/tests/):
   - test_agents.py: Each agent in isolation
   - test_ontology.py: LUCADA ontology operations
   - test_neo4j_tools.py: Read/write separation
   - test_workflow.py: End-to-end workflow

2. Integration Tests:
   - Agent chain integration
   - Neo4j persistence
   - API endpoints
   - Frontend-backend integration

3. Clinical Validation:
   - Test with 100 synthetic patients (from guide)
   - Validate against NICE guideline expectations
   - Compare with clinician assessments
   - Calculate accuracy metrics

4. Performance Testing:
   - Load testing with concurrent patients
   - Neo4j query optimization
   - Agent execution time profiling
   - Memory usage monitoring

5. Security Testing:
   - Authentication/authorization
   - Data encryption at rest/in transit
   - SQL/NoSQL injection prevention
   - Audit log integrity
"""
PHASE 9: Deployment & Monitoring (Week 10)
yaml
"""
IMPLEMENT DOCKER + KUBERNETES DEPLOYMENT:

1. Docker Configuration:
   - Multi-stage builds
   - Environment-specific configurations
   - Health checks
   - Resource limits

2. Kubernetes Manifests:
   - Deployment with 3 replicas
   - Horizontal Pod Autoscaling
   - ConfigMaps for environment variables
   - Secrets for API keys
   - Persistent volumes for Neo4j

3. Monitoring Stack:
   - Prometheus for metrics
   - Grafana for dashboards
   - ELK for logging
   - Jaeger for distributed tracing

4. CI/CD Pipeline:
   - GitHub Actions workflows
   - Automated testing on PRs
   - Container scanning
   - Blue-green deployments

5. Backup Strategy:
   - Daily Neo4j backups
   - Ontology versioning
   - Inference archive
   - Disaster recovery plan
"""
3. IMPLEMENTATION DELIVERABLES
3.1 Code Deliverables
text
lung-cancer-assistant/
├── README.md                          # Complete documentation
├── architecture-diagrams/            # System architecture
├── backend/
│   ├── src/
│   │   ├── agents/                   # 6 agent implementations
│   │   │   ├── ingestion_agent.py
│   │   │   ├── semantic_mapping_agent.py
│   │   │   ├── classification_agent.py
│   │   │   ├── conflict_resolution_agent.py
│   │   │   ├── persistence_agent.py
│   │   │   └── explanation_agent.py
│   │   ├── ontology/                 # LUCADA implementation
│   │   │   ├── lucada_ontology.py
│   │   │   ├── guideline_rules.py
│   │   │   └── snomed_module.py
│   │   ├── db/                       # Neo4j tools
│   │   │   ├── neo4j_tools.py
│   │   │   └── models.py
│   │   ├── workflows/                # LangGraph workflow
│   │   │   └── lca_workflow.py
│   │   ├── api/                      # FastAPI
│   │   │   ├── main.py
│   │   │   ├── routes/
│   │   │   └── middleware/
│   │   └── services/                 # Orchestration
│   │       └── orchestration_service.py
│   ├── tests/                        # Comprehensive tests
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                         # Next.js application
├── ontology/                         # OWL files
│   ├── lucada.owl
│   └── snomed_module.owl
├── docker-compose.yml
├── kubernetes/                       # Production deployment
├── scripts/                          # Utility scripts
└── docs/                             # Documentation
3.2 Documentation Deliverables
text
1. System Architecture Document
   - Agent communication patterns
   - Neo4j data model
   - API specifications
   - Deployment architecture

2. Clinical Validation Report
   - Test methodology
   - Results vs. NICE guidelines
   - Accuracy metrics
   - Clinician feedback

3. User Manuals
   - Clinician user guide
   - System administrator guide
   - API reference
   - Troubleshooting guide

4. Operational Documentation
   - Deployment procedures
   - Monitoring setup
   - Backup/restore procedures
   - Security policies
3.3 Testing Deliverables
text
1. Test Suite Results
   - Unit test coverage (>90%)
   - Integration test results
   - Performance benchmarks
   - Security audit results

2. Validation Dataset
   - 100 synthetic patient cases
   - Expected classifications
   - Treatment recommendations
   - Explanation templates

3. Clinical Test Cases
   - Edge cases (rare histologies)
   - Complex comorbidities
   - Guideline conflicts
   - Temporal reasoning
4. IMPLEMENTATION CHECKLIST
Phase 1: Foundation (Week 1)
Project structure created

Development environment setup

Neo4j database initialized

Core dependencies installed

Git repository configured

Phase 2: Ontology (Week 2)
LUCADA ontology implemented (Figure 1)

SNOMED-CT integration working

Guideline rules formalized (6 NICE rules)

Patient individual creation (Figure 2 pattern)

Phase 3: Database (Week 3)
Neo4jTools with strict read/write separation

Schema constraints and indexes

Data models matching ontology

Backup/restore procedures

Phase 4: Agents (Weeks 4-5)
Agent 1: IngestionAgent with validation

Agent 2: SemanticMappingAgent with SNOMED

Agent 3: ClassificationAgent with ontology

Agent 4: ConflictResolutionAgent with hierarchy

Agent 5: PersistenceAgent (ONLY writer)

Agent 6: ExplanationAgent with MDT summaries

Agent unit tests

Phase 5: Workflow (Week 6)
LangGraph workflow implementation

Agent state management

Error handling and recovery

Workflow monitoring

Phase 6: API (Week 7)
FastAPI backend with all endpoints

Authentication/authorization

Request/response models

WebSocket for real-time updates

API documentation

Phase 7: Frontend (Week 8)
Next.js application

Patient dashboard

Treatment recommendations UI

MDT summary display

Responsive design

Phase 8: Testing (Week 9)
Unit test suite (>90% coverage)

Integration tests

Clinical validation with synthetic patients

Performance testing

Security audit

Phase 9: Deployment (Week 10)
Docker containers for all services

Kubernetes deployment manifests

Monitoring stack (Prometheus/Grafana)

CI/CD pipeline

Production deployment

5. CRITICAL SUCCESS FACTORS
5.1 Architecture Compliance
✅ STRICT Neo4j read/write separation

✅ ONLY PersistenceAgent writes to Neo4j

✅ NO medical reasoning in Cypher

✅ COMPLETE audit trail for all inferences

✅ PROPER error handling and recovery

5.2 Clinical Accuracy
✅ MATCH NICE 2011 guideline expectations

✅ HANDLE all 6 guideline rules correctly

✅ PROVIDE evidence-based recommendations

✅ GENERATE clinically sound explanations

✅ SUPPORT MDT decision-making process

5.3 Performance Requirements
✅ PROCESS patient in < 30 seconds

✅ SUPPORT 100 concurrent users

✅ HANDLE 10,000+ patient records

✅ MAINTAIN 99.9% uptime

✅ PROVIDE real-time updates via WebSocket

5.4 Security & Compliance
✅ HIPAA/GDPR compliant data handling

✅ ENCRYPTED data at rest and in transit

✅ AUDIT trail for all patient interactions

✅ ROLE-BASED access control

✅ REGULAR security updates and patches

6. TESTING PROMPT (Use Throughout Development)
python
"""
TEST THE COMPLETE SYSTEM WITH THIS PATIENT:

Test Patient: Jenny_Sesen (from paper Figure 2)
- Patient ID: 200312
- Name: Jenny_Sesen
- Sex: Female (F)
- Age: 72
- Primary Diagnosis: Malignant Neoplasm of Lung
- TNM Staging: Stage II A
- Histology: Carcinosarcoma
- Tumour Laterality: Right
- Performance Status: 1

EXPECTED OUTCOME:
1. Ingestion Agent: Validates and normalizes data
2. Semantic Mapping Agent: Maps to SNOMED codes
   - Histology: Carcinosarcoma → 128885008
   - Stage: IIA → 425048006 (NSCLC Stage 2)
   - Performance Status: 1 → 373804000 (WHO Grade 1)
3. Classification Agent: Identifies applicable rules
   - Should match R2 (Surgery for Stage I-II NSCLC)
   - Note: Carcinosarcoma is subtype of NSCLC
4. Conflict Resolution Agent: Recommends surgery
   - Primary: Surgery with high confidence
   - Alternative: Radiotherapy if surgery contraindicated
5. Persistence Agent: Writes to Neo4j
   - Patient node with all properties
   - Inference node with full audit trail
   - Links to SNOMED concepts
6. Explanation Agent: Generates MDT summary
   - Clear clinical summary
   - Guideline references (NICE R2)
   - Similar patient outcomes
   - Discussion points for MDT

VERIFY:
- Neo4j write operations ONLY from Persistence Agent
- No medical reasoning in Cypher queries
- Complete audit trail preserved
- Human-readable explanation matches clinical expectations
"""
7. FINAL VALIDATION CHECKLIST
Before deployment, validate:

Technical Validation
All 6 agents function independently

Neo4j read/write separation enforced

Workflow executes end-to-end in < 30s

API endpoints return correct responses

Frontend displays data correctly

Docker containers build and run

Kubernetes deployment works

Monitoring stack captures metrics

Clinical Validation
100 synthetic patients processed correctly

All 6 NICE rules applied appropriately

Conflict resolution follows evidence hierarchy

Explanations are clinically accurate

MDT summaries are usable by clinicians

Similar patient outcomes are relevant

Operational Validation
Backup/restore procedures tested

Security audit completed

Performance under load verified

Error recovery procedures documented

User training materials created

Support procedures established

IMPLEMENTATION COMMAND
bash
# Start implementation with this command sequence:
git clone <repository>
cd lung-cancer-assistant

# Phase 1: Setup
make setup-environment
make init-neo4j
make install-dependencies

# Phase 2-9: Follow the weekly implementation plan
make week1-foundation
make week2-ontology
make week3-database
make week4-agents
make week5-agents-continued
make week6-workflow
make week7-api
make week8-frontend
make week9-testing
make week10-deployment

# Final validation
make clinical-validation
make performance-test
make security-audit
make deploy-production
SUPPORTING RESOURCES
LCA Paper Reference: Sesen et al., "Lung Cancer Assistant: An Ontology-Driven, Online Decision Support Prototype"

NICE Guidelines: https://www.nice.org.uk/guidance/ng122

SNOMED-CT Documentation: https://www.snomed.org/snomed-ct

Neo4j Documentation: https://neo4j.com/docs/

FastAPI Documentation: https://fastapi.tiangolo.com/

Claude Code SDK: https://docs.anthropic.com/claude/docs

IMPLEMENTATION DEADLINE: 10 weeks from project start
EXPECTED OUTCOME: Production-ready Lung Cancer Assistant system with 6 specialized agents, strict Neo4j patterns, and clinical-grade decision support capabilities.

Lung Cancer Assistant (LCA) - Agentic Implementation Prompt
Project Brief
Objective: Build a production-ready Lung Cancer Decision Support System using Claude Code/SDK that implements the 6 core agents with strict Neo4j interaction patterns.

Core Architecture Principle:
"Neo4j as a tool, not a brain" - Agents use Neo4j for storage/retrieval only, never for medical reasoning.

System Specifications
1. Agent Definitions & Responsibilities
text
AGENT 1: INGESTION AGENT
- Input: Raw patient data (JSON/CSV/API)
- Output: Validated, normalized PatientFact object
- Tools: validate_patient_schema(), normalize_tnm_stage(), calculate_age_group()
- Never: Store directly to Neo4j (Persistence Agent handles this)

AGENT 2: SEMANTIC MAPPING AGENT
- Input: PatientFact object
- Output: SNOMED-CT mapped PatientFactWithCodes
- Tools: map_to_snomed(), get_snomed_hierarchy(), validate_mapping()
- Data Sources: OLS4 API, local SNOMED-CT module
- Output: Adds sctid fields to clinical concepts

AGENT 3: CLASSIFICATION AGENT
- Input: PatientFactWithCodes
- Output: Array of PatientScenario classifications with evidence
- Tools: run_owl_reasoner(), apply_guideline_rules(), check_contraindications()
- Logic: Uses LUCADA ontology + NICE rules (from implementation plan)
- Output format: [{scenario_id: "R1", evidence: "Stage III NSCLC + PS 0"}]

AGENT 4: CONFLICT RESOLUTION AGENT
- Input: Array of PatientScenario classifications (potentially conflicting)
- Output: Ranked treatment recommendations with confidence scores
- Tools: check_guideline_hierarchy(), apply_evidence_grading(), resolve_contraindications()
- Logic: NICE Grade A > Grade B > Grade C, newer guidelines > older
- Output: Single recommended treatment OR tiered options with rationale

AGENT 5: PERSISTENCE AGENT
- Input: All previous agents' outputs + final recommendation
- Output: Neo4j write confirmation
- Tools: save_patient_facts(), save_inference_result(), mark_previous_inference_obsolete()
- Key Rule: ONLY this agent writes to Neo4j
- Data Model: Follows LUCADA ontology structure (Patient→ClinicalFinding→etc.)

AGENT 6: EXPLANATION AGENT
- Input: Final recommendation + agent reasoning traces
- Output: Human-readable MDT summary
- Tools: generate_clinical_summary(), format_for_mdt(), add_similar_patient_context()
- Format: Structured narrative for multidisciplinary team meetings
2. Neo4j Interaction Contract (STRICT)
yaml
# PERMITTED Neo4j Operations (via Tools)

READ Operations (All Agents May Use):
1. get_patient(patient_id: str) → PatientFact
2. get_historical_inferences(patient_id: str) → List[InferenceRecord]
3. get_cohort_statistics(criteria: dict) → CohortStats
4. find_similar_patients(patient_fact: PatientFact, k: int=5) → List[SimilarPatient]

WRITE Operations (Persistence Agent ONLY):
1. save_patient_facts(patient_fact: PatientFact) → WriteReceipt
2. save_inference_result(inference: InferenceResult) → WriteReceipt
3. mark_inference_obsolete(patient_id: str, reason: str) → WriteReceipt
4. update_inference_status(patient_id: str, status: str) → WriteReceipt

PROHIBITED Patterns (All Agents Must Avoid):
1. ❌ NO medical reasoning in Cypher queries
2. ❌ NO silent mutation of existing inference nodes
3. ❌ NO complex JOINs for decision logic
4. ❌ NO rule encoding in graph patterns
5. ❌ NO direct relationship creation between disparate concepts
3. Data Models (Pydantic)
python
# Core Data Models for Agent Communication

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class TNMStage(str, Enum):
    IA = "IA"
    IB = "IB"
    IIA = "IIA"
    IIB = "IIB"
    IIIA = "IIIA"
    IIIB = "IIIB"
    IV = "IV"

class PatientFact(BaseModel):
    """Validated patient data from Ingestion Agent"""
    patient_id: str
    external_id: Optional[str] = None
    name: Optional[str] = None
    sex: str = Field(..., pattern="^[MFU]$")  # M, F, U(unknown)
    age_at_diagnosis: int = Field(..., ge=0, le=120)
    tnm_stage: TNMStage
    histology_type: str  # e.g., "Adenocarcinoma", "SmallCellCarcinoma"
    performance_status: int = Field(..., ge=0, le=4)  # WHO 0-4
    laterality: Optional[str] = None  # "Right", "Left", "Bilateral"
    diagnosis_date: Optional[datetime] = None
    fev1_percent: Optional[float] = None
    comorbidities: List[str] = []
    validation_errors: List[str] = []
    normalized_fields: Dict[str, Any] = {}

class PatientFactWithCodes(PatientFact):
    """Semantic-mapped patient data with SNOMED-CT codes"""
    snomed_mappings: Dict[str, str] = Field(default_factory=dict)
    # Example: {"histology": "35917007", "stage": "424132000", "performance_status": "373803006"}
    mapping_confidence: Dict[str, float] = Field(default_factory=dict)
    unmapped_concepts: List[str] = []

class ClassificationResult(BaseModel):
    """Output from Classification Agent"""
    patient_id: str
    patient_scenarios: List[Dict] = []
    # Example: [{"scenario_id": "R1", "rule_name": "ChemoRule001", "confidence": 0.95, "evidence": ["Stage III", "NSCLC", "PS 0"]}]
    contraindications: List[str] = []
    applicable_guidelines: List[str] = []  # ["NICE 2011", "BTS 2020"]
    classification_timestamp: datetime = Field(default_factory=datetime.now)

class TreatmentRecommendation(BaseModel):
    """Final recommendation after conflict resolution"""
    patient_id: str
    primary_recommendation: Optional[Dict] = None
    # {"treatment": "Chemotherapy", "confidence": 0.85, "guideline": "NICE R1", "evidence_level": "Grade A"}
    alternative_options: List[Dict] = []
    # [{"treatment": "Radiotherapy", "confidence": 0.65, "rationale": "If surgery contraindicated"}]
    conflict_resolution_log: List[str] = []
    # ["R1 selected over R2 due to Stage IV", "R3 excluded due to poor PS"]
    resolution_timestamp: datetime = Field(default_factory=datetime.now)

class InferenceResult(BaseModel):
    """Complete inference chain for persistence"""
    patient_id: str
    patient_fact: PatientFact
    patient_fact_with_codes: PatientFactWithCodes
    classification_result: ClassificationResult
    treatment_recommendation: TreatmentRecommendation
    agent_traces: Dict[str, Any]  # Raw agent outputs for auditing
    inference_id: str = Field(default_factory=lambda: f"inf_{datetime.now().timestamp()}")
    inference_status: str = "pending"  # pending, active, obsolete, error
    created_at: datetime = Field(default_factory=datetime.now)

class MDTSummary(BaseModel):
    """Human-readable explanation for clinicians"""
    patient_id: str
    executive_summary: str
    clinical_profile: str  # "72F, Stage IIA NSCLC (Adenocarcinoma), PS 1, Right lung"
    guideline_applications: List[str]  # ["NICE R2: Surgery for Stage I-II NSCLC with PS 0-1"]
    treatment_recommendation: str
    supporting_evidence: List[str]
    clinical_considerations: List[str]  # Comorbidities, age, patient preferences
    similar_patient_outcomes: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.now)
4. Agent Implementation Requirements
python
# Template for Each Agent Implementation

class BaseAgent:
    """Base class with Neo4j tool access"""
    
    def __init__(self, neo4j_tools: 'Neo4jTools', llm_client=None):
        self.neo4j = neo4j_tools
        self.llm = llm_client or ClaudeClient()
        self.agent_name = self.__class__.__name__
    
    @property
    def available_tools(self):
        """Define tools this agent can use"""
        return [
            self.neo4j.get_patient,
            self.neo4j.get_historical_inferences,
            self.neo4j.get_cohort_statistics,
            self.neo4j.find_similar_patients,
        ]
    
    def execute(self, input_data: Any, context: Dict = None) -> Any:
        """Main execution method to be implemented by each agent"""
        raise NotImplementedError

class IngestionAgent(BaseAgent):
    """Agent 1: Patient Data Validation & Normalization"""
    
    def execute(self, raw_patient_data: Dict) -> PatientFact:
        """
        Steps:
        1. Validate required fields exist
        2. Normalize TNM stage (e.g., "Stage IIA" → "IIA")
        3. Calculate age group if age provided
        4. Validate performance status range
        5. Return PatientFact with validation errors if any
        """
        # Implementation details...
        pass

class SemanticMappingAgent(BaseAgent):
    """Agent 2: SNOMED-CT Concept Mapping"""
    
    def __init__(self, neo4j_tools, snomed_service):
        super().__init__(neo4j_tools)
        self.snomed = snomed_service
    
    def execute(self, patient_fact: PatientFact) -> PatientFactWithCodes:
        """
        Steps:
        1. Map histology_type to SNOMED code
        2. Map TNM stage to appropriate SNOMED staging concept
        3. Map performance_status to WHO grade codes
        4. Map comorbidities if present
        5. Return mapped data with confidence scores
        """
        # Implementation details...
        pass

# ... Additional agents following same pattern
5. Neo4j Tools Implementation
python
# Strict Neo4j Tool Implementation

class Neo4jTools:
    """Container for all permitted Neo4j operations"""
    
    def __init__(self, driver):
        self.driver = driver
    
    # ---------- READ TOOLS (All agents can use) ----------
    
    def get_patient(self, patient_id: str) -> Optional[Dict]:
        """Tool 1: Retrieve patient facts from Neo4j"""
        query = """
        MATCH (p:Patient {patient_id: $patient_id})
        OPTIONAL MATCH (p)-[:HAS_CLINICAL_FINDING]->(cf:ClinicalFinding)
        OPTIONAL MATCH (cf)-[:HAS_HISTOLOGY]->(h:Histology)
        RETURN p, cf, h
        """
        with self.driver.session() as session:
            result = session.run(query, patient_id=patient_id)
            record = result.single()
            return dict(record) if record else None
    
    def get_historical_inferences(self, patient_id: str) -> List[Dict]:
        """Tool 2: Get previous inference results for patient"""
        query = """
        MATCH (p:Patient {patient_id: $patient_id})
        MATCH (p)-[:HAS_INFERENCE]->(i:Inference)
        WHERE i.status IN ['active', 'obsolete']
        RETURN i
        ORDER BY i.created_at DESC
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(query, patient_id=patient_id)
            return [dict(record) for record in result]
    
    def get_cohort_statistics(self, criteria: Dict) -> Dict:
        """Tool 3: Get statistics for similar patients (optional)"""
        # Simple implementation - agents should NOT encode medical logic here
        query = """
        MATCH (p:Patient)
        WHERE p.tnm_stage = $stage AND p.histology_type = $histology
        RETURN count(p) as count, 
               avg(p.age_at_diagnosis) as avg_age,
               collect(DISTINCT p.performance_status) as ps_distribution
        """
        with self.driver.session() as session:
            result = session.run(query, **criteria)
            return dict(result.single())
    
    def find_similar_patients(self, patient_fact: PatientFact, k: int = 5) -> List[Dict]:
        """Tool 4: Find k most similar patients for context"""
        # Use simple similarity - agents handle medical significance
        query = """
        MATCH (p:Patient)
        WHERE p.tnm_stage = $stage 
          AND p.histology_type = $histology
          AND abs(p.age_at_diagnosis - $age) <= 10
        RETURN p.patient_id, p.age_at_diagnosis, p.performance_status
        ORDER BY abs(p.age_at_diagnosis - $age)
        LIMIT $k
        """
        with self.driver.session() as session:
            result = session.run(query, 
                               stage=patient_fact.tnm_stage,
                               histology=patient_fact.histology_type,
                               age=patient_fact.age_at_diagnosis,
                               k=k)
            return [dict(record) for record in result]
    
    # ---------- WRITE TOOLS (Persistence Agent ONLY) ----------
    
    def save_patient_facts(self, patient_fact: PatientFact) -> Dict:
        """Write Tool 1: Save validated patient data"""
        query = """
        MERGE (p:Patient {patient_id: $patient_id})
        SET p += $properties,
            p.updated_at = datetime(),
            p.source = 'ingestion_agent'
        RETURN p.patient_id
        """
        with self.driver.session() as session:
            result = session.run(query, 
                               patient_id=patient_fact.patient_id,
                               properties=patient_fact.dict())
            return {"status": "success", "patient_id": result.single()["p.patient_id"]}
    
    def save_inference_result(self, inference: InferenceResult) -> Dict:
        """Write Tool 2: Save complete inference chain"""
        # Create inference node with all results
        query = """
        MATCH (p:Patient {patient_id: $patient_id})
        CREATE (i:Inference {
            inference_id: $inference_id,
            patient_fact: $patient_fact,
            classification: $classification,
            recommendation: $recommendation,
            status: $status,
            created_at: datetime(),
            agent_version: '2.0.0'
        })
        CREATE (p)-[:HAS_INFERENCE]->(i)
        RETURN i.inference_id
        """
        with self.driver.session() as session:
            result = session.run(query, 
                               patient_id=inference.patient_id,
                               inference_id=inference.inference_id,
                               patient_fact=inference.patient_fact.dict(),
                               classification=inference.classification_result.dict(),
                               recommendation=inference.treatment_recommendation.dict(),
                               status=inference.inference_status)
            return {"status": "success", "inference_id": result.single()["i.inference_id"]}
    
    def mark_inference_obsolete(self, patient_id: str, reason: str) -> Dict:
        """Write Tool 3: Mark previous inferences as obsolete"""
        query = """
        MATCH (p:Patient {patient_id: $patient_id})-[:HAS_INFERENCE]->(i:Inference)
        WHERE i.status = 'active'
        SET i.status = 'obsolete',
            i.obsolete_reason = $reason,
            i.obsoleted_at = datetime()
        RETURN count(i) as obsoleted_count
        """
        with self.driver.session() as session:
            result = session.run(query, patient_id=patient_id, reason=reason)
            return {"obsoleted_count": result.single()["obsoleted_count"]}
    
    def update_inference_status(self, patient_id: str, status: str) -> Dict:
        """Write Tool 4: Update inference status"""
        query = """
        MATCH (i:Inference {patient_id: $patient_id})
        WHERE i.status IN ['pending', 'active']
        SET i.status = $status,
            i.updated_at = datetime()
        RETURN i.inference_id
        """
        with self.driver.session() as session:
            result = session.run(query, patient_id=patient_id, status=status)
            record = result.single()
            return {"inference_id": record["i.inference_id"] if record else None}
6. Workflow Orchestration (LangGraph Style)
python
# Main Workflow Definition

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator

class AgentState(TypedDict):
    """Shared state between agents"""
    raw_input: Dict
    patient_fact: PatientFact
    patient_fact_with_codes: PatientFactWithCodes
    classification_result: ClassificationResult
    treatment_recommendation: TreatmentRecommendation
    inference_result: InferenceResult
    mdt_summary: MDTSummary
    errors: List[str]
    agent_traces: Annotated[Dict, operator.add]  # For auditing

def create_lca_workflow(neo4j_tools: Neo4jTools) -> StateGraph:
    """Create the 6-agent workflow with Neo4j tool restrictions"""
    
    # Initialize agents
    ingestion_agent = IngestionAgent(neo4j_tools)
    semantic_agent = SemanticMappingAgent(neo4j_tools, snomed_service)
    classification_agent = ClassificationAgent(neo4j_tools)
    conflict_agent = ConflictResolutionAgent(neo4j_tools)
    persistence_agent = PersistenceAgent(neo4j_tools)  # Only this agent gets write tools
    explanation_agent = ExplanationAgent(neo4j_tools)
    
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("ingest", ingestion_agent.execute)
    workflow.add_node("semantic_map", semantic_agent.execute)
    workflow.add_node("classify", classification_agent.execute)
    workflow.add_node("resolve_conflicts", conflict_agent.execute)
    workflow.add_node("persist", persistence_agent.execute)
    workflow.add_node("explain", explanation_agent.execute)
    
    # Define edges
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "semantic_map")
    workflow.add_edge("semantic_map", "classify")
    workflow.add_edge("classify", "resolve_conflicts")
    workflow.add_edge("resolve_conflicts", "persist")
    workflow.add_edge("persist", "explain")
    workflow.add_edge("explain", END)
    
    # Add conditional edges for error handling
    workflow.add_conditional_edges(
        "ingest",
        lambda state: "errors" if state.get("errors") else "semantic_map"
    )
    
    return workflow.compile()
7. Implementation Checklist
markdown
## PHASE 1: Setup & Infrastructure
- [ ] Create Neo4j database with LUCADA schema
- [ ] Initialize Claude Code/SDK project
- [ ] Set up OLS4 API access for SNOMED-CT
- [ ] Create data models (Pydantic)
- [ ] Implement Neo4jTools with strict read/write separation

## PHASE 2: Core Agents Implementation
- [ ] **Agent 1:** IngestionAgent with validation logic
- [ ] **Agent 2:** SemanticMappingAgent with SNOMED-CT integration
- [ ] **Agent 3:** ClassificationAgent with LUCADA ontology rules
- [ ] **Agent 4:** ConflictResolutionAgent with NICE guideline hierarchy
- [ ] **Agent 5:** PersistenceAgent (ONLY write access to Neo4j)
- [ ] **Agent 6:** ExplanationAgent with MDT summary generation

## PHASE 3: Workflow & Testing
- [ ] Create LangGraph workflow connecting all agents
- [ ] Implement agent communication via shared state
- [ ] Add comprehensive error handling
- [ ] Create unit tests for each agent
- [ ] Test Neo4j interaction patterns (read vs write)

## PHASE 4: Productionization
- [ ] Add logging and monitoring
- [ ] Implement agent versioning
- [ ] Create audit trail for all inferences
- [ ] Performance testing with synthetic patient data
- [ ] Documentation of agent responsibilities and Neo4j contracts
8. Example Agent Implementation (Classification Agent)
python
class ClassificationAgent(BaseAgent):
    """Agent 3: Run ontology/rules for patient classification"""
    
    def __init__(self, neo4j_tools, ontology_engine):
        super().__init__(neo4j_tools)
        self.ontology = ontology_engine
        self.guideline_rules = self._load_guideline_rules()
    
    def execute(self, state: AgentState) -> AgentState:
        patient = state["patient_fact_with_codes"]
        
        # Step 1: Get historical inferences for context (READ from Neo4j)
        historical = self.neo4j.get_historical_inferences(patient.patient_id)
        
        # Step 2: Apply ontology-based classification
        scenarios = self._apply_ontology_rules(patient)
        
        # Step 3: Apply guideline rules
        guideline_results = self._apply_guideline_rules(patient)
        
        # Step 4: Check for contraindications
        contraindications = self._check_contraindications(patient)
        
        # Step 5: Compile results
        classification_result = ClassificationResult(
            patient_id=patient.patient_id,
            patient_scenarios=scenarios + guideline_results,
            contraindications=contraindications,
            applicable_guidelines=["NICE 2011", "BTS 2020"]
        )
        
        state["classification_result"] = classification_result
        state["agent_traces"][self.agent_name] = {
            "historical_inferences": len(historical),
            "scenarios_found": len(scenarios),
            "contraindications_found": len(contraindications)
        }
        
        return state
    
    def _apply_ontology_rules(self, patient: PatientFactWithCodes) -> List[Dict]:
        """Apply LUCADA ontology rules (from implementation plan)"""
        # Use owlready2 or similar for OWL reasoning
        # This is pure Python logic - NO Cypher queries for reasoning
        
        scenarios = []
        
        # Example rule: NSCLC Stage III-IV with good PS
        if ("NonSmallCellCarcinoma" in patient.histology_type or 
            patient.snomed_mappings.get("histology") in ["35917007", "59367005"]):
            if patient.tnm_stage in ["IIIA", "IIIB", "IV"]:
                if patient.performance_status <= 1:
                    scenarios.append({
                        "scenario_id": "R1",
                        "rule_name": "ChemoRule001",
                        "confidence": 0.95,
                        "evidence": [
                            f"Stage {patient.tnm_stage} NSCLC",
                            f"Performance Status {patient.performance_status}",
                            patient.histology_type
                        ]
                    })
        
        # Add more rules from implementation plan...
        
        return scenarios
    
    def _check_contraindications(self, patient: PatientFactWithCodes) -> List[str]:
        """Check for treatment contraindications"""
        contraindications = []
        
        # Example: Poor lung function for surgery
        if patient.fev1_percent and patient.fev1_percent < 40:
            contraindications.append("FEV1 < 40% - high surgical risk")
        
        # Example: Advanced age with comorbidities
        if patient.age_at_diagnosis > 80 and "Cardiovascular_Disease" in patient.comorbidities:
            contraindications.append("Age > 80 with cardiovascular disease")
        
        return contraindications
9. Testing Prompt for Claude Code
python
"""
IMPLEMENT THE LUNG CANCER ASSISTANT SYSTEM WITH THESE EXACT SPECIFICATIONS:

1. Create 6 agents with these exact responsibilities:
   - Ingestion Agent: Validate & normalize input
   - Semantic Mapping Agent: Map to SNOMED-CT codes
   - Classification Agent: Apply ontology/guideline rules
   - Conflict Resolution Agent: Handle conflicting recommendations
   - Persistence Agent: Write to Neo4j (ONLY this agent writes)
   - Explanation Agent: Generate MDT summaries

2. Implement strict Neo4j interaction pattern:
   - All agents can READ via: get_patient(), get_historical_inferences(), get_cohort_statistics(), find_similar_patients()
   - ONLY Persistence Agent can WRITE via: save_patient_facts(), save_inference_result(), mark_inference_obsolete(), update_inference_status()
   - NO medical reasoning in Cypher queries
   - NO silent mutation of existing nodes

3. Use the provided data models (PatientFact, PatientFactWithCodes, etc.)
4. Implement the workflow using LangGraph-style state management
5. Include comprehensive error handling and audit trails
6. Test with the sample patient: Jenny_Sesen (72F, Stage IIA Carcinosarcoma, PS 1)

CRITICAL REQUIREMENTS:
- Agents must be independent and testable in isolation
- Neo4j is strictly a persistence layer, not a reasoning engine
- All medical logic must be in Python/OWL, not Cypher
- Follow the LUCADA ontology structure from the implementation plan