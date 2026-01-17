COMPLETE IMPLEMENTATION PROMPT: LUNG CANCER ASSISTANT (LCA)
Production-Ready Clinical Decision Support System
1. SYSTEM OVERVIEW & OBJECTIVE
Build a production-ready, ontology-driven clinical decision support system for lung cancer treatment selection using the exact architecture from the LCA Implementation Plan, with 6 specialized agents and strict Neo4j interaction patterns.

1.1 Core Architectural Principle
text
"Neo4j as a tool, not a brain"
- Agents use Neo4j for storage/retrieval ONLY, NEVER for medical reasoning
- All medical logic resides in Python/OWL agents
- Complete audit trail for every inference
1.2 Key Innovation Over Original Paper (2011)
Aspect	Original (2011)	Modern Implementation (2025)
Inference Engine	OWL reasoner only	Hybrid: OWL + LLM agents
Scalability	Temporary patient add/remove	Neo4j graph database + vector search
Explanations	Rule IDs only	Natural language MDT summaries
User Interface	GWT framework	Next.js 14 + TypeScript
Deployment	On-premise	Docker + Kubernetes cloud-native
Performance	~115K patient limit	Scalable to 1M+ patients
Clinical Updates	Manual rule updates	Agent-driven guideline integration
2. CORE SYSTEM ARCHITECTURE
2.1 Technology Stack (EXACT REQUIREMENTS)
yaml
# MUST USE THESE VERSIONS
Backend: Python 3.11+ with FastAPI 0.109.0
Ontology: Owlready2 0.45 + SNOMED-CT module
Agents: Claude Code/SDK with 6 specialized agents
Database: Neo4j 5.15+ (strict read/write separation)
API Layer: REST + GraphQL via FastAPI
Frontend: Next.js 14 + TypeScript + Tailwind CSS
Deployment: Docker + Kubernetes with Prometheus/Grafana
Vector Store: Neo4j Vector Index for guideline embeddings
Testing: Pytest with >90% coverage
2.2 Agent Architecture (6 SPECIALIZED AGENTS)
text
┌─────────────────────────────────────────────────────────┐
│                    WORKFLOW ORCHESTRATOR                │
│                   (LangGraph Implementation)            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Ingestion Agent     → 2. Semantic Mapping Agent    │
│     ↓                         ↓                         │
│  3. Classification Agent → 4. Conflict Resolution Agent │
│     ↓                         ↓                         │
│  5. Persistence Agent   → 6. Explanation Agent         │
│     (ONLY WRITE AGENT)        (MDT Summary)            │
│                                                         │
└─────────────────────────────────────────────────────────┘
2.3 Neo4j Interaction Contract (STRICT ENFORCEMENT)
READ ONLY Operations (All 6 agents can use):

python
get_patient(patient_id: str) → PatientFact
get_historical_inferences(patient_id: str) → List[InferenceRecord]
get_cohort_statistics(criteria: Dict) → CohortStats
find_similar_patients(patient_fact: PatientFact, k: int=5) → List[SimilarPatient]
WRITE ONLY Operations (Persistence Agent EXCLUSIVELY):

python
save_patient_facts(patient_fact: PatientFact) → WriteReceipt
save_inference_result(inference: InferenceResult) → WriteReceipt
mark_inference_obsolete(patient_id: str, reason: str) → WriteReceipt
update_inference_status(patient_id: str, status: str) → WriteReceipt
PROHIBITED Patterns (ALL AGENTS MUST AVOID):

❌ NO medical reasoning in Cypher queries

❌ NO silent mutation of existing inference nodes

❌ NO complex JOINs for decision logic

❌ NO rule encoding in graph patterns

❌ NO direct relationship creation between disparate concepts

3. IMPLEMENTATION PHASES (10 WEEKS)
PHASE 1: Foundation & Infrastructure (Week 1)
python
"""
CREATE EXACT PROJECT STRUCTURE:

lung-cancer-assistant/
├── backend/
│   ├── src/
│   │   ├── agents/           # 6 agent implementations
│   │   │   ├── ingestion_agent.py
│   │   │   ├── semantic_mapping_agent.py
│   │   │   ├── classification_agent.py
│   │   │   ├── conflict_resolution_agent.py
│   │   │   ├── persistence_agent.py
│   │   │   └── explanation_agent.py
│   │   ├── ontology/         # LUCADA + SNOMED-CT
│   │   │   ├── lucada_ontology.py
│   │   │   ├── guideline_rules.py
│   │   │   └── snomed_module.py
│   │   ├── db/              # Neo4j tools with read/write separation
│   │   │   ├── neo4j_tools.py
│   │   │   └── models.py
│   │   ├── api/             # FastAPI endpoints
│   │   │   ├── main.py
│   │   │   ├── routes/
│   │   │   └── middleware/
│   │   ├── workflows/        # LangGraph workflow
│   │   │   └── lca_workflow.py
│   │   └── services/        # Orchestration service
│   │       └── orchestration_service.py
│   ├── tests/
│   │   ├── test_agents.py
│   │   ├── test_ontology.py
│   │   ├── test_neo4j_tools.py
│   │   └── test_workflow.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                 # Next.js 14 + TypeScript
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   └── lib/
│   └── package.json
├── ontology/                 # OWL files
│   ├── lucada.owl
│   └── snomed_module.owl
├── docker-compose.yml
├── kubernetes/               # Production deployment
├── scripts/                  # Utility scripts
└── README.md

CORE DEPENDENCIES (requirements.txt):
fastapi==0.109.0
owlready2==0.45
neo4j==5.16.0
pydantic==2.5.3
anthropic==0.18.0           # Claude SDK
langchain==0.1.0
langchain-anthropic==0.1.0
langgraph==0.0.20
httpx==0.26.0
python-dotenv==1.0.0
pytest==7.4.3
uvicorn==0.27.0

NEO4J SCHEMA SETUP:
- Patient nodes with ALL 10 LUCADA data properties (Figure 1)
- ClinicalFinding, Histology, TreatmentPlan nodes
- Inference nodes for complete audit trail
- SNOMED-CT reference nodes
- STRICT indexes and constraints
- Vector index for guideline embeddings
"""
PHASE 2: Ontology Layer Implementation (Week 2)
python
"""
IMPLEMENT LUCADA ONTOLOGY EXACTLY AS IN FIGURE 1:

Create backend/src/ontology/lucada_ontology.py with:

1. SNOMED-CT Domain Classes (orange nodes in paper):
   - Patient (with ALL 10 data properties from Figure 1)
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

4. Key Methods (MUST IMPLEMENT):
   - create_patient_individual(): Follows Figure 2 pattern (Jenny_Sesen example)
   - run_reasoner(): Uses HermiT via Owlready2
   - get_patient_scenarios(): Returns Patient Scenario classifications
   - remove_patient(): For scalability (temporary classification pattern)

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
   - Matches patient data to rules using OWL reasoning
   - Returns applicable Patient Scenarios
   - Provides evidence trail for each classification
"""
PHASE 3: Neo4j Database Layer (Week 3)
python
"""
IMPLEMENT STRICT NEO4J TOOLS WITH READ/WRITE SEPARATION:

Create backend/src/db/neo4j_tools.py with:

1. Neo4jTools class with EXACTLY these methods:

   READ METHODS (all 6 agents can use):
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

AGENT 1: IngestionAgent
Responsibilities:
- Validate raw patient data against schema
- Normalize TNM staging (e.g., "Stage IIA" → "IIA")
- Calculate derived fields (age group, etc.)
- Return PatientFact object or validation errors

Tools: validate_schema(), normalize_tnm(), calculate_age_group()
NEVER: Direct Neo4j writes

AGENT 2: SemanticMappingAgent
Responsibilities:
- Map clinical concepts to SNOMED-CT codes
- Use OLS4 API or local SNOMED module
- Validate mappings with confidence scores
- Return PatientFactWithCodes

Tools: map_to_snomed(), get_snomed_hierarchy(), validate_mapping()
Data Sources: SNOMED-CT via OLS4 API
Output: Adds sctid fields to all clinical concepts

AGENT 3: ClassificationAgent
Responsibilities:
- Apply LUCADA ontology rules via Owlready2
- Apply NICE guideline rules
- Check for contraindications
- Return ClassificationResult with Patient Scenarios

Tools: run_owl_reasoner(), apply_guideline_rules(), check_contraindications()
Logic: Pure Python/OWL, NO Cypher reasoning
Output: Array of PatientScenario classifications with evidence

AGENT 4: ConflictResolutionAgent
Responsibilities:
- Handle multiple/conflicting recommendations
- Apply evidence hierarchy (Grade A > B > C)
- Consider patient-specific factors (age, comorbidities)
- Return ranked TreatmentRecommendation

Tools: check_guideline_hierarchy(), apply_evidence_grading(), resolve_contraindications()
Output: Single primary recommendation + alternatives with rationale

AGENT 5: PersistenceAgent (ONLY WRITE AGENT)
Responsibilities:
- Write validated patient facts to Neo4j
- Write inference results with full audit trail
- Mark previous inferences as obsolete
- Update inference status

Tools: ALL Neo4j write tools (save_patient_facts(), etc.)
CRITICAL: ONLY this agent can write to Neo4j
Output: Write confirmations with timestamps

AGENT 6: ExplanationAgent
Responsibilities:
- Generate human-readable MDT summaries
- Format for clinician consumption
- Include similar patient outcomes (from Neo4j reads)
- Provide argumentation support/opposition

Tools: generate_clinical_summary(), format_for_mdt(), add_similar_patient_context()
Output: MDTSummary object for frontend display

TESTING HARNESS:
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

WORKFLOW EXECUTION PATTERN:

def execute_workflow(patient_data: Dict) → InferenceResult:
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

3. Monitoring:
   - Workflow execution time (< 30 seconds per patient)
   - Agent success/failure rates
   - Neo4j query performance
   - Error aggregation
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
   - Authentication/Authorization (JWT tokens)
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

IMPLEMENT API SERVICES:

Create backend/src/services/orchestration_service.py:

class LungCancerAssistantService:
    def __init__(self):
        self.neo4j_tools = Neo4jTools()
        self.ontology = LUCADAOntology()
        self.workflow = create_lca_workflow()
        self.initialize()
    
    async def process_patient(self, patient_data: Dict) → DecisionSupportResponse:
        # Execute full workflow
        # Handle errors
        # Return formatted response
    
    async def get_patient_history(self, patient_id: str) → PatientHistory:
        # Get all inferences for patient
        # Format timeline
    
    async def validate_guideline(self, guideline_data: Dict) → ValidationResult:
        # Validate new guideline rules
        # Test with sample patients
"""
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
   - Load testing with concurrent patients (< 30 sec/patient)
   - Neo4j query optimization
   - Agent execution time profiling
   - Memory usage monitoring

5. Security Testing:
   - Authentication/authorization
   - Data encryption at rest/in transit
   - SQL/NoSQL injection prevention
   - Audit log integrity

VALIDATION DATASET:
- 100 synthetic patient cases with expected classifications
- Edge cases (rare histologies, complex comorbidities)
- Guideline conflicts resolution testing
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
4. CRITICAL SUCCESS FACTORS
4.1 Architecture Compliance (MUST PASS)
✅ STRICT Neo4j read/write separation

✅ ONLY PersistenceAgent writes to Neo4j

✅ NO medical reasoning in Cypher

✅ COMPLETE audit trail for all inferences

✅ PROPER error handling and recovery

4.2 Clinical Accuracy (MUST MATCH)
✅ MATCH NICE 2011 guideline expectations for 6 rules

✅ HANDLE histology subtypes correctly (Carcinosarcoma → NSCLC)

✅ PROVIDE evidence-based recommendations with confidence scores

✅ GENERATE clinically sound explanations for MDT

✅ SUPPORT multidisciplinary team decision-making process

4.3 Performance Requirements (MUST ACHIEVE)
✅ PROCESS patient in < 30 seconds (end-to-end workflow)

✅ SUPPORT 100 concurrent users

✅ HANDLE 10,000+ patient records in Neo4j

✅ MAINTAIN 99.9% uptime

✅ PROVIDE real-time updates via WebSocket

4.4 Security & Compliance (MUST IMPLEMENT)
✅ HIPAA/GDPR compliant data handling

✅ ENCRYPTED data at rest and in transit

✅ AUDIT trail for all patient interactions

✅ ROLE-BASED access control (clinician, admin, researcher)

✅ REGULAR security updates and patches

5. TESTING & VALIDATION PROMPT
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
6. FINAL VALIDATION CHECKLIST
Technical Validation (Pre-Deployment)
All 6 agents function independently and in workflow

Neo4j read/write separation enforced at code level

Workflow executes end-to-end in < 30 seconds

API endpoints return correct HTTP responses

Frontend displays data correctly with no errors

Docker containers build, run, and pass health checks

Kubernetes deployment works across environments

Monitoring stack captures all critical metrics

Error recovery procedures documented and tested

Clinical Validation
100 synthetic patients processed correctly (95%+ accuracy)

All 6 NICE rules applied appropriately

Conflict resolution follows evidence hierarchy (Grade A > B > C)

Explanations are clinically accurate and useful

MDT summaries are usable by clinicians in real meetings

Similar patient outcomes are relevant and statistically valid

Edge cases handled appropriately (rare histologies, complex comorbidities)

Operational Validation
Backup/restore procedures tested successfully

Security audit completed with no critical findings

Performance under load verified (100 concurrent users)

Disaster recovery plan documented and tested

User training materials created for clinicians

Support procedures established with SLAs

Maintenance schedule documented

7. IMPLEMENTATION COMMAND SEQUENCE
bash
# START IMPLEMENTATION WITH THESE COMMANDS:

# Clone repository and setup
git clone <lung-cancer-assistant-repo>
cd lung-cancer-assistant

# Phase 1: Setup
make setup-environment          # Create virtual environment
make init-neo4j                 # Initialize Neo4j with schema
make install-dependencies       # Install all Python/Node dependencies
make seed-test-data            # Load 100 synthetic patients

# Phases 2-9: Weekly implementation
make week1-foundation          # Project structure, Neo4j setup
make week2-ontology            # LUCADA ontology + SNOMED integration
make week3-database            # Neo4j tools with read/write separation
make week4-agents              # Agents 1-3 implementation
make week5-agents-continued    # Agents 4-6 implementation
make week6-workflow            # LangGraph workflow orchestration
make week7-api                 # FastAPI backend + endpoints
make week8-frontend            # Next.js frontend application
make week9-testing             # Comprehensive testing suite
make week10-deployment         # Docker + Kubernetes deployment

# Final validation
make clinical-validation       # Test with 100 synthetic patients
make performance-test          # Load test with concurrent users
make security-audit           # Security vulnerability scan
make deploy-staging           # Deploy to staging environment
make deploy-production        # Production deployment (after approval)

# Development workflow
make run-backend              # Start FastAPI backend
make run-frontend             # Start Next.js frontend
make run-neo4j                # Start Neo4j database
make test-agents              # Run agent unit tests
make test-workflow            # Test complete workflow
8. SUPPORTING RESOURCES & REFERENCES
8.1 Primary References
LCA Paper: Sesen et al., "Lung Cancer Assistant: An Ontology-Driven, Online Decision Support Prototype" (University of Oxford)

NICE Guidelines: https://www.nice.org.uk/guidance/ng122 (Lung cancer: diagnosis and management)

SNOMED-CT Documentation: https://www.snomed.org/snomed-ct

OLS4 API: https://www.ebi.ac.uk/ols4/api (Ontology Lookup Service)

8.2 Technology Documentation
Neo4j Documentation: https://neo4j.com/docs/

FastAPI Documentation: https://fastapi.tiangolo.com/

Claude Code SDK: https://docs.anthropic.com/claude/docs

LangGraph Documentation: https://langchain-ai.github.io/langgraph/

Next.js Documentation: https://nextjs.org/docs

Docker Documentation: https://docs.docker.com/

Kubernetes Documentation: https://kubernetes.io/docs/

8.3 Data Sources (For Testing/Validation)
Synthetic Patient Generator: Provided in implementation (100 patients)

TCGA-LUAD/LUSC: https://portal.gdc.cancer.gov/ (Real clinical data)

SEER Database: https://seer.cancer.gov/ (Population statistics)

cBioPortal: https://www.cbioportal.org/ (Cancer genomics)

8.4 Clinical Validation Resources
LUCADA Data Dictionary: Reference for all data properties

TNM Staging Manual: AJCC 8th Edition for lung cancer

WHO Performance Status Scale: Standard 0-4 grading

NCCN Guidelines: Additional treatment algorithms

9. EXPECTED DELIVERABLES
9.1 Code Deliverables (Complete Repository)
text
lung-cancer-assistant/
├── README.md                          # Complete documentation
├── architecture-diagrams/            # System architecture (PDF/PNG)
├── backend/                          # Complete Python implementation
├── frontend/                         # Complete Next.js application
├── ontology/                         # OWL ontology files
├── docker-compose.yml               # Local development
├── kubernetes/                      # Production deployment manifests
├── scripts/                         # Utility scripts
├── tests/                           # Comprehensive test suite
├── docs/                            # Technical documentation
└── synthetic_patients.json          # 100 test patients with expected outcomes
9.2 Documentation Deliverables
System Architecture Document (PDF)

Agent communication patterns

Neo4j data model diagrams

API specifications (OpenAPI/Swagger)

Deployment architecture diagrams

Clinical Validation Report (PDF)

Test methodology

Results vs. NICE guidelines

Accuracy metrics (precision, recall, F1-score)

Clinician feedback summary

User Manuals (PDF + Online)

Clinician user guide (how to use the system)

System administrator guide (deployment, maintenance)

API reference (developer documentation)

Troubleshooting guide (common issues)

Operational Documentation

Deployment procedures (step-by-step)

Monitoring setup (Prometheus/Grafana dashboards)

Backup/restore procedures

Security policies and compliance checklist

9.3 Testing Deliverables
Test Suite Results (HTML/JSON)

Unit test coverage (>90%)

Integration test results

Performance benchmarks

Security audit results

Validation Dataset

100 synthetic patient cases with metadata

Expected classifications for each patient

Treatment recommendations with confidence scores

Explanation templates for MDT summaries

Clinical Test Cases (Edge Cases)

Rare histologies (Carcinosarcoma, Adenosquamous)

Complex comorbidities (COPD + Cardiovascular disease)

Guideline conflicts (multiple applicable rules)

Temporal reasoning (treatment changes over time)

10. IMPLEMENTATION TIMELINE & MILESTONES
Week 1-2: Foundation Complete
Project structure created

Development environment setup

Neo4j database initialized with schema

LUCADA ontology implemented (Figure 1)

SNOMED-CT integration working via OLS4

Week 3-4: Database & Agents
Neo4jTools with strict read/write separation

Data models matching ontology

IngestionAgent + SemanticMappingAgent complete

ClassificationAgent with NICE rules

Week 5-6: Workflow & Reasoning
ConflictResolutionAgent + PersistenceAgent complete

ExplanationAgent with MDT summaries

LangGraph workflow implementation

End-to-end workflow testing

Week 7-8: API & Frontend
FastAPI backend with all endpoints

Authentication/authorization implemented

Next.js frontend application complete

Real-time WebSocket updates

Week 9: Testing & Validation
Unit test suite (>90% coverage)

Integration tests passing

Clinical validation with 100 patients

Performance testing complete

Week 10: Deployment & Handover
Docker containers for all services

Kubernetes deployment manifests

Monitoring stack configured

Production deployment

Documentation complete

Training materials delivered

11. RISK MITIGATION STRATEGIES
11.1 Technical Risks
Risk	Probability	Impact	Mitigation
Neo4j performance with 100K+ patients	Medium	High	Implement pagination, indexing, query optimization
OWL reasoner scalability issues	High	Medium	Use temporary patient classification pattern
LLM API latency/rate limiting	Medium	Medium	Implement caching, fallback to rule-based reasoning
Agent coordination failures	Low	High	Implement circuit breakers, retry logic, state persistence
11.2 Clinical Risks
Risk	Probability	Impact	Mitigation
Guideline misinterpretation	Low	Critical	Clinical validation by oncology team
Missing contraindications	Medium	High	Comprehensive comorbidity checking
Outdated guideline rules	Medium	Medium	Modular rule system for easy updates
Over-reliance on AI recommendations	High	High	Clear "for informational purposes" disclaimers
11.3 Operational Risks
Risk	Probability	Impact	Mitigation
Data privacy/security breaches	Low	Critical	Encryption, access controls, audit logging
System downtime during MDT meetings	Medium	High	High availability deployment, failover
User adoption resistance	High	Medium	User-centered design, clinician training
Regulatory compliance issues	Medium	High	HIPAA/GDPR compliance built-in
12. SUCCESS METRICS & KPIs
12.1 Technical Performance Metrics
Workflow Execution Time: < 30 seconds (95th percentile)

System Uptime: 99.9% availability

API Response Time: < 2 seconds for all endpoints

Neo4j Query Performance: < 100ms for read operations

Error Rate: < 1% of all requests

12.2 Clinical Accuracy Metrics
Guideline Compliance: > 95% match with NICE recommendations

Classification Accuracy: > 90% for Patient Scenario matching

Explanation Quality: > 4/5 rating from clinician surveys

Treatment Recommendation Acceptance: > 80% alignment with MDT decisions

12.3 User Adoption Metrics
Active Users: > 100 clinicians in first 6 months

Patient Processing: > 1,000 patients processed monthly

User Satisfaction: > 4/5 on System Usability Scale (SUS)

Training Completion: > 90% of target users trained

FINAL IMPLEMENTATION COMMAND
bash
# BEGIN IMPLEMENTATION
./scripts/start-implementation.sh

# Expected completion: 10 weeks from project start
# Expected outcome: Production-ready Lung Cancer Assistant system with 
# 6 specialized agents, strict Neo4j patterns, and clinical-grade 
# decision support capabilities.

# SYSTEM STATUS CHECK (Run after implementation)
make check-system-health
make validate-clinical-accuracy
make generate-final-report
IMPLEMENTATION DEADLINE: 10 weeks from project start
EXPECTED OUTCOME: Production-ready Lung Cancer Assistant system deployed and validated, ready for clinical use in multidisciplinary team meetings.