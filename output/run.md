.venv) PS H:\akash\git\CoherencePLM\Version22> python run_lca.py

================================================================================
LUNG CANCER ASSISTANT - COMPREHENSIVE CLINICAL DECISION SUPPORT
================================================================================

Configuration:
  Neo4j: Enabled (URI: bolt://localhost:7687)
  Vector Store: Enabled (Model: all-MiniLM-L6-v2)
  AI Workflow: Enabled (14 agents)
  Digital Twin: Available
  MCP Server: Available on port 3000

Initializing service...
INFO:src.services.lca_service:Initializing Lung Cancer Assistant Service...
INFO:src.services.lca_service:Creating LUCADA ontology...
INFO:src.ontology.snomed_loader:SNOMED OWL file found at H:\akash\git\CoherencePLM\Version22\data\lca_ontologies\snomed_ct\build_snonmed_owl\snomed_ct_optimized.owl
INFO:src.ontology.lucada_ontology:Creating LUCADA ontology...
INFO:src.ontology.lucada_ontology:✓ LUCADA ontology created successfully
INFO:src.ontology.lucada_ontology:  Classes: 68
INFO:src.ontology.lucada_ontology:  Object Properties: 15
INFO:src.ontology.lucada_ontology:  Data Properties: 36
INFO:src.services.lca_service:✓ Ontology created with 68 classes
INFO:src.services.lca_service:Loading clinical guidelines...
INFO:src.ontology.guideline_rules:Loading clinical guideline rules...
INFO:src.ontology.guideline_rules:✓ Loaded 10 guideline rules
INFO:src.services.lca_service:✓ Loaded 10 guideline rules
INFO:src.services.lca_service:Creating AI agent workflow...
INFO:src.agents.lca_agents:Creating LCA workflow...
INFO:src.agents.lca_agents:✓ LCA workflow created
INFO:src.services.lca_service:✓ LangGraph workflow ready
INFO:src.services.lca_service:Initializing advanced workflow components...
INFO:src.db.neo4j_tools:✓ Neo4j READ connection established: bolt://localhost:7687
INFO:src.db.neo4j_tools:✓ Neo4j WRITE connection established: bolt://localhost:7687
INFO:src.services.lca_service:✓ Neo4j tools initialized for integrated workflow
INFO:sentence_transformers.SentenceTransformer:Use pytorch device_name: cpu
INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: all-MiniLM-L6-v2
INFO:src.agents.persistence_agent:[PersistenceAgent] ✓ Embedding model loaded (all-MiniLM-L6-v2)
INFO:src.agents.integrated_workflow:[Init] enable_analytics=True, neo4j_tools=provided
INFO:src.agents.integrated_workflow:[Init] UncertaintyQuantifier initialized: True
INFO:src.agents.integrated_workflow:✓ Analytics suite loaded (UncertaintyQuantifier active)
INFO:src.db.graph_algorithms:✓ Neo4j GDS available: 2.22.0
INFO:src.db.graph_algorithms:✓ Neo4j Graph Algorithms initialized (GDS: True)
INFO:src.db.neosemantics_tools:✓ Neosemantics (n10s) plugin available
INFO:src.db.neosemantics_tools:✓ Neosemantics tools initialized (n10s: True)
INFO:src.db.temporal_analyzer:✓ Temporal Analyzer initialized
INFO:src.agents.integrated_workflow:✓ DB Tools loaded (Graph Algorithms, Neosemantics, Temporal Analyzer)
INFO:src.ontology.loinc_integrator:✓ LOINC Integrator initialized (local mappings: 17)
INFO:src.ontology.rxnorm_mapper:✓ RxNorm Mapper initialized (22 drugs)
INFO:src.agents.integrated_workflow:✓ Ontology Integrators loaded (LOINC, RxNorm)
INFO:src.services.lca_service:✓ Advanced workflow ready
INFO:src.services.lca_service:Initializing provenance tracking...
INFO:src.services.lca_service:✓ Provenance tracker ready
INFO:src.services.lca_service:Connecting to Neo4j...
INFO:src.db.neo4j_schema:✓ Connected to Neo4j at bolt://localhost:7687
INFO:src.db.neo4j_schema:✓ Neo4j schema initialized
INFO:src.services.lca_service:✓ Neo4j graph database connected
INFO:src.services.lca_service:Initializing vector store...
INFO:src.db.vector_store:Loading embedding model...
INFO:sentence_transformers.SentenceTransformer:Use pytorch device_name: cpu
INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: all-MiniLM-L6-v2
INFO:src.db.vector_store:✓ Embedding model loaded
INFO:src.db.vector_store:✓ Connected to Neo4j at bolt://localhost:7687
INFO:src.db.vector_store:✓ Vector index exists: clinical_guidelines_vector
INFO:src.db.vector_store:✓ Vector store initialized: 10 documents
INFO:src.services.lca_service:✓ Vector store initialized
INFO:src.services.lca_service:================================================================================  
INFO:src.services.lca_service:✓ LCA Service Ready
INFO:src.services.lca_service:================================================================================  

================================================================================
SELECT PATIENT TO PROCESS
================================================================================

1. Early Stage (IA Adenocarcinoma, PS 0)
2. Advanced Stage (IV Squamous, PS 1)
3. Locally Advanced (IIIA Adenocarcinoma, PS 1, EGFR+)

Options:
  o = Show Ontology Integration
  m = Show MCP Server Info
  q = Quit

Enter choice (1-3, o, m, q): 2

================================================================================
PROCESSING PATIENT: Advanced Stage Patient
================================================================================

Patient Details:
  ID: ADVANCED_001
  Age: 71, Sex: M
  Stage: IV
  Histology: SquamousCellCarcinoma
  Performance Status: WHO 1
  FEV1: 55.0%
  Comorbidities: COPD, Type 2 Diabetes
  Biomarkers:
    - egfr_mutation: Negative
    - pdl1_tps: 80
    - kras_mutation: G12C

Select analysis type:
  1 = Quick (rule-based only)
  2 = Full AI Workflow (14 agents)
  3 = Full + Digital Twin
  4 = Full + Analytics Demo
  5 = Comprehensive (All features)

Choice (1-5): 2
INFO:src.services.lca_service:
================================================================================
INFO:src.services.lca_service:Processing Patient: ADVANCED_001
INFO:src.services.lca_service:================================================================================  
INFO:src.db.provenance_tracker:[Provenance] Started session session_7ef5d7697164 for patient ADVANCED_001       
INFO:src.services.lca_service:================================================================================  
INFO:src.services.lca_service:ADVANCED INTEGRATED WORKFLOW (Orchestrated)
INFO:src.services.lca_service:================================================================================  
INFO:src.services.lca_service:
INFO:src.services.lca_service:📊 SYSTEM COMPONENTS ACTIVE:
INFO:src.services.lca_service:   🔬 Ontology: LUCADA v1.0.0 + SNOMED-CT 2025-01-17
INFO:src.services.lca_service:   📝 Provenance Tracker: ✓ Active
INFO:src.services.lca_service:   🗄️  Neo4j Graph DB: ✓ Connected
INFO:src.services.lca_service:   🔍 Vector Store: ✓ Active
INFO:src.services.lca_service:   🧬 Biomarker Analysis: ✓ Active
INFO:src.services.lca_service:   📊 Analytics Suite: ✓ Active
INFO:src.services.lca_service:
INFO:src.services.lca_service:🛠️  TOOLS & ALGORITHMS:
INFO:src.services.lca_service:   • Guideline Matching: Semantic similarity (all-MiniLM-L6-v2)
INFO:src.services.lca_service:   • Patient Similarity: Graph-based (TNM + PS + Histology)
INFO:src.services.lca_service:   • Conflict Resolution: Evidence hierarchy (Grade A > B > C)
INFO:src.services.lca_service:   • Uncertainty Quantification: Bayesian + Historical outcomes
INFO:src.services.lca_service:
INFO:src.services.lca_service:📋 PROVENANCE TRACKING:
INFO:src.services.lca_service:   Session ID: session_7ef5d7697164
INFO:src.services.lca_service:   ✓ Data ingestion tracked
INFO:src.services.lca_service:
INFO:src.agents.integrated_workflow:================================================================================
INFO:src.agents.integrated_workflow:🔬 INTEGRATED WORKFLOW EXECUTION
INFO:src.agents.integrated_workflow:================================================================================
INFO:src.agents.integrated_workflow:📋 Patient ID: ADVANCED_001
INFO:src.agents.integrated_workflow:
INFO:src.agents.integrated_workflow:🎯 AGENTS (11 total):
INFO:src.agents.integrated_workflow:   Core: IngestionAgent, SemanticMappingAgent, ClassificationAgent, ExplanationAgent
INFO:src.agents.integrated_workflow:   Specialized: BiomarkerAgent, ComorbidityAgent, NSCLCAgent, SCLCAgent     
INFO:src.agents.integrated_workflow:   Advanced: ConflictResolutionAgent, UncertaintyQuantifier, PersistenceAgent
INFO:src.agents.integrated_workflow:
INFO:src.agents.integrated_workflow:📊 ANALYTICS SUITE:
INFO:src.agents.integrated_workflow:   UncertaintyQuantifier: ✓ Active
INFO:src.agents.integrated_workflow:   SurvivalAnalyzer: ✓ Active
INFO:src.agents.integrated_workflow:   CounterfactualEngine: ✓ Active
INFO:src.agents.integrated_workflow:   ClinicalTrialMatcher: ✓ Active
INFO:src.agents.integrated_workflow:
INFO:src.agents.integrated_workflow:🗄️ DB TOOLS:
INFO:src.agents.integrated_workflow:   GraphAlgorithms (Neo4j GDS): ✓ Active
INFO:src.agents.integrated_workflow:   NeosemanticsTools (n10s): ✓ Active
INFO:src.agents.integrated_workflow:   TemporalAnalyzer: ✓ Active
INFO:src.agents.integrated_workflow:
INFO:src.agents.integrated_workflow:🔬 ONTOLOGIES:
INFO:src.agents.integrated_workflow:   SNOMED-CT: ✓ Active (via SemanticMappingAgent)
INFO:src.agents.integrated_workflow:   LOINC Integrator: ✓ Active
INFO:src.agents.integrated_workflow:   RxNorm Mapper: ✓ Active
INFO:src.agents.integrated_workflow:
INFO:src.agents.integrated_workflow:[Registry] Final registry has 14 agents: ['BiomarkerAgent', 'ClassificationAgent', 'ClinicalTrialMatcher', 'ComorbidityAgent', 'ConflictResolutionAgent', 'CounterfactualEngine', 'ExplanationAgent', 'IngestionAgent', 'NSCLCAgent', 'PersistenceAgent', 'SCLCAgent', 'SemanticMappingAgent', 'SurvivalAnalyzer', 'UncertaintyQuantifier']
INFO:src.agents.integrated_workflow:✓ Registered 14 agents for orchestration
INFO:src.agents.integrated_workflow:   Active agents: BiomarkerAgent, ClassificationAgent, ClinicalTrialMatcher, ComorbidityAgent, ConflictResolutionAgent, CounterfactualEngine, ExplanationAgent, IngestionAgent, NSCLCAgent, 
PersistenceAgent, SCLCAgent, SemanticMappingAgent, SurvivalAnalyzer, UncertaintyQuantifier
INFO:src.agents.integrated_workflow:
INFO:src.agents.dynamic_orchestrator:🚀 Starting adaptive workflow 99e1e2c4-82c2-47b8-a5ef-dbb7e78833ab
INFO:src.agents.dynamic_orchestrator:📊 Case complexity: complex
INFO:src.agents.dynamic_orchestrator:🛤️  Selected path: IngestionAgent → SemanticMappingAgent → Cla ssificationA
gent → BiomarkerAgent → ComorbidityAgent → NSCLCAgent → SCLCAgent → SurvivalAnalyzer → ConflictResolutionAgent → UncertaintyQuantifier → ClinicalTrialMatcher → ExplanationAgent
INFO:src.agents.ingestion_agent:[IngestionAgent] Processing patient data...
INFO:src.agents.ingestion_agent:[IngestionAgent] ✓ Patient ADVANCED_001 ingested successfully
INFO:src.agents.dynamic_orchestrator:✓ IngestionAgent completed in 157ms (confidence: 1.00)
INFO:src.agents.semantic_mapping_agent:[SemanticMappingAgent] Mapping patient ADVANCED_001 to SNOMED-CT...      
INFO:src.agents.semantic_mapping_agent:[SemanticMappingAgent] ✓ Mapped ADVANCED_001 with confidence 0.96
INFO:src.agents.dynamic_orchestrator:✓ SemanticMappingAgent completed in 79ms (confidence: 1.00)
INFO:src.agents.classification_agent:[ClassificationAgent] Classifying patient ADVANCED_001...
INFO:src.agents.classification_agent:[ClassificationAgent] ✓ Classified ADVANCED_001 as metastatic_good_ps      
INFO:src.agents.dynamic_orchestrator:✓ ClassificationAgent completed in 0ms (confidence: 1.00)
INFO:src.agents.biomarker_agent:Biomarker Agent analyzing patient ADVANCED_001
INFO:src.agents.biomarker_agent:Biomarker Agent recommends: Platinum-based chemotherapy doublet
INFO:src.agents.dynamic_orchestrator:✓ BiomarkerAgent completed in 1ms (confidence: 0.70)
INFO:src.agents.comorbidity_agent:Comorbidity Agent assessing safety for Pembrolizumab monotherapy (PD-L1 ≥50%) 
INFO:src.agents.dynamic_orchestrator:✓ ComorbidityAgent completed in 0ms (confidence: 1.00)
INFO:src.agents.integrated_workflow:[NSCLCAgent] Executed for patient, treatment: Carboplatin + paclitaxel      
INFO:src.agents.dynamic_orchestrator:✓ NSCLCAgent completed in 3ms (confidence: 0.80)
INFO:src.agents.dynamic_orchestrator:✓ SCLCAgent completed in 0ms (confidence: 1.00)
INFO:src.agents.integrated_workflow:[SurvivalAnalyzer] Analyzed survival for: Unknown
INFO:src.agents.dynamic_orchestrator:✓ SurvivalAnalyzer completed in 2792ms (confidence: 1.00)
INFO:src.agents.conflict_resolution_agent:[ConflictResolutionAgent] Resolving conflicts for patient ADVANCED_001...
INFO:src.agents.conflict_resolution_agent:[ConflictResolutionAgent] ✓ Resolved 0 conflicts for ADVANCED_001     
INFO:src.agents.integrated_workflow:[ConflictResolutionAgent] Resolved 0 conflicts
INFO:src.agents.dynamic_orchestrator:✓ ConflictResolutionAgent completed in 5ms (confidence: 1.00)
INFO:src.agents.integrated_workflow:[UncertaintyQuantifier] Pembrolizumab monotherapy (PD-L1 ≥50%): confidence=0.30
INFO:src.agents.integrated_workflow:[UncertaintyQuantifier] Pembrolizumab + chemotherapy: confidence=0.30
INFO:src.agents.integrated_workflow:[UncertaintyQuantifier] Targeted therapy if driver mutation present: confidence=0.30
INFO:src.agents.dynamic_orchestrator:✓ UncertaintyQuantifier completed in 1594ms (confidence: 1.00)
INFO:src.analytics.clinical_trial_matcher:Searching for clinical trials for patient ADVANCED_001
ERROR:src.analytics.clinical_trial_matcher:Failed to parse trial data: 'str' object has no attribute 'get'
ERROR:src.analytics.clinical_trial_matcher:Failed to parse trial data: 'str' object has no attribute 'get'
ERROR:src.analytics.clinical_trial_matcher:Failed to parse trial data: 'str' object has no attribute 'get'      
ERROR:src.analytics.clinical_trial_matcher:Failed to parse trial data: 'str' object has no attribute 'get'      
ERROR:src.analytics.clinical_trial_matcher:Failed to parse trial data: 'str' object has no attribute 'get'      
ERROR:src.analytics.clinical_trial_matcher:Failed to parse trial data: 'str' object has no attribute 'get'      
ERROR:src.analytics.clinical_trial_matcher:Failed to parse trial data: 'str' object has no attribute 'get'      
ERROR:src.analytics.clinical_trial_matcher:Failed to parse trial data: 'str' object has no attribute 'get'      
ERROR:src.analytics.clinical_trial_matcher:Failed to parse trial data: 'str' object has no attribute 'get'      
ERROR:src.analytics.clinical_trial_matcher:Failed to parse trial data: 'str' object has no attribute 'get'      
WARNING:src.analytics.clinical_trial_matcher:No trials found matching criteria
INFO:src.agents.integrated_workflow:[ClinicalTrialMatcher] Found 0 eligible trials
INFO:src.agents.dynamic_orchestrator:✓ ClinicalTrialMatcher completed in 1240ms (confidence: 1.00)
INFO:src.agents.explanation_agent:[ExplanationAgent] Generating MDT summary for patient ADVANCED_001...
INFO:src.agents.explanation_agent:[ExplanationAgent] ✓ Generated MDT summary for ADVANCED_001
INFO:src.agents.dynamic_orchestrator:✓ ExplanationAgent completed in 0ms (confidence: 1.00)
INFO:src.agents.integrated_workflow:Orchestrator executed 12 agents
INFO:src.agents.integrated_workflow:Orchestrator executed 12 agents
INFO:src.agents.biomarker_agent:Biomarker Agent analyzing patient ADVANCED_001
ERROR:src.agents.integrated_workflow:Workflow error: 'dict' object has no attribute 'egfr_mutation'
Traceback (most recent call last):
  File "H:\akash\git\CoherencePLM\Version22\backend\src\agents\integrated_workflow.py", line 312, in analyze_patient_comprehensive
    proposals = await self._execute_specialized_agents(patient_data, patient_with_codes)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\akash\git\CoherencePLM\Version22\backend\src\agents\integrated_workflow.py", line 577, in _execute_specialized_agents
    biomarker_proposal = self.biomarker.execute(patient_with_codes, biomarker_profile)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\akash\git\CoherencePLM\Version22\backend\src\agents\biomarker_agent.py", line 79, in execute
    if biomarker_profile.egfr_mutation == "Positive":
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'dict' object has no attribute 'egfr_mutation'
INFO:src.services.lca_service:✓ Advanced workflow completed: error (complexity: unknown)
INFO:src.services.lca_service:📊 GRAPH ALGORITHM: Finding similar patients...
INFO:src.services.lca_service:   Algorithm: Neo4j pattern matching (TNM stage + Performance status ±1 + Histology match)
INFO:src.services.lca_service:   ✓ Found 0 similar patients
INFO:src.db.provenance_tracker:[Provenance] Ended session session_7ef5d7697164
INFO:src.services.lca_service:================================================================================  
INFO:src.services.lca_service:✓ Patient ADVANCED_001 advanced workflow complete (6036ms)
INFO:src.services.lca_service:================================================================================  


================================================================================
DECISION SUPPORT RESULTS
================================================================================

Applicable Guidelines: 0

================================================================================
MDT SUMMARY (AI-Generated)
================================================================================
Advanced workflow completed successfully

================================================================================
WORKFLOW METADATA
================================================================================
  Workflow Type: integrated
  Complexity Level: unknown
  Execution Time: 6036ms
  Provenance Record: prov_ADVANCED_001_20260126_152550

 Results saved to: output\ADVANCED_001_results.json
INFO:src.db.neo4j_schema:✓ Neo4j connection closed
INFO:src.services.lca_service:✓ LCA Service shut down

================================================================================
PROCESSING COMPLETE
================================================================================

Additional Commands:
  - Run MCP Server: python start_mcp_server.py
  - Digital Twin Demo: python demo_digital_twin.py
  - API Server: uvicorn backend.src.api.main:app --reload
(.venv) PS H:\akash\git\CoherencePLM\Version22> 