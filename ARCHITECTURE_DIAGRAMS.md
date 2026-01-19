# LCA Enhanced System Architecture

## System Overview

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Next.js UI]
        Chat[Chat Interface]
        Upload[File Upload]
        Dashboard[Dashboard]
    end
    
    subgraph "API Layer"
        API[FastAPI Server]
        ChatRouter[Chat Router]
        EnhancedRouter[Enhanced Router]
        SSE[SSE Streaming]
    end
    
    subgraph "Service Layer"
        Conv[Conversation Service]
        LCA[LCA Service]
        File[File Processor]
        LLM[LLM Extractor]
        Trans[Transparency Service]
        Export[Export Service]
        Cache[Cache Service]
    end
    
    subgraph "Agent Layer"
        Orch[Dynamic Orchestrator]
        Integrated[Integrated Workflow]
        NSCLC[NSCLC Agent]
        SCLC[SCLC Agent]
        Bio[Biomarker Agent]
        Class[Classification Agent]
    end
    
    subgraph "Data Layer"
        Neo4j[(Neo4j Graph)]
        Vector[(Vector Store)]
        Ontology[(OWL Ontology)]
        Ollama[Ollama LLM]
    end
    
    UI --> API
    Chat --> ChatRouter
    Upload --> EnhancedRouter
    Dashboard --> API
    
    ChatRouter --> Conv
    EnhancedRouter --> File
    EnhancedRouter --> LLM
    EnhancedRouter --> Trans
    EnhancedRouter --> Export
    
    Conv --> LCA
    Conv --> Cache
    File --> Cache
    LLM --> Ollama
    
    LCA --> Orch
    Orch --> Integrated
    Integrated --> NSCLC
    Integrated --> SCLC
    Integrated --> Bio
    Integrated --> Class
    
    NSCLC --> Neo4j
    SCLC --> Neo4j
    Bio --> Vector
    Class --> Ontology
    
    Trans -.->|Track| Orch
    Cache -.->|Cache| Neo4j
    Cache -.->|Cache| Vector
    Cache -.->|Cache| Ontology
    
    style UI fill:#e0e7ff
    style Chat fill:#dbeafe
    style Upload fill:#dbeafe
    style Conv fill:#fef3c7
    style LLM fill:#fef3c7
    style Trans fill:#dcfce7
    style Cache fill:#fce7f3
```

## Data Flow

### 1. Chat Conversation Flow
```mermaid
sequenceDiagram
    participant User
    participant Chat
    participant Conv
    participant LLM
    participant LCA
    participant Cache
    participant Agents
    
    User->>Chat: Send message
    Chat->>Conv: chat_stream()
    
    alt Use LLM
        Conv->>LLM: extract_from_conversation()
        LLM->>Ollama: Query
        Ollama-->>LLM: Extracted data
        LLM-->>Conv: Patient data
    else Use Regex
        Conv->>Conv: regex_extract()
    end
    
    Conv->>Cache: Check cache
    alt Cache hit
        Cache-->>Conv: Cached result
    else Cache miss
        Conv->>LCA: analyze_patient()
        LCA->>Agents: Execute workflow
        Agents-->>LCA: Results
        LCA-->>Conv: Analysis
        Conv->>Cache: Store result
    end
    
    Conv-->>Chat: SSE stream
    Chat-->>User: Real-time updates
```

### 2. File Upload Flow
```mermaid
sequenceDiagram
    participant User
    participant Upload
    participant File
    participant LLM
    participant LCA
    
    User->>Upload: Upload PDF
    Upload->>File: process_file()
    
    File->>File: Extract text
    File->>File: Regex extraction
    
    alt Use LLM enhancement
        File->>LLM: extract_from_document()
        LLM->>Ollama: Query
        Ollama-->>LLM: Structured data
        LLM->>LLM: Merge with regex
    end
    
    File->>File: Validate extraction
    File->>File: Format for LCA
    
    File-->>Upload: Extracted data
    
    alt Auto-analyze
        Upload->>LCA: analyze_patient()
        LCA-->>Upload: Results
    end
    
    Upload-->>User: Response
```

### 3. Agent Execution with Transparency
```mermaid
sequenceDiagram
    participant LCA
    participant Trans
    participant Orch
    participant Agents
    participant UI
    
    LCA->>Trans: create_workflow_graph()
    Trans->>Trans: Initialize graph
    
    LCA->>Orch: orchestrate_adaptive_workflow()
    
    loop For each agent
        Orch->>Trans: start_agent()
        Trans->>UI: SSE update (agent starting)
        
        Orch->>Agents: execute_agent()
        Agents->>Agents: Process
        Agents-->>Orch: Results + confidence
        
        Orch->>Trans: complete_agent(confidence)
        Trans->>UI: SSE update (agent complete)
    end
    
    Orch->>Trans: complete_workflow()
    Trans-->>UI: SSE final state
```

### 4. Export Flow
```mermaid
sequenceDiagram
    participant User
    participant API
    participant Export
    participant Session
    
    User->>API: Request export
    API->>Session: Get session data
    Session-->>API: Patient + analysis
    
    alt PDF Export
        API->>Export: generate_clinical_report()
        Export->>Export: Build PDF
        Export-->>API: PDF buffer
        API-->>User: Download PDF
    else FHIR Export
        API->>Export: export_fhir_bundle()
        Export->>Export: Build FHIR resources
        Export-->>API: FHIR JSON
        API-->>User: Download JSON
    end
```

## Component Architecture

### Service Layer Components

```mermaid
graph LR
    subgraph "Conversation Service"
        CS[ConversationService]
        Regex[Regex Extraction]
        Session[Session Manager]
    end
    
    subgraph "File Processor"
        FP[FileProcessor]
        PDF[PDF Parser]
        DOCX[DOCX Parser]
        Pattern[Pattern Matching]
    end
    
    subgraph "LLM Extractor"
        LE[LLMExtractor]
        Hybrid[HybridExtractor]
        Valid[Validator]
    end
    
    subgraph "Transparency"
        TS[TransparencyService]
        Graph[WorkflowGraph]
        Conf[ConfidenceCalc]
    end
    
    subgraph "Cache"
        RC[ResultCache]
        Patient[Patient Cache]
        Onto[Ontology Cache]
        Analysis[Analysis Cache]
    end
    
    CS --> Regex
    CS --> Session
    FP --> PDF
    FP --> DOCX
    FP --> Pattern
    LE --> Hybrid
    LE --> Valid
    TS --> Graph
    TS --> Conf
    RC --> Patient
    RC --> Onto
    RC --> Analysis
```

## Technology Stack

### Frontend
- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **TailwindCSS** - Styling
- **React Markdown** - Markdown rendering
- **EventSource** - SSE streaming

### Backend
- **FastAPI** - REST API + SSE
- **Python 3.10+** - Core language
- **LangChain** - Agent orchestration
- **Ollama** - LLM inference

### Services
- **PyPDF2** - PDF parsing
- **python-docx** - DOCX parsing
- **ReportLab** - PDF generation
- **FHIR Resources** - FHIR R4 models

### Data
- **Neo4j** - Graph database
- **FAISS** - Vector store
- **OWLReady2** - Ontology reasoning

## Deployment Architecture

```mermaid
graph TB
    subgraph "User Layer"
        Browser[Web Browser]
    end
    
    subgraph "Application Layer"
        LB[Load Balancer]
        Next1[Next.js Instance 1]
        Next2[Next.js Instance 2]
        API1[FastAPI Instance 1]
        API2[FastAPI Instance 2]
    end
    
    subgraph "Service Layer"
        Ollama1[Ollama Server]
        Cache1[Redis Cache]
    end
    
    subgraph "Data Layer"
        Neo4j1[(Neo4j Primary)]
        Neo4j2[(Neo4j Replica)]
        S3[(S3 Storage)]
    end
    
    Browser --> LB
    LB --> Next1
    LB --> Next2
    Next1 --> API1
    Next1 --> API2
    Next2 --> API1
    Next2 --> API2
    
    API1 --> Ollama1
    API2 --> Ollama1
    API1 --> Cache1
    API2 --> Cache1
    API1 --> Neo4j1
    API2 --> Neo4j1
    Neo4j1 --> Neo4j2
    
    API1 -.->|Store files| S3
    API2 -.->|Store files| S3
    
    style Browser fill:#e0e7ff
    style LB fill:#dbeafe
    style Cache1 fill:#fce7f3
    style Ollama1 fill:#fef3c7
```

## Security Architecture

```mermaid
graph TB
    subgraph "Authentication"
        Login[Login]
        JWT[JWT Token]
        OAuth[OAuth2]
    end
    
    subgraph "Authorization"
        RBAC[RBAC Engine]
        Roles[Roles: Clinician/Admin/Viewer]
        Permissions[Permissions]
    end
    
    subgraph "Audit"
        Logger[Audit Logger]
        Events[Event Stream]
        Storage[(Audit DB)]
    end
    
    subgraph "Encryption"
        TLS[TLS 1.3]
        AtRest[Encryption at Rest]
        InTransit[Encryption in Transit]
    end
    
    Login --> JWT
    Login --> OAuth
    JWT --> RBAC
    RBAC --> Roles
    Roles --> Permissions
    
    Permissions --> Logger
    Logger --> Events
    Events --> Storage
    
    TLS --> InTransit
    AtRest --> Storage
    
    style Login fill:#fef3c7
    style JWT fill:#dcfce7
    style RBAC fill:#e0e7ff
    style Logger fill:#fce7f3
```

## Caching Strategy

```mermaid
graph LR
    subgraph "L1: Memory Cache"
        LRU[LRU Cache]
        TTL1[TTL: Variable]
    end
    
    subgraph "L2: Redis Cache"
        Redis[(Redis)]
        TTL2[TTL: Hours-Days]
    end
    
    subgraph "L3: Database"
        DB[(Neo4j/Vector)]
        TTL3[TTL: Persistent]
    end
    
    Request[Request] --> LRU
    LRU -->|Miss| Redis
    Redis -->|Miss| DB
    DB -->|Store| Redis
    Redis -->|Store| LRU
    
    LRU --> TTL1
    Redis --> TTL2
    DB --> TTL3
    
    style Request fill:#e0e7ff
    style LRU fill:#fef3c7
    style Redis fill:#fce7f3
    style DB fill:#dbeafe
```

## Confidence Scoring System

```mermaid
graph TB
    subgraph "Extraction Confidence"
        EC[Extraction Confidence]
        Comp[Completeness 40%]
        Demo[Demographics 20%]
        Diag[Diagnosis 20%]
        Bio[Biomarkers 20%]
    end
    
    subgraph "Classification Confidence"
        CC[Classification Confidence]
        Base[Base 50%]
        Hist[Histology +30%]
        BioEv[Biomarker +20%]
    end
    
    subgraph "Overall Confidence"
        OC[Overall Confidence]
        Avg[Agent Average]
        Conflicts[Conflict Penalty -10%]
        Low[Low Agent Penalty]
    end
    
    EC --> Comp
    EC --> Demo
    EC --> Diag
    EC --> Bio
    
    CC --> Base
    CC --> Hist
    CC --> BioEv
    
    OC --> Avg
    Avg --> Conflicts
    Conflicts --> Low
    
    EC --> OC
    CC --> OC
    
    style EC fill:#dcfce7
    style CC fill:#fef3c7
    style OC fill:#e0e7ff
```

---

## Key Metrics

### Performance Targets
- **API Response**: < 200ms (cached)
- **First Analysis**: < 30s (uncached)
- **Cached Analysis**: < 100ms
- **File Upload**: < 5s (10MB PDF)
- **PDF Export**: < 2s
- **SSE Latency**: < 500ms

### Availability Targets
- **Uptime**: 99.9% (3 nines)
- **Error Rate**: < 0.1%
- **Cache Hit Rate**: > 85%

### Scalability Targets
- **Concurrent Users**: 100+
- **Requests/sec**: 1000+
- **Database Connections**: 100 pool

---

## Monitoring & Observability

### Metrics to Track
1. **Cache Performance**
   - Hit rate per cache type
   - Average latency
   - Memory usage

2. **Agent Performance**
   - Execution time per agent
   - Confidence scores
   - Failure rate

3. **API Performance**
   - Request latency (p50, p95, p99)
   - Error rate
   - Throughput

4. **LLM Performance**
   - Extraction accuracy
   - Response time
   - Ollama uptime

---

## References
- [ENHANCED_FEATURES_GUIDE.md](ENHANCED_FEATURES_GUIDE.md) - Implementation details
- [REMAINING_GAPS_ROADMAP.md](REMAINING_GAPS_ROADMAP.md) - Future work
- [PROJECT_GAPS_ANALYSIS.md](PROJECT_GAPS_ANALYSIS.md) - Gap analysis
