# Lung Cancer Assistant (LCA) - Modern Implementation

**Ontology-Driven Clinical Decision Support System with Agentic AI**

> Based on the seminal work by Sesen et al., University of Oxford
>
> Modernized with LangChain, LangGraph, Ollama, and SNOMED-CT integration

---

## 🎯 Overview

The Lung Cancer Assistant (LCA) is a hybrid clinical decision support system that combines:

- **OWL 2 Ontological Reasoning** for formal guideline compliance
- **SNOMED-CT Integration** for standardized medical terminology
- **LangGraph Agentic Workflows** for explainable AI decision support
- **Ollama Local LLMs** for privacy-preserving natural language generation
- **MCP Server** for seamless EHR integration

**Key Features:**
- ✅ NICE Lung Cancer Guidelines (CG121) formalized as OWL expressions
- ✅ Automatic patient classification and treatment recommendations
- ✅ Pro/con argumentation for MDT meetings
- ✅ Natural language MDT summaries
- ✅ 100% local inference - no external APIs required
- ✅ Complete SNOMED-CT OWL ontology support

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed
- 16GB+ RAM (for full SNOMED loading)
- Git

### 5-Minute Setup

```bash
# 1. Navigate to project
cd h:/akash/git/CoherencePLM/Version22

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama
ollama serve

# 4. Pull LLM model
ollama pull llama3.2:latest

# 5. Configure environment
cp .env.example .env

# 6. Run example
python -c "
from backend.src.ontology.lucada_ontology import LUCADAOntology
from backend.src.ontology.guideline_rules import GuidelineRuleEngine

# Create ontology
lucada = LUCADAOntology()
onto = lucada.create()

# Load guidelines
engine = GuidelineRuleEngine(lucada)

# Test patient
patient = {
    'patient_id': 'TEST001',
    'age': 68,
    'sex': 'M',
    'tnm_stage': 'IIIA',
    'histology_type': 'Adenocarcinoma',
    'performance_status': 1
}

# Get recommendations
recs = engine.classify_patient(patient)
print(f'Found {len(recs)} recommendations:')
for r in recs:
    print(f\"  - {r['recommended_treatment']} ({r['evidence_level']})\")
"
```

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────┐
│              Lung Cancer Assistant                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐    ┌──────────────────────────┐  │
│  │ SNOMED-CT    │───▶│  LUCADA Ontology         │  │
│  │ (350K+ terms)│    │  (OWL 2)                 │  │
│  └──────────────┘    └──────────────┬───────────┘  │
│                                     │               │
│                      ┌──────────────▼───────────┐  │
│                      │  Guideline Rule Engine   │  │
│                      │  (NICE CG121)            │  │
│                      └──────────────┬───────────┘  │
│                                     │               │
│        ┌────────────────────────────▼──────────┐   │
│        │     LangGraph Agent Workflow          │   │
│        │  ┌──────────────────────────────────┐ │   │
│        │  │ 1. Classification Agent          │ │   │
│        │  │ 2. Recommendation Agent          │ │   │
│        │  │ 3. Argumentation Agent           │ │   │
│        │  │ 4. Explanation Agent             │ │   │
│        │  └──────────────────────────────────┘ │   │
│        └────────────────┬──────────────────────┘   │
│                         │                           │
│               ┌─────────▼─────────┐                 │
│               │  Ollama LLM       │                 │
│               │  (llama3.2, etc.) │                 │
│               └─────────┬─────────┘                 │
│                         │                           │
│         ┌───────────────▼───────────────┐           │
│         │      MCP Server               │           │
│         │  (6 clinical tools)           │           │
│         └───────────────┬───────────────┘           │
│                         │                           │
│                         ▼                           │
│              ┌──────────────────┐                   │
│              │  EHR Integration │                   │
│              └──────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input:** Patient clinical data (demographics, TNM stage, histology, PS)
2. **Ontology:** Create OWL individual in LUCADA ontology
3. **Classification:** Match against guideline rule criteria
4. **AI Workflow:** LangGraph agents generate recommendations + arguments
5. **Output:** MDT-ready clinical summary with evidence citations

---

## ✨ Features

### 1. Ontology-Based Reasoning

- **SNOMED-CT Integration:** Complete OWL file with 350K+ medical concepts
- **LUCADA Ontology:** 60+ classes, 35 object properties, 60 data properties
- **Patient Modeling:** Follows paper's pattern (Figure 2)
- **Subsumption Reasoning:** Automatic classification via OWL semantics

### 2. Clinical Guideline Formalization

| Guideline | Treatment | Stage | Evidence | Implementation |
|-----------|-----------|-------|----------|----------------|
| R1 | Chemotherapy | III-IV NSCLC | Grade A | ✅ |
| R2 | Surgery | I-II NSCLC | Grade A | ✅ |
| R3 | Radiotherapy | I-IIIA NSCLC | Grade B | ✅ |
| R4 | Palliative Care | IIIB-IV, poor PS | Grade C | ✅ |
| R5 | Chemotherapy | SCLC | Grade A | ✅ |
| R6 | Chemoradiotherapy | IIIA-IIIB NSCLC | Grade A | ✅ |
| R7 | Immunotherapy | IIIB-IV NSCLC | Grade A | ✅ |

### 3. Agentic AI Workflow

**Powered by LangGraph + Ollama:**

- **Classification Agent:** Identifies applicable guidelines
- **Recommendation Agent:** Ranks treatment options by evidence
- **Argumentation Agent:** Generates pro/con clinical arguments
- **Explanation Agent:** Synthesizes MDT summary

**Benefits:**
- 100% local inference (privacy-preserving)
- Natural language explanations
- Transparent reasoning chains
- Configurable LLM models

### 4. MCP Server Integration

**6 Clinical Tools:**

```python
1. create_patient       # Add patient to ontology
2. classify_patient     # Get applicable guidelines
3. generate_recommendations  # Full AI workflow
4. list_guidelines      # View all rules
5. query_ontology       # Search concepts
6. get_ontology_stats   # System info
```

**Use Cases:**
- EHR integration via MCP protocol
- Third-party clinical applications
- Research data analysis
- Educational platforms

### 5. Synthetic Patient Generation

- Realistic demographic distributions
- Correlated clinical features (stage → PS → FEV1)
- Batch processing capabilities
- JSON export for testing

---

## 📦 Installation

### Step 1: Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [https://ollama.ai](https://ollama.ai)

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `owlready2==0.46` - OWL 2 ontology manipulation
- `langchain==0.3.12` - LLM orchestration
- `langgraph==0.2.57` - Agent workflows
- `ollama==0.4.5` - Local LLM client
- `mcp==1.3.2` - Model Context Protocol

### Step 3: Pull LLM Models

```bash
# Recommended: Llama 3.2 (balanced performance)
ollama pull llama3.2:latest

# Alternatives:
ollama pull mistral:latest    # Faster
ollama pull mixtral:latest    # More capable
ollama pull phi3:latest       # Lightweight
```

### Step 4: Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```ini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest
SNOMED_OWL_PATH=ontology-2026-01-17_12-36-08.owl
```

---

## 🎮 Usage

### Example 1: Basic Patient Classification

```python
from backend.src.ontology.lucada_ontology import LUCADAOntology
from backend.src.ontology.guideline_rules import GuidelineRuleEngine

# Initialize system
lucada = LUCADAOntology()
onto = lucada.create()
engine = GuidelineRuleEngine(lucada)

# Define patient
patient = {
    "patient_id": "P001",
    "name": "John Doe",
    "age": 68,
    "sex": "M",
    "tnm_stage": "IIIA",
    "histology_type": "Adenocarcinoma",
    "performance_status": 1,
    "fev1_percent": 65.0
}

# Get recommendations
recommendations = engine.classify_patient(patient)

# Display results
for rec in recommendations:
    print(f"{rec['recommended_treatment']}")
    print(f"  Evidence: {rec['evidence_level']}")
    print(f"  Priority: {rec['priority']}")
    print(f"  Survival: {rec['survival_benefit']}")
    print()
```

**Output:**
```
Chemoradiotherapy
  Evidence: Grade A
  Priority: 85
  Survival: 15-20% 5-year survival

Radiotherapy
  Evidence: Grade B
  Priority: 70
  Survival: 40-50% 5-year survival for Stage I with SABR
```

### Example 2: Full AI Workflow with Ollama

```python
from backend.src.agents.lca_agents import create_lca_workflow
import os

# Configure Ollama
os.environ["OLLAMA_MODEL"] = "llama3.2:latest"

# Create workflow
workflow = create_lca_workflow()

# Run workflow
initial_state = {
    "patient_id": patient["patient_id"],
    "patient_data": patient,
    "applicable_rules": recommendations,
    "treatment_recommendations": recommendations,
    "arguments": [],
    "explanation": "",
    "messages": []
}

final_state = workflow.invoke(initial_state)

# Get MDT summary
print(final_state["explanation"])
```

**Output:**
```
PATIENT SUMMARY
68-year-old male with Stage IIIA adenocarcinoma, WHO PS 1.

TREATMENT RECOMMENDATIONS
1. Concurrent Chemoradiotherapy (Grade A)
   - Standard of care for locally advanced NSCLC
   - 15-20% 5-year survival
   - Monitor renal function for cisplatin

2. Radical Radiotherapy (Grade B)
   - Alternative if chemo-unsuitable
   - 10-15% 5-year survival

KEY ARGUMENTS
Supporting Chemoradiotherapy:
- Grade A evidence (Furuse, Curran RCTs)
- Good PS indicates tolerability
- Curative intent possible

DISCUSSION POINTS
- Confirm staging with PET-CT
- Assess renal function for cisplatin
- Discuss intensive treatment with patient
```

### Example 3: Jupyter Notebook Experiments

```bash
cd notebooks
jupyter notebook LCA_Experiments.ipynb
```

**Included Experiments:**
1. SNOMED-CT ontology loading
2. LUCADA ontology creation
3. Guideline rule matching
4. Patient classification
5. AI agent workflow
6. Synthetic patient generation
7. Batch processing analysis

### Example 4: Run MCP Server

```bash
python backend/src/mcp_server/lca_mcp_server.py
```

**Test with MCP client:**
```python
import mcp

client = mcp.Client("http://localhost:3000")

# Classify patient
result = client.call_tool("classify_patient", {
    "patient_data": patient
})

print(result)
```

---

## 📁 Project Structure

```
Version22/
├── backend/
│   └── src/
│       ├── ontology/
│       │   ├── __init__.py
│       │   ├── snomed_loader.py           # SNOMED-CT OWL loader
│       │   ├── lucada_ontology.py         # LUCADA ontology creation
│       │   └── guideline_rules.py         # NICE guidelines engine
│       ├── agents/
│       │   ├── __init__.py
│       │   └── lca_agents.py              # LangGraph workflow
│       └── mcp_server/
│           └── lca_mcp_server.py          # MCP integration
├── data/
│   ├── synthetic_patient_generator.py     # Test data generator
│   └── synthetic_patients.json            # Generated cohort
├── notebooks/
│   └── LCA_Experiments.ipynb              # Complete experiments
├── docs/
│   └── Technical_Paper.md                 # Full technical paper
├── files/
│   ├── LCA_Complete_Implementation_Plan.md
│   └── Patient_Records_SNOMED_Guide.md
├── ontology-2026-01-17_12-36-08.owl       # Complete SNOMED OWL
├── requirements.txt                        # Python dependencies
├── .env.example                            # Configuration template
└── README.md                               # This file
```

---

## 📚 Documentation

### Core Documents

1. **[Technical Paper](docs/Technical_Paper.md)** - Complete system description with diagrams
2. **[Implementation Plan](files/LCA_Complete_Implementation_Plan.md)** - Original specification from papers
3. **[SNOMED Guide](files/Patient_Records_SNOMED_Guide.md)** - Patient data sources and SNOMED integration

### API Reference

#### LUCADA Ontology

```python
lucada = LUCADAOntology(ontology_iri="http://www.ox.ac.uk/lucada")
onto = lucada.create()  # Create ontology
lucada.save("lucada.owl")  # Save to file
lucada.load("lucada.owl")  # Load existing
```

#### Guideline Rule Engine

```python
engine = GuidelineRuleEngine(lucada)
rules = engine.get_all_rules()  # List all guidelines
rule = engine.get_rule_by_id("R1")  # Get specific rule
recs = engine.classify_patient(patient_data)  # Classify
```

#### LangGraph Agents

```python
workflow = create_lca_workflow()
final_state = workflow.invoke(initial_state)
mdt_summary = final_state["explanation"]
```

---

## 💡 Examples

### Jupyter Notebook

See `notebooks/LCA_Experiments.ipynb` for:
- Complete workflow demonstrations
- Synthetic patient generation
- Batch processing
- Visualization of results

### Python Scripts

```bash
# Generate 100 synthetic patients
cd data
python synthetic_patient_generator.py

# Test ontology creation
python backend/src/ontology/lucada_ontology.py

# Test guideline engine
python backend/src/ontology/guideline_rules.py

# Test AI agents (requires Ollama)
python backend/src/agents/lca_agents.py
```

---

## 🤝 Contributing

We welcome contributions! Areas for development:

### High Priority
- [ ] Expand to complete NICE CG121 guidelines
- [ ] Add BTS, ASCO, NCCN guideline sets
- [ ] Integrate genomic biomarkers (EGFR, ALK, PD-L1)
- [ ] Clinical validation with real patient data

### Medium Priority
- [ ] Web-based UI (FastAPI + React)
- [ ] Neo4j graph database integration
- [ ] Outcome tracking and learning
- [ ] Multi-modal data (imaging, pathology)

### Research
- [ ] Fine-tune LLM on clinical trial literature
- [ ] Federated learning across hospitals
- [ ] Survival prediction models
- [ ] Patient-facing decision support

---

## 📄 License

This project is based on research by Sesen et al., University of Oxford.

**Citation:**
```bibtex
@article{sesen2013lca,
  title={Lung Cancer Assistant: An ontology-driven, online decision support prototype},
  author={Sesen, M Berkan and Banares-Alcantara, Rene and Fox, John and Kadir, Timor and Brady, J Michael},
  journal={University of Oxford},
  year={2013}
}
```

---

## 📞 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email:** your-email@example.com

---

## 🙏 Acknowledgments

- **Original LCA Team:** M. Berkan Sesen, Rene Banares-Alcantara, John Fox, Timor Kadir, J. Michael Brady (University of Oxford)
- **SNOMED International:** For SNOMED-CT terminology
- **NICE:** For clinical guideline development
- **LangChain/LangGraph:** For agentic AI framework
- **Ollama:** For local LLM infrastructure

---

**Built with ❤️ for improving lung cancer patient outcomes**
