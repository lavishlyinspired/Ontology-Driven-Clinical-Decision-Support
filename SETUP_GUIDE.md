# Lung Cancer Assistant - Setup Guide

Complete setup instructions for the LCA system with ontology integration.

---

## 📋 Prerequisites

### Required Software

- **Python 3.10+** (tested with 3.10, 3.11)
- **Neo4j 5.x** with plugins:
  - Graph Data Science (GDS) library
  - Neosemantics (n10s) plugin
- **Git** (for cloning repository)

### Required Ontology Files

You need the following ontology files in your `data/lca_ontologies/` directory:

1. **SNOMED-CT OWL File**
   - Location: `data/lca_ontologies/snomed_ct/build_snonmed_owl/snomed_ct_optimized.owl`
   - Source: Built from SNOMED-CT International Edition RF2

2. **LOINC Directory**
   - Location: `data/lca_ontologies/loinc/Loinc_2.81/`
   - Source: LOINC Ontology 2.81 release

3. **RxNorm Directory**
   - Location: `data/lca_ontologies/rxnorm/RxNorm_full_01052026/`
   - Source: RxNorm full release

---

## 🚀 Quick Start

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd Version22
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
# Copy the example environment file
copy .env.example .env

# Edit .env with your settings (see below)
```

### Step 5: Verify Setup

```bash
# Run the environment setup script
python setup_environment.py
```

This will:
- ✅ Validate all ontology file paths
- ✅ Check Python dependencies
- ✅ Print configuration summary
- ✅ Provide setup instructions for any missing components

---

## ⚙️ Environment Configuration

### `.env` File Structure

The `.env` file should contain the following variables:

```bash
# ============================================================================
# ONTOLOGY PATHS
# ============================================================================

# SNOMED-CT OWL File (optimized version)
SNOMED_CT_PATH=H:\akash\git\CoherencePLM\Version22\data\lca_ontologies\snomed_ct\build_snonmed_owl\snomed_ct_optimized.owl

# LOINC Directory (contains LOINC 2.81 files)
LOINC_PATH=H:\akash\git\CoherencePLM\Version22\data\lca_ontologies\loinc\Loinc_2.81

# RxNorm Directory (contains RxNorm full release)
RXNORM_PATH=H:\akash\git\CoherencePLM\Version22\data\lca_ontologies\rxnorm\RxNorm_full_01052026

# LUCADA Ontology Output Directory (where generated ontology will be saved)
LUCADA_ONTOLOGY_OUTPUT=H:\akash\git\CoherencePLM\Version22\data\lca_ontologies\lucada

# LUCADA OWL File Name
LUCADA_OWL_FILE=lucada_ontology.owl

# ============================================================================
# NEO4J CONFIGURATION
# ============================================================================

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=lucada
NEO4J_VECTOR_INDEX=clinical_guidelines_vector

# ============================================================================
# OLLAMA CONFIGURATION (for local LLM)
# ============================================================================

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest

# Alternative models:
# OLLAMA_MODEL=mistral:latest
# OLLAMA_MODEL=mixtral:latest
# OLLAMA_MODEL=phi3:latest

# ============================================================================
# MCP SERVER CONFIGURATION
# ============================================================================

MCP_SERVER_PORT=3000
MCP_SERVER_HOST=localhost

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL=INFO
```

### Path Configuration Notes

**Important**: Update the paths in your `.env` file to match your actual file locations:

- Use **absolute paths** for reliability
- On Windows, use forward slashes `/` or escaped backslashes `\\`
- Or use the raw string prefix in your .env: `H:\path\to\file`

---

## 📁 Directory Structure

After setup, your directory structure should look like:

```
Version22/
├── .env                          # Your environment configuration
├── .env.example                  # Example configuration
├── setup_environment.py          # Environment validation script
├── run_tests.py                  # Test runner
├── requirements.txt              # Python dependencies
│
├── backend/
│   └── src/
│       ├── config.py            # Configuration loader
│       ├── agents/              # 13 agents
│       ├── analytics/           # 4 analytics modules
│       ├── db/                  # Neo4j tools
│       ├── ontology/            # Ontology loaders
│       │   ├── lucada_ontology.py
│       │   ├── snomed_loader.py
│       │   ├── loinc_integrator.py
│       │   └── rxnorm_mapper.py
│       ├── api/                 # FastAPI server
│       └── mcp_server/          # MCP tools
│
├── data/
│   └── lca_ontologies/
│       ├── snomed_ct/
│       │   └── build_snonmed_owl/
│       │       └── snomed_ct_optimized.owl  ✅ Required
│       ├── loinc/
│       │   └── Loinc_2.81/                  ✅ Required
│       ├── rxnorm/
│       │   └── RxNorm_full_01052026/        ✅ Required
│       └── lucada/                          📝 Generated here
│           └── lucada_ontology.owl
│
├── tests/
│   ├── test_components.py
│   └── test_integrated_workflow.py
│
└── notebooks/
    └── *.ipynb
```

---

## 🧪 Validate Installation

### 1. Check Configuration

```bash
python setup_environment.py
```

Expected output:
```
================================================================================
LCA SYSTEM CONFIGURATION
================================================================================

Project Root: H:\akash\git\CoherencePLM\Version22
Data Directory: H:\akash\git\CoherencePLM\Version22\data

Ontology Paths:
  SNOMED-CT: H:\...\snomed_ct_optimized.owl
  LOINC:     H:\...\Loinc_2.81
  RxNorm:    H:\...\RxNorm_full_01052026
  LUCADA Output: H:\...\lucada\lucada_ontology.owl

...

Path Validation:
  SNOMED_CT            ✅ EXISTS     H:\...\snomed_ct_optimized.owl
  LOINC                ✅ EXISTS     H:\...\Loinc_2.81
  RXNORM               ✅ EXISTS     H:\...\RxNorm_full_01052026
  LUCADA_OUTPUT        ✅ EXISTS     H:\...\lucada

🎉 SETUP COMPLETE - All checks passed!
```

### 2. Generate LUCADA Ontology

```bash
python -m backend.src.ontology.lucada_ontology
```

Expected output:
```
Creating LUCADA ontology...
✓ Created 158 classes
✓ Created 89 properties
✓ LUCADA ontology created and saved to: H:\...\lucada\lucada_ontology.owl
```

### 3. Run Tests

```bash
python run_tests.py
```

Expected output:
```
================================================================================
LUNG CANCER ASSISTANT - TEST SUITE
================================================================================

Running tests from: H:\...\tests

📦 Component Tests
--------------------------------------------------------------------------------
test_components.py::TestDynamicOrchestrator::test_simple_complexity_assessment PASSED
test_components.py::TestDynamicOrchestrator::test_moderate_complexity_assessment PASSED
...

✅ ALL TESTS PASSED!
```

---

## 🏃 Running the System

### Option 1: Interactive CLI

```bash
python cli.py
```

### Option 2: REST API Server

```bash
python -m backend.src.api.main
```

Then access:
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Option 3: MCP Server (for AI Assistants)

```bash
python -m backend.src.mcp_server.lca_mcp_server
```

### Option 4: Jupyter Notebooks

```bash
jupyter notebook
# Open notebooks/LCA_Complete_Implementation_Demo.ipynb
```

---

## 🔧 Troubleshooting

### Problem: "SNOMED OWL file not found"

**Solution**:
1. Verify the file exists at the specified path
2. Check the path in your `.env` file
3. Ensure you're using absolute paths
4. Re-run `python setup_environment.py`

### Problem: "Module 'owlready2' not found"

**Solution**:
```bash
pip install owlready2
```

### Problem: "Neo4j connection failed"

**Solution**:
1. Ensure Neo4j is running: `neo4j status`
2. Check credentials in `.env` file
3. Test connection:
   ```bash
   python -c "from neo4j import GraphDatabase; print('OK')"
   ```

### Problem: "ImportError: cannot import name 'LCAConfig'"

**Solution**:
Ensure you're running from the project root and the backend/src directory is in your Python path:
```bash
# Add to your Python path
set PYTHONPATH=%PYTHONPATH%;H:\akash\git\CoherencePLM\Version22\backend\src
```

---

## 📊 Performance Tips

### 1. SNOMED-CT Loading

The full SNOMED-CT ontology is large (~111 seconds to load). For development:
- Use minimal mode (default): `snomed_loader.load(load_full=False)`
- Or use the mapping dictionaries only (no OWL loading)

### 2. Neo4j Performance

- Install Graph Data Science plugin for 50x faster queries
- Use indexes on frequently queried properties
- Enable vector index for semantic search

### 3. Caching

The system caches ontology loads. First run is slower, subsequent runs are fast.

---

## 📚 Next Steps

1. ✅ **Verify setup**: Run `python setup_environment.py`
2. ✅ **Generate ontology**: Run LUCADA ontology generator
3. ✅ **Run tests**: Ensure all components work
4. 📖 **Read documentation**: See `LCA_2026_Complete_Guide.md`
5. 🧪 **Try examples**: Open Jupyter notebooks
6. 🚀 **Deploy**: Follow deployment guide for production

---

## 🆘 Getting Help

- **Documentation**: See `LCA_Architecture.md` and `LCA_2026_Complete_Guide.md`
- **Implementation Audit**: See `IMPLEMENTATION_AUDIT_2026.md`
- **Issues**: Check GitHub issues
- **Environment validation**: Run `python setup_environment.py`

---

**Version**: 3.0.0 Final
**Last Updated**: January 2026
**Status**: Production Ready
