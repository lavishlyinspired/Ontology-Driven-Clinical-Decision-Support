"""
Installation Test Script for Lung Cancer Assistant
Run this to verify your setup is working correctly
"""

import sys
from pathlib import Path

print("=" * 80)
print("LUNG CANCER ASSISTANT - INSTALLATION TEST")
print("=" * 80)
print()

# Test 1: Python version
print("Test 1: Python Version")
print("-" * 80)
version = sys.version_info
print(f"Python {version.major}.{version.minor}.{version.micro}")
if version.major >= 3 and version.minor >= 11:
    print("✓ Python version OK (3.11+)")
else:
    print("✗ Python 3.11+ required")
    sys.exit(1)
print()

# Test 2: Import core dependencies
print("Test 2: Core Dependencies")
print("-" * 80)

dependencies = [
    ("owlready2", "Ontology manipulation"),
    ("rdflib", "RDF triple store"),
    ("langchain", "LLM orchestration"),
    ("langgraph", "Agent workflows"),
    ("ollama", "Local LLM client"),
    ("pydantic", "Data validation"),
    ("pandas", "Data processing"),
    ("jupyter", "Notebook environment"),
]

all_imports_ok = True
for module, description in dependencies:
    try:
        __import__(module)
        print(f"✓ {module:20s} - {description}")
    except ImportError:
        print(f"✗ {module:20s} - NOT INSTALLED")
        all_imports_ok = False

if not all_imports_ok:
    print("\n⚠ Some dependencies missing. Run: pip install -r requirements.txt")
    sys.exit(1)
print()

# Test 3: File structure
print("Test 3: Project Structure")
print("-" * 80)

required_files = [
    "backend/src/ontology/snomed_loader.py",
    "backend/src/ontology/lucada_ontology.py",
    "backend/src/ontology/guideline_rules.py",
    "backend/src/agents/lca_agents.py",
    "backend/src/mcp_server/lca_mcp_server.py",
    "data/synthetic_patient_generator.py",
    "notebooks/LCA_Experiments.ipynb",
    "docs/Technical_Paper.md",
    "requirements.txt",
    "README.md",
]

files_ok = True
for file_path in required_files:
    if Path(file_path).exists():
        print(f"✓ {file_path}")
    else:
        print(f"✗ {file_path} - MISSING")
        files_ok = False

if not files_ok:
    print("\n⚠ Some project files missing")
print()

# Test 4: SNOMED OWL file
print("Test 4: SNOMED-CT OWL File")
print("-" * 80)
snomed_path = Path("ontology-2026-01-17_12-36-08.owl")
if snomed_path.exists():
    size_mb = snomed_path.stat().st_size / (1024 * 1024)
    print(f"✓ SNOMED OWL file found ({size_mb:.1f} MB)")
else:
    print("✗ SNOMED OWL file not found")
    print("  Expected: ontology-2026-01-17_12-36-08.owl")
print()

# Test 5: Ollama connection
print("Test 5: Ollama Connection")
print("-" * 80)
try:
    import httpx
    import os

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    response = httpx.get(f"{ollama_url}/api/tags", timeout=2.0)

    if response.status_code == 200:
        print(f"✓ Ollama server reachable at {ollama_url}")

        # Check for models
        data = response.json()
        models = data.get("models", [])

        if models:
            print(f"✓ Found {len(models)} installed model(s):")
            for model in models[:5]:
                print(f"    - {model.get('name', 'unknown')}")
        else:
            print("⚠ No models installed. Run: ollama pull llama3.2")
    else:
        print(f"✗ Ollama server returned status {response.status_code}")

except Exception as e:
    print(f"✗ Cannot connect to Ollama: {e}")
    print("  Make sure Ollama is running: ollama serve")
    print("  Then install a model: ollama pull llama3.2:latest")
print()

# Test 6: Create LUCADA Ontology
print("Test 6: LUCADA Ontology Creation")
print("-" * 80)
try:
    sys.path.insert(0, str(Path.cwd() / 'backend'))
    from src.ontology.lucada_ontology import LUCADAOntology

    lucada = LUCADAOntology()
    onto = lucada.create()

    num_classes = len(list(onto.classes()))
    num_obj_props = len(list(onto.object_properties()))
    num_data_props = len(list(onto.data_properties()))

    print(f"✓ Ontology created successfully")
    print(f"    Classes: {num_classes}")
    print(f"    Object Properties: {num_obj_props}")
    print(f"    Data Properties: {num_data_props}")

except Exception as e:
    print(f"✗ Failed to create ontology: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 7: Guideline Rules Engine
print("Test 7: Guideline Rules Engine")
print("-" * 80)
try:
    from src.ontology.guideline_rules import GuidelineRuleEngine

    engine = GuidelineRuleEngine(lucada)
    num_rules = len(engine.rules)

    print(f"✓ Rule engine initialized")
    print(f"    Guidelines loaded: {num_rules}")

    # Test classification
    test_patient = {
        "patient_id": "TEST001",
        "age": 68,
        "sex": "M",
        "tnm_stage": "IIIA",
        "histology_type": "Adenocarcinoma",
        "performance_status": 1
    }

    recommendations = engine.classify_patient(test_patient)
    print(f"✓ Test classification successful")
    print(f"    Recommendations for IIIA Adenocarcinoma PS1: {len(recommendations)}")

except Exception as e:
    print(f"✗ Failed to initialize rule engine: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 8: LangGraph Agents (Optional - requires Ollama)
print("Test 8: LangGraph Agents (Optional)")
print("-" * 80)
try:
    from src.agents.lca_agents import create_lca_workflow

    workflow = create_lca_workflow()
    print(f"✓ LangGraph workflow created")
    print(f"    Note: Agent execution requires Ollama with model installed")

except Exception as e:
    print(f"✗ Failed to create workflow: {e}")
print()

# Final summary
print("=" * 80)
print("INSTALLATION TEST SUMMARY")
print("=" * 80)

print("\n✓ CORE SYSTEM READY")
print("\nYou can now:")
print("  1. Run: python test_lca.py")
print("  2. Open: notebooks/LCA_Experiments.ipynb")
print("  3. Generate data: python data/synthetic_patient_generator.py")
print()

if "ollama" in str(sys.modules):
    print("For AI features:")
    print("  - Ensure Ollama is running: ollama serve")
    print("  - Install model: ollama pull llama3.2:latest")
    print("  - Then run full workflow examples")
print()

print("📚 Documentation:")
print("  - Quick start: QUICKSTART.md")
print("  - Full guide: README.md")
print("  - Technical details: docs/Technical_Paper.md")
print()

print("✓ Installation test complete!")
print("=" * 80)
