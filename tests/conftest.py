"""
Pytest configuration and shared fixtures for LCA test suite
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any

# Add backend to path
backend_path = Path(__file__).parent / "backend" / "src"
sys.path.insert(0, str(backend_path))


# ============================================================================
# Shared Fixtures - Sample Patient Data
# ============================================================================

@pytest.fixture
def simple_patient() -> Dict[str, Any]:
    """Simple early-stage patient with good prognosis"""
    return {
        "patient_id": "TEST-SIMPLE-001",
        "name": "Simple Test Patient",
        "age": 55,
        "sex": "M",
        "tnm_stage": "IA",
        "histology_type": "Adenocarcinoma",
        "performance_status": 0,
        "laterality": "Right",
        "comorbidities": [],
        "biomarker_profile": {}
    }


@pytest.fixture
def moderate_patient() -> Dict[str, Any]:
    """Moderate complexity with EGFR mutation"""
    return {
        "patient_id": "TEST-MOD-002",
        "name": "Moderate Test Patient",
        "age": 62,
        "sex": "F",
        "tnm_stage": "IIIA",
        "histology_type": "Adenocarcinoma",
        "performance_status": 1,
        "laterality": "Left",
        "comorbidities": ["hypertension"],
        "biomarker_profile": {
            "EGFR": "Positive",
            "egfr_mutation_type": "Ex19del",
            "PDL1": "45%"
        },
        "egfr_mutation": "Positive",
        "egfr_mutation_type": "Ex19del",
        "pdl1_tps": 45
    }


@pytest.fixture
def complex_patient() -> Dict[str, Any]:
    """Complex case with multiple comorbidities"""
    return {
        "patient_id": "TEST-COMPLEX-003",
        "name": "Complex Test Patient",
        "age": 72,
        "sex": "M",
        "tnm_stage": "IV",
        "histology_type": "Adenocarcinoma",
        "performance_status": 2,
        "laterality": "Right",
        "comorbidities": ["COPD", "diabetes", "chronic_kidney_disease"],
        "biomarker_profile": {
            "EGFR": "Negative",
            "ALK": "Negative",
            "ROS1": "Negative",
            "PDL1": "75%"
        },
        "egfr_mutation": "Negative",
        "alk_rearrangement": "Negative",
        "pdl1_tps": 75
    }


@pytest.fixture
def critical_patient() -> Dict[str, Any]:
    """Critical SCLC patient with poor prognosis"""
    return {
        "patient_id": "TEST-CRITICAL-004",
        "name": "Critical Test Patient",
        "age": 68,
        "sex": "F",
        "tnm_stage": "IV",
        "histology_type": "Small Cell Carcinoma",
        "performance_status": 3,
        "laterality": "Bilateral",
        "comorbidities": ["COPD", "heart_failure", "diabetes", "chronic_kidney_disease"],
        "biomarker_profile": {}
    }


@pytest.fixture
def sclc_patient() -> Dict[str, Any]:
    """SCLC specific patient"""
    return {
        "patient_id": "TEST-SCLC-005",
        "name": "SCLC Test Patient",
        "age": 65,
        "sex": "M",
        "tnm_stage": "IV",  # SCLC often presents at advanced stage
        "histology_type": "Small Cell Carcinoma",
        "performance_status": 1,
        "laterality": "Left",
        "comorbidities": ["COPD"],
        "biomarker_profile": {}
    }


# ============================================================================
# Biomarker Fixtures
# ============================================================================

@pytest.fixture
def egfr_ex19del_profile():
    """EGFR Ex19del mutation profile"""
    from backend.src.agents.biomarker_agent import BiomarkerProfile
    return BiomarkerProfile(
        egfr_mutation="Positive",
        egfr_mutation_type="Ex19del"
    )


@pytest.fixture
def alk_positive_profile():
    """ALK rearrangement profile"""
    from backend.src.agents.biomarker_agent import BiomarkerProfile
    return BiomarkerProfile(
        alk_rearrangement="Positive"
    )


@pytest.fixture
def high_pdl1_profile():
    """High PD-L1 expression profile"""
    from backend.src.agents.biomarker_agent import BiomarkerProfile
    return BiomarkerProfile(
        pdl1_tps=75
    )


# ============================================================================
# Mock Objects
# ============================================================================

@pytest.fixture
def mock_neo4j_tools():
    """Mock Neo4j tools for testing without database"""
    class MockNeo4jTools:
        def find_similar_patients(self, patient_id, limit=10):
            return []
        
        def get_treatment_outcomes(self, treatment):
            return []
        
        def execute_query(self, query, params=None):
            return []
    
    return MockNeo4jTools()


# ============================================================================
# Environment Setup
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment variables"""
    import os
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "ERROR"
    yield
    # Cleanup after all tests
    del os.environ["TESTING"]


# ============================================================================
# Skip Markers
# ============================================================================

def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line(
        "markers", "requires_neo4j: mark test as requiring Neo4j database"
    )
    config.addinivalue_line(
        "markers", "requires_ollama: mark test as requiring Ollama LLM"
    )
    config.addinivalue_line(
        "markers", "requires_internet: mark test as requiring internet connection"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on availability"""
    import os
    
    # Skip Neo4j tests if not available
    if not os.getenv("NEO4J_URI"):
        skip_neo4j = pytest.mark.skip(reason="Neo4j not configured")
        for item in items:
            if "neo4j" in item.keywords or "requires_neo4j" in item.keywords:
                item.add_marker(skip_neo4j)
    
    # Skip Ollama tests if not available
    try:
        import ollama
        ollama.list()
    except:
        skip_ollama = pytest.mark.skip(reason="Ollama not available")
        for item in items:
            if "requires_ollama" in item.keywords:
                item.add_marker(skip_ollama)
