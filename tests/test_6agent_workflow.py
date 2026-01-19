"""
Test Script for 6-Agent Workflow Implementation
Tests the complete implementation per final.md specification.

Run with: python -m pytest test_6agent_workflow.py -v
Or directly: python test_6agent_workflow.py
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

from backend.src.agents.ingestion_agent import IngestionAgent
from backend.src.agents.semantic_mapping_agent import SemanticMappingAgent
from backend.src.agents.classification_agent import ClassificationAgent, PatientScenario
from backend.src.agents.conflict_resolution_agent import ConflictResolutionAgent
from backend.src.agents.persistence_agent import PersistenceAgent
from backend.src.agents.explanation_agent import ExplanationAgent
from backend.src.agents.lca_workflow import LCAWorkflow, analyze_patient

from backend.src.db.models import (
    PatientFact,
    PatientFactWithCodes,
    ClassificationResult,
    Sex,
    HistologyType,
    EvidenceLevel
)


class TestIngestionAgent(unittest.TestCase):
    """Tests for IngestionAgent (Agent 1)"""

    def setUp(self):
        self.agent = IngestionAgent()

    def test_valid_patient_data(self):
        """Test ingestion of valid patient data"""
        raw_data = {
            "patient_id": "LC-TEST-001",
            "sex": "Male",
            "age_at_diagnosis": 68,
            "tnm_stage": "IIIA",
            "histology_type": "Adenocarcinoma",
            "performance_status": 1,
            "laterality": "Right",
            "diagnosis": "Lung Cancer"
        }
        
        patient_fact, errors = self.agent.execute(raw_data)
        
        self.assertIsNotNone(patient_fact)
        self.assertEqual(len(errors), 0)
        self.assertEqual(patient_fact.patient_id, "LC-TEST-001")
        self.assertEqual(patient_fact.tnm_stage, "IIIA")

    def test_tnm_normalization(self):
        """Test TNM stage normalization"""
        raw_data = {
            "patient_id": "LC-TEST-002",
            "tnm_stage": "Stage IIA",  # Should normalize to "IIA"
            "histology_type": "Adenocarcinoma"
        }
        
        patient_fact, errors = self.agent.execute(raw_data)
        
        self.assertIsNotNone(patient_fact)
        self.assertEqual(patient_fact.tnm_stage, "IIA")

    def test_missing_required_field(self):
        """Test handling of missing required fields"""
        raw_data = {
            "patient_id": "LC-TEST-003"
            # Missing tnm_stage and histology_type
        }
        
        patient_fact, errors = self.agent.execute(raw_data)
        
        self.assertIsNone(patient_fact)
        self.assertTrue(len(errors) > 0)


class TestSemanticMappingAgent(unittest.TestCase):
    """Tests for SemanticMappingAgent (Agent 2)"""

    def setUp(self):
        self.agent = SemanticMappingAgent()

    def test_snomed_mapping(self):
        """Test SNOMED-CT code mapping"""
        patient_fact = PatientFact(
            patient_id="LC-TEST-001",
            tnm_stage="IIIA",
            histology_type=HistologyType.ADENOCARCINOMA,
            performance_status=1,
            laterality="Right",
            diagnosis="Lung Cancer"
        )
        
        patient_with_codes, confidence = self.agent.execute(patient_fact)
        
        self.assertIsNotNone(patient_with_codes)
        self.assertIsNotNone(patient_with_codes.snomed_histology_code)
        self.assertEqual(patient_with_codes.snomed_histology_code, "35917007")  # Adenocarcinoma
        self.assertTrue(confidence > 0.5)

    def test_is_nsclc_subtype(self):
        """Test NSCLC subtype detection"""
        # Adenocarcinoma SNOMED code
        self.assertTrue(self.agent.is_nsclc_subtype("35917007"))
        
        # SCLC SNOMED code
        self.assertFalse(self.agent.is_nsclc_subtype("254632001"))


class TestClassificationAgent(unittest.TestCase):
    """Tests for ClassificationAgent (Agent 3)"""

    def setUp(self):
        self.agent = ClassificationAgent()

    def test_early_stage_operable(self):
        """Test classification of early stage operable patient"""
        patient = PatientFactWithCodes(
            patient_id="LC-TEST-001",
            tnm_stage="IIA",
            histology_type=HistologyType.ADENOCARCINOMA,
            performance_status=0,
            laterality="Right",
            diagnosis="Lung Cancer",
            snomed_histology_code="35917007",
            mapping_confidence=0.95
        )
        
        result = self.agent.execute(patient)
        
        self.assertEqual(result.scenario, PatientScenario.EARLY_STAGE_OPERABLE.value)
        self.assertTrue(len(result.recommendations) > 0)
        
        # Should recommend surgery
        treatments = [r.treatment.lower() for r in result.recommendations]
        self.assertTrue(any("surgery" in t or "resection" in t for t in treatments))

    def test_metastatic_classification(self):
        """Test classification of metastatic patient"""
        patient = PatientFactWithCodes(
            patient_id="LC-TEST-002",
            tnm_stage="IV",
            histology_type=HistologyType.ADENOCARCINOMA,
            performance_status=1,
            laterality="Right",
            diagnosis="Lung Cancer",
            snomed_histology_code="35917007",
            mapping_confidence=0.95
        )
        
        result = self.agent.execute(patient)
        
        self.assertEqual(result.scenario, PatientScenario.METASTATIC_GOOD_PS.value)
        
        # Should recommend systemic therapy
        treatments = [r.treatment.lower() for r in result.recommendations]
        self.assertTrue(any("pembrolizumab" in t or "chemotherapy" in t for t in treatments))

    def test_sclc_classification(self):
        """Test classification of SCLC patient"""
        patient = PatientFactWithCodes(
            patient_id="LC-TEST-003",
            tnm_stage="IIIA",
            histology_type=HistologyType.SMALL_CELL_CARCINOMA,
            performance_status=1,
            laterality="Left",
            diagnosis="Small Cell Lung Cancer",
            snomed_histology_code="254632001",
            mapping_confidence=0.95
        )
        
        result = self.agent.execute(patient)
        
        self.assertEqual(result.scenario, PatientScenario.SCLC_LIMITED.value)


class TestConflictResolutionAgent(unittest.TestCase):
    """Tests for ConflictResolutionAgent (Agent 4)"""

    def setUp(self):
        self.agent = ConflictResolutionAgent()

    def test_evidence_ranking(self):
        """Test evidence level ranking"""
        self.assertGreater(
            self.agent.compare_evidence_levels(EvidenceLevel.GRADE_A, EvidenceLevel.GRADE_B),
            0
        )
        self.assertLess(
            self.agent.compare_evidence_levels(EvidenceLevel.GRADE_C, EvidenceLevel.GRADE_A),
            0
        )


class TestExplanationAgent(unittest.TestCase):
    """Tests for ExplanationAgent (Agent 6)"""

    def setUp(self):
        self.agent = ExplanationAgent()

    def test_mdt_summary_generation(self):
        """Test MDT summary generation"""
        patient = PatientFactWithCodes(
            patient_id="LC-TEST-001",
            sex=Sex.MALE,
            age_at_diagnosis=68,
            tnm_stage="IIIA",
            histology_type=HistologyType.ADENOCARCINOMA,
            performance_status=1,
            laterality="Right",
            diagnosis="Lung Cancer",
            snomed_histology_code="35917007",
            snomed_stage_code="422968005",
            snomed_ps_code="373804000",
            mapping_confidence=0.95
        )
        
        # Create a mock classification result
        from backend.src.db.models import TreatmentRecommendation, TreatmentIntent
        classification = ClassificationResult(
            patient_id="LC-TEST-001",
            scenario="locally_advanced_resectable",
            scenario_confidence=0.85,
            recommendations=[
                TreatmentRecommendation(
                    rank=1,
                    treatment="Concurrent chemoradiotherapy",
                    evidence_level=EvidenceLevel.GRADE_A,
                    intent=TreatmentIntent.CURATIVE,
                    guideline_reference="NICE NG122",
                    rationale="Standard for locally advanced NSCLC"
                )
            ],
            reasoning_chain=["Stage IIIA", "Good PS"],
            ontology_concepts_matched=["SCTID:35917007"],
            guideline_refs=["NICE NG122"]
        )
        
        summary = self.agent.execute(patient, classification)
        
        self.assertIsNotNone(summary.clinical_summary)
        self.assertIn("68", summary.clinical_summary)  # Age
        self.assertIn("IIIA", summary.clinical_summary)  # Stage
        self.assertTrue(len(summary.formatted_recommendations) > 0)
        self.assertTrue(len(summary.discussion_points) > 0)


class TestLCAWorkflow(unittest.TestCase):
    """Tests for complete LCA Workflow"""

    def test_full_workflow_without_neo4j(self):
        """Test complete workflow without Neo4j persistence"""
        workflow = LCAWorkflow(persist_results=False)
        
        patient_data = {
            "patient_id": "LC-WORKFLOW-001",
            "sex": "Male",
            "age_at_diagnosis": 65,
            "tnm_stage": "IIA",
            "histology_type": "Adenocarcinoma",
            "performance_status": 0,
            "laterality": "Right",
            "diagnosis": "Lung Cancer"
        }
        
        result = workflow.run(patient_data)
        
        self.assertTrue(result.success)
        self.assertEqual(result.workflow_status, "complete")
        self.assertIn("IngestionAgent", result.agent_chain)
        self.assertIn("SemanticMappingAgent", result.agent_chain)
        self.assertIn("ClassificationAgent", result.agent_chain)
        self.assertIn("ConflictResolutionAgent", result.agent_chain)
        self.assertIn("ExplanationAgent", result.agent_chain)
        
        self.assertIsNotNone(result.scenario)
        self.assertTrue(len(result.recommendations) > 0)
        self.assertIsNotNone(result.mdt_summary)
        
        workflow.close()

    def test_analyze_patient_convenience_function(self):
        """Test the analyze_patient convenience function"""
        patient_data = {
            "patient_id": "LC-QUICK-001",
            "tnm_stage": "IV",
            "histology_type": "Adenocarcinoma",
            "performance_status": 2
        }
        
        result = analyze_patient(patient_data, persist=False)
        
        self.assertTrue(result.success)
        self.assertIn("metastatic", result.scenario.lower())


def run_tests():
    """Run all tests"""
    print("=" * 80)
    print("Testing 6-Agent Workflow Implementation")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestIngestionAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticMappingAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestClassificationAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestConflictResolutionAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestExplanationAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestLCAWorkflow))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("✓ All tests passed!")
    else:
        print(f"✗ {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
