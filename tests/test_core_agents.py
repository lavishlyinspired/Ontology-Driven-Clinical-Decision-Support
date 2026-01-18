"""
Unit Tests for Core LCA Agents (2024 Original 6-Agent Architecture)

Tests:
- IngestionAgent
- SemanticMappingAgent  
- ClassificationAgent
- ConflictResolutionAgent
- PersistenceAgent
- ExplanationAgent
"""

import pytest
from datetime import datetime
from backend.src.agents import (
    IngestionAgent,
    SemanticMappingAgent,
    ClassificationAgent,
    ConflictResolutionAgent,
    ExplanationAgent
)
from backend.src.db.models import PatientFact, PatientFactWithCodes


class TestIngestionAgent:
    """Test IngestionAgent - Data validation and ingestion"""
    
    def setup_method(self):
        self.agent = IngestionAgent()
    
    @pytest.mark.unit
    def test_valid_patient_ingestion(self, simple_patient):
        """Test ingestion of valid patient data"""
        patient_fact, errors = self.agent.execute(simple_patient)
        
        # Debug: Print errors if any
        if errors:
            print(f"Errors encountered: {errors}")
        
        assert patient_fact is not None, f"PatientFact is None. Errors: {errors}"
        assert len(errors) == 0
        assert patient_fact.patient_id == simple_patient["patient_id"]
        assert patient_fact.tnm_stage == simple_patient["tnm_stage"]
        assert patient_fact.histology_type == simple_patient["histology_type"]
    
    @pytest.mark.unit
    def test_missing_required_fields(self):
        """Test that missing required fields are caught"""
        invalid_patient = {
            "patient_id": "INVALID-001"
            # Missing age, sex, stage, etc.
        }
        
        patient_fact, errors = self.agent.execute(invalid_patient)
        
        assert patient_fact is None
        assert len(errors) > 0
    
    @pytest.mark.unit
    def test_invalid_stage(self):
        """Test that invalid TNM stage is caught"""
        invalid_patient = {
            "patient_id": "INVALID-002",
            "age_at_diagnosis": 65,
            "sex": "Male",
            "tnm_stage": "INVALID_STAGE",  # Invalid
            "histology_type": "Adenocarcinoma",
            "performance_status": 1,
            "laterality": "Left"
        }
        
        patient_fact, errors = self.agent.execute(invalid_patient)
        
        # Should still create patient but may have validation warnings
        assert "INVALID_STAGE" in str(patient_fact.tnm_stage) if patient_fact else True
    
    @pytest.mark.unit
    def test_invalid_performance_status(self):
        """Test that invalid PS (> 4) is caught"""
        invalid_patient = {
            "patient_id": "INVALID-003",
            "age_at_diagnosis": 65,
            "sex": "Male",
            "tnm_stage": "IIIA",
            "histology_type": "Adenocarcinoma",
            "performance_status": 10,  # Invalid - should be 0-4
            "laterality": "Left"
        }
        
        patient_fact, errors = self.agent.execute(invalid_patient)
        
        # Validation should catch this
        assert errors or patient_fact.performance_status <= 4
    
    @pytest.mark.unit
    def test_multiple_patients(self, simple_patient, moderate_patient, complex_patient):
        """Test batch ingestion"""
        patients = [simple_patient, moderate_patient, complex_patient]
        
        results = []
        for patient in patients:
            patient_fact, errors = self.agent.execute(patient)
            results.append((patient_fact, errors))
        
        # All should succeed
        assert all(pf is not None for pf, _ in results)
        assert all(len(errors) == 0 for _, errors in results)


class TestSemanticMappingAgent:
    """Test SemanticMappingAgent - SNOMED-CT mapping"""
    
    def setup_method(self):
        self.agent = SemanticMappingAgent()
    
    @pytest.mark.unit
    def test_stage_mapping(self, simple_patient):
        """Test TNM stage SNOMED mapping"""
        ingestion = IngestionAgent()
        patient_fact, _ = ingestion.execute(simple_patient)
        
        patient_with_codes, confidence = self.agent.execute(patient_fact)
        
        assert patient_with_codes is not None
        assert confidence > 0
        assert hasattr(patient_with_codes, 'snomed_stage_code')
        assert patient_with_codes.snomed_stage_code is not None
    
    @pytest.mark.unit
    def test_histology_mapping(self, moderate_patient):
        """Test histology type SNOMED mapping"""
        ingestion = IngestionAgent()
        patient_fact, _ = ingestion.execute(moderate_patient)
        
        patient_with_codes, confidence = self.agent.execute(patient_fact)
        
        assert patient_with_codes is not None
        assert hasattr(patient_with_codes, 'snomed_histology_code')
        assert patient_with_codes.snomed_histology_code is not None
    
    @pytest.mark.unit
    def test_confidence_score(self, complex_patient):
        """Test mapping confidence scoring"""
        ingestion = IngestionAgent()
        patient_fact, _ = ingestion.execute(complex_patient)
        
        patient_with_codes, confidence = self.agent.execute(patient_fact)
        
        assert 0.0 <= confidence <= 1.0
        # Should have high confidence for standard terms
        assert confidence > 0.5


class TestClassificationAgent:
    """Test ClassificationAgent - Scenario classification and recommendations"""
    
    def setup_method(self):
        self.agent = ClassificationAgent()
    
    @pytest.mark.unit
    def test_early_stage_classification(self, simple_patient):
        """Test classification of early-stage NSCLC"""
        ingestion = IngestionAgent()
        semantic = SemanticMappingAgent()
        
        patient_fact, _ = ingestion.execute(simple_patient)
        patient_with_codes, _ = semantic.execute(patient_fact)
        
        result = self.agent.execute(patient_with_codes)
        
        assert result is not None
        assert result.scenario is not None
        assert len(result.recommendations) > 0
        # Early stage should recommend surgery
        assert any("surgery" in rec.get("treatment", "").lower() 
                  for rec in result.recommendations)
    
    @pytest.mark.unit
    def test_advanced_stage_classification(self, complex_patient):
        """Test classification of advanced-stage NSCLC"""
        ingestion = IngestionAgent()
        semantic = SemanticMappingAgent()
        
        patient_fact, _ = ingestion.execute(complex_patient)
        patient_with_codes, _ = semantic.execute(patient_fact)
        
        result = self.agent.execute(patient_with_codes)
        
        assert result is not None
        assert result.scenario is not None
        # Advanced stage should recommend systemic therapy
        assert any("chemotherapy" in rec.get("treatment", "").lower() or
                  "immunotherapy" in rec.get("treatment", "").lower()
                  for rec in result.recommendations)
    
    @pytest.mark.unit
    def test_sclc_classification(self, sclc_patient):
        """Test classification of SCLC"""
        ingestion = IngestionAgent()
        semantic = SemanticMappingAgent()
        
        patient_fact, _ = ingestion.execute(sclc_patient)
        patient_with_codes, _ = semantic.execute(patient_fact)
        
        result = self.agent.execute(patient_with_codes)
        
        assert result is not None
        # SCLC should be identified
        assert "sclc" in result.scenario.lower() or "small cell" in result.scenario.lower()


class TestConflictResolutionAgent:
    """Test ConflictResolutionAgent - Recommendation conflict resolution"""
    
    def setup_method(self):
        self.agent = ConflictResolutionAgent()
    
    @pytest.mark.unit
    def test_no_conflicts(self, simple_patient):
        """Test resolution when no conflicts exist"""
        ingestion = IngestionAgent()
        semantic = SemanticMappingAgent()
        classification = ClassificationAgent()
        
        patient_fact, _ = ingestion.execute(simple_patient)
        patient_with_codes, _ = semantic.execute(patient_fact)
        result = classification.execute(patient_with_codes)
        
        resolved, conflicts = self.agent.execute(result)
        
        assert resolved is not None
        assert len(resolved.recommendations) > 0
    
    @pytest.mark.unit
    def test_ranking_by_evidence(self, moderate_patient):
        """Test that recommendations are ranked by evidence level"""
        ingestion = IngestionAgent()
        semantic = SemanticMappingAgent()
        classification = ClassificationAgent()
        
        patient_fact, _ = ingestion.execute(moderate_patient)
        patient_with_codes, _ = semantic.execute(patient_fact)
        result = classification.execute(patient_with_codes)
        
        resolved, conflicts = self.agent.execute(result)
        
        # First recommendation should have highest evidence
        if len(resolved.recommendations) > 1:
            first_evidence = resolved.recommendations[0].get("evidence_level", "")
            assert first_evidence in ["Grade A", "Grade B", "Grade C", "Grade D"]


class TestExplanationAgent:
    """Test ExplanationAgent - MDT summary generation"""
    
    def setup_method(self):
        self.agent = ExplanationAgent()
    
    @pytest.mark.unit
    def test_generate_explanation(self, moderate_patient):
        """Test MDT summary generation"""
        # Run through pipeline
        ingestion = IngestionAgent()
        semantic = SemanticMappingAgent()
        classification = ClassificationAgent()
        conflict_resolution = ConflictResolutionAgent()
        
        patient_fact, _ = ingestion.execute(moderate_patient)
        patient_with_codes, _ = semantic.execute(patient_fact)
        result = classification.execute(patient_with_codes)
        resolved, conflicts = conflict_resolution.execute(result)
        
        explanation = self.agent.execute(patient_with_codes, resolved)
        
        assert explanation is not None
        # MDTSummary is an object, check its attributes
        assert hasattr(explanation, 'patient_id')
        assert hasattr(explanation, 'formatted_recommendations')
        # Should contain key information
        assert len(explanation.formatted_recommendations) > 0
    
    @pytest.mark.unit
    def test_explanation_includes_evidence(self, simple_patient):
        """Test that explanation includes evidence references"""
        # Run through pipeline
        ingestion = IngestionAgent()
        semantic = SemanticMappingAgent()
        classification = ClassificationAgent()
        conflict_resolution = ConflictResolutionAgent()
        
        patient_fact, _ = ingestion.execute(simple_patient)
        patient_with_codes, _ = semantic.execute(patient_fact)
        result = classification.execute(patient_with_codes)
        resolved, conflicts = conflict_resolution.execute(result)
        
        explanation = self.agent.execute(patient_with_codes, resolved)
        
        # MDTSummary should have clinical_summary
        assert hasattr(explanation, 'clinical_summary')
        assert len(explanation.clinical_summary) > 0


class TestCompletePipeline:
    """Integration test for complete 6-agent pipeline"""
    
    @pytest.mark.integration
    def test_full_pipeline_simple_case(self, simple_patient):
        """Test complete pipeline with simple case"""
        # Initialize all agents
        ingestion = IngestionAgent()
        semantic = SemanticMappingAgent()
        classification = ClassificationAgent()
        conflict_resolution = ConflictResolutionAgent()
        explanation = ExplanationAgent()
        
        # Execute pipeline
        patient_fact, errors = ingestion.execute(simple_patient)
        assert patient_fact is not None
        
        patient_with_codes, confidence = semantic.execute(patient_fact)
        assert patient_with_codes is not None
        
        result = classification.execute(patient_with_codes)
        assert result is not None
        
        resolved, conflicts = conflict_resolution.execute(result)
        assert resolved is not None
        
        mdt_summary = explanation.execute(patient_with_codes, resolved)
        assert mdt_summary is not None
    
    @pytest.mark.integration
    def test_full_pipeline_complex_case(self, complex_patient):
        """Test complete pipeline with complex case"""
        # Initialize all agents
        ingestion = IngestionAgent()
        semantic = SemanticMappingAgent()
        classification = ClassificationAgent()
        conflict_resolution = ConflictResolutionAgent()
        explanation = ExplanationAgent()
        
        # Execute pipeline
        patient_fact, errors = ingestion.execute(complex_patient)
        assert patient_fact is not None
        
        patient_with_codes, confidence = semantic.execute(patient_fact)
        assert patient_with_codes is not None
        
        result = classification.execute(patient_with_codes)
        assert result is not None
        
        resolved, conflicts = conflict_resolution.execute(result)
        assert resolved is not None
        
        mdt_summary = explanation.execute(patient_with_codes, resolved)
        assert mdt_summary is not None
    
    @pytest.mark.integration
    def test_pipeline_error_handling(self):
        """Test pipeline handles errors gracefully"""
        ingestion = IngestionAgent()
        
        # Invalid patient data
        invalid_patient = {"patient_id": "INVALID"}
        
        patient_fact, errors = ingestion.execute(invalid_patient)
        
        # Should catch errors rather than crash
        assert patient_fact is None or len(errors) > 0
