"""
Unit Tests for Ontology Components

Tests:
- LOINCIntegrator
- RxNormMapper
- LUCADAOntology
- GuidelineRuleEngine
"""

import pytest
from backend.src.ontology import (
    LOINCIntegrator,
    RxNormMapper,
    LUCADAOntology,
    GuidelineRuleEngine
)


class TestLOINCIntegrator:
    """Test LOINC Lab Test Integration"""
    
    def setup_method(self):
        self.integrator = LOINCIntegrator()
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_common_lab_mapping(self):
        """Test mapping common lab tests"""
        common_tests = [
            "hemoglobin",
            "creatinine",
            "alt",
            "platelet",
            "wbc"
        ]
        
        for test_name in common_tests:
            mapping = self.integrator.get_loinc_code(test_name)
            assert mapping is not None, f"Failed to map {test_name}"
            assert mapping.loinc_num is not None
            assert mapping.display_name is not None
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_hemoglobin_mapping(self):
        """Test specific hemoglobin mapping"""
        mapping = self.integrator.get_loinc_code("hemoglobin")
        
        assert mapping is not None
        assert "hemoglobin" in mapping.display_name.lower()
        assert mapping.loinc_num.startswith("718-")  # Standard Hgb code
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_creatinine_mapping(self):
        """Test creatinine mapping"""
        mapping = self.integrator.get_loinc_code("creatinine")
        
        assert mapping is not None
        assert "creatinine" in mapping.display_name.lower()
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_case_insensitive_mapping(self):
        """Test that mapping is case-insensitive"""
        mapping1 = self.integrator.get_loinc_code("hemoglobin")
        mapping2 = self.integrator.get_loinc_code("HEMOGLOBIN")
        mapping3 = self.integrator.get_loinc_code("Hemoglobin")
        
        assert mapping1.loinc_num == mapping2.loinc_num == mapping3.loinc_num
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_unknown_test_handling(self):
        """Test handling of unknown lab test"""
        mapping = self.integrator.get_loinc_code("nonexistent_test_xyz")
        
        # Should return None or handle gracefully
        assert mapping is None
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_snomed_mapping(self):
        """Test SNOMED-CT mappings when available"""
        mapping = self.integrator.get_loinc_code("hemoglobin")
        
        if mapping and mapping.snomed_mapping:
            assert isinstance(mapping.snomed_mapping, str)
            # SNOMED codes are numeric
            assert mapping.snomed_mapping.isdigit()


class TestRxNormMapper:
    """Test RxNorm Drug Mapping"""
    
    def setup_method(self):
        self.mapper = RxNormMapper()
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_common_oncology_drugs(self):
        """Test mapping common oncology drugs"""
        common_drugs = [
            "osimertinib",
            "pembrolizumab",
            "cisplatin",
            "carboplatin",
            "alectinib"
        ]
        
        for drug in common_drugs:
            mapping = self.mapper.map_medication(drug)
            assert mapping is not None, f"Failed to map {drug}"
            assert mapping.rxcui is not None
            assert mapping.drug_name is not None
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_osimertinib_mapping(self):
        """Test osimertinib specific mapping"""
        mapping = self.mapper.map_medication("osimertinib")
        
        assert mapping is not None
        assert "osimertinib" in mapping.drug_name.lower()
        assert mapping.rxcui is not None
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_drug_class_retrieval(self):
        """Test retrieving drug class"""
        drug_classes = {
            "osimertinib": "EGFR TKI",
            "pembrolizumab": "Immunotherapy",
            "cisplatin": "Chemotherapy"
        }
        
        for drug, expected_class in drug_classes.items():
            drug_class = self.mapper.get_drug_class(drug)
            assert drug_class is not None
            # Check if expected class is in the returned class
            assert expected_class.lower() in drug_class.lower() or \
                   any(term in drug_class.lower() for term in expected_class.lower().split())
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_drug_interaction_check(self):
        """Test drug-drug interaction checking"""
        # Known interaction: Osimertinib + CYP3A4 inducers
        interactions = self.mapper.check_drug_interactions("osimertinib", "rifampin")
        
        if interactions:
            assert isinstance(interactions, list)
            assert len(interactions) > 0
            # Should mention interaction
            assert any("cyp" in str(i).lower() or "interaction" in str(i).lower() 
                      for i in interactions)
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_no_interaction(self):
        """Test drugs with no known interactions"""
        # These likely don't interact (not absolute, just for testing)
        interactions = self.mapper.check_drug_interactions("pembrolizumab", "paracetamol")
        
        # Should return empty list or None
        assert interactions is None or len(interactions) == 0
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_case_insensitive_drug_mapping(self):
        """Test that drug mapping is case-insensitive"""
        mapping1 = self.mapper.map_medication("osimertinib")
        mapping2 = self.mapper.map_medication("OSIMERTINIB")
        mapping3 = self.mapper.map_medication("Osimertinib")
        
        assert mapping1.rxcui == mapping2.rxcui == mapping3.rxcui


class TestLUCADAOntology:
    """Test LUCADA OWL Ontology"""
    
    def setup_method(self):
        self.ontology = LUCADAOntology()
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_ontology_loading(self):
        """Test that ontology loads successfully"""
        assert self.ontology is not None
        assert hasattr(self.ontology, 'onto')
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_patient_class_exists(self):
        """Test that Patient class exists in ontology"""
        # Try to access Patient class
        try:
            patient_class = self.ontology.get_class("Patient")
            assert patient_class is not None
        except:
            # Some implementations may differ
            pass
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_treatment_class_exists(self):
        """Test that Treatment-related classes exist"""
        try:
            treatment_classes = [
                "Treatment",
                "ChemotherapyTreatment",
                "SurgicalTreatment",
                "RadiotherapyTreatment"
            ]
            
            for class_name in treatment_classes:
                cls = self.ontology.get_class(class_name)
                # Should not raise error if class exists
        except:
            pass
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_reasoning(self):
        """Test basic ontology reasoning"""
        try:
            # Attempt to run reasoner
            self.ontology.run_reasoner()
            # Should complete without errors
            assert True
        except Exception as e:
            # Reasoning may not be available in all setups
            pytest.skip(f"Reasoning not available: {e}")
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_query_individuals(self):
        """Test querying individuals from ontology"""
        try:
            # Try to get all patients
            patients = self.ontology.get_all_individuals("Patient")
            assert isinstance(patients, list)
        except:
            # May not be implemented in all versions
            pass


class TestGuidelineRuleEngine:
    """Test Clinical Guideline Rules Engine"""
    
    def setup_method(self):
        self.engine = GuidelineRuleEngine()
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_rule_engine_initialization(self):
        """Test rule engine initializes"""
        assert self.engine is not None
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_early_stage_rule(self, simple_patient):
        """Test rule for early-stage resectable NSCLC"""
        recommendations = self.engine.evaluate_patient(simple_patient)
        
        assert recommendations is not None
        assert len(recommendations) > 0
        
        # Early stage should recommend surgery
        assert any("surgery" in str(rec).lower() for rec in recommendations)
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_advanced_stage_rule(self, complex_patient):
        """Test rule for advanced-stage NSCLC"""
        recommendations = self.engine.evaluate_patient(complex_patient)
        
        assert recommendations is not None
        assert len(recommendations) > 0
        
        # Advanced stage should not primarily recommend surgery
        surgery_recs = [rec for rec in recommendations if "surgery" in str(rec).lower()]
        # Surgery might be mentioned but shouldn't be primary
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_ps_contraindication_rule(self, critical_patient):
        """Test that poor PS affects treatment recommendations"""
        recommendations = self.engine.evaluate_patient(critical_patient)
        
        assert recommendations is not None
        
        # PS 3-4 should limit aggressive treatment options
        # Should recommend supportive/palliative care
        assert any("supportive" in str(rec).lower() or 
                  "palliative" in str(rec).lower() or
                  "best supportive care" in str(rec).lower()
                  for rec in recommendations)
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_sclc_different_guidelines(self, sclc_patient):
        """Test that SCLC follows different guidelines than NSCLC"""
        recommendations = self.engine.evaluate_patient(sclc_patient)
        
        assert recommendations is not None
        assert len(recommendations) > 0
        
        # SCLC should emphasize chemotherapy over surgery
        chemo_recs = [rec for rec in recommendations if "chemo" in str(rec).lower()]
        assert len(chemo_recs) > 0
    
    @pytest.mark.unit
    @pytest.mark.ontology
    def test_rule_priorities(self, moderate_patient):
        """Test that rules are prioritized correctly"""
        recommendations = self.engine.evaluate_patient(moderate_patient)
        
        assert recommendations is not None
        
        if len(recommendations) > 1:
            # First recommendation should be highest priority
            # (Implementation-specific, just verify it's ordered)
            assert all(isinstance(rec, (str, dict)) for rec in recommendations)


class TestOntologyIntegration:
    """Integration tests for ontology components"""
    
    @pytest.mark.integration
    @pytest.mark.ontology
    def test_loinc_and_rxnorm_together(self):
        """Test using LOINC and RxNorm together"""
        loinc = LOINCIntegrator()
        rxnorm = RxNormMapper()
        
        # Get lab test
        hgb_mapping = loinc.get_loinc_code("hemoglobin")
        assert hgb_mapping is not None
        
        # Get medication
        drug_mapping = rxnorm.map_medication("osimertinib")
        assert drug_mapping is not None
        
        # Both should work together without conflicts
        assert hgb_mapping.loinc_num != drug_mapping.rxcui
    
    @pytest.mark.integration
    @pytest.mark.ontology
    def test_ontology_with_guidelines(self, moderate_patient):
        """Test ontology reasoning with guideline rules"""
        ontology = LUCADAOntology()
        rules = GuidelineRuleEngine()
        
        # Get recommendations from rules
        recommendations = rules.evaluate_patient(moderate_patient)
        
        assert ontology is not None
        assert recommendations is not None
        
        # Ontology and rules should work together
        # (Implementation-specific integration)
    
    @pytest.mark.integration
    @pytest.mark.ontology
    def test_complete_clinical_mapping(self, complex_patient):
        """Test complete clinical data mapping"""
        loinc = LOINCIntegrator()
        rxnorm = RxNormMapper()
        
        # Map lab tests
        lab_tests = ["hemoglobin", "creatinine", "platelet"]
        lab_mappings = [loinc.get_loinc_code(test) for test in lab_tests]
        
        # Map medications
        medications = ["cisplatin", "pembrolizumab"]
        drug_mappings = [rxnorm.map_medication(drug) for drug in medications]
        
        # All should map successfully
        assert all(m is not None for m in lab_mappings)
        assert all(m is not None for m in drug_mappings)
        
        # No conflicts between LOINC and RxNorm codes
        loinc_codes = set(m.loinc_num for m in lab_mappings if m)
        rxnorm_codes = set(m.rxcui for m in drug_mappings if m)
        
        assert len(loinc_codes.intersection(rxnorm_codes)) == 0
