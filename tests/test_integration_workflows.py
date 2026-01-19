"""
Integration tests for complete end-to-end workflows.
Tests patient analysis pipeline, FHIR import, HITL workflow, and batch processing.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from backend.src.agents.integrated_workflow import IntegratedWorkflow
from backend.src.services.fhir_service import FHIRService
from backend.src.services.hitl_service import HITLService
from backend.src.services.batch_service import BatchService


# ============================================================================
# PATIENT ANALYSIS WORKFLOW TESTS
# ============================================================================

class TestPatientAnalysisWorkflow:
    """Test complete patient analysis pipeline."""
    
    @pytest.fixture
    async def workflow(self):
        """Initialize integrated workflow."""
        workflow = IntegratedWorkflow()
        await workflow.initialize()
        return workflow
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_nsclc_patient_analysis(self, workflow):
        """Test complete NSCLC patient analysis workflow."""
        patient_data = {
            "patient_id": "PT-12345",
            "demographics": {
                "age": 62,
                "gender": "male",
                "smoking_history": "former smoker, 40 pack-years"
            },
            "diagnosis": {
                "condition": "Non-small cell lung cancer",
                "snomed_code": "254637007",
                "stage": "IIIB",
                "histology": "adenocarcinoma"
            },
            "biomarkers": {
                "PD-L1": "50%",
                "EGFR": "negative",
                "ALK": "negative",
                "KRAS": "G12C mutation"
            },
            "performance_status": {
                "ecog": 1
            }
        }
        
        # Run complete workflow
        result = await workflow.analyze_patient(patient_data)
        
        # Verify all workflow stages completed
        assert result["status"] == "success"
        assert "ingestion" in result["stages"]
        assert "semantic_mapping" in result["stages"]
        assert "classification" in result["stages"]
        assert "biomarker_analysis" in result["stages"]
        assert "recommendations" in result
        
        # Verify recommendations structure
        recommendations = result["recommendations"]
        assert "treatment_plan" in recommendations
        assert "clinical_trials" in recommendations
        assert "confidence_score" in recommendations
        assert 0 <= recommendations["confidence_score"] <= 1
        
        # Verify knowledge graph integration
        assert "graph_node_id" in result
        assert result["graph_node_id"] is not None
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sclc_patient_analysis(self, workflow):
        """Test SCLC patient analysis workflow."""
        patient_data = {
            "patient_id": "PT-67890",
            "demographics": {"age": 58, "gender": "female"},
            "diagnosis": {
                "condition": "Small cell lung cancer",
                "snomed_code": "254632001",
                "stage": "extensive"
            },
            "performance_status": {"ecog": 2}
        }
        
        result = await workflow.analyze_patient(patient_data)
        
        assert result["status"] == "success"
        assert result["diagnosis_type"] == "SCLC"
        assert "treatment_plan" in result["recommendations"]
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_low_confidence_triggers_hitl(self, workflow):
        """Test that low confidence scores trigger HITL review."""
        patient_data = {
            "patient_id": "PT-EDGE-001",
            "demographics": {"age": 75, "gender": "male"},
            "diagnosis": {
                "condition": "Lung cancer NOS",
                "snomed_code": "363358000",
                "stage": "unclear"  # Ambiguous staging
            },
            "comorbidities": [
                "heart_failure",
                "chronic_kidney_disease",
                "diabetes"
            ]
        }
        
        result = await workflow.analyze_patient(patient_data)
        
        # Should complete but flag for review
        assert result["status"] == "success"
        assert result["recommendations"]["confidence_score"] < 0.7
        assert result["requires_hitl_review"] is True
        assert "hitl_case_id" in result


# ============================================================================
# FHIR IMPORT WORKFLOW TESTS
# ============================================================================

class TestFHIRImportWorkflow:
    """Test FHIR Bundle import and integration."""
    
    @pytest.fixture
    def fhir_service(self):
        return FHIRService(fhir_server_url="http://localhost:8080/fhir")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_fhir_bundle_import(self, fhir_service):
        """Test importing complete FHIR Bundle with multiple resources."""
        bundle = {
            "resourceType": "Bundle",
            "type": "transaction",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "fhir-pt-001",
                        "name": [{"family": "Johnson", "given": ["Emily"]}],
                        "gender": "female",
                        "birthDate": "1962-05-15"
                    },
                    "request": {"method": "POST", "url": "Patient"}
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "subject": {"reference": "Patient/fhir-pt-001"},
                        "code": {
                            "coding": [{
                                "system": "http://snomed.info/sct",
                                "code": "254637007",
                                "display": "Non-small cell lung cancer"
                            }]
                        },
                        "stage": [{
                            "summary": {
                                "coding": [{
                                    "code": "IIIA",
                                    "display": "Stage IIIA"
                                }]
                            }
                        }]
                    },
                    "request": {"method": "POST", "url": "Condition"}
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "subject": {"reference": "Patient/fhir-pt-001"},
                        "code": {
                            "coding": [{
                                "code": "PD-L1",
                                "display": "PD-L1 Expression"
                            }]
                        },
                        "valueQuantity": {
                            "value": 65,
                            "unit": "%"
                        }
                    },
                    "request": {"method": "POST", "url": "Observation"}
                }
            ]
        }
        
        with patch.object(fhir_service, 'client') as mock_client:
            mock_client.post.return_value.status_code = 200
            mock_client.post.return_value.json.return_value = {
                "resourceType": "Bundle",
                "type": "transaction-response",
                "entry": [
                    {"response": {"status": "201 Created", "location": "Patient/fhir-pt-001"}},
                    {"response": {"status": "201 Created"}},
                    {"response": {"status": "201 Created"}}
                ]
            }
            
            result = await fhir_service.import_bundle(bundle)
            
            assert result["total_resources"] == 3
            assert result["successful"] == 3
            assert result["failed"] == 0
            assert "patient_ids" in result
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_fhir_to_lca_transformation(self, fhir_service):
        """Test transforming FHIR resources into LCA patient model."""
        fhir_patient = {
            "resourceType": "Patient",
            "id": "fhir-pt-002",
            "name": [{"family": "Williams", "given": ["Robert"]}],
            "gender": "male",
            "birthDate": "1958-08-20"
        }
        
        fhir_condition = {
            "resourceType": "Condition",
            "subject": {"reference": "Patient/fhir-pt-002"},
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "254632001",
                    "display": "Small cell lung cancer"
                }]
            }
        }
        
        lca_patient = await fhir_service.transform_to_lca_model(
            fhir_patient, [fhir_condition]
        )
        
        assert lca_patient["patient_id"] == "fhir-pt-002"
        assert lca_patient["demographics"]["age"] == 65  # Calculated from birthDate
        assert lca_patient["demographics"]["gender"] == "male"
        assert lca_patient["diagnosis"]["snomed_code"] == "254632001"


# ============================================================================
# HITL WORKFLOW TESTS
# ============================================================================

class TestHITLWorkflow:
    """Test human-in-the-loop review workflow."""
    
    @pytest.fixture
    def hitl_service(self):
        return HITLService()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_hitl_workflow(self, hitl_service):
        """Test complete HITL review cycle."""
        # 1. Submit case for review
        case_data = {
            "patient_id": "PT-HITL-001",
            "recommendations": [
                "Carboplatin + Pemetrexed",
                "Consider immunotherapy"
            ],
            "confidence_score": 0.62,
            "priority": "high",
            "clinical_context": {
                "diagnosis": "NSCLC Stage IIIB",
                "biomarkers": {"PD-L1": "45%"},
                "comorbidities": ["COPD", "diabetes"]
            }
        }
        
        with patch.object(hitl_service, 'db') as mock_db:
            mock_db.hitl_cases.insert_one.return_value = Mock(inserted_id="case_001")
            
            case_id = await hitl_service.submit_case(case_data)
            assert case_id == "case_001"
            
            # 2. Retrieve from queue
            mock_db.hitl_cases.find.return_value.sort.return_value = [{
                "case_id": "case_001",
                "status": "pending",
                "priority": "high",
                "submitted_at": datetime.utcnow()
            }]
            
            queue = await hitl_service.get_review_queue(status="pending")
            assert len(queue) == 1
            assert queue[0]["case_id"] == "case_001"
            
            # 3. Submit review
            review_data = {
                "case_id": "case_001",
                "reviewer_id": "clinician_123",
                "decision": "approved_with_modifications",
                "comments": "Added consideration for pembrolizumab based on PD-L1 level",
                "modifications": {
                    "recommendations": [
                        "Carboplatin + Pemetrexed + Pembrolizumab",
                        "Monitor for immune-related adverse events"
                    ]
                }
            }
            
            mock_db.hitl_cases.find_one.return_value = {
                "case_id": "case_001",
                "status": "pending"
            }
            mock_db.hitl_reviews.insert_one.return_value = Mock(inserted_id="review_001")
            
            review_id = await hitl_service.submit_review(review_data)
            assert review_id == "review_001"
            
            # 4. Verify case updated
            mock_db.hitl_cases.update_one.assert_called()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_hitl_metrics_tracking(self, hitl_service):
        """Test HITL metrics and performance tracking."""
        with patch.object(hitl_service, 'db') as mock_db:
            mock_db.hitl_reviews.aggregate.return_value = [
                {
                    "total_reviews": 150,
                    "avg_review_time_minutes": 12.5,
                    "approval_rate": 0.78,
                    "modification_rate": 0.15,
                    "rejection_rate": 0.07
                }
            ]
            
            metrics = await hitl_service.get_review_metrics(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31)
            )
            
            assert metrics["total_reviews"] == 150
            assert metrics["approval_rate"] > 0.7
            assert metrics["avg_review_time_minutes"] < 15


# ============================================================================
# BATCH PROCESSING WORKFLOW TESTS
# ============================================================================

class TestBatchProcessingWorkflow:
    """Test population-level batch processing."""
    
    @pytest.fixture
    def batch_service(self):
        return BatchService()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_population_analysis_batch(self, batch_service):
        """Test batch analysis of patient population."""
        job_data = {
            "job_type": "population_analysis",
            "input_data": {
                "cohort_definition": {
                    "diagnosis": "NSCLC",
                    "stage": ["IIIA", "IIIB"],
                    "age_range": [50, 75]
                }
            },
            "config": {
                "use_latest_guidelines": True,
                "include_survival_analysis": True
            }
        }
        
        with patch.object(batch_service, 'db') as mock_db:
            mock_db.batch_jobs.insert_one.return_value = Mock(inserted_id="batch_001")
            
            job_id = await batch_service.create_batch_job(job_data)
            assert job_id == "batch_001"
            
            # Simulate job processing
            mock_db.batch_jobs.find_one.return_value = {
                "job_id": "batch_001",
                "status": "processing",
                "total_items": 500,
                "processed_items": 250,
                "progress": 50
            }
            
            status = await batch_service.get_job_status("batch_001")
            assert status["progress"] == 50
            
            # Simulate completion
            mock_db.batch_jobs.find_one.return_value = {
                "job_id": "batch_001",
                "status": "completed",
                "total_items": 500,
                "processed_items": 500,
                "results_summary": {
                    "analyzed_patients": 500,
                    "avg_confidence_score": 0.82,
                    "hitl_reviews_required": 75
                }
            }
            
            final_status = await batch_service.get_job_status("batch_001")
            assert final_status["status"] == "completed"
            assert final_status["results_summary"]["analyzed_patients"] == 500
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_bulk_fhir_import_batch(self, batch_service):
        """Test batch import of FHIR resources."""
        job_data = {
            "job_type": "bulk_fhir_import",
            "input_data": {
                "file_path": "/uploads/fhir_bundle_batch.json",
                "bundle_size": 100
            }
        }
        
        with patch.object(batch_service, 'db') as mock_db:
            mock_db.batch_jobs.insert_one.return_value = Mock(inserted_id="batch_002")
            
            job_id = await batch_service.create_batch_job(job_data)
            
            # Verify job created
            assert job_id == "batch_002"


# ============================================================================
# CONCURRENT WORKFLOW TESTS
# ============================================================================

class TestConcurrentWorkflows:
    """Test multiple concurrent workflow executions."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_concurrent_patient_analyses(self):
        """Test analyzing multiple patients concurrently."""
        workflow = IntegratedWorkflow()
        await workflow.initialize()
        
        patients = [
            {"patient_id": f"PT-CONCURRENT-{i}", "diagnosis": {"snomed_code": "254637007"}}
            for i in range(10)
        ]
        
        # Analyze all patients concurrently
        tasks = [workflow.analyze_patient(p) for p in patients]
        results = await asyncio.gather(*tasks)
        
        # Verify all completed successfully
        assert len(results) == 10
        for result in results:
            assert result["status"] == "success"
            assert "recommendations" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration", "--tb=short"])
