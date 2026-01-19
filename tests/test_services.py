"""
Unit tests for new service implementations.
Tests auth, audit, HITL, analytics, RAG, websocket, version, batch, and FHIR services.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import jwt

from backend.src.services.auth_service import AuthService, UserCreate, UserInDB
from backend.src.services.audit_service import AuditService, AuditEvent
from backend.src.services.hitl_service import HITLService, HITLCase, HITLReview
from backend.src.services.analytics_service import AnalyticsService
from backend.src.services.rag_service import RAGService
from backend.src.services.websocket_service import WebSocketService
from backend.src.services.version_service import VersionService, GuidelineVersion
from backend.src.services.batch_service import BatchService, BatchJob
from backend.src.services.fhir_service import FHIRService


# ============================================================================
# AUTH SERVICE TESTS
# ============================================================================

class TestAuthService:
    """Test authentication and user management."""
    
    @pytest.fixture
    def auth_service(self):
        return AuthService(
            secret_key="test_secret_key",
            algorithm="HS256",
            access_token_expire_minutes=30
        )
    
    def test_password_hashing(self, auth_service):
        """Test password is hashed correctly."""
        password = "SecurePassword123!"
        hashed = auth_service.get_password_hash(password)
        
        assert hashed != password
        assert auth_service.verify_password(password, hashed)
        assert not auth_service.verify_password("wrong_password", hashed)
    
    def test_create_access_token(self, auth_service):
        """Test JWT token creation."""
        user_data = {"sub": "user@example.com", "role": "clinician"}
        token = auth_service.create_access_token(user_data)
        
        assert isinstance(token, str)
        
        # Decode and verify
        payload = jwt.decode(token, "test_secret_key", algorithms=["HS256"])
        assert payload["sub"] == "user@example.com"
        assert payload["role"] == "clinician"
        assert "exp" in payload
    
    @pytest.mark.asyncio
    async def test_create_user(self, auth_service):
        """Test user creation."""
        user_create = UserCreate(
            email="test@example.com",
            password="SecurePass123!",
            full_name="Test User",
            role="clinician"
        )
        
        with patch.object(auth_service, 'db') as mock_db:
            mock_db.users.find_one.return_value = None  # No existing user
            mock_db.users.insert_one.return_value = Mock(inserted_id="user_123")
            
            user = await auth_service.create_user(user_create)
            
            assert user.email == "test@example.com"
            assert user.full_name == "Test User"
            assert user.role == "clinician"
            assert user.hashed_password != "SecurePass123!"
    
    @pytest.mark.asyncio
    async def test_authenticate_user(self, auth_service):
        """Test user authentication."""
        hashed_password = auth_service.get_password_hash("password123")
        
        with patch.object(auth_service, 'db') as mock_db:
            mock_db.users.find_one.return_value = {
                "email": "user@example.com",
                "hashed_password": hashed_password,
                "full_name": "Test User",
                "role": "clinician",
                "is_active": True
            }
            
            user = await auth_service.authenticate_user("user@example.com", "password123")
            
            assert user is not None
            assert user.email == "user@example.com"
            
            # Wrong password
            user_wrong = await auth_service.authenticate_user("user@example.com", "wrong")
            assert user_wrong is None


# ============================================================================
# AUDIT SERVICE TESTS
# ============================================================================

class TestAuditService:
    """Test audit logging and compliance tracking."""
    
    @pytest.fixture
    def audit_service(self):
        return AuditService()
    
    @pytest.mark.asyncio
    async def test_log_event(self, audit_service):
        """Test audit event logging."""
        event = AuditEvent(
            event_type="user_login",
            user_id="user_123",
            resource_type="User",
            action="login",
            details={"ip_address": "192.168.1.1"}
        )
        
        with patch.object(audit_service, 'db') as mock_db:
            mock_db.audit_logs.insert_one.return_value = Mock(inserted_id="log_123")
            
            log_id = await audit_service.log_event(event)
            
            assert log_id == "log_123"
            mock_db.audit_logs.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_user_audit_trail(self, audit_service):
        """Test retrieving user audit trail."""
        with patch.object(audit_service, 'db') as mock_db:
            mock_db.audit_logs.find.return_value.sort.return_value.limit.return_value = [
                {
                    "event_type": "user_login",
                    "user_id": "user_123",
                    "timestamp": datetime.utcnow(),
                    "action": "login"
                }
            ]
            
            events = await audit_service.get_user_audit_trail("user_123", limit=10)
            
            assert len(events) == 1
            assert events[0]["user_id"] == "user_123"


# ============================================================================
# HITL SERVICE TESTS
# ============================================================================

class TestHITLService:
    """Test human-in-the-loop review workflow."""
    
    @pytest.fixture
    def hitl_service(self):
        return HITLService()
    
    @pytest.mark.asyncio
    async def test_submit_case(self, hitl_service):
        """Test submitting a case for review."""
        case = HITLCase(
            patient_id="patient_123",
            recommendations=["Recommendation 1", "Recommendation 2"],
            confidence_score=0.65,
            priority="high",
            clinical_context={"diagnosis": "NSCLC"}
        )
        
        with patch.object(hitl_service, 'db') as mock_db:
            mock_db.hitl_cases.insert_one.return_value = Mock(inserted_id="case_123")
            
            case_id = await hitl_service.submit_case(case)
            
            assert case_id == "case_123"
    
    @pytest.mark.asyncio
    async def test_get_review_queue(self, hitl_service):
        """Test retrieving review queue."""
        with patch.object(hitl_service, 'db') as mock_db:
            mock_db.hitl_cases.find.return_value.sort.return_value = [
                {
                    "case_id": "case_123",
                    "status": "pending",
                    "priority": "high",
                    "submitted_at": datetime.utcnow()
                }
            ]
            
            queue = await hitl_service.get_review_queue(status="pending")
            
            assert len(queue) == 1
            assert queue[0]["status"] == "pending"
    
    @pytest.mark.asyncio
    async def test_submit_review(self, hitl_service):
        """Test submitting a clinical review."""
        review = HITLReview(
            case_id="case_123",
            reviewer_id="clinician_456",
            decision="approved",
            comments="Recommendations look good",
            modifications=None
        )
        
        with patch.object(hitl_service, 'db') as mock_db:
            mock_db.hitl_cases.find_one.return_value = {
                "case_id": "case_123",
                "status": "pending"
            }
            mock_db.hitl_cases.update_one.return_value = Mock()
            mock_db.hitl_reviews.insert_one.return_value = Mock(inserted_id="review_123")
            
            review_id = await hitl_service.submit_review(review)
            
            assert review_id == "review_123"


# ============================================================================
# ANALYTICS SERVICE TESTS
# ============================================================================

class TestAnalyticsService:
    """Test advanced analytics computations."""
    
    @pytest.fixture
    def analytics_service(self):
        return AnalyticsService()
    
    @pytest.mark.asyncio
    async def test_generate_survival_curve(self, analytics_service):
        """Test survival curve generation."""
        patient_cohort = ["patient_1", "patient_2", "patient_3"]
        
        with patch.object(analytics_service, 'neo4j') as mock_neo4j:
            mock_neo4j.execute_query.return_value = [
                {"time_points": [0, 12, 24, 36], "survival_prob": [1.0, 0.9, 0.75, 0.6]}
            ]
            
            result = await analytics_service.generate_survival_curve(patient_cohort)
            
            assert "time_points" in result
            assert "survival_prob" in result
            assert len(result["time_points"]) == 4
    
    @pytest.mark.asyncio
    async def test_quantify_uncertainty(self, analytics_service):
        """Test uncertainty quantification."""
        recommendations = {
            "treatment_plan": "Carboplatin + Pemetrexed",
            "confidence": 0.85
        }
        
        result = await analytics_service.quantify_uncertainty(recommendations)
        
        assert "overall_uncertainty" in result
        assert "epistemic_uncertainty" in result
        assert "aleatoric_uncertainty" in result
        assert 0 <= result["overall_uncertainty"] <= 1


# ============================================================================
# RAG SERVICE TESTS
# ============================================================================

class TestRAGService:
    """Test retrieval-augmented generation."""
    
    @pytest.fixture
    def rag_service(self):
        return RAGService()
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant_guidelines(self, rag_service):
        """Test guideline retrieval."""
        query = "Treatment for stage III NSCLC"
        
        with patch.object(rag_service, 'vector_store') as mock_vs:
            mock_vs.similarity_search.return_value = [
                {"content": "Guideline 1", "score": 0.95},
                {"content": "Guideline 2", "score": 0.88}
            ]
            
            results = await rag_service.retrieve_relevant_guidelines(query, top_k=2)
            
            assert len(results) == 2
            assert results[0]["score"] >= results[1]["score"]
    
    @pytest.mark.asyncio
    async def test_generate_with_context(self, rag_service):
        """Test context-aware generation."""
        query = "What is the recommended dosage?"
        context_docs = ["Document 1", "Document 2"]
        
        with patch.object(rag_service, 'llm') as mock_llm:
            mock_llm.generate.return_value = "Recommended dosage is 500mg twice daily"
            
            response = await rag_service.generate_with_context(query, context_docs)
            
            assert isinstance(response, str)
            assert len(response) > 0


# ============================================================================
# WEBSOCKET SERVICE TESTS
# ============================================================================

class TestWebSocketService:
    """Test real-time WebSocket messaging."""
    
    @pytest.fixture
    def websocket_service(self):
        return WebSocketService()
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self, websocket_service):
        """Test WebSocket connection lifecycle."""
        mock_ws = Mock()
        
        await websocket_service.connect("user_123", mock_ws)
        assert "user_123" in websocket_service.active_connections
        
        await websocket_service.disconnect("user_123")
        assert "user_123" not in websocket_service.active_connections
    
    @pytest.mark.asyncio
    async def test_subscribe_to_channel(self, websocket_service):
        """Test channel subscription."""
        await websocket_service.subscribe("user_123", "analysis_updates")
        
        assert "user_123" in websocket_service.channel_subscriptions["analysis_updates"]
    
    @pytest.mark.asyncio
    async def test_broadcast_to_channel(self, websocket_service):
        """Test broadcasting to channel subscribers."""
        mock_ws = AsyncMock()
        await websocket_service.connect("user_123", mock_ws)
        await websocket_service.subscribe("user_123", "system_alerts")
        
        message = {"alert": "System maintenance in 10 minutes"}
        await websocket_service.broadcast_to_channel("system_alerts", message)
        
        mock_ws.send_json.assert_called_once()


# ============================================================================
# VERSION SERVICE TESTS
# ============================================================================

class TestVersionService:
    """Test guideline version management."""
    
    @pytest.fixture
    def version_service(self):
        return VersionService()
    
    @pytest.mark.asyncio
    async def test_create_version(self, version_service):
        """Test creating a new guideline version."""
        version = GuidelineVersion(
            version_number="2024.1",
            guideline_type="NCCN_NSCLC",
            content={"treatment_protocols": []},
            changelog="Updated staging criteria"
        )
        
        with patch.object(version_service, 'db') as mock_db:
            mock_db.guideline_versions.insert_one.return_value = Mock(inserted_id="v_123")
            
            version_id = await version_service.create_version(version)
            
            assert version_id == "v_123"
    
    @pytest.mark.asyncio
    async def test_activate_version(self, version_service):
        """Test activating a guideline version."""
        with patch.object(version_service, 'db') as mock_db:
            mock_db.guideline_versions.update_many.return_value = Mock()
            mock_db.guideline_versions.update_one.return_value = Mock()
            
            await version_service.activate_version("v_123")
            
            # Should deactivate all other versions
            assert mock_db.guideline_versions.update_many.called
            assert mock_db.guideline_versions.update_one.called


# ============================================================================
# BATCH SERVICE TESTS
# ============================================================================

class TestBatchService:
    """Test batch processing and population-level analysis."""
    
    @pytest.fixture
    def batch_service(self):
        return BatchService()
    
    @pytest.mark.asyncio
    async def test_create_batch_job(self, batch_service):
        """Test batch job creation."""
        job = BatchJob(
            job_type="population_analysis",
            input_data={"patient_ids": ["p1", "p2", "p3"]},
            config={"use_latest_guidelines": True}
        )
        
        with patch.object(batch_service, 'db') as mock_db:
            mock_db.batch_jobs.insert_one.return_value = Mock(inserted_id="job_123")
            
            job_id = await batch_service.create_batch_job(job)
            
            assert job_id == "job_123"
    
    @pytest.mark.asyncio
    async def test_get_job_status(self, batch_service):
        """Test retrieving job status."""
        with patch.object(batch_service, 'db') as mock_db:
            mock_db.batch_jobs.find_one.return_value = {
                "job_id": "job_123",
                "status": "processing",
                "progress": 50
            }
            
            status = await batch_service.get_job_status("job_123")
            
            assert status["status"] == "processing"
            assert status["progress"] == 50


# ============================================================================
# FHIR SERVICE TESTS
# ============================================================================

class TestFHIRService:
    """Test FHIR R4 integration."""
    
    @pytest.fixture
    def fhir_service(self):
        return FHIRService(fhir_server_url="http://localhost:8080/fhir")
    
    @pytest.mark.asyncio
    async def test_import_bundle(self, fhir_service):
        """Test FHIR Bundle import."""
        bundle = {
            "resourceType": "Bundle",
            "type": "transaction",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient-123",
                        "name": [{"family": "Smith", "given": ["John"]}]
                    }
                }
            ]
        }
        
        with patch.object(fhir_service, 'client') as mock_client:
            mock_client.post.return_value.json.return_value = {
                "resourceType": "Bundle",
                "type": "transaction-response",
                "entry": [{"response": {"status": "201 Created"}}]
            }
            
            result = await fhir_service.import_bundle(bundle)
            
            assert result["total_resources"] == 1
    
    @pytest.mark.asyncio
    async def test_get_patient(self, fhir_service):
        """Test retrieving FHIR patient resource."""
        with patch.object(fhir_service, 'client') as mock_client:
            mock_client.get.return_value.json.return_value = {
                "resourceType": "Patient",
                "id": "patient-123",
                "name": [{"family": "Smith"}]
            }
            
            patient = await fhir_service.get_patient("patient-123")
            
            assert patient["resourceType"] == "Patient"
            assert patient["id"] == "patient-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
