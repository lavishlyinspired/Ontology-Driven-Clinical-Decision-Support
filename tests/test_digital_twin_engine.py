"""
Unit Tests for Digital Twin Engine

Tests the core DigitalTwinEngine functionality
"""

import pytest
import asyncio
from datetime import datetime
from backend.src.digital_twin import (
    DigitalTwinEngine,
    TwinState,
    UpdateType,
    create_digital_twin
)


class TestDigitalTwinEngine:
    """Test Digital Twin Engine core functionality"""
    
    @pytest.mark.asyncio
    async def test_create_twin(self):
        """Test creating a digital twin"""
        twin = DigitalTwinEngine(patient_id="P001")
        
        assert twin.patient_id == "P001"
        assert twin.state == TwinState.INITIALIZING
        assert len(twin.context_graph.nodes) == 0
        
    @pytest.mark.asyncio
    async def test_initialize_twin(self):
        """Test initializing a digital twin with patient data"""
        patient_data = {
            "patient_id": "P001",
            "age": 68,
            "stage": "IIIA",
            "histology": "Adenocarcinoma",
            "cancer_type": "NSCLC"
        }
        
        twin = DigitalTwinEngine(patient_id="P001")
        result = await twin.initialize(patient_data)
        
        assert result["patient_id"] == "P001"
        assert twin.state == TwinState.ACTIVE
        assert len(twin.context_graph.nodes) > 0
        assert len(twin.snapshots) == 1
        
    @pytest.mark.asyncio
    async def test_update_twin_lab_result(self):
        """Test updating twin with lab result"""
        patient_data = {
            "patient_id": "P001",
            "age": 68,
            "stage": "IIIA",
            "histology": "Adenocarcinoma",
            "cancer_type": "NSCLC"
        }
        
        twin = DigitalTwinEngine(patient_id="P001")
        await twin.initialize(patient_data)
        
        # Update with lab result
        update_result = await twin.update({
            "type": UpdateType.LAB_RESULT.value,
            "data": {
                "test": "EGFR mutation",
                "result": "T790M detected"
            }
        })
        
        assert update_result["update_type"] == UpdateType.LAB_RESULT.value
        assert "biomarker" in update_result.get("changes_detected", []) or \
               "new_lab_data" in update_result.get("changes_detected", [])
        assert len(twin.snapshots) == 2  # Initial + update
        
    @pytest.mark.asyncio
    async def test_update_twin_progression(self):
        """Test updating twin with progression event"""
        patient_data = {
            "patient_id": "P001",
            "age": 68,
            "stage": "IIIA",
            "histology": "Adenocarcinoma",
            "cancer_type": "NSCLC"
        }
        
        twin = DigitalTwinEngine(patient_id="P001")
        await twin.initialize(patient_data)
        
        # Update with progression
        update_result = await twin.update({
            "type": UpdateType.IMAGING.value,
            "data": {
                "recist_status": "PD",
                "tumor_size_mm": 45,
                "findings": "Disease progression detected"
            }
        })
        
        assert len(update_result.get("new_alerts", [])) > 0
        assert any(alert["category"] == "progression" for alert in update_result.get("new_alerts", []))
        
    @pytest.mark.asyncio
    async def test_get_current_state(self):
        """Test getting current twin state"""
        patient_data = {
            "patient_id": "P001",
            "age": 68,
            "stage": "IIIA",
            "cancer_type": "NSCLC"
        }
        
        twin = DigitalTwinEngine(patient_id="P001")
        await twin.initialize(patient_data)
        
        state = twin.get_current_state()
        
        assert state["patient_id"] == "P001"
        assert state["state"] == TwinState.ACTIVE.value
        assert "clinical_state" in state
        assert "context_graph" in state
        assert "active_alerts" in state
        
    @pytest.mark.asyncio
    async def test_predict_trajectories(self):
        """Test trajectory prediction"""
        patient_data = {
            "patient_id": "P001",
            "age": 68,
            "stage": "IIIA",
            "cancer_type": "NSCLC",
            "performance_status": 1
        }
        
        twin = DigitalTwinEngine(patient_id="P001")
        await twin.initialize(patient_data)
        
        predictions = await twin.predict_trajectories()
        
        assert "pathways" in predictions
        assert "confidence" in predictions
        assert len(predictions["pathways"]) > 0
        
    @pytest.mark.asyncio
    async def test_export_twin(self):
        """Test exporting twin state"""
        patient_data = {
            "patient_id": "P001",
            "age": 68,
            "stage": "IIIA",
            "cancer_type": "NSCLC"
        }
        
        twin = DigitalTwinEngine(patient_id="P001")
        await twin.initialize(patient_data)
        
        export = twin.export_twin()
        
        assert "twin_metadata" in export
        assert "patient_data" in export
        assert "context_graph" in export
        assert "snapshots" in export
        assert export["twin_metadata"]["patient_id"] == "P001"
        
    @pytest.mark.asyncio
    async def test_context_graph_layers(self):
        """Test context graph layer organization"""
        patient_data = {
            "patient_id": "P001",
            "age": 68,
            "stage": "IIIA",
            "cancer_type": "NSCLC"
        }
        
        twin = DigitalTwinEngine(patient_id="P001")
        await twin.initialize(patient_data)
        
        # Add various updates
        await twin.update({
            "type": UpdateType.LAB_RESULT.value,
            "data": {"test": "CBC", "result": "Normal"}
        })
        
        await twin.update({
            "type": UpdateType.IMAGING.value,
            "data": {"recist_status": "PR", "tumor_size_mm": 30}
        })
        
        state = twin.get_current_state()
        layers = state["context_graph"]["layers"]
        
        # Should have nodes in multiple layers
        assert layers["clinical_facts"] > 0
        assert layers["temporal_events"] > 0
        
    @pytest.mark.asyncio
    async def test_alert_generation(self):
        """Test alert generation on significant events"""
        patient_data = {
            "patient_id": "P001",
            "age": 68,
            "stage": "IIIA",
            "cancer_type": "NSCLC"
        }
        
        twin = DigitalTwinEngine(patient_id="P001")
        await twin.initialize(patient_data)
        
        # Update with resistance mutation (should trigger alert)
        update_result = await twin.update({
            "type": UpdateType.LAB_RESULT.value,
            "data": {
                "test": "EGFR Resistance Panel",
                "results": {
                    "T790M": "Detected",
                    "allele_frequency": 0.15
                },
                "resistance_mutations": ["T790M"]
            }
        })
        
        # Should generate alerts
        assert len(twin.active_alerts) > 0 or len(update_result.get("new_alerts", [])) > 0
        
    @pytest.mark.asyncio
    async def test_create_digital_twin_convenience(self):
        """Test convenience function for creating twins"""
        patient_data = {
            "patient_id": "P002",
            "age": 70,
            "stage": "IIIB",
            "cancer_type": "NSCLC"
        }
        
        twin = await create_digital_twin("P002", patient_data)
        
        assert twin.patient_id == "P002"
        assert twin.state == TwinState.ACTIVE
        assert len(twin.context_graph.nodes) > 0
        

class TestTwinLifecycle:
    """Test complete twin lifecycle"""
    
    @pytest.mark.asyncio
    async def test_complete_patient_journey(self):
        """Test a complete patient journey through twin"""
        # Initial presentation
        patient_data = {
            "patient_id": "P003",
            "age": 68,
            "stage": "IIIA",
            "histology": "Adenocarcinoma",
            "cancer_type": "NSCLC",
            "biomarkers": {"EGFR": "Ex19del"}
        }
        
        twin = DigitalTwinEngine(patient_id="P003")
        await twin.initialize(patient_data)
        
        initial_nodes = len(twin.context_graph.nodes)
        
        # Month 3: Good response
        await twin.update({
            "type": UpdateType.IMAGING.value,
            "data": {
                "recist_status": "PR",
                "tumor_size_mm": 28,
                "findings": "Partial response"
            }
        })
        
        # Month 8: Progression
        await twin.update({
            "type": UpdateType.IMAGING.value,
            "data": {
                "recist_status": "PD",
                "tumor_size_mm": 38,
                "findings": "Disease progression"
            }
        })
        
        # Resistance testing
        await twin.update({
            "type": UpdateType.LAB_RESULT.value,
            "data": {
                "test": "T790M",
                "result": "Detected"
            }
        })
        
        # Treatment change
        await twin.update({
            "type": UpdateType.TREATMENT_CHANGE.value,
            "data": {
                "new_treatment": "Osimertinib",
                "reason": "T790M resistance"
            }
        })
        
        # Verify journey tracked
        assert len(twin.snapshots) == 5  # Initial + 4 updates
        assert len(twin.context_graph.nodes) > initial_nodes
        assert len(twin.active_alerts) > 0
        
        # Export for review
        export = twin.export_twin()
        assert len(export["snapshots"]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
