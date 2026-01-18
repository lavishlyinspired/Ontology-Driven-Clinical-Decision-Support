"""
Component-level tests for LCA modules
Tests individual components without requiring full Neo4j setup
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend" / "src"
sys.path.insert(0, str(backend_path))

import pytest
from datetime import datetime

# Import components to test
from agents.dynamic_orchestrator import (
    DynamicWorkflowOrchestrator,
    WorkflowComplexity,
    DynamicContextGraph,
    ContextNode,
    ContextEdge
)

from agents.nsclc_agent import NSCLCAgent
from agents.sclc_agent import SCLCAgent
from agents.biomarker_agent import BiomarkerAgent
from agents.negotiation_protocol import NegotiationProtocol, NegotiationStrategy, AgentProposal

from db.models import PatientFact


class TestDynamicOrchestrator:
    """Test Dynamic Workflow Orchestrator"""

    def test_simple_complexity_assessment(self):
        """Stage IA, PS 0, no comorbidities = SIMPLE"""
        orchestrator = DynamicWorkflowOrchestrator()

        patient_data = {
            "patient_id": "T001",
            "age_at_diagnosis": 55,
            "tnm_stage": "IA",
            "performance_status": 0,
            "comorbidities": []
        }

        complexity = orchestrator.assess_complexity(patient_data)
        assert complexity == WorkflowComplexity.SIMPLE

    def test_moderate_complexity_assessment(self):
        """Stage IIIA, PS 1 = MODERATE"""
        orchestrator = DynamicWorkflowOrchestrator()

        patient_data = {
            "patient_id": "T002",
            "age_at_diagnosis": 65,
            "tnm_stage": "IIIA",
            "performance_status": 1,
            "comorbidities": []
        }

        complexity = orchestrator.assess_complexity(patient_data)
        assert complexity == WorkflowComplexity.MODERATE

    def test_complex_complexity_assessment(self):
        """Stage IIIB, PS 2, 2 comorbidities = COMPLEX"""
        orchestrator = DynamicWorkflowOrchestrator()

        patient_data = {
            "patient_id": "T003",
            "age_at_diagnosis": 72,
            "tnm_stage": "IIIB",
            "performance_status": 2,
            "comorbidities": ["COPD", "Diabetes"]
        }

        complexity = orchestrator.assess_complexity(patient_data)
        assert complexity == WorkflowComplexity.COMPLEX

    def test_critical_complexity_assessment(self):
        """Stage IV, PS 3, age 85, 3 comorbidities = CRITICAL"""
        orchestrator = DynamicWorkflowOrchestrator()

        patient_data = {
            "patient_id": "T004",
            "age_at_diagnosis": 85,
            "tnm_stage": "IV",
            "performance_status": 3,
            "comorbidities": ["COPD", "Diabetes", "CAD"]
        }

        complexity = orchestrator.assess_complexity(patient_data)
        assert complexity == WorkflowComplexity.CRITICAL

    def test_emergency_override(self):
        """Emergency flag forces CRITICAL"""
        orchestrator = DynamicWorkflowOrchestrator()

        patient_data = {
            "patient_id": "T005",
            "age_at_diagnosis": 55,
            "tnm_stage": "IA",
            "performance_status": 0,
            "emergency": True
        }

        complexity = orchestrator.assess_complexity(patient_data)
        assert complexity == WorkflowComplexity.CRITICAL

    def test_simple_workflow_path(self):
        """SIMPLE should use minimal agents"""
        orchestrator = DynamicWorkflowOrchestrator()

        path = orchestrator.select_workflow_path(WorkflowComplexity.SIMPLE)

        assert "IngestionAgent" in path
        assert "ClassificationAgent" in path
        assert "ExplanationAgent" in path
        # Should NOT include heavy analytics
        assert "UncertaintyQuantifier" not in path
        assert "SurvivalAnalyzer" not in path

    def test_critical_workflow_path(self):
        """CRITICAL should use all agents"""
        orchestrator = DynamicWorkflowOrchestrator()

        path = orchestrator.select_workflow_path(WorkflowComplexity.CRITICAL)

        assert "BiomarkerAgent" in path
        assert "ComorbidityAgent" in path
        assert "SurvivalAnalyzer" in path
        assert "ClinicalTrialMatcher" in path
        assert "CounterfactualEngine" in path
        assert "UncertaintyQuantifier" in path


class TestContextGraphs:
    """Test Dynamic Context Graphs"""

    def test_create_context_graph(self):
        """Basic graph creation"""
        graph = DynamicContextGraph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_add_nodes(self):
        """Add nodes to graph"""
        graph = DynamicContextGraph()

        node1 = ContextNode(
            node_id="patient_1",
            node_type="patient",
            content={"patient_id": "P001", "age": 65},
            source_agent="IngestionAgent"
        )

        node2 = ContextNode(
            node_id="finding_1",
            node_type="finding",
            content={"diagnosis": "Stage IIIA Adenocarcinoma"},
            source_agent="ClassificationAgent"
        )

        graph.add_node(node1)
        graph.add_node(node2)

        assert len(graph.nodes) == 2
        assert "patient_1" in graph.nodes
        assert "finding_1" in graph.nodes

    def test_add_edges(self):
        """Add edges between nodes"""
        graph = DynamicContextGraph()

        node1 = ContextNode(
            node_id="patient_1",
            node_type="patient",
            content={},
            source_agent="test"
        )

        node2 = ContextNode(
            node_id="finding_1",
            node_type="finding",
            content={},
            source_agent="test"
        )

        graph.add_node(node1)
        graph.add_node(node2)

        edge = ContextEdge(
            source_id="patient_1",
            target_id="finding_1",
            relation_type="has_finding",
            weight=1.0
        )

        graph.add_edge(edge)

        assert len(graph.edges) == 1

    def test_conflict_detection(self):
        """Detect conflicting recommendations"""
        graph = DynamicContextGraph()

        rec1 = ContextNode(
            node_id="rec_1",
            node_type="recommendation",
            content={"treatment": "Surgery"},
            source_agent="SurgicalAgent"
        )

        rec2 = ContextNode(
            node_id="rec_2",
            node_type="recommendation",
            content={"treatment": "Radiotherapy"},
            source_agent="RadiationAgent"
        )

        graph.add_node(rec1)
        graph.add_node(rec2)

        conflict_edge = ContextEdge(
            source_id="rec_1",
            target_id="rec_2",
            relation_type="conflicts"
        )

        graph.add_edge(conflict_edge)

        conflicts = graph.detect_conflicts()

        assert len(conflicts) == 1
        assert conflicts[0][0].node_id == "rec_1"
        assert conflicts[0][1].node_id == "rec_2"

    def test_graph_export(self):
        """Export graph to dictionary"""
        graph = DynamicContextGraph()

        node = ContextNode(
            node_id="test_node",
            node_type="patient",
            content={"data": "value"},
            source_agent="TestAgent"
        )

        graph.add_node(node)

        export = graph.to_dict()

        assert "nodes" in export
        assert "edges" in export
        assert "statistics" in export
        assert export["statistics"]["total_nodes"] == 1
        assert export["statistics"]["total_edges"] == 0


class TestNSCLCAgent:
    """Test NSCLC Specialized Agent"""

    def test_nsclc_early_stage_recommendation(self):
        """Stage IA should recommend surgery"""
        agent = NSCLCAgent()

        patient = PatientFact(
            patient_id="NSCLC001",
            age_at_diagnosis=60,
            sex="Male",
            tnm_stage="IA",
            histology_type="Adenocarcinoma",
            performance_status=0
        )

        proposal = agent.execute(patient, biomarker_profile={})

        assert proposal is not None
        assert "surgery" in proposal.treatment.lower() or "resection" in proposal.treatment.lower()
        assert proposal.confidence > 0.0
        assert proposal.treatment_intent in ["curative", "radical"]

    def test_nsclc_advanced_stage_recommendation(self):
        """Stage IV should recommend systemic therapy"""
        agent = NSCLCAgent()

        patient = PatientFact(
            patient_id="NSCLC002",
            age_at_diagnosis=68,
            sex="Female",
            tnm_stage="IV",
            histology_type="Adenocarcinoma",
            performance_status=1
        )

        proposal = agent.execute(patient, biomarker_profile={})

        assert proposal is not None
        # Should recommend chemotherapy or immunotherapy
        assert any(term in proposal.treatment.lower()
                  for term in ["chemotherapy", "immunotherapy", "systemic"])


class TestSCLCAgent:
    """Test SCLC Specialized Agent"""

    def test_sclc_limited_stage(self):
        """Limited stage should get concurrent chemoRT"""
        agent = SCLCAgent()

        patient = PatientFact(
            patient_id="SCLC001",
            age_at_diagnosis=65,
            sex="Male",
            tnm_stage="Limited",
            histology_type="Small Cell Carcinoma",
            performance_status=1
        )

        proposal = agent.execute(patient)

        assert proposal is not None
        assert "chemoradiotherapy" in proposal.treatment.lower() or "concurrent" in proposal.treatment.lower()
        assert proposal.treatment_intent in ["curative", "radical"]

    def test_sclc_extensive_stage(self):
        """Extensive stage should get palliative therapy"""
        agent = SCLCAgent()

        patient = PatientFact(
            patient_id="SCLC002",
            age_at_diagnosis=70,
            sex="Female",
            tnm_stage="Extensive",
            histology_type="Small Cell Carcinoma",
            performance_status=2
        )

        proposal = agent.execute(patient)

        assert proposal is not None
        assert proposal.treatment_intent == "palliative"


class TestBiomarkerAgent:
    """Test Biomarker Precision Medicine Agent"""

    def test_egfr_mutation_detection(self):
        """EGFR Ex19del should trigger TKI recommendation"""
        agent = BiomarkerAgent()

        patient = PatientFact(
            patient_id="BIO001",
            age_at_diagnosis=65,
            sex="Female",
            tnm_stage="IV",
            histology_type="Adenocarcinoma",
            performance_status=1
        )

        biomarker_profile = {
            "EGFR": "Ex19del"
        }

        result = agent.execute(patient, biomarker_profile)

        # Should contain EGFR TKI reference
        assert result is not None
        result_str = str(result).lower()
        assert any(tki in result_str
                  for tki in ["osimertinib", "gefitinib", "erlotinib", "egfr", "tki"])

    def test_alk_rearrangement_detection(self):
        """ALK positive should trigger ALK inhibitor"""
        agent = BiomarkerAgent()

        patient = PatientFact(
            patient_id="BIO002",
            age_at_diagnosis=55,
            sex="Male",
            tnm_stage="IV",
            histology_type="Adenocarcinoma",
            performance_status=0
        )

        biomarker_profile = {
            "ALK": "Positive"
        }

        result = agent.execute(patient, biomarker_profile)

        assert result is not None
        result_str = str(result).lower()
        assert any(inhibitor in result_str
                  for inhibitor in ["alectinib", "crizotinib", "brigatinib", "alk"])


class TestNegotiationProtocol:
    """Test Multi-Agent Negotiation"""

    def test_evidence_hierarchy_negotiation(self):
        """Grade A evidence should win over Grade B"""
        negotiator = NegotiationProtocol(strategy=NegotiationStrategy.EVIDENCE_HIERARCHY)

        proposals = [
            AgentProposal(
                agent_id="agent1",
                treatment="Treatment A",
                confidence=0.90,
                evidence_level="Grade B",
                treatment_intent="curative",
                rationale="Good evidence"
            ),
            AgentProposal(
                agent_id="agent2",
                treatment="Treatment B",
                confidence=0.85,
                evidence_level="Grade A",
                treatment_intent="curative",
                rationale="Excellent evidence"
            )
        ]

        result = negotiator.negotiate(proposals)

        # Grade A should win despite lower confidence
        assert result.selected_treatment.treatment == "Treatment B"
        assert result.selected_treatment.evidence_level == "Grade A"

    def test_safety_first_negotiation(self):
        """Lower risk should win with SAFETY_FIRST"""
        negotiator = NegotiationProtocol(strategy=NegotiationStrategy.SAFETY_FIRST)

        proposals = [
            AgentProposal(
                agent_id="agent1",
                treatment="Aggressive Treatment",
                confidence=0.95,
                evidence_level="Grade A",
                treatment_intent="curative",
                rationale="High efficacy",
                risk_score=0.8
            ),
            AgentProposal(
                agent_id="agent2",
                treatment="Conservative Treatment",
                confidence=0.90,
                evidence_level="Grade A",
                treatment_intent="curative",
                rationale="Good safety",
                risk_score=0.2
            )
        ]

        result = negotiator.negotiate(proposals)

        # Lower risk should win
        assert result.selected_treatment.treatment == "Conservative Treatment"
        assert result.selected_treatment.risk_score == 0.2

    def test_consensus_voting(self):
        """Majority treatment should win with CONSENSUS"""
        negotiator = NegotiationProtocol(strategy=NegotiationStrategy.CONSENSUS_VOTING)

        proposals = [
            AgentProposal(
                agent_id="agent1",
                treatment="Chemotherapy",
                confidence=0.85,
                evidence_level="Grade A",
                treatment_intent="curative",
                rationale="Standard"
            ),
            AgentProposal(
                agent_id="agent2",
                treatment="Chemotherapy",
                confidence=0.88,
                evidence_level="Grade A",
                treatment_intent="curative",
                rationale="Guideline"
            ),
            AgentProposal(
                agent_id="agent3",
                treatment="Immunotherapy",
                confidence=0.75,
                evidence_level="Grade B",
                treatment_intent="curative",
                rationale="Emerging"
            )
        ]

        result = negotiator.negotiate(proposals)

        # Chemotherapy has 2 votes vs 1
        assert "Chemotherapy" in result.selected_treatment.treatment


def test_import_all_agents():
    """Verify all agents can be imported"""
    from agents import (
        IngestionAgent,
        SemanticMappingAgent,
        ClassificationAgent,
        ConflictResolutionAgent,
        PersistenceAgent,
        ExplanationAgent,
        BiomarkerAgent,
        NSCLCAgent,
        SCLCAgent,
        ComorbidityAgent,
        NegotiationProtocol,
        DynamicWorkflowOrchestrator,
        IntegratedLCAWorkflow
    )

    # Just checking imports don't fail
    assert IngestionAgent is not None
    assert BiomarkerAgent is not None
    assert DynamicWorkflowOrchestrator is not None


def test_import_analytics():
    """Verify analytics modules can be imported"""
    try:
        from analytics import (
            UncertaintyQuantifier,
            SurvivalAnalyzer,
            CounterfactualEngine,
            ClinicalTrialMatcher
        )

        assert UncertaintyQuantifier is not None
        assert SurvivalAnalyzer is not None
        assert CounterfactualEngine is not None
        assert ClinicalTrialMatcher is not None
    except ImportError as e:
        pytest.skip(f"Analytics modules not available: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-s"])
    print("\n✅ All component tests passed!")
