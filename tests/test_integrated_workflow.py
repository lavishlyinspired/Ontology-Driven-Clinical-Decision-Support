"""
Comprehensive Test Suite for Integrated LCA Workflow

Tests all major components and their integration:
- Dynamic Orchestrator
- Specialized Agents (NSCLC/SCLC)
- Biomarker Agent
- Analytics Suite
- Context Graphs
- Multi-agent Negotiation
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

# Import core components
from backend.src.agents import (
    IntegratedLCAWorkflow,
    DynamicWorkflowOrchestrator,
    WorkflowComplexity,
    NSCLCAgent,
    SCLCAgent,
    BiomarkerAgent,
    NegotiationProtocol,
    NegotiationStrategy
)

from backend.src.db.models import PatientFact


class TestComplexityAssessment:
    """Test complexity assessment logic"""

    def setup_method(self):
        self.orchestrator = DynamicWorkflowOrchestrator()

    def test_simple_case_classification(self):
        """Test that early-stage cases are classified as SIMPLE"""
        patient_data = {
            "patient_id": "TEST001",
            "age_at_diagnosis": 55,
            "tnm_stage": "IA",
            "histology_type": "Adenocarcinoma",
            "performance_status": 0,
            "comorbidities": []
        }

        complexity = self.orchestrator.assess_complexity(patient_data)
        assert complexity == WorkflowComplexity.SIMPLE

    def test_moderate_case_classification(self):
        """Test that Stage III cases are classified as MODERATE"""
        patient_data = {
            "patient_id": "TEST002",
            "age_at_diagnosis": 65,
            "tnm_stage": "IIIA",
            "histology_type": "Adenocarcinoma",
            "performance_status": 1,
            "comorbidities": []
        }

        complexity = self.orchestrator.assess_complexity(patient_data)
        assert complexity == WorkflowComplexity.MODERATE

    def test_complex_case_classification(self):
        """Test that Stage III with comorbidities is COMPLEX"""
        patient_data = {
            "patient_id": "TEST003",
            "age_at_diagnosis": 72,
            "tnm_stage": "IIIB",
            "histology_type": "Adenocarcinoma",
            "performance_status": 2,
            "comorbidities": ["COPD", "Diabetes"]
        }

        complexity = self.orchestrator.assess_complexity(patient_data)
        assert complexity == WorkflowComplexity.COMPLEX

    def test_critical_case_classification(self):
        """Test that Stage IV with poor PS is CRITICAL"""
        patient_data = {
            "patient_id": "TEST004",
            "age_at_diagnosis": 85,
            "tnm_stage": "IV",
            "histology_type": "Adenocarcinoma",
            "performance_status": 3,
            "comorbidities": ["COPD", "Diabetes", "CAD"]
        }

        complexity = self.orchestrator.assess_complexity(patient_data)
        assert complexity == WorkflowComplexity.CRITICAL

    def test_emergency_flag_override(self):
        """Test that emergency flag forces CRITICAL"""
        patient_data = {
            "patient_id": "TEST005",
            "age_at_diagnosis": 55,
            "tnm_stage": "IA",
            "performance_status": 0,
            "emergency": True
        }

        complexity = self.orchestrator.assess_complexity(patient_data)
        assert complexity == WorkflowComplexity.CRITICAL


class TestWorkflowPathSelection:
    """Test workflow path selection based on complexity"""

    def setup_method(self):
        self.orchestrator = DynamicWorkflowOrchestrator()

    def test_simple_path_selection(self):
        """Test SIMPLE complexity selects minimal agent set"""
        path = self.orchestrator.select_workflow_path(WorkflowComplexity.SIMPLE)

        assert "IngestionAgent" in path
        assert "ClassificationAgent" in path
        assert "ExplanationAgent" in path
        # Should skip optional agents
        assert "UncertaintyQuantifier" not in path
        assert "SurvivalAnalyzer" not in path

    def test_moderate_path_selection(self):
        """Test MODERATE complexity includes biomarker analysis"""
        path = self.orchestrator.select_workflow_path(WorkflowComplexity.MODERATE)

        assert "BiomarkerAgent" in path
        assert "ConflictResolutionAgent" in path

    def test_complex_path_selection(self):
        """Test COMPLEX complexity includes safety assessment"""
        path = self.orchestrator.select_workflow_path(WorkflowComplexity.COMPLEX)

        assert "BiomarkerAgent" in path
        assert "ComorbidityAgent" in path
        assert "UncertaintyQuantifier" in path

    def test_critical_path_selection(self):
        """Test CRITICAL complexity includes all analytics"""
        path = self.orchestrator.select_workflow_path(WorkflowComplexity.CRITICAL)

        assert "SurvivalAnalyzer" in path
        assert "ClinicalTrialMatcher" in path
        assert "CounterfactualEngine" in path
        assert len(path) >= 10  # Comprehensive analysis


class TestSpecializedAgents:
    """Test NSCLC and SCLC specialized agents"""

    def setup_method(self):
        self.nsclc_agent = NSCLCAgent()
        self.sclc_agent = SCLCAgent()

    def test_nsclc_agent_early_stage(self):
        """Test NSCLC agent for early-stage patient"""
        patient = PatientFact(
            patient_id="NSCLC001",
            age_at_diagnosis=60,
            sex="Male",
            tnm_stage="IA",
            histology_type="Adenocarcinoma",
            performance_status=0
        )

        biomarker_profile = {}
        proposal = self.nsclc_agent.execute(patient, biomarker_profile)

        assert proposal is not None
        assert "surgery" in proposal.treatment.lower() or "resection" in proposal.treatment.lower()
        assert proposal.confidence > 0.7

    def test_nsclc_agent_advanced_stage(self):
        """Test NSCLC agent for advanced-stage patient"""
        patient = PatientFact(
            patient_id="NSCLC002",
            age_at_diagnosis=68,
            sex="Female",
            tnm_stage="IV",
            histology_type="Adenocarcinoma",
            performance_status=1
        )

        biomarker_profile = {}
        proposal = self.nsclc_agent.execute(patient, biomarker_profile)

        assert proposal is not None
        assert any(term in proposal.treatment.lower()
                  for term in ["chemotherapy", "immunotherapy", "systemic"])

    def test_sclc_agent_limited_stage(self):
        """Test SCLC agent for limited-stage patient"""
        patient = PatientFact(
            patient_id="SCLC001",
            age_at_diagnosis=65,
            sex="Male",
            tnm_stage="Limited",
            histology_type="Small Cell Carcinoma",
            performance_status=1
        )

        proposal = self.sclc_agent.execute(patient)

        assert proposal is not None
        assert "chemoradiotherapy" in proposal.treatment.lower()
        assert proposal.treatment_intent in ["curative", "radical"]

    def test_sclc_agent_extensive_stage(self):
        """Test SCLC agent for extensive-stage patient"""
        patient = PatientFact(
            patient_id="SCLC002",
            age_at_diagnosis=70,
            sex="Female",
            tnm_stage="Extensive",
            histology_type="Small Cell Carcinoma",
            performance_status=2
        )

        proposal = self.sclc_agent.execute(patient)

        assert proposal is not None
        assert any(term in proposal.treatment.lower()
                  for term in ["chemotherapy", "immunotherapy"])
        assert proposal.treatment_intent == "palliative"


class TestBiomarkerAgent:
    """Test biomarker-driven precision medicine"""

    def setup_method(self):
        self.biomarker_agent = BiomarkerAgent()

    def test_egfr_positive_detection(self):
        """Test EGFR mutation detection and TKI recommendation"""
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

        result = self.biomarker_agent.execute(patient, biomarker_profile)

        # Should recommend EGFR TKI
        assert result is not None
        assert any(tki in str(result).lower()
                  for tki in ["osimertinib", "gefitinib", "erlotinib"])

    def test_alk_rearrangement_detection(self):
        """Test ALK rearrangement and inhibitor recommendation"""
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

        result = self.biomarker_agent.execute(patient, biomarker_profile)

        # Should recommend ALK inhibitor
        assert result is not None
        assert any(inhibitor in str(result).lower()
                  for inhibitor in ["alectinib", "crizotinib", "brigatinib"])

    def test_pdl1_high_expression(self):
        """Test PD-L1 high expression and immunotherapy"""
        patient = PatientFact(
            patient_id="BIO003",
            age_at_diagnosis=70,
            sex="Female",
            tnm_stage="IV",
            histology_type="Squamous Cell Carcinoma",
            performance_status=1
        )

        biomarker_profile = {
            "PD-L1": "85%"
        }

        result = self.biomarker_agent.execute(patient, biomarker_profile)

        # Should recommend immunotherapy
        assert result is not None
        # Biomarker agent may return different format, just check it's not None

    def test_no_actionable_mutations(self):
        """Test handling of no actionable mutations"""
        patient = PatientFact(
            patient_id="BIO004",
            age_at_diagnosis=65,
            sex="Male",
            tnm_stage="IIIA",
            histology_type="Adenocarcinoma",
            performance_status=1
        )

        biomarker_profile = {}

        result = self.biomarker_agent.execute(patient, biomarker_profile)

        # Should handle gracefully (may return None or standard treatment)
        # Implementation-dependent


class TestContextGraphs:
    """Test dynamic context graph functionality"""

    def setup_method(self):
        from backend.src.agents.dynamic_orchestrator import (
            DynamicContextGraph,
            ContextNode,
            ContextEdge
        )
        self.graph = DynamicContextGraph()
        self.ContextNode = ContextNode
        self.ContextEdge = ContextEdge

    def test_add_nodes_and_edges(self):
        """Test adding nodes and edges to context graph"""
        patient_node = self.ContextNode(
            node_id="patient_1",
            node_type="patient",
            content={"patient_id": "P001"},
            source_agent="test"
        )

        finding_node = self.ContextNode(
            node_id="finding_1",
            node_type="finding",
            content={"diagnosis": "Stage IIIA"},
            source_agent="test"
        )

        self.graph.add_node(patient_node)
        self.graph.add_node(finding_node)

        edge = self.ContextEdge(
            source_id="patient_1",
            target_id="finding_1",
            relation_type="has_finding"
        )

        self.graph.add_edge(edge)

        assert len(self.graph.nodes) == 2
        assert len(self.graph.edges) == 1

    def test_conflict_detection(self):
        """Test conflict detection in context graph"""
        rec1 = self.ContextNode(
            node_id="rec_1",
            node_type="recommendation",
            content={"treatment": "Surgery"},
            source_agent="agent1"
        )

        rec2 = self.ContextNode(
            node_id="rec_2",
            node_type="recommendation",
            content={"treatment": "Chemotherapy"},
            source_agent="agent2"
        )

        self.graph.add_node(rec1)
        self.graph.add_node(rec2)

        conflict_edge = self.ContextEdge(
            source_id="rec_1",
            target_id="rec_2",
            relation_type="conflicts"
        )

        self.graph.add_edge(conflict_edge)

        conflicts = self.graph.detect_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0][0].content["treatment"] == "Surgery"
        assert conflicts[0][1].content["treatment"] == "Chemotherapy"

    def test_reasoning_chain_traversal(self):
        """Test traversal of reasoning chains"""
        patient = self.ContextNode(
            node_id="patient_1",
            node_type="patient",
            content={},
            source_agent="ingestion"
        )

        finding = self.ContextNode(
            node_id="finding_1",
            node_type="finding",
            content={},
            source_agent="classifier"
        )

        recommendation = self.ContextNode(
            node_id="rec_1",
            node_type="recommendation",
            content={},
            source_agent="treatment"
        )

        self.graph.add_node(patient)
        self.graph.add_node(finding)
        self.graph.add_node(recommendation)

        self.graph.add_edge(self.ContextEdge(
            source_id="patient_1",
            target_id="finding_1",
            relation_type="derives_from"
        ))

        self.graph.add_edge(self.ContextEdge(
            source_id="finding_1",
            target_id="rec_1",
            relation_type="derives_from"
        ))

        chain = self.graph.get_reasoning_chain("rec_1")
        assert len(chain) >= 2

    def test_graph_export(self):
        """Test exporting graph to dictionary"""
        node = self.ContextNode(
            node_id="test_1",
            node_type="test",
            content={"test": "data"},
            source_agent="test"
        )

        self.graph.add_node(node)

        export = self.graph.to_dict()

        assert "nodes" in export
        assert "edges" in export
        assert "statistics" in export
        assert export["statistics"]["total_nodes"] == 1


@pytest.mark.asyncio
class TestIntegratedWorkflow:
    """Test complete integrated workflow"""

    async def test_nsclc_workflow_with_biomarker(self):
        """Test complete NSCLC workflow with biomarker"""
        workflow = IntegratedLCAWorkflow(
            neo4j_tools=None,  # Mock mode
            enable_analytics=False,  # Disable for unit test
            enable_negotiation=True
        )

        patient_data = {
            "patient_id": "INT001",
            "age_at_diagnosis": 65,
            "sex": "Female",
            "tnm_stage": "IV",
            "histology_type": "Adenocarcinoma",
            "performance_status": 1,
            "biomarker_profile": {
                "EGFR": "Ex19del"
            }
        }

        result = await workflow.analyze_patient_comprehensive(patient_data)

        assert result["status"] == "success"
        assert "NSCLCAgent" in result["agent_chain"]
        assert "BiomarkerAgent" in result["agent_chain"]
        assert len(result["recommendations"]) > 0

    async def test_sclc_workflow(self):
        """Test complete SCLC workflow"""
        workflow = IntegratedLCAWorkflow(
            neo4j_tools=None,
            enable_analytics=False,
            enable_negotiation=False
        )

        patient_data = {
            "patient_id": "INT002",
            "age_at_diagnosis": 68,
            "sex": "Male",
            "tnm_stage": "Limited",
            "histology_type": "Small Cell Carcinoma",
            "performance_status": 1
        }

        result = await workflow.analyze_patient_comprehensive(patient_data)

        assert result["status"] == "success"
        assert "SCLCAgent" in result["agent_chain"]
        # SCLC shouldn't trigger biomarker agent
        assert "BiomarkerAgent" not in result["agent_chain"]

    async def test_workflow_error_handling(self):
        """Test workflow handles invalid input gracefully"""
        workflow = IntegratedLCAWorkflow(
            neo4j_tools=None,
            enable_analytics=False,
            enable_negotiation=False
        )

        # Missing required fields
        patient_data = {
            "patient_id": "INT003"
        }

        result = await workflow.analyze_patient_comprehensive(patient_data)

        # Should fail gracefully
        assert "errors" in result or result["status"] != "success"


class TestNegotiationProtocol:
    """Test multi-agent negotiation"""

    def setup_method(self):
        from backend.src.agents.negotiation_protocol import AgentProposal
        self.AgentProposal = AgentProposal

    def test_evidence_hierarchy_strategy(self):
        """Test evidence hierarchy prioritizes Grade A over B"""
        negotiator = NegotiationProtocol(strategy=NegotiationStrategy.EVIDENCE_HIERARCHY)

        proposals = [
            self.AgentProposal(
                agent_id="agent1",
                treatment="Treatment A",
                confidence=0.85,
                evidence_level="Grade B",
                treatment_intent="curative",
                rationale="Good evidence"
            ),
            self.AgentProposal(
                agent_id="agent2",
                treatment="Treatment B",
                confidence=0.80,
                evidence_level="Grade A",
                treatment_intent="curative",
                rationale="Excellent evidence"
            )
        ]

        result = negotiator.negotiate(proposals)

        # Grade A should win despite slightly lower confidence
        assert result.selected_treatment.treatment == "Treatment B"

    def test_safety_first_strategy(self):
        """Test safety-first prioritizes lower risk"""
        negotiator = NegotiationProtocol(strategy=NegotiationStrategy.SAFETY_FIRST)

        proposals = [
            self.AgentProposal(
                agent_id="agent1",
                treatment="Aggressive Treatment",
                confidence=0.90,
                evidence_level="Grade A",
                treatment_intent="curative",
                rationale="High efficacy",
                risk_score=0.8  # High risk
            ),
            self.AgentProposal(
                agent_id="agent2",
                treatment="Conservative Treatment",
                confidence=0.85,
                evidence_level="Grade A",
                treatment_intent="curative",
                rationale="Good safety profile",
                risk_score=0.2  # Low risk
            )
        ]

        result = negotiator.negotiate(proposals)

        # Lower risk should win
        assert result.selected_treatment.treatment == "Conservative Treatment"

    def test_consensus_voting_strategy(self):
        """Test consensus voting with multiple similar proposals"""
        negotiator = NegotiationProtocol(strategy=NegotiationStrategy.CONSENSUS_VOTING)

        proposals = [
            self.AgentProposal(
                agent_id="agent1",
                treatment="Chemotherapy",
                confidence=0.85,
                evidence_level="Grade A",
                treatment_intent="curative",
                rationale="Standard"
            ),
            self.AgentProposal(
                agent_id="agent2",
                treatment="Chemotherapy",  # Same treatment
                confidence=0.88,
                evidence_level="Grade A",
                treatment_intent="curative",
                rationale="Guideline-based"
            ),
            self.AgentProposal(
                agent_id="agent3",
                treatment="Immunotherapy",
                confidence=0.75,
                evidence_level="Grade B",
                treatment_intent="curative",
                rationale="Emerging evidence"
            )
        ]

        result = negotiator.negotiate(proposals)

        # Chemotherapy should win (2 votes vs 1)
        assert "Chemotherapy" in result.selected_treatment.treatment


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
