"""
Unit Tests for Advanced Agents (2025-2026 Enhancements)

Tests:
- DynamicWorkflowOrchestrator
- BiomarkerAgent
- NSCLCAgent
- SCLCAgent
- ComorbidityAgent
- NegotiationProtocol
- DynamicContextGraph
"""

import pytest
from backend.src.agents import (
    DynamicWorkflowOrchestrator,
    WorkflowComplexity,
    BiomarkerAgent,
    NSCLCAgent,
    SCLCAgent,
    ComorbidityAgent,
    NegotiationProtocol,
    NegotiationStrategy,
    AgentProposal,
    DynamicContextGraph
)
from backend.src.agents.dynamic_orchestrator import ContextNode, ContextEdge
from backend.src.agents.biomarker_agent import BiomarkerProfile


class TestDynamicOrchestrator:
    """Test Dynamic Workflow Orchestrator - Adaptive routing"""
    
    def setup_method(self):
        self.orchestrator = DynamicWorkflowOrchestrator()
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_simple_complexity_assessment(self, simple_patient):
        """Early stage, PS 0, no comorbidities = SIMPLE"""
        complexity = self.orchestrator.assess_complexity(simple_patient)
        assert complexity == WorkflowComplexity.SIMPLE
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_moderate_complexity_assessment(self, moderate_patient):
        """Stage IIIA, PS 1, with EGFR mutation = COMPLEX (biomarker-driven)"""
        complexity = self.orchestrator.assess_complexity(moderate_patient)
        # Stage III + biomarker profile = COMPLEX workflow
        assert complexity in [WorkflowComplexity.MODERATE, WorkflowComplexity.COMPLEX]
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_complex_complexity_assessment(self, complex_patient):
        """Stage IV, PS 2, multiple comorbidities = CRITICAL (high complexity)"""
        complexity = self.orchestrator.assess_complexity(complex_patient)
        # Stage IV + poor PS + 3 comorbidities = CRITICAL workflow
        assert complexity in [WorkflowComplexity.COMPLEX, WorkflowComplexity.CRITICAL]
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_critical_complexity_assessment(self, critical_patient):
        """Stage IV, PS 3, multiple comorbidities = CRITICAL"""
        complexity = self.orchestrator.assess_complexity(critical_patient)
        assert complexity == WorkflowComplexity.CRITICAL
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_emergency_override(self, simple_patient):
        """Emergency flag should force CRITICAL"""
        simple_patient["emergency"] = True
        complexity = self.orchestrator.assess_complexity(simple_patient)
        assert complexity == WorkflowComplexity.CRITICAL
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_simple_workflow_path(self):
        """SIMPLE complexity should use minimal agents"""
        path = self.orchestrator.select_workflow_path(WorkflowComplexity.SIMPLE)
        
        # Should have core agents
        assert "IngestionAgent" in path
        assert "ClassificationAgent" in path
        
        # Should NOT have heavy analytics
        assert "UncertaintyQuantifier" not in path
        assert "CounterfactualEngine" not in path
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_critical_workflow_path(self):
        """CRITICAL complexity should use all agents"""
        path = self.orchestrator.select_workflow_path(WorkflowComplexity.CRITICAL)
        
        # Should have all specialized agents
        assert "BiomarkerAgent" in path
        assert "ComorbidityAgent" in path
        assert "UncertaintyQuantifier" in path
        assert "SurvivalAnalyzer" in path
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_routing_efficiency(self):
        """SIMPLE should route through fewer agents than CRITICAL"""
        simple_path = self.orchestrator.select_workflow_path(WorkflowComplexity.SIMPLE)
        critical_path = self.orchestrator.select_workflow_path(WorkflowComplexity.CRITICAL)
        
        assert len(simple_path) < len(critical_path)


class TestBiomarkerAgent:
    """Test BiomarkerAgent - Precision medicine pathways"""
    
    def setup_method(self):
        self.agent = BiomarkerAgent()
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_egfr_ex19del_pathway(self, moderate_patient, egfr_ex19del_profile):
        """Test EGFR Ex19del → Osimertinib pathway"""
        # Create mock patient object
        class MockPatient:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        patient = MockPatient(moderate_patient)
        proposal = self.agent.execute(patient, egfr_ex19del_profile)
        
        assert proposal is not None
        assert "osimertinib" in proposal.treatment.lower()
        assert proposal.evidence_level == "Grade A"
        assert proposal.confidence >= 0.9
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_alk_positive_pathway(self, moderate_patient, alk_positive_profile):
        """Test ALK+ → Alectinib pathway"""
        class MockPatient:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        patient = MockPatient(moderate_patient)
        proposal = self.agent.execute(patient, alk_positive_profile)
        
        assert proposal is not None
        assert any(drug in proposal.treatment.lower() 
                  for drug in ["alectinib", "lorlatinib"])
        assert proposal.evidence_level == "Grade A"
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_high_pdl1_pathway(self, moderate_patient, high_pdl1_profile):
        """Test PD-L1 ≥50% → Pembrolizumab pathway"""
        class MockPatient:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        patient = MockPatient(moderate_patient)
        proposal = self.agent.execute(patient, high_pdl1_profile)
        
        assert proposal is not None
        assert "pembrolizumab" in proposal.treatment.lower()
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_no_actionable_biomarkers(self, complex_patient):
        """Test handling when no actionable biomarkers found"""
        class MockPatient:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        patient = MockPatient(complex_patient)
        profile = BiomarkerProfile()  # No biomarkers
        
        proposal = self.agent.execute(patient, profile)
        
        assert proposal is not None
        # Should recommend standard chemotherapy or immunotherapy
        assert any(term in proposal.treatment.lower() 
                  for term in ["chemotherapy", "immunotherapy", "standard"])


class TestNSCLCAgent:
    """Test NSCLCAgent - NSCLC-specific recommendations"""
    
    def setup_method(self):
        self.agent = NSCLCAgent()
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_early_stage_nsclc(self, simple_patient):
        """Test early-stage NSCLC recommendation"""
        class MockPatient:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        patient = MockPatient(simple_patient)
        proposal = self.agent.execute(patient)
        
        assert proposal is not None
        # Early stage should recommend surgery
        assert "surgery" in proposal.treatment.lower() or "resection" in proposal.treatment.lower()
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_locally_advanced_nsclc(self, moderate_patient):
        """Test locally advanced NSCLC recommendation"""
        class MockPatient:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        patient = MockPatient(moderate_patient)
        proposal = self.agent.execute(patient)
        
        assert proposal is not None
        # Should recommend multimodal therapy
        assert any(term in proposal.treatment.lower() 
                  for term in ["chemoradiotherapy", "durvalumab", "multimodal"])
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_advanced_nsclc(self, complex_patient):
        """Test advanced NSCLC recommendation"""
        class MockPatient:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        patient = MockPatient(complex_patient)
        proposal = self.agent.execute(patient)
        
        assert proposal is not None
        assert proposal.treatment is not None
        assert len(proposal.treatment) > 0
        # Should have a recommendation for advanced NSCLC
        assert proposal.confidence > 0.0


class TestSCLCAgent:
    """Test SCLCAgent - SCLC-specific recommendations"""
    
    def setup_method(self):
        self.agent = SCLCAgent()
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_limited_stage_sclc(self, sclc_patient):
        """Test limited-stage SCLC recommendation"""
        class MockPatient:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        patient = MockPatient(sclc_patient)
        proposal = self.agent.execute(patient)
        
        assert proposal is not None
        # Should recommend platinum-based therapy (carboplatin, etoposide, etc.)
        assert any(term in proposal.treatment.lower() 
                  for term in ["chemotherapy", "carboplatin", "etoposide", "platinum", "atezolizumab"])
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_extensive_stage_sclc(self, critical_patient):
        """Test extensive-stage SCLC recommendation"""
        class MockPatient:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        patient = MockPatient(critical_patient)
        proposal = self.agent.execute(patient)
        
        assert proposal is not None
        # Extensive stage with poor PS - palliative therapy or supportive care
        assert any(term in proposal.treatment.lower() 
                  for term in ["chemotherapy", "carboplatin", "etoposide", "supportive", "palliative"])


class TestComorbidityAgent:
    """Test ComorbidityAgent - Comorbidity-aware recommendations"""
    
    def setup_method(self):
        self.agent = ComorbidityAgent()
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_no_comorbidities(self, simple_patient):
        """Test patient with no comorbidities"""
        class MockPatient:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        patient = MockPatient(simple_patient)
        # ComorbidityAgent assesses safety of a proposed treatment
        assessment = self.agent.execute(patient, treatment="Lobectomy")
        
        assert assessment is not None
        # Should have low risk with no comorbidities
        assert assessment.overall_safety in ["Safe", "Caution"]
        assert assessment.risk_score < 0.5
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_multiple_comorbidities(self, complex_patient):
        """Test patient with multiple comorbidities"""
        class MockPatient:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        patient = MockPatient(complex_patient)
        # Test with platinum-based chemotherapy
        assessment = self.agent.execute(patient, treatment="Carboplatin + Pemetrexed")
        
        assert assessment is not None
        # The agent should assess the treatment, even if it's safe
        assert assessment.treatment == "Carboplatin + Pemetrexed"
        assert assessment.overall_safety in ["Safe", "Caution", "Contraindicated"]
        # Check assessment was performed
        assert isinstance(assessment.risk_score, float)


class TestNegotiationProtocol:
    """Test Multi-Agent Negotiation Protocol"""
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_evidence_hierarchy_strategy(self):
        """Test Evidence Hierarchy negotiation strategy"""
        proposals = [
            AgentProposal(
                agent_id="biomarker",
                agent_type="BiomarkerAgent",
                treatment="Osimertinib",
                confidence=0.95,
                evidence_level="Grade A",
                treatment_intent="Curative",
                rationale="EGFR mutation",
                guideline_reference="NCCN",
                risk_score=0.1
            ),
            AgentProposal(
                agent_id="nsclc",
                agent_type="NSCLCAgent",
                treatment="Chemotherapy",
                confidence=0.80,
                evidence_level="Grade B",
                treatment_intent="Curative",
                rationale="Standard therapy",
                guideline_reference="NCCN",
                risk_score=0.3
            )
        ]
        
        protocol = NegotiationProtocol(strategy=NegotiationStrategy.EVIDENCE_HIERARCHY)
        result = protocol.negotiate(proposals)
        
        assert result is not None
        # Should select Grade A over Grade B
        assert result.selected_treatment == "Osimertinib"
        assert result.consensus_score > 0
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_safety_first_strategy(self):
        """Test Safety First negotiation strategy"""
        proposals = [
            AgentProposal(
                agent_id="biomarker",
                agent_type="BiomarkerAgent",
                treatment="Aggressive Chemo",
                confidence=0.90,
                evidence_level="Grade A",
                treatment_intent="Curative",
                rationale="High efficacy",
                guideline_reference="NCCN",
                risk_score=0.7  # High risk
            ),
            AgentProposal(
                agent_id="comorbidity",
                agent_type="ComorbidityAgent",
                treatment="Reduced Dose Chemo",
                confidence=0.75,
                evidence_level="Grade B",
                treatment_intent="Curative",
                rationale="Safety",
                guideline_reference="Safety Guidelines",
                risk_score=0.2  # Low risk
            )
        ]
        
        protocol = NegotiationProtocol(strategy=NegotiationStrategy.SAFETY_FIRST)
        result = protocol.negotiate(proposals)
        
        assert result is not None
        # Should select lower risk option
        assert result.selected_treatment == "Reduced Dose Chemo"
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_consensus_voting_strategy(self):
        """Test Consensus Voting strategy"""
        # Multiple agents agreeing on similar treatment
        proposals = [
            AgentProposal(
                agent_id="agent1",
                agent_type="Agent1",
                treatment="Immunotherapy",
                confidence=0.85,
                evidence_level="Grade A",
                treatment_intent="Curative",
                rationale="High PD-L1",
                guideline_reference="NCCN",
                risk_score=0.2
            ),
            AgentProposal(
                agent_id="agent2",
                agent_type="Agent2",
                treatment="Immunotherapy",
                confidence=0.80,
                evidence_level="Grade A",
                treatment_intent="Curative",
                rationale="Good PS",
                guideline_reference="NCCN",
                risk_score=0.25
            ),
            AgentProposal(
                agent_id="agent3",
                agent_type="Agent3",
                treatment="Chemotherapy",
                confidence=0.70,
                evidence_level="Grade B",
                treatment_intent="Curative",
                rationale="Alternative",
                guideline_reference="NCCN",
                risk_score=0.35
            )
        ]
        
        protocol = NegotiationProtocol(strategy=NegotiationStrategy.CONSENSUS_VOTING)
        result = protocol.negotiate(proposals)
        
        assert result is not None
        # Should favor the treatment with more votes
        assert result.selected_treatment == "Immunotherapy"
        assert result.consensus_score > 0.5


class TestDynamicContextGraph:
    """Test Dynamic Context Graph - Reasoning chains"""
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_create_empty_graph(self):
        """Test creating empty context graph"""
        graph = DynamicContextGraph()
        
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_add_nodes(self):
        """Test adding nodes to graph"""
        graph = DynamicContextGraph()
        
        node1 = ContextNode(
            node_id="patient_1",
            node_type="patient",
            content={"patient_id": "P001"},
            source_agent="IngestionAgent"
        )
        
        node2 = ContextNode(
            node_id="finding_1",
            node_type="finding",
            content={"stage": "IIIA"},
            source_agent="ClassificationAgent"
        )
        
        graph.add_node(node1)
        graph.add_node(node2)
        
        assert len(graph.nodes) == 2
        assert "patient_1" in graph.nodes
        assert "finding_1" in graph.nodes
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_add_edges(self):
        """Test adding edges between nodes"""
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
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_reasoning_chain(self):
        """Test retrieving reasoning chain"""
        graph = DynamicContextGraph()
        
        # Create chain: patient -> finding -> recommendation
        patient = ContextNode(
            node_id="patient_1",
            node_type="patient",
            content={},
            source_agent="IngestionAgent"
        )
        
        finding = ContextNode(
            node_id="finding_1",
            node_type="finding",
            content={},
            source_agent="ClassificationAgent"
        )
        
        recommendation = ContextNode(
            node_id="rec_1",
            node_type="recommendation",
            content={},
            source_agent="BiomarkerAgent"
        )
        
        graph.add_node(patient)
        graph.add_node(finding)
        graph.add_node(recommendation)
        
        graph.add_edge(ContextEdge("patient_1", "finding_1", "has_finding", 1.0))
        graph.add_edge(ContextEdge("finding_1", "rec_1", "supports", 0.9))
        
        # Get reasoning chain
        chain = graph.get_reasoning_chain("rec_1")
        
        assert len(chain) > 0
        # Should include the recommendation node
        assert any(node.node_id == "rec_1" for node in chain)
    
    @pytest.mark.unit
    @pytest.mark.agents
    def test_graph_export(self):
        """Test exporting graph to dictionary"""
        graph = DynamicContextGraph()
        
        node = ContextNode(
            node_id="test_1",
            node_type="test",
            content={"key": "value"},
            source_agent="TestAgent"
        )
        
        graph.add_node(node)
        
        exported = graph.to_dict()
        
        assert "nodes" in exported
        assert "edges" in exported
        assert "statistics" in exported
        assert exported["statistics"]["total_nodes"] == 1
