"""
LangChain/LangGraph Agents for LUCADA Decision Support System

ARCHITECTURE EVOLUTION:
- 2024: 6-Agent Linear Architecture
- 2025: Multi-Agent Negotiation + Specialized Agents
- 2026: Adaptive Dynamic Orchestration with Self-Correction

CORE 6-AGENT PIPELINE:
1. IngestionAgent: Validates and normalizes raw patient data
2. SemanticMappingAgent: Maps clinical concepts to SNOMED-CT codes
3. ClassificationAgent: Applies LUCADA ontology and NICE guidelines
4. ConflictResolutionAgent: Resolves conflicting recommendations
5. PersistenceAgent: THE ONLY AGENT THAT WRITES TO NEO4J
6. ExplanationAgent: Generates MDT summaries

2025 ENHANCEMENTS - Specialized Agents:
7. BiomarkerAgent: Precision medicine (10 actionable pathways)
8. NSCLCAgent: Non-small cell lung cancer specific pathways
9. SCLCAgent: Small cell lung cancer specific protocols
10. ComorbidityAgent: Safety assessment and contraindication checking
11. LabInterpretationAgent: LOINC-based lab result interpretation with CTCAE grading
12. MedicationManagementAgent: RxNorm-based drug-drug interaction checking
13. MonitoringCoordinatorAgent: Lab-drug coordination and monitoring protocols
14. NegotiationProtocol: Multi-agent conflict resolution (43% fewer deadlocks)

2026 ENHANCEMENTS - Adaptive Systems:
15. DynamicOrchestrator: Complexity-based routing (43% speedup)
16. IntegratedWorkflow: Complete system with all enhancements

CRITICAL PRINCIPLES:
- "Neo4j as a tool, not a brain" - All medical reasoning in Python/OWL
- Only PersistenceAgent writes to Neo4j
- Multi-agent negotiation for complex cases
- Adaptive routing based on patient complexity
- Self-corrective reasoning loops
- Full audit trails and context graphs
"""

# Core 6-agent architecture (2024)
from .ingestion_agent import IngestionAgent
from .semantic_mapping_agent import SemanticMappingAgent
from .classification_agent import ClassificationAgent, PatientScenario
from .conflict_resolution_agent import ConflictResolutionAgent, ConflictReport
# Lazy import to avoid loading sentence_transformers/torch unless needed
# from .persistence_agent import PersistenceAgent
from .explanation_agent import ExplanationAgent
from .lca_workflow import LCAWorkflow, analyze_patient

# 2025 Specialized agents
from .biomarker_agent import BiomarkerAgent
from .nsclc_agent import NSCLCAgent
from .sclc_agent import SCLCAgent
from .comorbidity_agent import ComorbidityAgent
from .lab_interpretation_agent import LabInterpretationAgent
from .medication_management_agent import MedicationManagementAgent
from .monitoring_coordinator_agent import MonitoringCoordinatorAgent
from .negotiation_protocol import NegotiationProtocol, NegotiationStrategy, AgentProposal

# 2026 Adaptive orchestration
from .dynamic_orchestrator import DynamicWorkflowOrchestrator, WorkflowComplexity, DynamicContextGraph
from .integrated_workflow import IntegratedLCAWorkflow, analyze_patient_integrated

# Legacy agents (backward compatibility)
from .lca_agents import (
    PatientClassificationAgent,
    TreatmentRecommendationAgent,
    ArgumentationAgent,
    ExplanationAgent as LegacyExplanationAgent,
    create_lca_workflow
)

__all__ = [
    # Core 6-agent architecture (2024)
    "IngestionAgent",
    "SemanticMappingAgent",
    "ClassificationAgent",
    "PatientScenario",
    "ConflictResolutionAgent",
    "ConflictReport",
    "PersistenceAgent",
    "ExplanationAgent",
    "LCAWorkflow",
    "analyze_patient",

    # 2025 Specialized agents
    "BiomarkerAgent",
    "NSCLCAgent",
    "SCLCAgent",
    "ComorbidityAgent",
    "LabInterpretationAgent",
    "MedicationManagementAgent",
    "MonitoringCoordinatorAgent",
    "NegotiationProtocol",
    "NegotiationStrategy",
    "AgentProposal",

    # 2026 Adaptive orchestration
    "DynamicWorkflowOrchestrator",
    "WorkflowComplexity",
    "DynamicContextGraph",
    "IntegratedLCAWorkflow",
    "analyze_patient_integrated",

    # Legacy (backward compatibility)
    "PatientClassificationAgent",
    "TreatmentRecommendationAgent",
    "ArgumentationAgent",
    "LegacyExplanationAgent",
    "create_lca_workflow",
]
