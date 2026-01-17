"""
LangChain/LangGraph Agents for LUCADA Decision Support System

6-Agent Architecture per final.md specification:
1. IngestionAgent: Validates and normalizes raw patient data
2. SemanticMappingAgent: Maps clinical concepts to SNOMED-CT codes
3. ClassificationAgent: Applies LUCADA ontology and NICE guidelines
4. ConflictResolutionAgent: Resolves conflicting recommendations
5. PersistenceAgent: THE ONLY AGENT THAT WRITES TO NEO4J
6. ExplanationAgent: Generates MDT summaries

CRITICAL PRINCIPLE: "Neo4j as a tool, not a brain"
- All medical reasoning happens in Python/OWL
- Neo4j is only for storage and retrieval
- Only PersistenceAgent writes to Neo4j
"""

# New 6-agent architecture
from .ingestion_agent import IngestionAgent
from .semantic_mapping_agent import SemanticMappingAgent
from .classification_agent import ClassificationAgent, PatientScenario
from .conflict_resolution_agent import ConflictResolutionAgent, ConflictReport
from .persistence_agent import PersistenceAgent
from .explanation_agent import ExplanationAgent
from .lca_workflow import LCAWorkflow, analyze_patient

# Legacy agents (for backward compatibility)
from .lca_agents import (
    PatientClassificationAgent,
    TreatmentRecommendationAgent,
    ArgumentationAgent,
    ExplanationAgent as LegacyExplanationAgent,
    create_lca_workflow
)

__all__ = [
    # New 6-agent architecture
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
    
    # Legacy (backward compatibility)
    "PatientClassificationAgent",
    "TreatmentRecommendationAgent",
    "ArgumentationAgent",
    "LegacyExplanationAgent",
    "create_lca_workflow",
]
