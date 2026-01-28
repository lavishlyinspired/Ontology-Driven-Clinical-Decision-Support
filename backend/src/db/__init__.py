"""
Database Layer for LUCADA
Includes Neo4j graph database, vector store integration, and data models.

CRITICAL PRINCIPLE from final.md: "Neo4j as a tool, not a brain"
- All medical reasoning happens in Python/OWL
- Neo4j is only for storage and retrieval
- Only PersistenceAgent can use Neo4jWriteTools
"""

from .neo4j_schema import LUCADAGraphDB, Neo4jConfig

# Lazy import for vector store to avoid loading heavy dependencies unless needed
def get_vector_store():
    """Lazy loader for LUCADAVectorStore to defer PyTorch/transformers imports"""
    from .vector_store import LUCADAVectorStore
    return LUCADAVectorStore

# For backwards compatibility
LUCADAVectorStore = None  # Will be imported on demand

# New models per final.md PHASE 3
from .models import (
    PatientFact,
    PatientFactWithCodes,
    ClassificationResult,
    TreatmentRecommendation,
    InferenceRecord,
    WriteReceipt,
    MDTSummary,
    DecisionSupportResponse,
    Sex,
    TNMStage,
    HistologyType,
    PerformanceStatus,
    Laterality,
    InferenceStatus,
    EvidenceLevel,
    TreatmentIntent,
    SimilarPatient
)

# Alias for backward compatibility
Recommendation = TreatmentRecommendation

# New Neo4j tools with strict read/write separation
from .neo4j_tools import (
    Neo4jReadTools,
    Neo4jWriteTools,
    setup_neo4j_schema
)

# 2025 Enhancements: Graph algorithms and analytics
from .graph_algorithms import Neo4jGraphAlgorithms
from .neosemantics_tools import NeosemanticsTools, setup_neosemantics
from .temporal_analyzer import TemporalAnalyzer

__all__ = [
    # Legacy
    "LUCADAGraphDB",
    "Neo4jConfig",
    "LUCADAVectorStore",
    
    # Models
    "PatientFact",
    "PatientFactWithCodes",
    "ClassificationResult",
    "TreatmentRecommendation",
    "Recommendation",  # Alias for backward compatibility
    "SimilarPatient",
    "InferenceRecord",
    "WriteReceipt",
    "MDTSummary",
    "DecisionSupportResponse",
    
    # Enums
    "Sex",
    "TNMStage",
    "HistologyType",
    "PerformanceStatus",
    "Laterality",
    "InferenceStatus",
    "EvidenceLevel",
    "TreatmentIntent",
    
    # Neo4j Tools
    "Neo4jReadTools",
    "Neo4jWriteTools",
    "setup_neo4j_schema",
    
    # 2025 Graph Analytics
    "Neo4jGraphAlgorithms",
    "NeosemanticsTools",
    "setup_neosemantics",
    "TemporalAnalyzer",
]
