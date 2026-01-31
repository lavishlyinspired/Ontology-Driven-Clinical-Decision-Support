"""
Graph Algorithms API Routes
Provides endpoints for Neo4j Graph Data Science algorithms
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph-algorithms", tags=["graph-algorithms"])

# Lazy load graph algorithms to avoid startup issues
_graph_algorithms = None

def get_graph_algorithms():
    """Get or create graph algorithms instance"""
    global _graph_algorithms
    if _graph_algorithms is None:
        try:
            from ...db.graph_algorithms import Neo4jGraphAlgorithms
            _graph_algorithms = Neo4jGraphAlgorithms()
            logger.info(f"Graph algorithms initialized: available={_graph_algorithms.is_available}")
        except Exception as e:
            logger.warning(f"Graph algorithms not available: {e}")
    return _graph_algorithms


# =============================================================================
# Response Models
# =============================================================================

class SimilarPatientResponse(BaseModel):
    patient_id: str
    similarity_score: float
    shared_features: Dict[str, Any] = {}


class CommunityResponse(BaseModel):
    community_id: int
    members: List[str]
    size: int
    treatments: List[str] = []


class TreatmentPathResponse(BaseModel):
    path: List[str]
    outcome: str
    success_rate: Optional[float] = None
    patient_count: int = 0


class InfluentialTreatmentResponse(BaseModel):
    treatment: str
    score: float
    patient_count: int
    success_rate: Optional[float] = None


class GraphAlgorithmsStatusResponse(BaseModel):
    neo4j_available: bool
    gds_available: bool
    algorithms: List[str]


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/status", response_model=GraphAlgorithmsStatusResponse)
async def get_graph_algorithms_status():
    """
    Get status of Neo4j graph algorithms availability
    """
    ga = get_graph_algorithms()

    if ga is None:
        return GraphAlgorithmsStatusResponse(
            neo4j_available=False,
            gds_available=False,
            algorithms=[]
        )

    algorithms = []
    if ga.is_available:
        algorithms = [
            "node_similarity",
            "louvain_community_detection",
            "shortest_path",
            "degree_centrality",
            "vector_similarity"
        ]

    return GraphAlgorithmsStatusResponse(
        neo4j_available=ga.is_available,
        gds_available=ga._gds_available if hasattr(ga, '_gds_available') else False,
        algorithms=algorithms
    )


@router.get("/similar-patients/{patient_id}", response_model=List[SimilarPatientResponse])
async def find_similar_patients(
    patient_id: str,
    k: int = Query(default=10, ge=1, le=50, description="Number of similar patients to return"),
    min_similarity: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum similarity threshold")
):
    """
    Find similar patients using graph-based similarity algorithm.

    Uses Neo4j GDS Node Similarity algorithm when available,
    otherwise falls back to custom Cypher-based similarity.
    """
    ga = get_graph_algorithms()

    if ga is None or not ga.is_available:
        raise HTTPException(
            status_code=503,
            detail="Graph algorithms not available. Check Neo4j connection."
        )

    try:
        similar_patients = ga.find_similar_patients_graph_based(
            patient_id=patient_id,
            k=k,
            min_similarity=min_similarity
        )

        return [
            SimilarPatientResponse(
                patient_id=p.patient_id if hasattr(p, 'patient_id') else p.get('patient_id', ''),
                similarity_score=p.similarity_score if hasattr(p, 'similarity_score') else p.get('similarity_score', 0),
                shared_features=p.shared_features if hasattr(p, 'shared_features') else p.get('shared_features', {})
            )
            for p in similar_patients
        ]
    except Exception as e:
        logger.error(f"Error finding similar patients: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/communities", response_model=List[CommunityResponse])
async def detect_treatment_communities(
    resolution: float = Query(default=1.0, ge=0.1, le=5.0, description="Louvain resolution parameter")
):
    """
    Detect treatment communities using Louvain algorithm.

    Higher resolution = more smaller communities.
    Lower resolution = fewer larger communities.
    """
    ga = get_graph_algorithms()

    if ga is None or not ga.is_available:
        raise HTTPException(
            status_code=503,
            detail="Graph algorithms not available. Check Neo4j connection."
        )

    try:
        communities = ga.detect_treatment_communities(resolution=resolution)

        if communities is None:
            return []

        return [
            CommunityResponse(
                community_id=c.get('community_id', idx),
                members=c.get('members', []),
                size=c.get('size', len(c.get('members', []))),
                treatments=c.get('treatments', [])
            )
            for idx, c in enumerate(communities)
        ]
    except Exception as e:
        logger.error(f"Error detecting communities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/treatment-paths/{patient_id}", response_model=List[TreatmentPathResponse])
async def find_optimal_treatment_paths(
    patient_id: str,
    target_outcome: str = Query(default="complete_response", description="Target outcome to optimize for"),
    max_depth: int = Query(default=5, ge=1, le=10, description="Maximum path depth")
):
    """
    Find optimal treatment paths for a patient.

    Returns treatment sequences that have led to the target outcome
    for similar patients.
    """
    ga = get_graph_algorithms()

    if ga is None or not ga.is_available:
        raise HTTPException(
            status_code=503,
            detail="Graph algorithms not available. Check Neo4j connection."
        )

    try:
        paths = ga.find_optimal_treatment_paths(
            patient_id=patient_id,
            target_outcome=target_outcome,
            max_depth=max_depth
        )

        if paths is None:
            return []

        return [
            TreatmentPathResponse(
                path=p.get('path', []),
                outcome=p.get('outcome', target_outcome),
                success_rate=p.get('success_rate'),
                patient_count=p.get('patient_count', 0)
            )
            for p in paths
        ]
    except Exception as e:
        logger.error(f"Error finding treatment paths: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/influential-treatments", response_model=List[InfluentialTreatmentResponse])
async def find_influential_treatments(
    top_n: int = Query(default=10, ge=1, le=50, description="Number of treatments to return")
):
    """
    Find most influential treatments using centrality analysis.

    Returns treatments ranked by their influence in the treatment graph.
    """
    ga = get_graph_algorithms()

    if ga is None or not ga.is_available:
        raise HTTPException(
            status_code=503,
            detail="Graph algorithms not available. Check Neo4j connection."
        )

    try:
        treatments = ga.find_influential_treatments(top_n=top_n)

        if treatments is None:
            return []

        return [
            InfluentialTreatmentResponse(
                treatment=t.get('treatment', ''),
                score=t.get('score', 0),
                patient_count=t.get('patient_count', 0),
                success_rate=t.get('success_rate')
            )
            for t in treatments
        ]
    except Exception as e:
        logger.error(f"Error finding influential treatments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vector-search")
async def vector_similarity_search(
    query_vector: List[float],
    top_k: int = Query(default=10, ge=1, le=100),
    index_name: str = Query(default="patient_embeddings")
):
    """
    Search for similar items using vector similarity.

    Requires Neo4j 5.11+ with vector index support.
    """
    ga = get_graph_algorithms()

    if ga is None or not ga.is_available:
        raise HTTPException(
            status_code=503,
            detail="Graph algorithms not available. Check Neo4j connection."
        )

    try:
        results = ga.vector_similarity_search(
            query_vector=query_vector,
            top_k=top_k,
            index_name=index_name
        )

        return {
            "results": results or [],
            "count": len(results) if results else 0
        }
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient-graph/{patient_id}")
async def get_patient_subgraph(
    patient_id: str,
    depth: int = Query(default=2, ge=1, le=5, description="Graph traversal depth")
):
    """
    Get the subgraph around a patient.

    Returns nodes and relationships within the specified depth.
    """
    ga = get_graph_algorithms()

    if ga is None or not ga.is_available:
        raise HTTPException(
            status_code=503,
            detail="Graph algorithms not available. Check Neo4j connection."
        )

    try:
        # This would need to be implemented in graph_algorithms.py
        # For now, return a basic structure
        return {
            "patient_id": patient_id,
            "depth": depth,
            "nodes": [],
            "relationships": [],
            "message": "Subgraph extraction requires implementation"
        }
    except Exception as e:
        logger.error(f"Error getting patient subgraph: {e}")
        raise HTTPException(status_code=500, detail=str(e))
