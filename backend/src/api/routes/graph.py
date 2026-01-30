"""
Graph Visualization API Routes
==============================

REST API endpoints for context graph visualization with NVL.
Provides graph data for the frontend visualization component.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from pydantic import BaseModel

from ...logging_config import get_logger
from ...db.context_graph_client import get_context_graph_client, GraphData, GraphNode, GraphRelationship

logger = get_logger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])


class NodeIdsRequest(BaseModel):
    """Request body for fetching relationships between nodes."""
    node_ids: List[str]


class GraphStatistics(BaseModel):
    """Graph statistics response."""
    node_counts: dict
    relationship_counts: dict
    total_nodes: int
    total_relationships: int


class GraphSchema(BaseModel):
    """Graph schema response."""
    node_labels: List[str]
    relationship_types: List[str]


@router.get("", response_model=GraphData)
async def get_graph(
    center_node_id: Optional[str] = Query(None, description="ID of the center node"),
    center_node_type: Optional[str] = Query(None, description="Type of the center node"),
    depth: int = Query(2, ge=1, le=5, description="Depth of traversal from center node"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of nodes to return"),
):
    """
    Get a subgraph for visualization.

    If center_node_id is provided, returns a subgraph centered on that node.
    Otherwise, returns a sample of the graph.

    Parameters:
    - center_node_id: The ID of the node to center the graph on (patient_id, decision_id, etc.)
    - center_node_type: Optional type hint for the center node
    - depth: How many hops from the center node to include (1-5)
    - limit: Maximum number of nodes to return

    Returns:
    - GraphData with nodes and relationships for NVL visualization
    """
    try:
        client = get_context_graph_client()

        if not client.verify_connectivity():
            raise HTTPException(status_code=503, detail="Neo4j database is not available")

        graph_data = client.get_graph_data(
            center_node_id=center_node_id,
            center_node_type=center_node_type,
            depth=depth,
            limit=limit,
        )

        logger.info(f"[GraphAPI] Returned {len(graph_data.nodes)} nodes, {len(graph_data.relationships)} relationships")
        return graph_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GraphAPI] Error fetching graph: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch graph data: {str(e)}")


@router.get("/expand/{node_id}", response_model=GraphData)
async def expand_node(
    node_id: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum nodes to return"),
):
    """
    Expand a node to get all directly connected nodes.

    Used for interactive graph exploration - double-click a node to expand it.

    Parameters:
    - node_id: The element ID or custom ID of the node to expand
    - limit: Maximum number of connected nodes to return

    Returns:
    - GraphData with the expanded node and all its direct connections
    """
    try:
        client = get_context_graph_client()

        if not client.verify_connectivity():
            raise HTTPException(status_code=503, detail="Neo4j database is not available")

        graph_data = client.expand_node(node_id, limit=limit)

        logger.info(f"[GraphAPI] Expanded node {node_id}: {len(graph_data.nodes)} nodes")
        return graph_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GraphAPI] Error expanding node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to expand node: {str(e)}")


@router.post("/relationships", response_model=List[GraphRelationship])
async def get_relationships_between(request: NodeIdsRequest):
    """
    Get all relationships between a set of nodes.

    Used after graph expansion to find any missing relationships
    between the nodes currently displayed.

    Parameters:
    - node_ids: List of node element IDs to check for relationships

    Returns:
    - List of relationships connecting any of the provided nodes
    """
    try:
        client = get_context_graph_client()

        if not client.verify_connectivity():
            raise HTTPException(status_code=503, detail="Neo4j database is not available")

        if len(request.node_ids) < 2:
            return []

        relationships = client.get_relationships_between(request.node_ids)

        logger.info(f"[GraphAPI] Found {len(relationships)} relationships between {len(request.node_ids)} nodes")
        return relationships

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GraphAPI] Error getting relationships: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get relationships: {str(e)}")


@router.get("/schema", response_model=GraphSchema)
async def get_graph_schema():
    """
    Get the graph database schema.

    Returns all node labels and relationship types in the database.
    Useful for understanding the data model and building queries.

    Returns:
    - node_labels: List of all node labels (e.g., Patient, TreatmentDecision)
    - relationship_types: List of all relationship types (e.g., ABOUT, BASED_ON)
    """
    try:
        client = get_context_graph_client()

        if not client.verify_connectivity():
            raise HTTPException(status_code=503, detail="Neo4j database is not available")

        schema = client.get_schema()

        if "error" in schema:
            raise HTTPException(status_code=500, detail=schema["error"])

        return GraphSchema(**schema)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GraphAPI] Error getting schema: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get schema: {str(e)}")


@router.get("/statistics", response_model=GraphStatistics)
async def get_graph_statistics():
    """
    Get graph statistics.

    Returns counts of nodes and relationships by type.
    Useful for understanding the size and composition of the graph.

    Returns:
    - node_counts: Dict mapping node labels to counts
    - relationship_counts: Dict mapping relationship types to counts
    - total_nodes: Total number of nodes
    - total_relationships: Total number of relationships
    """
    try:
        client = get_context_graph_client()

        if not client.verify_connectivity():
            raise HTTPException(status_code=503, detail="Neo4j database is not available")

        stats = client.get_statistics()

        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])

        return GraphStatistics(**stats)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GraphAPI] Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.get("/patient/{patient_id}", response_model=GraphData)
async def get_patient_graph(
    patient_id: str,
    depth: int = Query(2, ge=1, le=4, description="Depth of traversal"),
    include_decisions: bool = Query(True, description="Include treatment decisions"),
    include_biomarkers: bool = Query(True, description="Include biomarkers"),
):
    """
    Get a patient-centered subgraph.

    Convenience endpoint for fetching a graph centered on a specific patient,
    with options to include or exclude related entity types.

    Parameters:
    - patient_id: The patient_id to center the graph on
    - depth: How many hops from the patient to include
    - include_decisions: Whether to include TreatmentDecision nodes
    - include_biomarkers: Whether to include Biomarker nodes

    Returns:
    - GraphData with the patient and related entities
    """
    try:
        client = get_context_graph_client()

        if not client.verify_connectivity():
            raise HTTPException(status_code=503, detail="Neo4j database is not available")

        # Use the general get_graph_data method with patient_id
        graph_data = client.get_graph_data(
            center_node_id=patient_id,
            center_node_type="Patient",
            depth=depth,
            limit=100,
        )

        # Filter nodes if requested
        if not include_decisions:
            graph_data.nodes = [n for n in graph_data.nodes if "TreatmentDecision" not in n.labels]

        if not include_biomarkers:
            graph_data.nodes = [n for n in graph_data.nodes if "Biomarker" not in n.labels]

        # Filter relationships to only include those between remaining nodes
        node_ids = {n.id for n in graph_data.nodes}
        graph_data.relationships = [
            r for r in graph_data.relationships
            if r.startNodeId in node_ids and r.endNodeId in node_ids
        ]

        logger.info(f"[GraphAPI] Patient graph for {patient_id}: {len(graph_data.nodes)} nodes")
        return graph_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GraphAPI] Error getting patient graph: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get patient graph: {str(e)}")


@router.get("/decision/{decision_id}", response_model=GraphData)
async def get_decision_graph(
    decision_id: str,
    include_causal: bool = Query(True, description="Include causal chain"),
    causal_depth: int = Query(2, ge=1, le=3, description="Causal chain depth"),
):
    """
    Get a decision-centered subgraph.

    Fetches a graph centered on a treatment decision, including the patient,
    biomarkers, guidelines, and optionally the causal chain.

    Parameters:
    - decision_id: The decision ID to center the graph on
    - include_causal: Whether to include CAUSED/INFLUENCED relationships
    - causal_depth: How many hops in the causal chain to include

    Returns:
    - GraphData with the decision and related entities
    """
    try:
        client = get_context_graph_client()

        if not client.verify_connectivity():
            raise HTTPException(status_code=503, detail="Neo4j database is not available")

        depth = causal_depth + 1 if include_causal else 2

        graph_data = client.get_graph_data(
            center_node_id=decision_id,
            center_node_type="TreatmentDecision",
            depth=depth,
            limit=75,
        )

        logger.info(f"[GraphAPI] Decision graph for {decision_id}: {len(graph_data.nodes)} nodes")
        return graph_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GraphAPI] Error getting decision graph: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get decision graph: {str(e)}")
