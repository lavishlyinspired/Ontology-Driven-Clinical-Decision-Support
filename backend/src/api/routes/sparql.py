"""
SPARQL Endpoint
===============

Exposes a POST /sparql endpoint that delegates to the Neosemantics
(n10s) SPARQL execution within Neo4j.

Requires the n10s plugin to be installed and configured.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from ...logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["SPARQL"])


class SPARQLRequest(BaseModel):
    """SPARQL query request body."""
    query: str
    format: Optional[str] = "json"  # json or table


class SPARQLResponse(BaseModel):
    """SPARQL query response."""
    success: bool
    results: List[Dict[str, Any]] = []
    result_count: int = 0
    error: Optional[str] = None


@router.post("/sparql", response_model=SPARQLResponse)
async def execute_sparql(request: SPARQLRequest):
    """
    Execute a SPARQL query against the Neo4j graph via Neosemantics (n10s).

    The query is passed to n10s.rdf.export.sn / n10s.rdf.export.sparql
    for execution.

    Example queries:
    - SELECT ?s WHERE {?s a onco:Drug}
    - SELECT ?s ?label WHERE {?s rdfs:label ?label} LIMIT 10

    Args:
        request: SPARQLRequest with query string

    Returns:
        Query results as list of dictionaries
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Empty SPARQL query")

    try:
        from ...db.neosemantics_tools import NeosemanticsTools

        tools = NeosemanticsTools()
        if not tools._available:
            raise HTTPException(
                status_code=503,
                detail="Neo4j not available. Ensure database is running."
            )

        if not tools._n10s_available:
            raise HTTPException(
                status_code=503,
                detail="Neosemantics (n10s) plugin not available. "
                       "Install from https://neo4j.com/labs/neosemantics/"
            )

        results = tools.execute_sparql(request.query)

        return SPARQLResponse(
            success=True,
            results=results,
            result_count=len(results),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SPARQL execution failed: {e}", exc_info=True)
        return SPARQLResponse(
            success=False,
            error=str(e),
        )
