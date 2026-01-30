"""
Chat Graph API Routes
=====================

Enhanced chat endpoint with context graph integration.
Streams SSE events including graph visualization data.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import json
import asyncio

from ...logging_config import get_logger
from ...services.conversation_service import ConversationService
from ...services.lca_service import LungCancerAssistantService
from ...agents.graph_tools import GraphToolExecutor, create_graph_tools
from ...db.context_graph_client import get_context_graph_client

logger = get_logger(__name__)

router = APIRouter(prefix="/chat-graph", tags=["chat-graph"])

# Initialize services
use_neo4j = os.getenv("CHAT_USE_NEO4J", "true").lower() == "true"
use_vector_store = os.getenv("CHAT_USE_VECTOR_STORE", "true").lower() == "true"

lca_service = LungCancerAssistantService(
    use_neo4j=use_neo4j,
    use_vector_store=use_vector_store,
    enable_advanced_workflow=True,
    enable_provenance=True
)
conversation_service = ConversationService(lca_service)


class ChatGraphRequest(BaseModel):
    """Chat request with graph options."""
    message: str = Field(description="User message")
    session_id: Optional[str] = Field(default="default", description="Session ID for conversation history")
    include_graph: bool = Field(default=True, description="Include graph visualization data in responses")
    auto_expand_entities: bool = Field(default=True, description="Automatically fetch graph for mentioned entities")


class AgentContext(BaseModel):
    """Agent context for transparency."""
    model: str = "ollama/llama3.2"
    system_prompt: str
    available_tools: List[str]


def format_sse(data: Dict[str, Any]) -> str:
    """Format data as SSE event."""
    return f"data: {json.dumps(data)}\n\n"


async def enhanced_chat_stream(
    message: str,
    session_id: str,
    include_graph: bool = True,
    auto_expand_entities: bool = True,
):
    """
    Enhanced chat stream with graph visualization support.

    Yields SSE events with the following types:
    - agent_context: System prompt and available tools
    - status: Workflow step updates
    - patient_data: Extracted patient data
    - text: Streaming text content
    - graph_data: Graph visualization data
    - tool_use: Tool call information
    - tool_result: Tool execution results
    - done: Completion signal
    - error: Error messages
    """
    graph_executor = GraphToolExecutor()
    context_graph = get_context_graph_client()

    # Emit agent context for transparency
    if include_graph:
        agent_context = {
            "model": "ollama/llama3.2",
            "system_prompt": (
                "You are an AI assistant for lung cancer clinical decision support. "
                "You have access to a Context Graph that stores patient data, treatment decisions, "
                "biomarker findings, and clinical guidelines. Use the available tools to search for "
                "patients, find similar decisions, trace causal chains, and record new decisions."
            ),
            "available_tools": list(graph_executor.tools.keys()),
        }
        yield format_sse({"type": "agent_context", "context": agent_context})

    # Stream the main conversation
    accumulated_response = ""
    patient_data = None
    detected_entities = []

    try:
        async for chunk in conversation_service.chat_stream(
            session_id=session_id,
            message=message
        ):
            # Parse and forward the chunk
            try:
                # Extract the data from SSE format
                if chunk.startswith("data: "):
                    data = json.loads(chunk[6:].strip())

                    # Track patient data
                    if data.get("type") == "patient_data":
                        patient_data = data.get("content")
                        yield chunk

                        # Auto-fetch graph for detected patient
                        if include_graph and auto_expand_entities and patient_data:
                            patient_id = patient_data.get("patient_id")
                            if patient_id:
                                try:
                                    graph_result = graph_executor.execute(
                                        "get_patient_graph",
                                        patient_id=patient_id,
                                        depth=2
                                    )
                                    if graph_result.get("graph_data"):
                                        yield format_sse({
                                            "type": "graph_data",
                                            "content": graph_result["graph_data"]
                                        })
                                        yield format_sse({
                                            "type": "tool_use",
                                            "name": "get_patient_graph",
                                            "input": {"patient_id": patient_id, "depth": 2}
                                        })
                                        yield format_sse({
                                            "type": "tool_result",
                                            "name": "get_patient_graph",
                                            "output": graph_result
                                        })
                                except Exception as e:
                                    logger.warning(f"Failed to fetch patient graph: {e}")

                    # Track text for entity detection
                    elif data.get("type") == "text":
                        accumulated_response += data.get("content", "")
                        yield chunk

                    # Forward other events
                    else:
                        yield chunk

                else:
                    yield chunk

            except json.JSONDecodeError:
                yield chunk

        # After response, try to fetch relevant graph data
        if include_graph and auto_expand_entities and accumulated_response:
            # Try to extract and visualize mentioned patients/decisions
            await fetch_mentioned_entities_graph(
                accumulated_response,
                graph_executor,
                context_graph,
            )

            # Emit any additional tool calls made during entity extraction
            for tool_call in graph_executor.get_tool_calls():
                if tool_call.get("output", {}).get("graph_data"):
                    yield format_sse({
                        "type": "tool_use",
                        "name": tool_call["name"],
                        "input": tool_call["input"]
                    })
                    yield format_sse({
                        "type": "tool_result",
                        "name": tool_call["name"],
                        "output": tool_call["output"]
                    })

        yield format_sse({"type": "done"})

    except Exception as e:
        logger.error(f"Chat stream error: {e}", exc_info=True)
        yield format_sse({"type": "error", "content": str(e)})


async def fetch_mentioned_entities_graph(
    text: str,
    executor: GraphToolExecutor,
    graph_client,
) -> None:
    """
    Extract and fetch graph data for entities mentioned in the response.

    This is a simple heuristic-based approach. In production, you might use
    NER or more sophisticated entity extraction.
    """
    # Simple pattern matching for patient IDs (e.g., LC-2024-001)
    import re

    # Look for patient ID patterns
    patient_patterns = [
        r"LC-\d{4}-\d{3}",  # LC-YYYY-NNN format
        r"patient[_\s]?id[:\s]+([A-Za-z0-9-]+)",  # patient_id: XXX
    ]

    for pattern in patient_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches[:3]:  # Limit to 3 matches
            patient_id = match if isinstance(match, str) else match[0]
            try:
                executor.execute("get_patient_graph", patient_id=patient_id, depth=1)
            except Exception:
                pass


@router.post("/stream")
async def chat_graph_stream(request: ChatGraphRequest):
    """
    Stream chat responses with context graph integration.

    This endpoint extends the standard chat with:
    - Agent context transparency (system prompt, available tools)
    - Graph visualization data for mentioned entities
    - Tool call tracking for explainability
    - Auto-expansion of patient graphs

    SSE Event Types:
    - agent_context: Model info, system prompt, available tools
    - status: Workflow step updates
    - patient_data: Extracted patient data from user message
    - text: Streaming text content
    - graph_data: NVL-compatible graph visualization data
    - tool_use: Tool call name and input parameters
    - tool_result: Tool execution output
    - done: Stream completion
    - error: Error message

    Args:
        message: User message
        session_id: Session ID for conversation history
        include_graph: Whether to include graph visualization
        auto_expand_entities: Auto-fetch graphs for detected entities

    Returns:
        SSE stream with enhanced chat events
    """
    try:
        async def event_generator():
            async for chunk in enhanced_chat_stream(
                message=request.message,
                session_id=request.session_id,
                include_graph=request.include_graph,
                auto_expand_entities=request.auto_expand_entities,
            ):
                yield chunk

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"Chat graph stream error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query-with-graph")
async def query_with_graph(request: ChatGraphRequest):
    """
    Non-streaming chat that returns response with graph data.

    Useful for testing or when SSE is not available.

    Returns:
        Complete response with text, patient data, and graph visualization
    """
    try:
        graph_executor = GraphToolExecutor()
        context_graph = get_context_graph_client()

        # Collect all response parts
        response_text = ""
        patient_data = None
        graph_data = None
        tool_calls = []

        async for chunk in enhanced_chat_stream(
            message=request.message,
            session_id=request.session_id,
            include_graph=request.include_graph,
            auto_expand_entities=request.auto_expand_entities,
        ):
            try:
                if chunk.startswith("data: "):
                    data = json.loads(chunk[6:].strip())

                    if data.get("type") == "text":
                        response_text += data.get("content", "")
                    elif data.get("type") == "patient_data":
                        patient_data = data.get("content")
                    elif data.get("type") == "graph_data":
                        graph_data = data.get("content")
                    elif data.get("type") == "tool_use":
                        tool_calls.append({
                            "name": data.get("name"),
                            "input": data.get("input")
                        })
                    elif data.get("type") == "tool_result":
                        # Update last tool call with output
                        if tool_calls:
                            tool_calls[-1]["output"] = data.get("output")

            except json.JSONDecodeError:
                continue

        return {
            "session_id": request.session_id,
            "message": request.message,
            "response": response_text,
            "patient_data": patient_data,
            "graph_data": graph_data,
            "tool_calls": tool_calls,
        }

    except Exception as e:
        logger.error(f"Query with graph error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute-tool")
async def execute_graph_tool(
    tool_name: str,
    tool_input: Dict[str, Any],
):
    """
    Execute a graph tool directly.

    Useful for frontend to call tools independently of chat.

    Args:
        tool_name: Name of the tool to execute
        tool_input: Input parameters for the tool

    Returns:
        Tool execution result with graph data
    """
    try:
        executor = GraphToolExecutor()

        if tool_name not in executor.tools:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown tool: {tool_name}. Available: {list(executor.tools.keys())}"
            )

        result = executor.execute(tool_name, **tool_input)

        return {
            "tool_name": tool_name,
            "input": tool_input,
            "output": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Execute tool error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools")
async def list_graph_tools():
    """
    List available graph tools.

    Returns tool names, descriptions, and parameter schemas.
    """
    tools = create_graph_tools()

    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args_schema.model_json_schema() if tool.args_schema else None,
            }
            for tool in tools
        ],
        "count": len(tools),
    }


@router.get("/health")
async def chat_graph_health():
    """Health check for chat-graph service."""
    context_graph = get_context_graph_client()
    neo4j_connected = context_graph.verify_connectivity()

    return {
        "status": "healthy" if neo4j_connected else "degraded",
        "service": "chat-graph",
        "neo4j_connected": neo4j_connected,
        "features": [
            "streaming_responses",
            "graph_visualization",
            "tool_transparency",
            "entity_extraction",
            "session_management"
        ]
    }
