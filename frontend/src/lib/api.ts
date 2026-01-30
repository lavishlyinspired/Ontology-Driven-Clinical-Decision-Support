/**
 * API Client for Lung Cancer Assistant
 *
 * Provides typed API calls for graph visualization, chat, and clinical data.
 */

import axios from "axios";

// Base URL for API calls
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Axios instance with default config
export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// ============================================
// TYPES
// ============================================

export interface GraphNode {
  id: string;
  labels: string[];
  properties: Record<string, unknown>;
}

export interface GraphRelationship {
  id: string;
  type: string;
  startNodeId: string;
  endNodeId: string;
  properties: Record<string, unknown>;
}

export interface GraphData {
  nodes: GraphNode[];
  relationships: GraphRelationship[];
}

export interface Patient {
  id: string;
  name?: string;
  age?: number;
  stage?: string;
  histology?: string;
  ps?: number;
  decision_count?: number;
}

export interface TreatmentDecision {
  id: string;
  decision_type: string;
  category: string;
  status: string;
  reasoning?: string;
  reasoning_summary?: string;
  treatment?: string;
  confidence_score?: number;
  confidence?: number;
  risk_factors?: string[];
  guidelines_applied?: string[];
  biomarkers_considered?: string[];
  timestamp?: string;
  decision_timestamp?: string;
  evidence_level?: string;
  intent?: string;
  guideline_reference?: string;
  rank?: number;
}

export interface SimilarDecision {
  decision: TreatmentDecision;
  similarity_score: number;
  similarity_type: string;
}

export interface CausalChain {
  decision_id: string;
  causes: TreatmentDecision[];
  effects: TreatmentDecision[];
  depth: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface AgentContext {
  model: string;
  system_prompt: string;
  available_tools: string[];
  mcp_server?: string;
}

export interface ToolCall {
  name: string;
  input: Record<string, unknown>;
  output?: unknown;
}

// SSE Event types
export type StreamEventType =
  | "agent_context"
  | "status"
  | "patient_data"
  | "text"
  | "graph_data"
  | "tool_use"
  | "tool_result"
  | "complexity"
  | "recommendation"
  | "clinical_summary"
  | "treatment_plan"
  | "suggestions"
  | "reasoning"
  | "progress"
  | "log"
  | "done"
  | "error"
  | "ping";

export interface StreamEvent {
  type: StreamEventType;
  content?: unknown;
  context?: AgentContext;
  name?: string;
  input?: Record<string, unknown>;
  output?: unknown;
  error?: string;
  level?: string;
  timestamp?: string;
}

export interface GraphSchema {
  node_labels: string[];
  relationship_types: string[];
}

export interface GraphStatistics {
  node_counts: Record<string, number>;
  relationship_counts: Record<string, number>;
  total_nodes: number;
  total_relationships: number;
}

// ============================================
// GRAPH API
// ============================================

/**
 * Get a subgraph for visualization.
 */
export async function getGraphData(
  centerNodeId?: string,
  depth: number = 2,
  limit: number = 100
): Promise<GraphData> {
  const params = new URLSearchParams();
  if (centerNodeId) params.append("center_node_id", centerNodeId);
  params.append("depth", depth.toString());
  params.append("limit", limit.toString());

  const response = await apiClient.get<GraphData>(`/api/graph?${params}`);
  return response.data;
}

/**
 * Expand a node to get all connected nodes.
 */
export async function expandNode(
  nodeId: string,
  limit: number = 50
): Promise<GraphData> {
  const response = await apiClient.get<GraphData>(
    `/api/graph/expand/${encodeURIComponent(nodeId)}?limit=${limit}`
  );
  return response.data;
}

/**
 * Get relationships between a set of nodes.
 */
export async function getRelationshipsBetween(
  nodeIds: string[]
): Promise<GraphRelationship[]> {
  const response = await apiClient.post<GraphRelationship[]>(
    "/api/graph/relationships",
    { node_ids: nodeIds }
  );
  return response.data;
}

/**
 * Get the graph schema.
 */
export async function getGraphSchema(): Promise<GraphSchema> {
  const response = await apiClient.get<GraphSchema>("/api/graph/schema");
  return response.data;
}

/**
 * Get graph statistics.
 */
export async function getGraphStatistics(): Promise<GraphStatistics> {
  const response = await apiClient.get<GraphStatistics>("/api/graph/statistics");
  return response.data;
}

/**
 * Get a patient-centered graph.
 */
export async function getPatientGraph(
  patientId: string,
  depth: number = 2
): Promise<GraphData> {
  const response = await apiClient.get<GraphData>(
    `/api/graph/patient/${encodeURIComponent(patientId)}?depth=${depth}`
  );
  return response.data;
}

/**
 * Get a decision-centered graph.
 */
export async function getDecisionGraph(
  decisionId: string,
  includeCausal: boolean = true
): Promise<GraphData> {
  const response = await apiClient.get<GraphData>(
    `/api/graph/decision/${encodeURIComponent(decisionId)}?include_causal=${includeCausal}`
  );
  return response.data;
}

// ============================================
// DECISION API
// ============================================

/**
 * Get similar decisions for a given decision.
 */
export async function getSimilarDecisions(
  decisionId: string,
  limit: number = 5,
  method: string = "hybrid"
): Promise<SimilarDecision[]> {
  // This would call the backend API - for now return empty
  // In production, implement the backend endpoint
  console.log(`Getting similar decisions for ${decisionId}`);
  return [];
}

/**
 * Get the causal chain for a decision.
 */
export async function getCausalChain(
  decisionId: string,
  depth: number = 3
): Promise<CausalChain> {
  // This would call the backend API - for now return empty structure
  // In production, implement the backend endpoint
  console.log(`Getting causal chain for ${decisionId}`);
  return {
    decision_id: decisionId,
    causes: [],
    effects: [],
    depth,
  };
}

// ============================================
// CHAT API
// ============================================

/**
 * Stream chat messages with graph integration.
 *
 * Uses Server-Sent Events to stream response in real-time.
 */
export async function* streamChatMessage(
  message: string,
  conversationHistory: ChatMessage[] = [],
  sessionId: string = "default",
  includeGraph: boolean = true
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_BASE_URL}/api/chat-graph/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      include_graph: includeGraph,
      auto_expand_entities: true,
    }),
  });

  if (!response.ok) {
    throw new Error(`Chat request failed: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("No response body");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6));
          yield data as StreamEvent;
        } catch {
          // Skip invalid JSON
        }
      }
    }
  }

  // Process any remaining buffer
  if (buffer.startsWith("data: ")) {
    try {
      const data = JSON.parse(buffer.slice(6));
      yield data as StreamEvent;
    } catch {
      // Skip invalid JSON
    }
  }
}

/**
 * Send chat message and get response (non-streaming).
 */
export async function sendChatMessage(
  message: string,
  sessionId: string = "default"
): Promise<{
  response: string;
  patient_data?: unknown;
  graph_data?: GraphData;
  tool_calls?: ToolCall[];
}> {
  const response = await apiClient.post("/api/chat-graph/query-with-graph", {
    message,
    session_id: sessionId,
    include_graph: true,
    auto_expand_entities: true,
  });
  return response.data;
}

/**
 * Execute a graph tool directly.
 */
export async function executeGraphTool(
  toolName: string,
  toolInput: Record<string, unknown>
): Promise<{ tool_name: string; input: unknown; output: unknown }> {
  const response = await apiClient.post(
    `/api/chat-graph/execute-tool?tool_name=${encodeURIComponent(toolName)}`,
    toolInput
  );
  return response.data;
}

/**
 * Get available graph tools.
 */
export async function getGraphTools(): Promise<{
  tools: Array<{
    name: string;
    description: string;
    parameters: unknown;
  }>;
  count: number;
}> {
  const response = await apiClient.get("/api/chat-graph/tools");
  return response.data;
}

// ============================================
// LEGACY CHAT API (for backward compatibility)
// ============================================

/**
 * Stream chat using legacy endpoint.
 */
export async function* streamLegacyChatMessage(
  message: string,
  sessionId: string = "default"
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_BASE_URL}/api/v1/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      session_id: sessionId,
    }),
  });

  if (!response.ok) {
    throw new Error(`Chat request failed: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("No response body");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6));
          yield data as StreamEvent;
        } catch {
          // Skip invalid JSON
        }
      }
    }
  }
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Transform GraphData to NVL format.
 */
export function transformToNvlFormat(graphData: GraphData): {
  nodes: Array<{
    id: string;
    caption: string;
    color: string;
    size: number;
  }>;
  relationships: Array<{
    id: string;
    from: string;
    to: string;
    caption: string;
    color: string;
  }>;
} {
  const NODE_COLORS: Record<string, string> = {
    Patient: "#4299E1",
    TreatmentDecision: "#9F7AEA",
    Biomarker: "#48BB78",
    Guideline: "#38B2AC",
    Comorbidity: "#ED8936",
    ClinicalTrial: "#F56565",
  };

  const NODE_SIZES: Record<string, number> = {
    Patient: 30,
    TreatmentDecision: 28,
    Biomarker: 22,
    Guideline: 24,
    Comorbidity: 20,
    ClinicalTrial: 22,
  };

  const REL_COLORS: Record<string, string> = {
    ABOUT: "#A0AEC0",
    BASED_ON: "#48BB78",
    APPLIED_GUIDELINE: "#38B2AC",
    CAUSED: "#E53E3E",
    INFLUENCED: "#D69E2E",
    FOLLOWED_PRECEDENT: "#9F7AEA",
    HAS_BIOMARKER: "#48BB78",
    HAS_COMORBIDITY: "#ED8936",
  };

  const nodes = graphData.nodes.map((node) => {
    const primaryLabel = node.labels[0] || "Unknown";
    const caption =
      (node.properties.name as string) ||
      (node.properties.patient_id as string) ||
      (node.properties.decision_type as string) ||
      (node.properties.marker_type as string) ||
      node.id.slice(0, 8);

    return {
      id: node.id,
      caption,
      color: NODE_COLORS[primaryLabel] || "#718096",
      size: NODE_SIZES[primaryLabel] || 20,
    };
  });

  const relationships = graphData.relationships.map((rel) => ({
    id: rel.id,
    from: rel.startNodeId,
    to: rel.endNodeId,
    caption: rel.type,
    color: REL_COLORS[rel.type] || "#A0AEC0",
  }));

  return { nodes, relationships };
}

/**
 * Merge two GraphData objects, avoiding duplicates.
 */
export function mergeGraphData(
  existing: GraphData,
  newData: GraphData
): GraphData {
  const existingNodeIds = new Set(existing.nodes.map((n) => n.id));
  const existingRelIds = new Set(existing.relationships.map((r) => r.id));

  const mergedNodes = [
    ...existing.nodes,
    ...newData.nodes.filter((n) => !existingNodeIds.has(n.id)),
  ];

  const mergedRels = [
    ...existing.relationships,
    ...newData.relationships.filter((r) => !existingRelIds.has(r.id)),
  ];

  return {
    nodes: mergedNodes,
    relationships: mergedRels,
  };
}
