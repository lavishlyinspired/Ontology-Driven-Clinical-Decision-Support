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

// Lab Results Types
export interface LabResult {
  id?: string;
  loinc_code: string;
  loinc_name?: string;
  test_name?: string;
  value: number | string;
  units?: string;
  unit?: string;
  reference_range?: string;
  interpretation?: "normal" | "high" | "low" | "critical" | "abnormal";
  severity?: "normal" | "grade1" | "grade2" | "grade3" | "grade4" | "critical";
  test_date?: string;
  lab_category?: string;
}

export interface LabInterpretation extends LabResult {
  clinical_significance?: string;
  recommendation?: string;
  ctcae_grade?: string;
}

// Medication Types
export interface Medication {
  id?: string;
  rxcui?: string;
  drug_name: string;
  name?: string;
  drug_class?: string;
  dose?: string;
  route?: string;
  frequency?: string;
  start_date?: string;
  end_date?: string;
  status?: "active" | "inactive" | "discontinued";
}

export interface DrugInteraction {
  id?: string;
  drug1?: string;
  drug2?: string;
  drug1_rxcui?: string;
  drug2_rxcui?: string;
  severity: "SEVERE" | "MODERATE" | "MILD";
  mechanism?: string;
  clinical_effect?: string;
  recommendation?: string;
}

// Monitoring Protocol Types
export interface MonitoringProtocol {
  id?: string;
  protocol_name?: string;
  regimen: string;
  frequency?: string;
  duration?: string;
  tests_to_monitor?: string[] | Array<{
    loinc_code?: string;
    loinc_name?: string;
    name?: string;
    frequency?: string;
  }>;
  schedule?: Array<{
    week?: number;
    day?: number;
    tests?: string[];
  }>;
  dose_adjustments?: Array<{
    parameter?: string;
    threshold?: string;
    action?: string;
  }>;
  created_date?: string;
}

// Clinical Trial Types
export interface ClinicalTrial {
  nct_id: string;
  title?: string;
  brief_title?: string;
  phase?: string;
  status?: string;
  condition?: string;
  intervention?: string;
  eligibility_score?: number;
  match_score?: number;
  eligibility_criteria?: string[];
  matched_criteria?: string[];
}

// SSE Event Types
export type SSEEventType =
  | "status"
  | "progress"
  | "reasoning"
  | "patient_data"
  | "complexity"
  | "recommendation"
  | "text"
  | "log"
  | "error"
  | "graph_data"
  | "suggestions"
  | "lab_results"
  | "drug_interactions"
  | "monitoring_protocol"
  | "eligible_trials";

export interface SSEEvent {
  type: SSEEventType;
  content: any;
  level?: string;
  timestamp?: string;
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
// LABORATORY API
// ============================================

export async function interpretLabResult(
  loincCode: string,
  value: number,
  units: string,
  patientContext?: any
): Promise<LabInterpretation> {
  const response = await apiClient.post("/api/v1/laboratory/interpret", {
    loinc_code: loincCode,
    value,
    units,
    patient_context: patientContext,
  });
  return response.data;
}

export async function getLabPanel(panelType: string): Promise<LabResult[]> {
  const response = await apiClient.get(`/api/v1/laboratory/panels/${panelType}`);
  return response.data;
}

export async function batchInterpretLabs(
  patientId: string,
  labResults: LabResult[]
): Promise<{ interpretations: LabInterpretation[]; critical_flags: any[] }> {
  const response = await apiClient.post("/api/v1/laboratory/batch-interpret", {
    patient_id: patientId,
    lab_results: labResults,
  });
  return response.data;
}

export async function getPatientLabs(
  patientId: string,
  startDate?: string,
  endDate?: string,
  loincCode?: string
): Promise<LabResult[]> {
  const params: any = {};
  if (startDate) params.start_date = startDate;
  if (endDate) params.end_date = endDate;
  if (loincCode) params.loinc_code = loincCode;

  const response = await apiClient.get(`/api/v1/patients/${patientId}/labs`, {
    params,
  });
  return response.data;
}

export async function searchLoincCodes(
  query: string,
  category: string = "all"
): Promise<any[]> {
  const response = await apiClient.get("/api/v1/laboratory/search", {
    params: { q: query, category },
  });
  return response.data;
}

// ============================================
// MEDICATIONS API
// ============================================

export async function searchMedications(query: string): Promise<Medication[]> {
  const response = await apiClient.get("/api/v1/medications/search", {
    params: { q: query },
  });
  return response.data;
}

export async function checkDrugInteractions(
  drugList: string[]
): Promise<{ interactions: DrugInteraction[] }> {
  const response = await apiClient.post("/api/v1/medications/interactions", {
    drug_list: drugList,
  });
  return response.data;
}

export async function getTherapeuticAlternatives(
  rxcui: string
): Promise<Medication[]> {
  const response = await apiClient.get(
    `/api/v1/medications/${rxcui}/alternatives`
  );
  return response.data;
}

export async function getMedicationDetails(rxcui: string): Promise<any> {
  const response = await apiClient.get(`/api/v1/medications/${rxcui}/details`);
  return response.data;
}

export async function getPatientMedications(
  patientId: string,
  activeOnly: boolean = true
): Promise<Medication[]> {
  const response = await apiClient.get(
    `/api/v1/patients/${patientId}/medications`,
    {
      params: { active_only: activeOnly },
    }
  );
  return response.data;
}

export async function addPatientMedication(
  patientId: string,
  medication: Medication
): Promise<{ medication: Medication; interactions: DrugInteraction[] }> {
  const response = await apiClient.post(
    `/api/v1/patients/${patientId}/medications`,
    medication
  );
  return response.data;
}

// ============================================
// MONITORING API
// ============================================

export async function getMonitoringProtocol(
  regimen: string
): Promise<MonitoringProtocol> {
  const response = await apiClient.get(
    `/api/v1/monitoring/protocols/${regimen}`
  );
  return response.data;
}

export async function assessDoseAdjustment(
  drugName: string,
  labResults: LabResult[]
): Promise<{
  dose_adjustment_needed: boolean;
  recommendation: string;
  rationale: string;
}> {
  const response = await apiClient.post("/api/v1/monitoring/assess-dose", {
    drug_name: drugName,
    lab_results: labResults,
  });
  return response.data;
}

export async function predictLabEffects(drugName: string): Promise<any[]> {
  const response = await apiClient.get(
    `/api/v1/monitoring/predict-effects/${drugName}`
  );
  return response.data;
}

export async function getPatientMonitoringProtocol(
  patientId: string
): Promise<MonitoringProtocol> {
  const response = await apiClient.get(
    `/api/v1/patients/${patientId}/monitoring-protocol`
  );
  return response.data;
}

export async function createPatientMonitoringProtocol(
  patientId: string,
  regimen: string,
  startDate: string
): Promise<MonitoringProtocol> {
  const response = await apiClient.post(
    `/api/v1/patients/${patientId}/monitoring-protocol`,
    {
      regimen,
      start_date: startDate,
    }
  );
  return response.data;
}

// ============================================
// CLINICAL TRIALS API
// ============================================

export async function searchClinicalTrials(params: {
  condition?: string;
  intervention?: string;
  phase?: string;
  status?: string;
}): Promise<ClinicalTrial[]> {
  const response = await apiClient.get("/api/v1/clinical-trials/search", {
    params,
  });
  return response.data;
}

export async function matchPatientToTrials(
  patientId: string
): Promise<{ trials: ClinicalTrial[] }> {
  const response = await apiClient.post("/api/v1/clinical-trials/match-patient", {
    patient_id: patientId,
  });
  return response.data;
}

export async function getClinicalTrial(nctId: string): Promise<ClinicalTrial> {
  const response = await apiClient.get(`/api/v1/clinical-trials/${nctId}`);
  return response.data;
}

export async function storeClinicalTrial(nctId: string): Promise<any> {
  const response = await apiClient.post(
    `/api/v1/clinical-trials/${nctId}/store`
  );
  return response.data;
}

export async function getPatientEligibleTrials(
  patientId: string
): Promise<ClinicalTrial[]> {
  const response = await apiClient.get(
    `/api/v1/patients/${patientId}/eligible-trials`
  );
  return response.data;
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
