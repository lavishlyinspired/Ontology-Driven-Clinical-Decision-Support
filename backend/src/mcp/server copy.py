#!/usr/bin/env python3
"""
MCP Server for Lung Cancer Assistant (LCA)
Provides clinical decision support tools for Claude Desktop and other MCP clients.

Based on Model Context Protocol specification.
Compatible with Claude Desktop, Cursor, and other MCP-enabled applications.
"""

import json
import sys
import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ToolName(str, Enum):
    """Available MCP tools"""
    GET_PATIENT_CONTEXT = "get_patient_context"
    SEARCH_SIMILAR_PATIENTS = "search_similar_patients"
    GET_TREATMENT_OPTIONS = "get_treatment_options"
    CHECK_GUIDELINES = "check_guidelines"
    VALIDATE_ONTOLOGY = "validate_ontology"
    GET_BIOMARKER_INFO = "get_biomarker_info"
    FIND_CLINICAL_TRIALS = "find_clinical_trials"
    GET_SURVIVAL_DATA = "get_survival_data"
    QUERY_KNOWLEDGE_GRAPH = "query_knowledge_graph"
    GET_ARGUMENTATION = "get_argumentation"


@dataclass
class Tool:
    """MCP Tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]


@dataclass
class Resource:
    """MCP Resource definition"""
    uri: str
    name: str
    description: str
    mimeType: str


# Define available tools
TOOLS: List[Tool] = [
    Tool(
        name=ToolName.GET_PATIENT_CONTEXT,
        description="Get comprehensive context for a patient including diagnosis, biomarkers, treatments, and current status",
        inputSchema={
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "The patient identifier"
                }
            },
            "required": ["patient_id"]
        }
    ),
    Tool(
        name=ToolName.SEARCH_SIMILAR_PATIENTS,
        description="Find patients with similar characteristics for cohort analysis",
        inputSchema={
            "type": "object",
            "properties": {
                "diagnosis": {
                    "type": "string",
                    "description": "Primary diagnosis (e.g., 'NSCLC', 'SCLC')"
                },
                "stage": {
                    "type": "string",
                    "description": "Cancer stage (e.g., 'IV', 'IIIB')"
                },
                "biomarkers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of biomarkers (e.g., ['EGFR+', 'PD-L1 >= 50%'])"
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of similar patients to return"
                }
            },
            "required": ["diagnosis"]
        }
    ),
    Tool(
        name=ToolName.GET_TREATMENT_OPTIONS,
        description="Get evidence-based treatment options for a patient profile",
        inputSchema={
            "type": "object",
            "properties": {
                "diagnosis": {
                    "type": "string",
                    "description": "Primary diagnosis"
                },
                "stage": {
                    "type": "string",
                    "description": "Cancer stage"
                },
                "biomarkers": {
                    "type": "object",
                    "description": "Biomarker results (e.g., {\"EGFR\": \"L858R\", \"PD-L1\": 80})"
                },
                "prior_treatments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of prior treatments"
                },
                "ecog_ps": {
                    "type": "integer",
                    "description": "ECOG Performance Status (0-4)"
                }
            },
            "required": ["diagnosis", "stage"]
        }
    ),
    Tool(
        name=ToolName.CHECK_GUIDELINES,
        description="Check NCCN and other clinical guidelines for a specific scenario",
        inputSchema={
            "type": "object",
            "properties": {
                "guideline": {
                    "type": "string",
                    "enum": ["NCCN", "ESMO", "ASCO"],
                    "description": "Guideline source"
                },
                "disease": {
                    "type": "string",
                    "description": "Disease type (e.g., 'NSCLC')"
                },
                "scenario": {
                    "type": "string",
                    "description": "Clinical scenario description"
                }
            },
            "required": ["guideline", "disease"]
        }
    ),
    Tool(
        name=ToolName.VALIDATE_ONTOLOGY,
        description="Validate clinical data against LUCADA ontology and SNOMED-CT",
        inputSchema={
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "description": "Clinical data to validate"
                },
                "domain": {
                    "type": "string",
                    "enum": ["diagnosis", "treatment", "biomarker", "staging", "procedure"],
                    "description": "Ontology domain"
                }
            },
            "required": ["data"]
        }
    ),
    Tool(
        name=ToolName.GET_BIOMARKER_INFO,
        description="Get detailed information about a biomarker including testing, prevalence, and targeted therapies",
        inputSchema={
            "type": "object",
            "properties": {
                "biomarker": {
                    "type": "string",
                    "description": "Biomarker name (e.g., 'EGFR', 'ALK', 'PD-L1')"
                },
                "cancer_type": {
                    "type": "string",
                    "description": "Cancer type context"
                }
            },
            "required": ["biomarker"]
        }
    ),
    Tool(
        name=ToolName.FIND_CLINICAL_TRIALS,
        description="Search for matching clinical trials based on patient profile",
        inputSchema={
            "type": "object",
            "properties": {
                "diagnosis": {
                    "type": "string",
                    "description": "Primary diagnosis"
                },
                "biomarkers": {
                    "type": "object",
                    "description": "Biomarker profile"
                },
                "stage": {
                    "type": "string",
                    "description": "Cancer stage"
                },
                "prior_lines": {
                    "type": "integer",
                    "description": "Number of prior treatment lines"
                },
                "location": {
                    "type": "string",
                    "description": "Geographic location for trial search"
                }
            },
            "required": ["diagnosis"]
        }
    ),
    Tool(
        name=ToolName.GET_SURVIVAL_DATA,
        description="Get survival statistics for specific treatment regimens",
        inputSchema={
            "type": "object",
            "properties": {
                "treatment": {
                    "type": "string",
                    "description": "Treatment regimen"
                },
                "indication": {
                    "type": "string",
                    "description": "Treatment indication"
                },
                "biomarker_subgroup": {
                    "type": "string",
                    "description": "Biomarker-defined subgroup"
                },
                "endpoint": {
                    "type": "string",
                    "enum": ["OS", "PFS", "ORR", "DOR"],
                    "description": "Survival endpoint"
                }
            },
            "required": ["treatment", "indication"]
        }
    ),
    Tool(
        name=ToolName.QUERY_KNOWLEDGE_GRAPH,
        description="Execute a Cypher query against the clinical knowledge graph",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Cypher query to execute"
                },
                "parameters": {
                    "type": "object",
                    "description": "Query parameters"
                }
            },
            "required": ["query"]
        }
    ),
    Tool(
        name=ToolName.GET_ARGUMENTATION,
        description="Get guideline-based argumentation for a treatment decision",
        inputSchema={
            "type": "object",
            "properties": {
                "patient_data": {
                    "type": "object",
                    "description": "Patient clinical data"
                },
                "proposed_treatment": {
                    "type": "string",
                    "description": "Proposed treatment to evaluate"
                }
            },
            "required": ["patient_data", "proposed_treatment"]
        }
    )
]

# Define available resources
RESOURCES: List[Resource] = [
    Resource(
        uri="lca://guidelines/nccn/nsclc",
        name="NCCN NSCLC Guidelines",
        description="NCCN Clinical Practice Guidelines for Non-Small Cell Lung Cancer",
        mimeType="application/json"
    ),
    Resource(
        uri="lca://ontology/lucada",
        name="LUCADA Ontology",
        description="Lung Cancer Data Ontology extending SNOMED-CT",
        mimeType="application/json"
    ),
    Resource(
        uri="lca://knowledge-graph/schema",
        name="Knowledge Graph Schema",
        description="Schema of the clinical knowledge graph",
        mimeType="application/json"
    )
]


class MCPServer:
    """MCP Server implementation for Lung Cancer Assistant"""

    def __init__(self):
        self.tools = {tool.name: tool for tool in TOOLS}
        self.resources = {res.uri: res for res in RESOURCES}

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request"""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        try:
            if method == "initialize":
                return self._handle_initialize(request_id, params)
            elif method == "tools/list":
                return self._handle_tools_list(request_id)
            elif method == "tools/call":
                return await self._handle_tool_call(request_id, params)
            elif method == "resources/list":
                return self._handle_resources_list(request_id)
            elif method == "resources/read":
                return await self._handle_resource_read(request_id, params)
            elif method == "prompts/list":
                return self._handle_prompts_list(request_id)
            else:
                return self._error_response(request_id, -32601, f"Method not found: {method}")
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return self._error_response(request_id, -32603, str(e))

    def _handle_initialize(self, request_id: Any, params: Dict) -> Dict:
        """Handle initialize request"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"subscribe": False, "listChanged": True},
                    "prompts": {"listChanged": False}
                },
                "serverInfo": {
                    "name": "lca-mcp-server",
                    "version": "1.0.0"
                }
            }
        }

    def _handle_tools_list(self, request_id: Any) -> Dict:
        """Handle tools/list request"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema
                    }
                    for tool in TOOLS
                ]
            }
        }

    async def _handle_tool_call(self, request_id: Any, params: Dict) -> Dict:
        """Handle tools/call request"""
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        if tool_name not in self.tools:
            return self._error_response(request_id, -32602, f"Unknown tool: {tool_name}")

        # Execute tool
        result = await self._execute_tool(tool_name, tool_args)

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }
                ]
            }
        }

    async def _execute_tool(self, tool_name: str, args: Dict) -> Dict:
        """Execute a tool and return results"""

        if tool_name == ToolName.GET_PATIENT_CONTEXT:
            return self._get_patient_context(args.get("patient_id"))

        elif tool_name == ToolName.SEARCH_SIMILAR_PATIENTS:
            return self._search_similar_patients(
                args.get("diagnosis"),
                args.get("stage"),
                args.get("biomarkers", []),
                args.get("limit", 10)
            )

        elif tool_name == ToolName.GET_TREATMENT_OPTIONS:
            return self._get_treatment_options(
                args.get("diagnosis"),
                args.get("stage"),
                args.get("biomarkers", {}),
                args.get("prior_treatments", []),
                args.get("ecog_ps")
            )

        elif tool_name == ToolName.CHECK_GUIDELINES:
            return self._check_guidelines(
                args.get("guideline"),
                args.get("disease"),
                args.get("scenario")
            )

        elif tool_name == ToolName.VALIDATE_ONTOLOGY:
            return self._validate_ontology(
                args.get("data"),
                args.get("domain")
            )

        elif tool_name == ToolName.GET_BIOMARKER_INFO:
            return self._get_biomarker_info(
                args.get("biomarker"),
                args.get("cancer_type")
            )

        elif tool_name == ToolName.FIND_CLINICAL_TRIALS:
            return self._find_clinical_trials(
                args.get("diagnosis"),
                args.get("biomarkers", {}),
                args.get("stage"),
                args.get("prior_lines"),
                args.get("location")
            )

        elif tool_name == ToolName.GET_SURVIVAL_DATA:
            return self._get_survival_data(
                args.get("treatment"),
                args.get("indication"),
                args.get("biomarker_subgroup"),
                args.get("endpoint")
            )

        elif tool_name == ToolName.QUERY_KNOWLEDGE_GRAPH:
            return self._query_knowledge_graph(
                args.get("query"),
                args.get("parameters", {})
            )

        elif tool_name == ToolName.GET_ARGUMENTATION:
            return self._get_argumentation(
                args.get("patient_data"),
                args.get("proposed_treatment")
            )

        return {"error": "Tool not implemented"}

    # Tool implementations
    def _get_patient_context(self, patient_id: str) -> Dict:
        """Get patient context - mock implementation"""
        return {
            "patient_id": patient_id,
            "demographics": {
                "age": 65,
                "gender": "Male",
                "smoking_status": "Former"
            },
            "diagnosis": {
                "primary": "Non-Small Cell Lung Cancer",
                "histology": "Adenocarcinoma",
                "stage": "IV",
                "date": "2024-01-15"
            },
            "biomarkers": {
                "EGFR": {"status": "Positive", "variant": "L858R"},
                "ALK": {"status": "Negative"},
                "PD-L1": {"status": "Positive", "tps": 45},
                "KRAS": {"status": "Negative"}
            },
            "treatments": [
                {
                    "name": "Osimertinib",
                    "line": 1,
                    "start_date": "2024-02-01",
                    "status": "Active",
                    "response": "Partial Response"
                }
            ],
            "performance_status": {
                "ecog": 1,
                "date": "2024-03-01"
            }
        }

    def _search_similar_patients(self, diagnosis: str, stage: Optional[str],
                                  biomarkers: List[str], limit: int) -> Dict:
        """Search similar patients - mock implementation"""
        return {
            "query": {
                "diagnosis": diagnosis,
                "stage": stage,
                "biomarkers": biomarkers
            },
            "count": 3,
            "patients": [
                {"id": "P001", "similarity": 0.95, "diagnosis": diagnosis, "stage": stage},
                {"id": "P045", "similarity": 0.88, "diagnosis": diagnosis, "stage": stage},
                {"id": "P112", "similarity": 0.82, "diagnosis": diagnosis, "stage": stage}
            ]
        }

    def _get_treatment_options(self, diagnosis: str, stage: str,
                               biomarkers: Dict, prior_treatments: List[str],
                               ecog_ps: Optional[int]) -> Dict:
        """Get treatment options - mock implementation"""
        options = []

        if biomarkers.get("EGFR"):
            options.append({
                "treatment": "Osimertinib",
                "category": "Targeted Therapy",
                "evidence_level": "Category 1",
                "line": "1st line",
                "rationale": "Preferred for EGFR-mutated NSCLC",
                "efficacy": {"PFS": "18.9 months", "ORR": "80%"}
            })

        if biomarkers.get("PD-L1", {}).get("tps", 0) >= 50:
            options.append({
                "treatment": "Pembrolizumab",
                "category": "Immunotherapy",
                "evidence_level": "Category 1",
                "line": "1st line",
                "rationale": "For PD-L1 >= 50% without driver mutations",
                "efficacy": {"OS": "30 months", "ORR": "45%"}
            })

        if not options:
            options.append({
                "treatment": "Carboplatin/Pemetrexed/Pembrolizumab",
                "category": "Chemo-Immunotherapy",
                "evidence_level": "Category 1",
                "line": "1st line",
                "rationale": "Standard of care for non-squamous NSCLC",
                "efficacy": {"OS": "22 months", "ORR": "48%"}
            })

        return {
            "diagnosis": diagnosis,
            "stage": stage,
            "options": options
        }

    def _check_guidelines(self, guideline: str, disease: str,
                          scenario: Optional[str]) -> Dict:
        """Check clinical guidelines - mock implementation"""
        return {
            "guideline": guideline,
            "disease": disease,
            "version": "2024.1",
            "recommendations": [
                {
                    "category": "First-Line Therapy",
                    "level": "Category 1",
                    "recommendation": "Osimertinib for EGFR exon 19 deletion or L858R mutations",
                    "evidence": "FLAURA trial"
                },
                {
                    "category": "Biomarker Testing",
                    "level": "Category 1",
                    "recommendation": "Broad molecular profiling for all patients with advanced NSCLC",
                    "evidence": "Multiple studies"
                }
            ]
        }

    def _validate_ontology(self, data: Dict, domain: Optional[str]) -> Dict:
        """Validate against ontology - mock implementation"""
        validations = []
        for key, value in data.items():
            validations.append({
                "field": key,
                "value": value,
                "valid": True,
                "snomed_code": "254637007" if "cancer" in str(value).lower() else None,
                "mapped_concept": value
            })

        return {
            "valid": True,
            "domain": domain or "general",
            "validations": validations
        }

    def _get_biomarker_info(self, biomarker: str, cancer_type: Optional[str]) -> Dict:
        """Get biomarker information - mock implementation"""
        biomarker_data = {
            "EGFR": {
                "name": "Epidermal Growth Factor Receptor",
                "type": "Oncogene",
                "prevalence": {"NSCLC": "15-20%", "Asian_NSCLC": "40-50%"},
                "testing": ["NGS", "PCR", "IHC"],
                "targeted_therapies": ["Osimertinib", "Erlotinib", "Gefitinib", "Afatinib"],
                "common_mutations": ["Exon 19 deletion", "L858R", "T790M"]
            },
            "ALK": {
                "name": "Anaplastic Lymphoma Kinase",
                "type": "Fusion Oncogene",
                "prevalence": {"NSCLC": "3-7%"},
                "testing": ["FISH", "IHC", "NGS"],
                "targeted_therapies": ["Alectinib", "Brigatinib", "Lorlatinib", "Crizotinib"],
                "common_fusions": ["EML4-ALK"]
            },
            "PD-L1": {
                "name": "Programmed Death-Ligand 1",
                "type": "Immune Checkpoint",
                "prevalence": {"NSCLC": "25-30% (>= 50%)"},
                "testing": ["IHC (22C3, SP263)"],
                "targeted_therapies": ["Pembrolizumab", "Atezolizumab", "Durvalumab"],
                "thresholds": ["< 1%", "1-49%", ">= 50%"]
            }
        }

        info = biomarker_data.get(biomarker.upper(), {
            "name": biomarker,
            "message": "Limited information available"
        })

        return {
            "biomarker": biomarker,
            "cancer_type": cancer_type,
            **info
        }

    def _find_clinical_trials(self, diagnosis: str, biomarkers: Dict,
                              stage: Optional[str], prior_lines: Optional[int],
                              location: Optional[str]) -> Dict:
        """Find clinical trials - mock implementation"""
        return {
            "query": {
                "diagnosis": diagnosis,
                "biomarkers": biomarkers,
                "stage": stage
            },
            "count": 2,
            "trials": [
                {
                    "nct_id": "NCT04487080",
                    "title": "Osimertinib Plus Savolitinib in EGFRm NSCLC",
                    "phase": "Phase III",
                    "status": "Recruiting",
                    "match_score": 92
                },
                {
                    "nct_id": "NCT04538664",
                    "title": "Amivantamab + Lazertinib vs Osimertinib",
                    "phase": "Phase III",
                    "status": "Recruiting",
                    "match_score": 88
                }
            ]
        }

    def _get_survival_data(self, treatment: str, indication: str,
                           biomarker_subgroup: Optional[str],
                           endpoint: Optional[str]) -> Dict:
        """Get survival data - mock implementation"""
        return {
            "treatment": treatment,
            "indication": indication,
            "biomarker_subgroup": biomarker_subgroup,
            "endpoints": {
                "OS": {"median": "38.6 months", "HR": 0.80, "CI": "0.64-1.00"},
                "PFS": {"median": "18.9 months", "HR": 0.46, "CI": "0.37-0.57"},
                "ORR": {"rate": "80%", "CR": "3%", "PR": "77%"}
            },
            "source": "FLAURA trial",
            "publication": "NEJM 2020"
        }

    def _query_knowledge_graph(self, query: str, parameters: Dict) -> Dict:
        """Query knowledge graph - mock implementation"""
        return {
            "query": query,
            "parameters": parameters,
            "result": {
                "message": "Knowledge graph query executed",
                "nodes": 5,
                "relationships": 8
            }
        }

    def _get_argumentation(self, patient_data: Dict, proposed_treatment: str) -> Dict:
        """Get guideline argumentation - mock implementation"""
        return {
            "proposed_treatment": proposed_treatment,
            "patient_context": patient_data,
            "arguments_for": [
                {
                    "source": "NCCN Guidelines",
                    "strength": "Strong",
                    "text": f"{proposed_treatment} is Category 1 recommended for this indication"
                }
            ],
            "arguments_against": [],
            "recommendation": "Supported",
            "confidence": 0.95
        }

    def _handle_resources_list(self, request_id: Any) -> Dict:
        """Handle resources/list request"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "resources": [
                    {
                        "uri": res.uri,
                        "name": res.name,
                        "description": res.description,
                        "mimeType": res.mimeType
                    }
                    for res in RESOURCES
                ]
            }
        }

    async def _handle_resource_read(self, request_id: Any, params: Dict) -> Dict:
        """Handle resources/read request"""
        uri = params.get("uri")

        if uri not in self.resources:
            return self._error_response(request_id, -32602, f"Unknown resource: {uri}")

        # Return resource content
        content = self._get_resource_content(uri)

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(content, indent=2)
                    }
                ]
            }
        }

    def _get_resource_content(self, uri: str) -> Dict:
        """Get resource content"""
        if uri == "lca://guidelines/nccn/nsclc":
            return {
                "name": "NCCN NSCLC Guidelines",
                "version": "2024.1",
                "sections": ["Diagnosis", "Staging", "Treatment", "Surveillance"]
            }
        elif uri == "lca://ontology/lucada":
            return {
                "name": "LUCADA Ontology",
                "domains": ["Diagnosis", "Treatment", "Biomarker", "Staging"],
                "base": "SNOMED-CT"
            }
        elif uri == "lca://knowledge-graph/schema":
            return {
                "nodes": ["Patient", "Diagnosis", "Treatment", "Biomarker", "Guideline"],
                "relationships": ["HAS_DIAGNOSIS", "RECEIVED_TREATMENT", "HAS_BIOMARKER"]
            }
        return {}

    def _handle_prompts_list(self, request_id: Any) -> Dict:
        """Handle prompts/list request"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "prompts": []
            }
        }

    def _error_response(self, request_id: Any, code: int, message: str) -> Dict:
        """Create error response"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }


async def main():
    """Main entry point for MCP server"""
    server = MCPServer()

    logger.info("LCA MCP Server starting...")

    # Read from stdin, write to stdout (stdio transport)
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            request = json.loads(line)
            response = await server.handle_request(request)

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
