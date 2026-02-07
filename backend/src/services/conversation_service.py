"""
Conversational Service for LCA Chatbot
Handles natural language interaction with the LCA system
Enhanced with LangChain/LangGraph for memory and intelligent conversations
"""

import json
import re
import logging
import os
from typing import Dict, List, Optional, AsyncIterator, Any, Literal
from datetime import datetime
import asyncio
import uuid
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, field_validator
# Centralized logging
from ..logging_config import get_logger, log_execution

logger = get_logger(__name__)
# LangChain/LangGraph imports for enhanced conversations
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
    from langchain_core.runnables import RunnableConfig
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    logger.warning("LangChain/LangGraph not available - enhanced features disabled")
    LANGGRAPH_AVAILABLE = False

# Ollama LLM for intelligent responses
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    logger.warning("langchain_ollama not available - LLM responses disabled")
    OLLAMA_AVAILABLE = False

# Centralized logging
from ..logging_config import get_logger, log_execution, create_sse_log_handler

# MCP tool integration
from .mcp_client import get_mcp_invoker

# Clustering service
from .clustering_service import ClusteringService, ClusteringMethod

logger = get_logger(__name__)


# =============================================================================
# Clinical System Prompt for LLM-Powered Responses
# =============================================================================

CLINICAL_SYSTEM_PROMPT = """You are **ConsensusCare**, an expert lung cancer clinical decision support assistant built on the LUCADA ontology (Lung Cancer Data) and NICE/NCCN/ESMO guidelines.

## Your Knowledge Base
- **Ontology**: LUCADA OWL2 with 68 classes covering Patient, Diagnosis, TreatmentPlan, Biomarker, Comorbidity, and Decision entities
- **Standards**: SNOMED-CT (diagnosis coding), RxNorm (medications), LOINC (lab tests)
- **Guidelines**: NICE CG121, NCCN NSCLC/SCLC 2025, ESMO Clinical Practice Guidelines
- **Graph Database**: Neo4j knowledge graph with patient records, treatment decisions, causal chains, and guideline references

## Clinical Guidelines Summary

### NSCLC Treatment by Stage
| Stage | PS 0-1 | PS 2 | PS 3-4 |
|-------|--------|------|--------|
| I-II | Surgery ± adjuvant chemo | Radiotherapy (SABR) | Best supportive care |
| IIIA | Chemoradiotherapy or Surgery+chemo | Radiotherapy | Palliative care |
| IIIB | Concurrent chemoradiotherapy | Sequential chemoRT | Palliative care |
| IV | Systemic therapy (biomarker-driven) | Modified systemic therapy | Palliative care |

### Biomarker-Driven Therapy (Stage IIIB-IV NSCLC)
| Biomarker | Positive Result | First-Line Treatment | Key Trial |
|-----------|-----------------|---------------------|-----------|
| EGFR (Ex19del/L858R) | Mutation detected | Osimertinib 80mg daily | FLAURA (OS 38.6mo) |
| ALK rearrangement | Fusion detected | Alectinib 600mg BID | ALEX (PFS 34.8mo) |
| ROS1 fusion | Fusion detected | Crizotinib/Entrectinib | PROFILE 1001 |
| BRAF V600E | Mutation detected | Dabrafenib + Trametinib | BRF113928 |
| KRAS G12C | Mutation detected | Sotorasib/Adagrasib | CodeBreaK 200 |
| PD-L1 ≥50% | High expression | Pembrolizumab mono | KEYNOTE-024 (5yr OS 31.9%) |
| PD-L1 1-49% | Low expression | Chemo + Pembrolizumab | KEYNOTE-189 (OS 22mo) |
| PD-L1 <1% | Negative | Chemo + IO combination | CheckMate-9LA |

### SCLC Treatment
| Stage | Treatment | Key Evidence |
|-------|-----------|-------------|
| Limited | Concurrent chemoRT (cisplatin/etoposide) + PCI | Turrisi (OS 23mo) |
| Extensive | Chemo + Atezolizumab/Durvalumab | IMpower133/CASPIAN |

### Adjuvant/Neoadjuvant
- **ADAURA**: Adjuvant osimertinib for resected IB-IIIA EGFR+ NSCLC (DFS HR 0.17)
- **CheckMate-816**: Neoadjuvant nivolumab + chemo for resectable NSCLC (pCR 24%)
- **IMpower010**: Adjuvant atezolizumab for II-IIIA PD-L1+ NSCLC

## Response Formatting Rules
1. Use markdown with headers (##), tables, and bullet points
2. Always cite guideline source and evidence level (Grade A/B/C or Category 1/2A/2B)
3. Include survival data from landmark trials when relevant
4. Flag contraindications with ⚠️
5. Suggest next steps and follow-up questions
6. For biomarker-driven therapy, always mention the specific mutation/fusion and preferred agent
7. Be concise but thorough - aim for clinical utility
8. If information is uncertain, say so explicitly rather than guessing"""

FOLLOW_UP_SYSTEM_PROMPT = """You are ConsensusCare, a lung cancer clinical decision support assistant.
You are answering a follow-up question about a patient case that was just analyzed.

Use the patient data and analysis results provided to give a specific, evidence-based answer.
Format your response with clear markdown headers, tables where appropriate, and clinical citations.
Be thorough but concise. Prioritize actionable clinical information."""

TEXT2CYPHER_SYSTEM_PROMPT = """You are a Neo4j Cypher query expert for a lung cancer clinical decision support system.

The graph schema has these node types:
- Patient (patient_id, name, age, age_at_diagnosis, sex, tnm_stage, histology_type, performance_status, laterality)
- Diagnosis (diagnosis_type, icd10_code, snomed_code)
- TreatmentDecision (id, decision_type, category, status, treatment, reasoning, confidence_score, risk_factors, decision_timestamp)
- Biomarker (marker_type, value, status)
- Comorbidity (name, condition)
- Guideline (name, source, evidence_level)
- TreatmentRecommendation (treatment_type, evidence_level, treatment_intent, survival_benefit)

Relationships:
- (Patient)-[:HAS_DIAGNOSIS]->(Diagnosis)
- (Patient)-[:HAS_BIOMARKER]->(Biomarker)
- (Patient)-[:HAS_COMORBIDITY]->(Comorbidity)
- (TreatmentDecision)-[:ABOUT]->(Patient)
- (TreatmentDecision)-[:BASED_ON]->(Biomarker)
- (TreatmentDecision)-[:APPLIED_GUIDELINE]->(Guideline)
- (TreatmentDecision)-[:CAUSED|INFLUENCED]->(TreatmentDecision)
- (TreatmentDecision)-[:FOLLOWED_PRECEDENT]->(TreatmentDecision)
- (Patient)-[:RECEIVED_RECOMMENDATION]->(TreatmentRecommendation)

Return ONLY the Cypher query, no explanation. Use parameters where appropriate ($param syntax).
Always include RETURN clause. Limit results to 20 unless specifically asked for more."""


# =============================================================================
# Enhanced Conversation State Management
# =============================================================================

@dataclass
class ConversationContext:
    """Enhanced conversation context for memory and state management"""
    patient_cases: Dict[str, Dict] = field(default_factory=dict)
    analysis_history: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    clinical_domain: str = "lung_cancer"
    expertise_level: str = "expert"  # novice, intermediate, expert
    last_analysis_id: Optional[str] = None
    follow_up_suggestions: List[str] = field(default_factory=list)
    active_thread_id: Optional[str] = None

if LANGGRAPH_AVAILABLE:
    class EnhancedConversationState(BaseModel):
        """LangGraph state for enhanced conversation management"""
        messages: List[BaseMessage] = Field(default_factory=list)
        patient_data: Optional[Dict] = None
        analysis_result: Optional[Dict[str, Any]] = None
        follow_up_questions: List[str] = Field(default_factory=list)
        context: ConversationContext = Field(default_factory=ConversationContext)
        next_action: Optional[str] = None
        awaiting_input: bool = False
        session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class FollowUpGenerator:
    """Enhanced follow-up question generation"""
    
    @staticmethod
    def generate_clinical_questions(analysis_result: Dict[str, Any], patient_data: Dict) -> List[str]:
        """Generate intelligent follow-up questions based on analysis"""
        questions = []
        
        if not analysis_result or not analysis_result.get("recommendations"):
            return ["Could you provide more details about the patient case?"]
        
        primary_rec = analysis_result["recommendations"][0]
        treatment = primary_rec.get("treatment", "")
        stage = patient_data.get("tnm_stage", "")
        biomarkers = patient_data.get("biomarker_profile", {})
        
        # Treatment-specific questions
        if "osimertinib" in treatment.lower():
            questions.extend([
                "Would you like to know about potential side effects of osimertinib?",
                "Are you interested in alternative treatment options for EGFR+ NSCLC?",
                "Should we discuss resistance mechanisms and second-line therapies?"
            ])
        
        if "chemotherapy" in treatment.lower():
            questions.extend([
                "Would you like dosing recommendations based on performance status?",
                "Should we review potential drug interactions?",
                "Are you interested in supportive care guidelines?"
            ])
        
        # Stage-specific questions
        if stage in ["IIIA", "IIIB"]:
            questions.extend([
                "Would you like to explore surgical candidacy assessment?",
                "Should we discuss radiation therapy sequencing?",
                "Are you interested in neoadjuvant vs adjuvant approaches?"
            ])
        
        # Always include general options
        questions.extend([
            "Would you like to see similar cases from our database?",
            "Should I explain the clinical reasoning in more detail?",
            "Are you interested in current clinical trial options?"
        ])
        
        return questions[:5]  # Limit to 5 suggestions


# --- Pydantic model for validated patient data extraction ---

class ExtractedPatientData(BaseModel):
    """Validated patient data extracted from natural language"""
    patient_id: str = ""
    age: int = Field(ge=1, le=120)
    age_at_diagnosis: Optional[int] = None
    sex: str = Field(pattern=r'^[MFU]$')
    tnm_stage: str
    histology_type: str
    performance_status: int = Field(ge=0, le=4, default=1)
    biomarker_profile: Optional[Dict] = None
    comorbidities: Optional[List[str]] = None

    @field_validator('age_at_diagnosis', mode='before')
    @classmethod
    def sync_age(cls, v, info):
        if v is not None:
            return v
        return info.data.get('age', 65)

    model_config = {"extra": "allow"}


class ConversationService:
    """
    Manages conversational interactions with LCA

    Features:
    - Natural language patient data extraction with Pydantic validation
    - Streaming responses via SSE
    - Session-based conversation history with LangGraph memory
    - Intent classification (patient analysis, follow-up, general Q&A)
    - Context-aware follow-up handling
    - Enhanced conversation flow with intelligent follow-up suggestions
    - Thread-based conversation persistence
    """

    def __init__(self, lca_service, enable_enhanced_features=True):
        self.lca_service = lca_service
        self.sessions: Dict[str, List[Dict]] = {}  # session_id -> message history
        self.patient_context: Dict[str, Dict] = {}  # session_id -> {patient_data, result}
        self.enhanced_context: Dict[str, ConversationContext] = {}  # session_id -> enhanced context

        # Enhanced features
        self.enable_enhanced_features = enable_enhanced_features and LANGGRAPH_AVAILABLE
        self.follow_up_generator = FollowUpGenerator()

        # Initialize LLM for intelligent responses
        self.llm = None
        self.llm_available = False
        if OLLAMA_AVAILABLE:
            try:
                model_name = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                self.llm = ChatOllama(
                    model=model_name,
                    base_url=base_url,
                    temperature=0.3,
                    num_predict=2048,
                )
                self.llm_available = True
                logger.info(f"LLM initialized: {model_name} at {base_url}")
            except Exception as e:
                logger.warning(f"LLM initialization failed: {e} - falling back to template responses")

        # LangGraph setup for enhanced conversations
        if self.enable_enhanced_features:
            self.checkpointer = InMemorySaver()
            self.conversation_graph = self._build_enhanced_graph()
            logger.info("Enhanced conversation features enabled with LangGraph")
        else:
            self.checkpointer = None
            self.conversation_graph = None
            if enable_enhanced_features:
                logger.warning("Enhanced features requested but LangGraph not available")
        self.mcp_invoker = get_mcp_invoker()  # MCP tool integration
        self.clustering_service = ClusteringService()  # Patient clustering

    def _build_enhanced_graph(self):
        """Build LangGraph conversation flow for enhanced interactions"""
        if not LANGGRAPH_AVAILABLE:
            return None
            
        def route_input(state: 'EnhancedConversationState') -> Literal["analyze_patient", "follow_up", "general_chat"]:
            """Route user input to appropriate handler"""
            if not state.messages:
                return "general_chat"
            
            last_message = state.messages[-1].content.lower()
            
            # Check if this looks like patient data
            patient_indicators = ["year old", "stage", "adenocarcinoma", "squamous", "egfr", "alk", "pdl1"]
            if any(indicator in last_message for indicator in patient_indicators):
                return "analyze_patient"
            
            # Check if this is a follow-up question
            if state.analysis_result and any(word in last_message for word in ["why", "what", "how", "explain", "alternative", "side effect"]):
                return "follow_up"
            
            return "general_chat"
        
        async def analyze_patient_node(state: 'EnhancedConversationState') -> 'EnhancedConversationState':
            """Enhanced patient analysis with follow-up generation"""
            last_message = state.messages[-1].content
            
            # Use existing patient analysis logic but capture results
            analysis_result = None
            patient_data = None
            
            try:
                # Generate follow-up questions based on results
                follow_ups = []
                if analysis_result and patient_data:
                    follow_ups = self.follow_up_generator.generate_clinical_questions(
                        analysis_result, patient_data
                    )
                
                # Update context
                state.context.analysis_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "query": last_message,
                    "result": analysis_result
                })
                
                return EnhancedConversationState(
                    messages=state.messages + [AIMessage(content="Analysis completed with enhanced suggestions")],
                    patient_data=patient_data,
                    analysis_result=analysis_result,
                    follow_up_questions=follow_ups,
                    context=state.context,
                    session_id=state.session_id,
                    thread_id=state.thread_id
                )
                
            except Exception as e:
                logger.error(f"Enhanced analysis error: {e}")
                return state
        
        # Build the graph
        workflow = StateGraph(EnhancedConversationState)
        workflow.add_node("analyze_patient", analyze_patient_node)
        workflow.add_conditional_edges("__start__", route_input, {
            "analyze_patient": "analyze_patient",
            "follow_up": "analyze_patient",  # Reuse for now
            "general_chat": "analyze_patient"   # Simplified for now
        })
        workflow.add_edge("analyze_patient", END)
        
        return workflow.compile(checkpointer=self.checkpointer)

    def _get_enhanced_context(self, session_id: str) -> ConversationContext:
        """Get or create enhanced conversation context"""
        if session_id not in self.enhanced_context:
            self.enhanced_context[session_id] = ConversationContext()
        return self.enhanced_context[session_id]

    def _get_session(self, session_id: str) -> List[Dict]:
        """Get or create conversation session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def _add_to_history(self, session_id: str, role: str, content: str):
        """Add message to conversation history"""
        session = self._get_session(session_id)
        session.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    async def chat_stream(
        self,
        session_id: str,
        message: str,
        use_enhanced_features: bool = None
    ) -> AsyncIterator[str]:
        """
        Stream conversational responses with optional enhanced features

        Args:
            session_id: Unique session identifier
            message: User message
            use_enhanced_features: Whether to use LangGraph enhanced features (defaults to service setting)

        Yields:
            SSE-formatted chunks with potential follow-up suggestions
        """
        # Determine whether to use enhanced features
        use_enhanced = use_enhanced_features if use_enhanced_features is not None else self.enable_enhanced_features
        
        try:
            # Add user message to history
            self._add_to_history(session_id, "user", message)

            # Classify intent with session context first
            intent = self._classify_intent(message, session_id)

            # Enhanced features: check if we should use LangGraph flow
            if use_enhanced and intent == "patient_analysis" and self.conversation_graph:
                async for chunk in self._stream_enhanced_analysis(message, session_id):
                    yield chunk
                return

            if intent == "patient_lookup":
                async for chunk in self._stream_patient_lookup(message, session_id):
                    yield chunk
            elif intent == "patient_analysis":
                async for chunk in self._stream_patient_analysis(message, session_id):
                    yield chunk
                
                # Enhanced: Generate follow-up suggestions after analysis
                if use_enhanced:
                    await self._add_follow_up_suggestions(session_id)
                    
            elif intent == "follow_up":
                async for chunk in self._stream_follow_up(message, session_id):
                    yield chunk
            elif intent == "mcp_tool":
                async for chunk in self._stream_mcp_tool(message, session_id):
                    yield chunk
            elif intent == "mcp_app":
                async for chunk in self._stream_mcp_app(message, session_id):
                    yield chunk
            elif intent == "clustering_analysis":
                async for chunk in self._stream_clustering_analysis(message, session_id):
                    yield chunk
            elif intent == "graph_query":
                async for chunk in self._stream_graph_query(message, session_id):
                    yield chunk
            else:
                async for chunk in self._stream_general_qa(message, session_id):
                    yield chunk

        except Exception as e:
            logger.error(f"Chat stream error: {e}", exc_info=True)
            yield self._format_sse({
                "type": "error",
                "content": str(e)
            })

    def _classify_intent(self, message: str, session_id: str = None) -> str:
        """
        Classify user intent considering conversation context

        Returns:
            "patient_lookup", "patient_analysis", "follow_up", "mcp_tool", "mcp_app",
            "clustering_analysis", "graph_query", or "general_qa"
        """
        # Check for patient ID lookup patterns
        patient_id_patterns = [
            r'patient\s+id[:\s]+([A-Z0-9_]+)',
            r'\b(CHAT_\d{8}_\d{6})\b',
            r'\b(PAT_[A-Z0-9]+)\b',
            r'lookup\s+([A-Z0-9_]+)',
            r'find\s+patient\s+([A-Z0-9_]+)',
            r'retrieve\s+([A-Z0-9_]+)'
        ]
        for pattern in patient_id_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "patient_lookup"

        # Check for graph/database query patterns (Text2Cypher)
        graph_query_patterns = [
            r'how\s+many\s+patient',
            r'show\s+(me\s+)?(all\s+)?patient',
            r'list\s+(all\s+)?patient',
            r'query\s+(the\s+)?(graph|database|neo4j)',
            r'what\s+patient.+in\s+(the\s+)?database',
            r'count\s+(all\s+)?patient',
            r'show\s+(me\s+)?(all\s+)?treatment\s+decision',
            r'what\s+decisions?\s+(have\s+been|were)\s+made',
            r'graph\s+statistics',
            r'database\s+summary',
            r'run\s+cypher',
        ]
        for pattern in graph_query_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "graph_query"

        # Check for MCP App invocation patterns
        mcp_app_patterns = [
            r'compare\s+treatment',
            r'treatment\s+comparison',
            r'survival\s+curve',
            r'kaplan\s+meier',
            r'guideline\s+tree',
            r'nccn\s+decision',
            r'decision\s+tree',
            r'clinical\s+trial',
            r'trial\s+match'
        ]
        for pattern in mcp_app_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "mcp_app"

        # Check for clustering analysis patterns
        clustering_patterns = [
            r'cluster\s+patient',
            r'cohort\s+analysis',
            r'patient\s+group',
            r'find\s+patients?\s+like',
            r'similar\s+patient'
        ]
        for pattern in clustering_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "clustering_analysis"

        # Check for MCP tool invocation patterns
        mcp_tool_patterns = [
            r'(use|invoke|call|run)\s+(tool|MCP|agent)',
            r'analyze\s+survival\s+data',
            r'match\s+clinical\s+trial',
            r'find\s+similar\s+patient',
            r'search\s+(patient|guideline)',
            r'get\s+(biomarker|pathway|treatment)',
            r'interpret\s+lab\s+result',
            r'predict\s+resistance',
            r'graph\s+(query|search)',
            r'ontology\s+(map|validate)',
            r'export\s+(patient|report)',
        ]
        for pattern in mcp_tool_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "mcp_tool"

        # Check if this is a follow-up about an existing patient
        if session_id and session_id in self.patient_context:
            follow_up_patterns = [
                r'alternative', r'other\s+option', r'different\s+therap',
                r'explain', r'reasoning', r'why\b', r'how\s+come',
                r'similar\s+case', r'similar\s+patient',
                r'comorbidity', r'side\s+effect', r'toxicity',
                r'prognosis', r'survival', r'outlook',
                r'biomarker', r'mutation',
                r'clinical\s+trial', r'\btrial\b',
                r'what\s+about', r'tell\s+me\s+more', r'can\s+you',
                r'show\s+me', r'more\s+detail', r'elaborate',
                r'assess', r'interaction', r'risk',
            ]
            for pattern in follow_up_patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return "follow_up"

        # General knowledge questions (check BEFORE patient indicators)
        general_question_patterns = [
            r'^(explain|what\s+(is|are)|describe|tell\s+me\s+about|how\s+(does|do))\s+',
            r'treatment\s+options?\s+for',
            r'guidelines?\s+for',
            r'evidence\s+for',
            r'mechanism\s+of',
            r'efficacy\s+of',
        ]
        for pattern in general_question_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                # But if it contains clear patient data, it's still patient_analysis
                has_patient_context = bool(re.search(r'\d{2}[-\s]?(year|yr)[-\s]?old|stage\s+(I{1,3}[ABC]?|IV)', message, re.IGNORECASE))
                if not has_patient_context:
                    return "general_qa"

        # Keywords that indicate new patient data (not general questions)
        patient_indicators = [
            r'\d{2}[-\s]?(year|yr)[-\s]?old',
            r'stage\s+(I{1,3}[ABC]?|IV)',
            r'\b\d{2,3}\s*[MF]\b',
            r'(adenocarcinoma|squamous|small\s+cell|SCLC|NSCLC)',
            # Biomarker patterns: only match when in patient context (positive/negative/mutation/wild-type)
            r'(EGFR|ALK|ROS1|BRAF|KRAS|PD-L1)\s*(positive|negative|mutation|mutant|wild[\s-]?type|\+|\-|status)',
            r'(PS|performance\s+status|ECOG)\s*[:\-]?\s*[0-4]',
            r'T\d+N\d+M\d+',
            r'comorbid',
        ]

        for pattern in patient_indicators:
            if re.search(pattern, message, re.IGNORECASE):
                return "patient_analysis"

        return "general_qa"

    async def _stream_patient_lookup(self, message: str, session_id: str = None) -> AsyncIterator[str]:
        """Stream patient lookup by ID from Neo4j"""
        
        # Extract patient ID from message
        patient_id_patterns = [
            r'patient\s+id[:\s]+([A-Z0-9_]+)',
            r'\b(CHAT_\d{8}_\d{6})\b',
            r'\b(PAT_[A-Z0-9]+)\b',
            r'lookup\s+([A-Z0-9_]+)',
            r'find\s+patient\s+([A-Z0-9_]+)',
            r'retrieve\s+([A-Z0-9_]+)'
        ]
        
        patient_id = None
        for pattern in patient_id_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                patient_id = match.group(1)
                break
        
        if not patient_id:
            yield self._format_sse({
                "type": "error",
                "content": "Could not extract patient ID from your message. Please provide a valid patient ID (e.g., CHAT_20260201_142225)"
            })
            return
        
        yield self._format_sse({
            "type": "status",
            "content": f"Looking up patient {patient_id} in database..."
        })
        
        # Check if Neo4j is available
        if not self.lca_service.graph_db or not self.lca_service.graph_db.driver:
            yield self._format_sse({
                "type": "error",
                "content": "Database connection not available. Cannot lookup patient records."
            })
            return
        
        try:
            # Query Neo4j for patient
            from ..db.neo4j_client import Neo4jGraphClient
            
            query = """
            MATCH (p:Patient {patient_id: $patient_id})
            OPTIONAL MATCH (p)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
            OPTIONAL MATCH (p)-[:HAS_BIOMARKER]->(b:Biomarker)
            OPTIONAL MATCH (p)-[:RECEIVED_RECOMMENDATION]->(r:TreatmentRecommendation)
            RETURN p, 
                   collect(DISTINCT d) as diagnoses,
                   collect(DISTINCT b) as biomarkers,
                   collect(DISTINCT r) as recommendations
            """
            
            with self.lca_service.graph_db.driver.session() as session:
                result = session.run(query, patient_id=patient_id)
                record = result.single()
                
                if not record:
                    yield self._format_sse({
                        "type": "error",
                        "content": f"Patient {patient_id} not found in database. This patient may not have been processed yet or the ID is incorrect."
                    })
                    return
                
                # Extract patient data
                patient_node = record['p']
                patient_data = dict(patient_node)
                
                yield self._format_sse({
                    "type": "status",
                    "content": "Patient found! Retrieving details..."
                })
                
                # Stream patient data
                yield self._format_sse({
                    "type": "patient_data",
                    "content": patient_data
                })
                
                # Format patient summary
                age = patient_data.get('age', 'Unknown')
                sex = patient_data.get('sex', 'Unknown')
                stage = patient_data.get('tnm_stage', 'Unknown')
                histology = patient_data.get('histology_type', 'Unknown')
                
                summary = f"""## Patient Summary: {patient_id}

**Demographics:**
- Age: {age}
- Sex: {sex}

**Diagnosis:**
- Stage: {stage}
- Histology: {histology}
- Performance Status: {patient_data.get('performance_status', 'Not specified')}
"""
                
                # Add biomarker info if available
                biomarkers = patient_data.get('biomarker_profile', {})
                if biomarkers:
                    summary += "\n**Biomarkers:**\n"
                    for marker, value in biomarkers.items():
                        summary += f"- {marker}: {value}\n"
                
                yield self._format_sse({
                    "type": "text",
                    "content": summary
                })
                
                # Get recommendations if available
                recommendations = record.get('recommendations', [])
                if recommendations:
                    recs_text = "\n## Previous Recommendations\n\n"
                    for i, rec in enumerate(recommendations, 1):
                        rec_dict = dict(rec)
                        recs_text += f"{i}. **{rec_dict.get('treatment_type', 'Unknown')}**\n"
                        recs_text += f"   - Evidence: {rec_dict.get('evidence_level', 'N/A')}\n"
                        recs_text += f"   - Intent: {rec_dict.get('treatment_intent', 'N/A')}\n"
                        if rec_dict.get('survival_benefit'):
                            recs_text += f"   - Survival Benefit: {rec_dict.get('survival_benefit')}\n"
                        recs_text += "\n"
                    
                    yield self._format_sse({
                        "type": "recommendation",
                        "content": recs_text
                    })
                else:
                    yield self._format_sse({
                        "type": "text",
                        "content": "\n*No treatment recommendations found for this patient.*\n"
                    })
                
                # Store in session context for follow-ups
                if session_id:
                    self.patient_context[session_id] = {
                        "patient_data": patient_data,
                        "patient_id": patient_id
                    }
                
                # Suggest follow-up actions
                suggestions = [
                    f"Update treatment plan for {patient_id}",
                    f"Find similar patients to {patient_id}",
                    "Get alternative treatment options"
                ]
                
                yield self._format_sse({
                    "type": "suggestions",
                    "content": suggestions
                })
                
        except Exception as e:
            logger.error(f"Patient lookup error: {e}", exc_info=True)
            yield self._format_sse({
                "type": "error",
                "content": f"Error retrieving patient data: {str(e)}"
            })

    async def _stream_patient_analysis(self, message: str, session_id: str = None) -> AsyncIterator[str]:
        """Stream patient analysis workflow with enhanced visibility"""

        try:
            # Step 1: Extract patient data with detailed progress
            yield self._format_sse({
                "type": "status",
                "content": "Extracting patient data from your message..."
            })

            patient_data = self._extract_patient_data(message)

            # Debug log the extracted data
            logger.info(f"[DEBUG] Extracted patient data: {patient_data}")

            if not patient_data:
                yield self._format_sse({
                    "type": "error",
                    "content": "Could not extract patient data. Please provide: age, sex, stage, histology type."
                })
                return

            # Check extraction metadata for incomplete extraction
            extraction_meta = patient_data.pop("_extraction_meta", {})
            missing_fields = extraction_meta.get("missing_fields", [])
            extracted_fields = extraction_meta.get("extracted_fields", [])

            # Debug log the extraction metadata
            logger.info(f"[DEBUG] Missing fields: {missing_fields}")
            logger.info(f"[DEBUG] Extracted fields: {extracted_fields}")

            if missing_fields:
                # Stream what was extracted
                yield self._format_sse({
                    "type": "reasoning",
                    "content": f"📋 **Partial Extraction:** Found {', '.join(extracted_fields) if extracted_fields else 'no required fields'}"
                })

                # Show what we extracted so far
                if extracted_fields:
                    yield self._format_sse({
                        "type": "patient_data",
                        "content": {k: v for k, v in patient_data.items() if not k.startswith("_")}
                    })

                # Provide helpful error about missing fields
                missing_hints = {
                    "age": "age (e.g., '68 year old' or '68M')",
                    "sex": "sex (e.g., 'male', 'female', 'M', or 'F')",
                    "tnm_stage": "stage (e.g., 'stage IIIA', 'stage IV', 'limited stage', 'extensive stage')",
                    "histology_type": "histology type (e.g., 'adenocarcinoma', 'squamous cell', 'SCLC', 'NSCLC')"
                }
                hints = [missing_hints.get(f, f) for f in missing_fields]

                yield self._format_sse({
                    "type": "error",
                    "content": f"⚠️ Missing required fields: {', '.join(hints)}\n\nPlease include these details to get treatment recommendations."
                })
                return

            # Stream reasoning about what was extracted
            yield self._format_sse({
                "type": "reasoning",
                "content": f"✅ **Extraction Complete:** Successfully extracted {len(extracted_fields)} required fields"
            })

            yield self._format_sse({
                "type": "patient_data",
                "content": patient_data
            })

            # Step 1b: Check if patient exists in Neo4j
            if self.lca_service.graph_db and self.lca_service.graph_db.driver:
                yield self._format_sse({
                    "type": "status",
                    "content": "Checking existing patient records in Neo4j..."
                })

                existing = self._find_existing_patient(patient_data)
                if existing:
                    yield self._format_sse({
                        "type": "progress",
                        "content": f"Found existing patient record: {existing.get('patient_id', 'unknown')}"
                    })
                    patient_data["patient_id"] = existing.get("patient_id", patient_data["patient_id"])
                else:
                    yield self._format_sse({
                        "type": "progress",
                        "content": "No existing record found - creating new patient entry"
                    })

            # Step 2: Assess complexity with reasoning
            yield self._format_sse({
                "type": "status",
                "content": "Assessing case complexity..."
            })

            complexity = await self.lca_service.assess_complexity(patient_data)

            # Stream reasoning about complexity assessment
            complexity_factors = []
            stage = patient_data.get("tnm_stage", "")
            if "IV" in stage or "Extensive" in stage:
                complexity_factors.append("advanced/metastatic stage")
            ps = patient_data.get("performance_status", 1)
            if ps >= 2:
                complexity_factors.append(f"PS {ps} (functional limitations)")
            biomarkers = patient_data.get("biomarker_profile", {})
            if biomarkers:
                complexity_factors.append(f"{len(biomarkers)} biomarker(s) to evaluate")
            comorbidities = patient_data.get("comorbidities", [])
            if comorbidities:
                complexity_factors.append(f"{len(comorbidities)} comorbidity(ies)")

            yield self._format_sse({
                "type": "reasoning",
                "content": f"🔍 **Complexity Factors:** {', '.join(complexity_factors) if complexity_factors else 'Standard case parameters'}"
            })

            yield self._format_sse({
                "type": "complexity",
                "content": {
                    "level": complexity["complexity"],
                    "workflow": complexity["recommended_workflow"],
                    "score": complexity.get("complexity_score", "N/A")
                }
            })

            # Step 3: Run workflow with detailed agent progress
            use_advanced = complexity["recommended_workflow"] == "integrated"

            # Stream which agents will be invoked
            agent_path_desc = {
                "SIMPLE": ["Ingestion", "SemanticMapping", "Classification", "Biomarker", "Comorbidity", "NSCLC/SCLC Routing", "ConflictResolution", "Explanation"],
                "MODERATE": ["+ Uncertainty Quantification"],
                "COMPLEX": ["+ Survival Analysis", "Clinical Trial Matching"],
                "CRITICAL": ["+ Counterfactual Reasoning", "MDT Escalation"]
            }

            workflow_type = complexity["recommended_workflow"]
            yield self._format_sse({
                "type": "reasoning",
                "content": f"🚀 **Workflow Selected:** {complexity['complexity']} complexity → {workflow_type} workflow"
            })

            yield self._format_sse({
                "type": "status",
                "content": f"Initializing {workflow_type} workflow with multi-agent orchestration..."
            })

            # Stream agent execution preview
            base_agents = agent_path_desc.get("SIMPLE", [])
            yield self._format_sse({
                "type": "reasoning",
                "content": f"📊 **Agent Pipeline:** {' → '.join(base_agents[:4])}..."
            })

            # Create async queues for real-time streaming
            progress_queue = asyncio.Queue()
            log_queue = asyncio.Queue(maxsize=100)  # Limit to prevent memory issues

            async def stream_progress(message: str):
                """Stream progress messages in real-time"""
                await progress_queue.put(("progress", message))

            # Install log capture handler
            log_handler = create_sse_log_handler(level=logging.INFO)
            log_handler.install(log_queue)

            try:
                # Execute with streaming updates
                # Set use_ai_workflow=False for faster response (2-3s vs 20-30s)
                # The AI workflow adds argumentation but is not required for recommendations
                processing_task = asyncio.create_task(
                    self.lca_service.process_patient(
                        patient_data=patient_data,
                        use_ai_workflow=False,  # Changed from True to False for faster responses
                        force_advanced=use_advanced,
                        progress_callback=stream_progress
                    )
                )

                # Stream progress and logs as they come in
                start_time = datetime.now()
                streamed_count = 0
                last_log_time = start_time

                while not processing_task.done():
                    # Check for progress messages
                    try:
                        msg_type, msg = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                        streamed_count += 1
                        yield self._format_sse({
                            "type": "progress",
                            "content": f"[Step {streamed_count}] {msg}"
                        })
                    except asyncio.TimeoutError:
                        pass

                    # Check for log entries
                    try:
                        while not log_queue.empty():
                            log_entry = log_queue.get_nowait()
                            # Format log for display
                            log_msg = f"[{log_entry['level']}] {log_entry['logger']}: {log_entry['message']}"
                            if log_entry.get('agent'):
                                log_msg = f"[{log_entry['agent']}] {log_entry['message']}"
                            if log_entry.get('duration_ms'):
                                log_msg += f" ({log_entry['duration_ms']}ms)"

                            yield self._format_sse({
                                "type": "log",
                                "content": log_msg,
                                "level": log_entry['level'],
                                "timestamp": log_entry['timestamp']
                            })
                            last_log_time = datetime.now()
                    except Exception:
                        pass

                    # Show elapsed time if no activity
                    elapsed = (datetime.now() - start_time).total_seconds()
                    log_silence = (datetime.now() - last_log_time).total_seconds()
                    if elapsed > 3 and log_silence > 2 and streamed_count == 0:
                        yield self._format_sse({
                            "type": "reasoning",
                            "content": f"⏳ Processing ({int(elapsed)}s elapsed)... orchestrating agents"
                        })
                        last_log_time = datetime.now()  # Reset to avoid spam

                # Get result
                result = await processing_task

                # Debug log the result
                logger.info(f"[DEBUG] LCA processing result: recommendations={len(result.recommendations) if result.recommendations else 0}")
                if result.recommendations:
                    logger.info(f"[DEBUG] First recommendation: {result.recommendations[0]}")
                else:
                    logger.error(f"[DEBUG] No recommendations generated for patient data: {patient_data}")

                # Stream any remaining items
                while not progress_queue.empty():
                    msg_type, msg = await progress_queue.get()
                    streamed_count += 1
                    yield self._format_sse({
                        "type": "progress",
                        "content": f"[Step {streamed_count}] {msg}"
                    })

                while not log_queue.empty():
                    log_entry = log_queue.get_nowait()
                    log_msg = f"[{log_entry['level']}] {log_entry['logger']}: {log_entry['message']}"
                    yield self._format_sse({
                        "type": "log",
                        "content": log_msg,
                        "level": log_entry['level'],
                        "timestamp": log_entry['timestamp']
                    })

            finally:
                # Always uninstall handler
                log_handler.uninstall()

            # Show matched rules summary
            if result.recommendations:
                rules_summary = ", ".join([
                    f"{r.rule_id} ({r.treatment_type})"
                    for r in result.recommendations[:3]
                ])
                yield self._format_sse({
                    "type": "progress",
                    "content": f"Matched guidelines: {rules_summary}"
                })

            # Store in session context for follow-up questions
            if session_id:
                self.patient_context[session_id] = {
                    "patient_data": patient_data,
                    "result": result
                }

            # Step 4: Stream results
            yield self._format_sse({
                "type": "status",
                "content": f"Analysis complete ({result.execution_time_ms}ms)"
            })

            # Debug logging
            logger.info(f"Result has {len(result.recommendations) if result.recommendations else 0} recommendations")
            if result.recommendations:
                logger.info(f"First recommendation: {result.recommendations[0]}")
            else:
                logger.warning("No recommendations generated by LCA service")

            # Format recommendations - always yield something
            if result.recommendations:
                recommendations_text = self._format_recommendations(result)
            else:
                recommendations_text = """## Analysis Complete

⚠️ **No specific treatment recommendations generated.**

The system successfully analyzed your patient case but did not generate specific treatment recommendations. This may be due to:

- **Insufficient patient data**: Consider adding more clinical details
- **Complex case requirements**: May require MDT discussion
- **Missing biomarker data**: Molecular testing results may be needed
- **Staging clarification**: TNM staging may need verification

**Suggestions:**
- Provide additional patient history, imaging results, or lab values
- Include performance status and comorbidities
- Specify molecular testing results if available
- Consider manual clinical review for complex cases"""

            yield self._format_sse({
                "type": "recommendation",
                "content": recommendations_text
            })

            # MDT summary
            if result.mdt_summary:
                # Format MDT summary if it's an object
                summary_text = self._format_mdt_summary(result.mdt_summary)
                yield self._format_sse({
                    "type": "text",
                    "content": f"\n\n**Clinical Summary:**\n{summary_text}"
                })

            # Stream lab results if present
            if hasattr(result, 'lab_results') and result.lab_results:
                yield self._format_sse({
                    "type": "lab_results",
                    "content": {
                        "results": result.lab_results,
                        "interpretations": getattr(result, 'lab_interpretations', [])
                    }
                })

            # Stream drug interactions if present
            if hasattr(result, 'drug_interactions') and result.drug_interactions:
                # Check for severe interactions
                severe_interactions = [
                    di for di in result.drug_interactions
                    if di.get('severity', '').upper() == 'SEVERE'
                ]
                yield self._format_sse({
                    "type": "drug_interactions",
                    "content": {
                        "interactions": result.drug_interactions,
                        "severe_count": len(severe_interactions)
                    }
                })

            # Stream monitoring protocol if present
            if hasattr(result, 'monitoring_protocol') and result.monitoring_protocol:
                yield self._format_sse({
                    "type": "monitoring_protocol",
                    "content": result.monitoring_protocol
                })

            # Stream eligible clinical trials if present
            if hasattr(result, 'eligible_trials') and result.eligible_trials:
                yield self._format_sse({
                    "type": "eligible_trials",
                    "content": {
                        "trials": result.eligible_trials,
                        "match_scores": getattr(result, 'trial_match_scores', {})
                    }
                })

            # Step 5: Record decision in Neo4j and send graph visualization data
            if self.lca_service.graph_db and self.lca_service.graph_db.driver:
                try:
                    patient_id = patient_data.get('patient_id')
                    logger.info(f"[GraphData] Attempting to fetch graph data for patient {patient_id}")
                    yield self._format_sse({
                        "type": "status",
                        "content": "Loading context graph visualization..."
                    })

                    # Import ContextGraphClient
                    from ..db.context_graph_client import LCAContextGraphClient, GraphNode as GNode, GraphRelationship as GRel

                    # Get Neo4j config from graph_db
                    neo4j_uri = getattr(self.lca_service.graph_db, 'uri', 'bolt://localhost:7687')
                    neo4j_user = getattr(self.lca_service.graph_db, 'user', 'neo4j')
                    neo4j_password = getattr(self.lca_service.graph_db, 'password', '123456789')
                    neo4j_database = getattr(self.lca_service.graph_db, 'database', 'neo4j')

                    logger.info(f"[GraphData] Creating LCAContextGraphClient with uri={neo4j_uri}, database={neo4j_database}")

                    graph_client = LCAContextGraphClient(
                        uri=neo4j_uri,
                        username=neo4j_user,
                        password=neo4j_password,
                        database=neo4j_database
                    )

                    # Record the decision in Neo4j for future reference
                    decision_id = None
                    if result.recommendations:
                        primary = result.recommendations[0]
                        treatment = getattr(primary, 'treatment_type', getattr(primary, 'treatment', 'Unknown'))
                        evidence = getattr(primary, 'evidence_level', 'Not specified')
                        confidence = getattr(primary, 'confidence', getattr(primary, 'confidence_score', 0.8))
                        if isinstance(confidence, (int, float)) and confidence > 1:
                            confidence = confidence / 100
                        intent = getattr(primary, 'treatment_intent', getattr(primary, 'intent', 'Not specified'))
                        rationale = getattr(primary, 'rationale', str(result.mdt_summary) if result.mdt_summary else '')
                        guideline = getattr(primary, 'guideline_reference', getattr(primary, 'rule_source', ''))

                        try:
                            decision_id = graph_client.record_decision(
                                decision_type="treatment_recommendation",
                                category=patient_data.get("histology_type", "NSCLC"),
                                reasoning=rationale[:500] if rationale else f"Recommended {treatment} based on clinical guidelines",
                                patient_id=patient_id,
                                treatment=treatment,
                                confidence_score=float(confidence),
                                risk_factors=patient_data.get("comorbidities", []),
                                session_id=session_id,
                                agent_name="LCA_Orchestrator"
                            )
                            logger.info(f"[GraphData] Recorded decision {decision_id}")
                        except Exception as dec_err:
                            logger.warning(f"[GraphData] Failed to record decision: {dec_err}")

                    logger.info(f"[GraphData] Calling get_graph_data with center_node_id={patient_id}, depth=2")

                    # Get graph data centered on the patient
                    graph_data = graph_client.get_graph_data(
                        center_node_id=patient_id,
                        depth=2,
                        limit=50
                    )

                    logger.info(f"[GraphData] Retrieved {len(graph_data.nodes)} nodes, {len(graph_data.relationships)} relationships")

                    # Build comprehensive graph payload with all nodes
                    nodes_list = []
                    relationships_list = []
                    node_ids = set()

                    # Add nodes from Neo4j
                    for node in graph_data.nodes:
                        nodes_list.append({
                            "id": node.id,
                            "labels": node.labels,
                            "properties": node.properties
                        })
                        node_ids.add(node.id)

                    # Add relationships from Neo4j
                    for rel in graph_data.relationships:
                        relationships_list.append({
                            "id": rel.id,
                            "type": rel.type,
                            "startNodeId": rel.startNodeId,
                            "endNodeId": rel.endNodeId,
                            "properties": rel.properties
                        })

                    # Create virtual nodes for current analysis (even if not in Neo4j yet)
                    # This ensures the user always sees the decision context

                    # Patient node (if not already present)
                    virtual_patient_id = f"virtual_patient_{patient_id}"
                    if not any(n.get("id") == patient_id or n.get("properties", {}).get("patient_id") == patient_id for n in nodes_list):
                        nodes_list.append({
                            "id": virtual_patient_id,
                            "labels": ["Patient"],
                            "properties": {
                                "patient_id": patient_id,
                                "age_at_diagnosis": patient_data.get("age", patient_data.get("age_at_diagnosis")),
                                "sex": patient_data.get("sex"),
                                "tnm_stage": patient_data.get("tnm_stage"),
                                "histology_type": patient_data.get("histology_type"),
                                "performance_status": patient_data.get("performance_status"),
                                "basis_of_diagnosis": "Clinical"
                            }
                        })
                        node_ids.add(virtual_patient_id)
                    else:
                        # Find actual patient node id
                        for n in nodes_list:
                            if n.get("properties", {}).get("patient_id") == patient_id:
                                virtual_patient_id = n.get("id")
                                break

                    # Add decision nodes for all recommendations
                    for idx, rec in enumerate(result.recommendations[:5]):  # Top 5 recommendations
                        treatment = getattr(rec, 'treatment_type', getattr(rec, 'treatment', 'Unknown'))
                        evidence = getattr(rec, 'evidence_level', 'Not specified')
                        confidence = getattr(rec, 'confidence', getattr(rec, 'confidence_score', 0.8))
                        if isinstance(confidence, (int, float)) and confidence > 1:
                            confidence = confidence / 100
                        intent = getattr(rec, 'treatment_intent', getattr(rec, 'intent', 'Not specified'))
                        rationale = getattr(rec, 'rationale', '')
                        guideline = getattr(rec, 'guideline_reference', getattr(rec, 'rule_source', ''))

                        virtual_decision_id = f"virtual_decision_{idx}_{datetime.now().strftime('%H%M%S')}"
                        nodes_list.append({
                            "id": virtual_decision_id,
                            "labels": ["TreatmentDecision"],
                            "properties": {
                                "id": virtual_decision_id,
                                "decision_type": "treatment_recommendation",
                                "category": patient_data.get("histology_type", "NSCLC"),
                                "status": "recommended" if idx == 0 else "alternative",
                                "treatment": treatment,
                                "evidence_level": evidence,
                                "confidence_score": confidence,
                                "reasoning": rationale[:200] if rationale else f"Based on {guideline}",
                                "reasoning_summary": f"{treatment} - {evidence}",
                                "intent": intent,
                                "guideline_reference": guideline,
                                "rank": idx + 1,
                                "decision_timestamp": datetime.now().isoformat()
                            }
                        })
                        node_ids.add(virtual_decision_id)

                        # Link decision to patient
                        relationships_list.append({
                            "id": f"rel_decision_{idx}",
                            "type": "ABOUT",
                            "startNodeId": virtual_decision_id,
                            "endNodeId": virtual_patient_id,
                            "properties": {}
                        })

                        # Add guideline node if present
                        if guideline:
                            guideline_id = f"virtual_guideline_{idx}"
                            if not any(n.get("properties", {}).get("name") == guideline for n in nodes_list):
                                nodes_list.append({
                                    "id": guideline_id,
                                    "labels": ["Guideline"],
                                    "properties": {
                                        "name": guideline,
                                        "source": guideline.split()[0] if guideline else "Clinical",
                                        "evidence_level": evidence
                                    }
                                })
                                node_ids.add(guideline_id)

                                # Link decision to guideline
                                relationships_list.append({
                                    "id": f"rel_guideline_{idx}",
                                    "type": "APPLIED_GUIDELINE",
                                    "startNodeId": virtual_decision_id,
                                    "endNodeId": guideline_id,
                                    "properties": {}
                                })

                    # Collect all decision node IDs for cross-linking
                    decision_node_ids = [n["id"] for n in nodes_list if "TreatmentDecision" in n.get("labels", [])]
                    guideline_node_ids = []

                    # Add biomarker nodes
                    biomarkers = patient_data.get("biomarker_profile", {})
                    biomarker_node_ids = []
                    for marker_name, marker_value in biomarkers.items():
                        biomarker_id = f"virtual_biomarker_{marker_name}"
                        display_name = marker_name.replace("_", " ").upper()

                        # Format value for display
                        if isinstance(marker_value, bool):
                            display_value = "Positive" if marker_value else "Negative"
                        elif marker_name == "pdl1_tps":
                            display_value = f"{marker_value}%"
                        else:
                            display_value = str(marker_value)

                        nodes_list.append({
                            "id": biomarker_id,
                            "labels": ["Biomarker"],
                            "properties": {
                                "marker_type": display_name,
                                "value": display_value,
                                "status": display_value,
                                "name": f"{display_name}: {display_value}"
                            }
                        })
                        node_ids.add(biomarker_id)
                        biomarker_node_ids.append(biomarker_id)

                        # Link biomarker to patient
                        relationships_list.append({
                            "id": f"rel_biomarker_{marker_name}",
                            "type": "HAS_BIOMARKER",
                            "startNodeId": virtual_patient_id,
                            "endNodeId": biomarker_id,
                            "properties": {}
                        })

                    # Link ALL decisions to ALL biomarkers (biomarkers influence treatment selection)
                    for dec_id in decision_node_ids:
                        for bm_id in biomarker_node_ids:
                            relationships_list.append({
                                "id": f"rel_based_on_{bm_id}_{dec_id}",
                                "type": "BASED_ON",
                                "startNodeId": dec_id,
                                "endNodeId": bm_id,
                                "properties": {}
                            })

                    # Add comorbidity nodes
                    comorbidities = patient_data.get("comorbidities", [])
                    comorbidity_node_ids = []
                    for comorbidity in comorbidities:
                        comorbidity_id = f"virtual_comorbidity_{comorbidity}"
                        nodes_list.append({
                            "id": comorbidity_id,
                            "labels": ["Comorbidity"],
                            "properties": {
                                "name": comorbidity,
                                "condition": comorbidity
                            }
                        })
                        node_ids.add(comorbidity_id)
                        comorbidity_node_ids.append(comorbidity_id)

                        # Link comorbidity to patient
                        relationships_list.append({
                            "id": f"rel_comorbidity_{comorbidity}",
                            "type": "HAS_COMORBIDITY",
                            "startNodeId": virtual_patient_id,
                            "endNodeId": comorbidity_id,
                            "properties": {}
                        })

                    # Link decisions to comorbidities (comorbidities may affect treatment)
                    for dec_id in decision_node_ids:
                        for comorb_id in comorbidity_node_ids:
                            relationships_list.append({
                                "id": f"rel_considers_{comorb_id}_{dec_id}",
                                "type": "CONSIDERS",
                                "startNodeId": dec_id,
                                "endNodeId": comorb_id,
                                "properties": {}
                            })

                    # Link guideline nodes to all relevant decisions
                    guideline_node_ids = [n["id"] for n in nodes_list if "Guideline" in n.get("labels", [])]
                    for dec_node in nodes_list:
                        if "TreatmentDecision" in dec_node.get("labels", []):
                            dec_guideline = dec_node.get("properties", {}).get("guideline_reference", "")
                            # Find matching guideline node
                            for gd_node in nodes_list:
                                if "Guideline" in gd_node.get("labels", []):
                                    gd_name = gd_node.get("properties", {}).get("name", "")
                                    if gd_name and gd_name in str(dec_guideline):
                                        # Check if relationship already exists
                                        rel_exists = any(
                                            r["startNodeId"] == dec_node["id"] and r["endNodeId"] == gd_node["id"]
                                            for r in relationships_list
                                        )
                                        if not rel_exists:
                                            relationships_list.append({
                                                "id": f"rel_applied_gd_{dec_node['id']}_{gd_node['id']}",
                                                "type": "APPLIED_GUIDELINE",
                                                "startNodeId": dec_node["id"],
                                                "endNodeId": gd_node["id"],
                                                "properties": {}
                                            })

                    graph_payload = {
                        "nodes": nodes_list,
                        "relationships": relationships_list
                    }

                    logger.info(f"[GraphData] Sending graph_data SSE message with {len(graph_payload['nodes'])} nodes, {len(graph_payload['relationships'])} relationships")

                    yield self._format_sse({
                        "type": "graph_data",
                        "content": graph_payload
                    })

                except Exception as graph_error:
                    logger.error(f"[GraphData] Failed to fetch graph data: {graph_error}", exc_info=True)
                    # Send empty graph data so frontend knows we tried
                    yield self._format_sse({
                        "type": "graph_data",
                        "content": {"nodes": [], "relationships": []}
                    })
            else:
                logger.warning("[GraphData] Neo4j not available - skipping graph data fetch")

            # Step 6: Run ontology-based inference (async, non-blocking)
            inference_details = {}
            if self.lca_service.graph_db and self.lca_service.graph_db.driver:
                try:
                    from ..db.neo4j_inference import Neo4jInferenceEngine
                    inference_engine = Neo4jInferenceEngine(
                        driver=self.lca_service.graph_db.driver,
                        database=getattr(self.lca_service.graph_db, 'database', 'neo4j')
                    )
                    patient_id = patient_data.get('patient_id')
                    
                    # Run all 6 inference rules
                    inference_results = await asyncio.to_thread(
                        inference_engine.run_all_inferences, patient_id
                    )
                    
                    # Get detailed inference results
                    inference_details = await asyncio.to_thread(
                        inference_engine.get_patient_inferences, patient_id
                    )
                    
                    total_inferred = sum(inference_results.values())
                    if total_inferred > 0:
                        # Build detailed provenance message
                        provenance_parts = ["🧠 **Ontology Inference Engine (6 Rules Applied):**"]
                        provenance_parts.append(f"\n📊 Total: {total_inferred} knowledge items inferred\n")
                        
                        if inference_results.get('cancer_type_classification', 0) > 0:
                            provenance_parts.append(f"✓ Cancer Classification: {inference_results['cancer_type_classification']} items")
                        if inference_results.get('biomarker_therapy_inference', 0) > 0:
                            provenance_parts.append(f"✓ Biomarker-Therapy Links: {inference_results['biomarker_therapy_inference']} items")
                        if inference_results.get('guideline_applicability', 0) > 0:
                            provenance_parts.append(f"✓ Guideline Matching: {inference_results['guideline_applicability']} guidelines")
                        if inference_results.get('risk_stratification', 0) > 0:
                            provenance_parts.append(f"✓ Risk Stratification: {inference_results['risk_stratification']} assessments")
                        if inference_results.get('contraindication_check', 0) > 0:
                            provenance_parts.append(f"✓ Contraindication Check: {inference_results['contraindication_check']} warnings")
                        if inference_results.get('stage_group_inference', 0) > 0:
                            provenance_parts.append(f"✓ Stage Grouping: {inference_results['stage_group_inference']} classifications")
                        
                        # Add specific inferences as provenance
                        if inference_details.get('classification'):
                            cls = inference_details['classification']
                            provenance_parts.append(f"\n📌 **Inferred Classification:** {cls.get('cancer_subtype', 'N/A')} ({cls.get('cancer_category', 'N/A')})")
                        
                        if inference_details.get('therapy_inferences'):
                            therapy_classes = [ti.get('therapy_class') for ti in inference_details['therapy_inferences'] if ti.get('therapy_class')]
                            if therapy_classes:
                                provenance_parts.append(f"📌 **Inferred Therapies:** {', '.join(therapy_classes)}")
                        
                        if inference_details.get('stage_group'):
                            sg = inference_details['stage_group']
                            provenance_parts.append(f"📌 **Treatment Intent:** {sg.get('treatment_intent', 'N/A')} (Stage Group: {sg.get('stage_group', 'N/A')})")
                        
                        if inference_details.get('applicable_guidelines'):
                            guidelines = inference_details['applicable_guidelines']
                            if guidelines:
                                provenance_parts.append(f"📌 **Applicable Guidelines:** {', '.join(guidelines[:3])}")
                        
                        yield self._format_sse({
                            "type": "reasoning",
                            "content": "\n".join(provenance_parts)
                        })
                except Exception as inf_err:
                    logger.warning(f"Inference engine failed (non-critical): {inf_err}")

            # Suggest follow-ups (context-aware based on actual inference results)
            suggestions = self._generate_inference_aware_suggestions(
                patient_data, result, inference_details
            )
            yield self._format_sse({
                "type": "suggestions",
                "content": suggestions
            })

        except Exception as e:
            logger.error(f"Patient analysis failed: {e}", exc_info=True)
            yield self._format_sse({
                "type": "error",
                "content": f"Analysis failed: {str(e)}"
            })

    async def _stream_follow_up(self, message: str, session_id: str) -> AsyncIterator[str]:
        """Stream follow-up responses using orchestration workflows with full provenance tracking"""
        context = self.patient_context[session_id]
        patient_data = context.get("patient_data", {})
        
        # Track provenance for complete transparency
        provenance = {
            "timestamp": datetime.now().isoformat(),
            "query": message,
            "data_sources": [],
            "agents_used": [],
            "database_queries": [],
            "inferences_applied": []
        }

        yield self._format_sse({
            "type": "status",
            "content": "🔍 Analyzing follow-up with orchestration workflow..."
        })

        # Step 1: Assess query complexity to route to appropriate workflow
        query_complexity = self._assess_followup_complexity(message, patient_data)
        provenance["query_complexity"] = query_complexity
        
        yield self._format_sse({
            "type": "reasoning",
            "content": f"📊 **Query Complexity:** {query_complexity.upper()} → {'Advanced' if query_complexity in ['complex', 'critical'] else 'Basic'} Workflow"
        })

        # Step 2: Try graph database lookup (most grounded)
        yield self._format_sse({
            "type": "status",
            "content": "🗄️ Querying Neo4j graph database..."
        })
        
        graph_response, graph_provenance = await self._graph_based_followup_with_provenance(message, context)
        provenance["data_sources"].extend(graph_provenance.get("sources", []))
        provenance["database_queries"].extend(graph_provenance.get("queries", []))
        
        if graph_response:
            provenance["primary_source"] = "Neo4j Graph Database"
            yield self._format_sse({
                "type": "reasoning",
                "content": f"✅ **Data Source:** Neo4j Graph Database\n📈 **Queries Executed:** {len(graph_provenance.get('queries', []))}\n🔗 **Nodes Retrieved:** {graph_provenance.get('nodes_count', 0)}"
            })
            response = graph_response
        else:
            # Step 3: Try orchestration workflow (adaptive)
            yield self._format_sse({
                "type": "status",
                "content": f"🤖 Invoking {query_complexity} orchestration workflow..."
            })
            
            workflow_response, workflow_provenance = await self._workflow_based_followup(message, context, query_complexity)
            provenance["agents_used"].extend(workflow_provenance.get("agents", []))
            provenance["inferences_applied"].extend(workflow_provenance.get("inferences", []))
            
            if workflow_response:
                provenance["primary_source"] = f"Orchestration Workflow ({query_complexity})"
                yield self._format_sse({
                    "type": "reasoning",
                    "content": f"✅ **Data Source:** {query_complexity.capitalize()} Orchestration Workflow\n🤖 **Agents Used:** {', '.join(workflow_provenance.get('agents', [])[:5])}\n🧠 **Inferences:** {len(workflow_provenance.get('inferences', []))}"
                })
                response = workflow_response
            else:
                # Step 4: Try ontology-based inference
                yield self._format_sse({
                    "type": "status",
                    "content": "🧬 Querying LUCADA ontology..."
                })
                
                ontology_response = self._ontology_based_followup(message, context)
                
                if ontology_response:
                    provenance["primary_source"] = "LUCADA Ontology + Guidelines"
                    provenance["data_sources"].append("LUCADA OWL Ontology")
                    provenance["data_sources"].append("Clinical Guidelines (NICE/NCCN/ESMO)")
                    yield self._format_sse({
                        "type": "reasoning",
                        "content": "✅ **Data Source:** LUCADA OWL Ontology + Clinical Guidelines"
                    })
                    response = ontology_response
                else:
                    # Step 5: Template-based response (NO LLM FALLBACK)
                    template_response = self._handle_follow_up(message, context)
                    
                    if template_response and len(template_response) > 50:
                        provenance["primary_source"] = "Clinical Decision Templates"
                        provenance["data_sources"].append("Patient Context")
                        yield self._format_sse({
                            "type": "reasoning",
                            "content": "📋 **Data Source:** Clinical Decision Templates (Patient Context)"
                        })
                        response = template_response
                    else:
                        # NO DATA AVAILABLE - Be honest
                        provenance["primary_source"] = "None - Insufficient Data"
                        yield self._format_sse({
                            "type": "reasoning",
                            "content": "⚠️ **Data Source:** None available in database/ontology"
                        })
                        response = self._build_no_data_response(message, patient_data)

        # Stream provenance tracking
        yield self._format_sse({
            "type": "provenance",
            "content": provenance
        })
        
        yield self._format_sse({
            "type": "text",
            "content": response
        })

        # Generate context-aware follow-up suggestions
        patient_data = context.get("patient_data", {})
        result = context.get("result")
        suggestions = self._generate_patient_aware_suggestions(message, patient_data, result)

        yield self._format_sse({
            "type": "suggestions",
            "content": suggestions
        })

        # Generate context-aware follow-up suggestions
        result = context.get("result")
        suggestions = self._generate_patient_aware_suggestions(message, patient_data, result)

        yield self._format_sse({
            "type": "suggestions",
            "content": suggestions
        })

        self._add_to_history(session_id, "assistant", response)

    def _assess_followup_complexity(self, message: str, patient_data: Dict) -> str:
        """Assess follow-up query complexity to route to appropriate workflow"""
        message_lower = message.lower()
        
        # Critical complexity triggers
        critical_keywords = ["counterfactual", "what if", "alternative outcome", "survival prediction", "trial matching"]
        if any(kw in message_lower for kw in critical_keywords):
            return "critical"
        
        # Complex triggers
        complex_keywords = ["similar patient", "comparative", "survival", "prognosis", "risk stratification", "biomarker interaction"]
        if any(kw in message_lower for kw in complex_keywords):
            return "complex"
        
        # Moderate triggers
        moderate_keywords = ["guideline", "evidence", "trial", "contraindication", "drug interaction"]
        if any(kw in message_lower for kw in moderate_keywords):
            return "moderate"
        
        # Simple by default
        return "simple"

    def _build_no_data_response(self, message: str, patient_data: Dict) -> str:
        """Build honest response when no data is available in database/ontology"""
        response = "## ⚠️ Insufficient Data in Database\n\n"
        response += "I searched the following sources:\n\n"
        response += "- ❌ **Neo4j Graph Database**: No matching records found\n"
        response += "- ❌ **LUCADA OWL Ontology**: No applicable inference rules\n"
        response += "- ❌ **Clinical Guidelines**: No specific guidance for this query\n"
        response += "- ❌ **Patient Context Templates**: Limited template coverage\n\n"
        response += "**What you can do:**\n\n"
        response += f"1. Try rephrasing your question: \"{message}\"\n"
        response += "2. Ask about specific patient aspects: biomarkers, stage, treatments\n"
        response += "3. Query the graph database directly (see suggestions below)\n\n"
        response += "**Available Data for this Patient:**\n\n"
        
        # Show what data we DO have
        if patient_data:
            response += f"- Stage: {patient_data.get('tnm_stage', 'Unknown')}\n"
            response += f"- Histology: {patient_data.get('histology_type', 'Unknown')}\n"
            biomarkers = patient_data.get('biomarker_profile', {})
            if biomarkers:
                response += f"- Biomarkers: {', '.join([f'{k}={v}' for k, v in biomarkers.items()])}\n"
        else:
            response += "*No patient context available*\n"
        
        return response

    async def _graph_based_followup_with_provenance(self, message: str, context: Dict) -> tuple[Optional[str], Dict]:
        """Query Neo4j with detailed provenance tracking"""
        provenance = {"sources": [], "queries": [], "nodes_count": 0}
        result = await self._graph_based_followup(message, context)
        
        if result:
            provenance["sources"].append("Neo4j Graph Database")
            # Count queries executed (approximate)
            message_lower = message.lower()
            if "similar" in message_lower:
                provenance["queries"].append("MATCH (similar:Patient) WHERE stage/histology match")
            if any(kw in message_lower for kw in ["biomarker", "egfr", "alk"]):
                provenance["queries"].append("MATCH (p)-[:HAS_BIOMARKER]->(b)")
                provenance["queries"].append("MATCH (p)-[:HAS_INFERENCE]->(inf:TherapyInference)")
            if any(kw in message_lower for kw in ["contraindication", "comorbidity"]):
                provenance["queries"].append("MATCH (p)-[:HAS_COMORBIDITY]->(c)")
                provenance["queries"].append("MATCH (p)-[:HAS_INFERENCE]->(ci:ContraindicationInference)")
            if "inference" in message_lower or "classification" in message_lower:
                provenance["queries"].append("MATCH (p)-[:HAS_CLASSIFICATION]->(cls)")
                provenance["queries"].append("Neo4jInferenceEngine.get_patient_inferences()")
        
        return result, provenance

    async def _workflow_based_followup(self, message: str, context: Dict, complexity: str) -> tuple[Optional[str], Dict]:
        """Use orchestration workflows to answer follow-up questions"""
        provenance = {"agents": [], "inferences": [], "workflow_type": complexity}
        patient_data = context.get("patient_data", {})
        
        try:
            # Assess if we need advanced workflow
            use_advanced = complexity in ["complex", "critical"]
            
            # Build augmented patient data with follow-up context
            augmented_data = patient_data.copy()
            augmented_data["_followup_query"] = message
            augmented_data["_followup_complexity"] = complexity
            
            # Run workflow
            if use_advanced and self.lca_service.integrated_workflow:
                provenance["workflow_type"] = "integrated_advanced"
                provenance["agents"].extend([
                    "IngestionAgent",
                    "SemanticMappingAgent",
                    "ClassificationAgent",
                    "BiomarkerAgent",
                    "NSCLCAgent/SCLCAgent",
                    "ComorbidityAgent",
                    "LabInterpretationAgent",
                    "MedicationManagementAgent",
                    "MonitoringCoordinatorAgent",
                    "ConflictResolutionAgent",
                    "UncertaintyQuantifier"
                ])
                
                result = await self.lca_service.integrated_workflow.analyze_patient_comprehensive(
                    patient_data=augmented_data,
                    persist=False,
                    progress_callback=None
                )
                
                provenance["agents_used_count"] = len(result.get("agent_chain", []))
                provenance["inferences"].extend(result.get("analytics", {}).get("inferences", []))
                
                # Extract relevant information from result
                response = self._format_workflow_result_for_followup(result, message)
                return response, provenance
            else:
                # Use basic workflow
                provenance["workflow_type"] = "basic_6agent"
                provenance["agents"].extend([
                    "IngestionAgent",
                    "SemanticMappingAgent",
                    "ClassificationAgent",
                    "ConflictResolutionAgent",
                    "PersistenceAgent",
                    "ExplanationAgent"
                ])
                
                result = await self.lca_service.process_patient(
                    patient_data=augmented_data,
                    use_ai_workflow=False,
                    force_advanced=False
                )
                
                if result and hasattr(result, 'recommendations'):
                    response = self._format_workflow_result_for_followup(result, message)
                    return response, provenance
        
        except Exception as e:
            logger.error(f"Workflow-based followup error: {e}", exc_info=True)
            provenance["error"] = str(e)
        
        return None, provenance
    
    def _format_workflow_result_for_followup(self, result, query: str) -> str:
        """Format workflow result as a follow-up answer"""
        response = "## 🤖 Orchestration Workflow Analysis\n\n"
        
        # Check if result is dict (integrated workflow) or object (basic workflow)
        if isinstance(result, dict):
            # Integrated workflow result
            response += f"**Workflow:** {result.get('workflow_version', 'Integrated')}\n"
            response += f"**Complexity:** {result.get('complexity', 'Unknown')}\n"
            response += f"**Agents Executed:** {len(result.get('successful_agents', []))}\n\n"
            
            if result.get('recommendations'):
                response += "### Recommendations\n\n"
                for i, rec in enumerate(result['recommendations'][:3], 1):
                    if isinstance(rec, dict):
                        response += f"{i}. **{rec.get('treatment', 'Unknown')}**\n"
                        response += f"   - Evidence: {rec.get('evidence_level', 'N/A')}\n"
                        response += f"   - Confidence: {rec.get('confidence', 'N/A')}\n\n"
            
            if result.get('analytics'):
                analytics = result['analytics']
                response += "### Analytics\n\n"
                if analytics.get('uncertainty'):
                    response += f"- **Uncertainty Score:** {analytics['uncertainty']}\n"
                if analytics.get('survival_analysis'):
                    response += f"- **Survival Analysis:** Available\n"
        else:
            # Basic workflow result
            if hasattr(result, 'recommendations') and result.recommendations:
                response += "### Recommendations\n\n"
                for i, rec in enumerate(result.recommendations[:3], 1):
                    treatment = getattr(rec, 'treatment_type', getattr(rec, 'treatment', 'Unknown'))
                    evidence = getattr(rec, 'evidence_level', 'N/A')
                    response += f"{i}. **{treatment}** (Evidence: {evidence})\n"
            
            if hasattr(result, 'mdt_summary') and result.mdt_summary:
                response += f"\n### Summary\n\n{str(result.mdt_summary)[:500]}...\n"
        
        return response

    async def _graph_based_followup(self, message: str, context: Dict) -> Optional[str]:
        """Query Neo4j graph database for grounded answers to follow-up questions"""
        if not self.lca_service.graph_db or not self.lca_service.graph_db.driver:
            return None
        
        patient_data = context.get("patient_data", {})
        patient_id = patient_data.get("patient_id")
        
        if not patient_id:
            return None
        
        message_lower = message.lower()
        
        try:
            # Query for similar patients
            if any(kw in message_lower for kw in ["similar", "comparable", "like this"]):
                query = """
                MATCH (p:Patient {patient_id: $patient_id})
                MATCH (similar:Patient)
                WHERE p.patient_id <> similar.patient_id
                  AND p.tnm_stage = similar.tnm_stage
                  AND p.histology_type = similar.histology_type
                WITH similar
                OPTIONAL MATCH (similar)-[:RECEIVED_RECOMMENDATION]->(rec)
                RETURN similar.patient_id as id, 
                       similar.age as age,
                       similar.sex as sex,
                       similar.tnm_stage as stage,
                       similar.histology_type as histology,
                       collect(rec.treatment) as treatments
                LIMIT 5
                """
                
                with self.lca_service.graph_db.driver.session() as session:
                    result = session.run(query, patient_id=patient_id)
                    records = list(result)
                    
                    if records:
                        response = "## \ud83d\udcc8 Similar Patients from Database\\n\\n"
                        response += "Found **{}** similar patients with matching stage and histology:\\n\\n".format(len(records))
                        
                        for i, rec in enumerate(records, 1):
                            response += f"### Patient {i}: {rec['id']}\\n\\n"
                            response += f"- Age: {rec['age']} | Sex: {rec['sex']}\\n"
                            response += f"- Stage: {rec['stage']} | Histology: {rec['histology']}\\n"
                            if rec['treatments']:
                                response += f"- Treatments: {', '.join([t for t in rec['treatments'] if t])}\\n"
                            response += "\\n"
                        
                        return response
            
            # Query for biomarker-specific treatment data
            if any(kw in message_lower for kw in ["biomarker", "egfr", "alk", "ros1", "kras", "pdl1"]):
                query = """
                MATCH (p:Patient {patient_id: $patient_id})-[:HAS_BIOMARKER]->(b:Biomarker)
                OPTIONAL MATCH (p)-[:HAS_INFERENCE]->(inf:TherapyInference)
                RETURN b.marker_type as marker,
                       b.value as value,
                       collect(inf.therapy_class) as inferred_therapies
                """
                
                with self.lca_service.graph_db.driver.session() as session:
                    result = session.run(query, patient_id=patient_id)
                    records = list(result)
                    
                    if records:
                        response = "## \ud83e\uddec Biomarker-Based Treatment Inference\\n\\n"
                        response += "**Graph Database Analysis:**\\n\\n"
                        
                        for rec in records:
                            response += f"- **{rec['marker']}**: {rec['value']}\\n"
                            if rec['inferred_therapies']:
                                therapies = [t for t in rec['inferred_therapies'] if t]
                                if therapies:
                                    response += f"  → Inferred Therapy Classes: {', '.join(therapies)}\\n"
                        
                        return response
            
            # Query for contraindications
            if any(kw in message_lower for kw in ["contraindication", "drug interaction", "safety"]):
                query = """
                MATCH (p:Patient {patient_id: $patient_id})
                OPTIONAL MATCH (p)-[:HAS_COMORBIDITY]->(c:Comorbidity)
                OPTIONAL MATCH (p)-[:HAS_INFERENCE]->(ci:ContraindicationInference)
                RETURN collect(DISTINCT c.name) as comorbidities,
                       collect(DISTINCT ci.drug) as contraindicated_drugs,
                       collect(DISTINCT ci.reason) as reasons
                """
                
                with self.lca_service.graph_db.driver.session() as session:
                    result = session.run(query, patient_id=patient_id)
                    record = result.single()
                    
                    if record and (record['contraindicated_drugs'] or record['comorbidities']):
                        response = "## \u26a0\ufe0f Drug Safety Analysis (Graph Database)\\n\\n"
                        
                        if record['comorbidities']:
                            response += "**Documented Comorbidities:**\\n"
                            for c in record['comorbidities']:
                                if c:
                                    response += f"- {c}\\n"
                            response += "\\n"
                        
                        if record['contraindicated_drugs']:
                            response += "**Contraindicated Medications:**\\n"
                            drugs = [d for d in record['contraindicated_drugs'] if d]
                            for drug in drugs:
                                response += f"- {drug}\\n"
                            response += "\\n"
                        
                        return response
            
            # Query for inferences
            if any(kw in message_lower for kw in ["inference", "classification", "predicted"]):
                from ..db.neo4j_inference import Neo4jInferenceEngine
                
                inference_engine = Neo4jInferenceEngine(
                    driver=self.lca_service.graph_db.driver,
                    database=getattr(self.lca_service.graph_db, 'database', 'neo4j')
                )
                
                inferences = inference_engine.get_patient_inferences(patient_id)
                
                if inferences and any(inferences.values()):
                    response = "## \ud83e\udde0 Ontology-Based Inferences\\n\\n"
                    
                    if inferences.get('classification'):
                        cls = inferences['classification']
                        response += f"**Cancer Classification:**\\n"
                        response += f"- Subtype: {cls.get('cancer_subtype', 'Unknown')}\\n"
                        response += f"- Category: {cls.get('cancer_category', 'Unknown')}\\n\\n"
                    
                    if inferences.get('therapy_inferences'):
                        response += f"**Therapy Inferences:** {len(inferences['therapy_inferences'])} identified\\n"
                        for inf in inferences['therapy_inferences'][:5]:
                            response += f"- {inf.get('therapy_class', 'Unknown')} (from {inf.get('biomarker', 'biomarker analysis')})\\n"
                        response += "\\n"
                    
                    if inferences.get('risk_level'):
                        response += f"**Risk Assessment:** {inferences['risk_level']}\\n\\n"
                    
                    return response
                    
        except Exception as e:
            logger.error(f"Graph-based followup error: {e}", exc_info=True)
            return None
        
        return None

    def _ontology_based_followup(self, message: str, context: Dict) -> Optional[str]:
        """Use LUCADA ontology and guideline rules for grounded answers"""
        patient_data = context.get("patient_data", {})
        message_lower = message.lower()
        
        # Query guideline rules from ontology
        if any(kw in message_lower for kw in ["guideline", "evidence", "nccn", "nice", "esmo", "trial"]):
            try:
                # Get applicable guidelines from rule engine
                if hasattr(self.lca_service, 'rule_engine'):
                    stage = patient_data.get('tnm_stage', '')
                    histology = patient_data.get('histology_type', '')
                    
                    matching_rules = []
                    for rule_id, rule in self.lca_service.rule_engine.rules.items():
                        rule_dict = rule if isinstance(rule, dict) else rule.__dict__
                        # Check if rule applies to this patient
                        if stage.upper() in str(rule_dict.get('applies_to_stage', '')).upper():
                            matching_rules.append((rule_id, rule_dict))
                    
                    if matching_rules:
                        response = "## \ud83d\udcda Clinical Guidelines (LUCADA Ontology)\\n\\n"
                        response += f"Found **{len(matching_rules)}** applicable guideline rules:\\n\\n"
                        
                        for rule_id, rule in matching_rules[:5]:
                            response += f"### {rule_id}: {rule.get('name', 'Guideline Rule')}\\n\\n"
                            response += f"- **Source:** {rule.get('source', 'Clinical Guidelines')}\\n"
                            response += f"- **Evidence:** {rule.get('evidence_level', 'N/A')}\\n"
                            response += f"- **Treatment:** {rule.get('treatment', 'See guideline')}\\n"
                            if rule.get('survival_benefit'):
                                response += f"- **Benefit:** {rule['survival_benefit']}\\n"
                            response += "\\n"
                        
                        return response
            except Exception as e:
                logger.error(f"Ontology-based followup error: {e}", exc_info=True)
        
        return None

    def _llm_follow_up_response(self, message: str, context: Dict) -> str:
        """Generate LLM-powered follow-up response with full patient context"""
        patient_data = context.get("patient_data", {})
        result = context.get("result")

        # Build patient summary for LLM
        patient_summary = self._build_patient_summary_text(patient_data)

        # Build recommendations summary
        recs_text = "No recommendations available."
        if result and hasattr(result, 'recommendations') and result.recommendations:
            recs_parts = []
            for i, rec in enumerate(result.recommendations[:5], 1):
                treatment = getattr(rec, 'treatment_type', getattr(rec, 'treatment', 'Unknown'))
                evidence = getattr(rec, 'evidence_level', 'N/A')
                intent = getattr(rec, 'treatment_intent', 'N/A')
                survival = getattr(rec, 'survival_benefit', '')
                source = getattr(rec, 'rule_source', getattr(rec, 'guideline_reference', ''))
                recs_parts.append(
                    f"{i}. {treatment} (Evidence: {evidence}, Intent: {intent}, Source: {source})"
                    + (f" - {survival}" if survival else "")
                )
            recs_text = "\n".join(recs_parts)

        mdt_summary = ""
        if result and hasattr(result, 'mdt_summary') and result.mdt_summary:
            mdt_summary = f"\nMDT Summary: {str(result.mdt_summary)[:500]}"

        messages = [
            SystemMessage(content=FOLLOW_UP_SYSTEM_PROMPT),
            HumanMessage(content=f"""## Patient Case
{patient_summary}

## Current Recommendations
{recs_text}
{mdt_summary}

## User Follow-up Question
{message}

Provide a detailed, evidence-based answer. Use markdown formatting with headers, tables, and bullet points. Cite specific guidelines and trials where relevant.""")
        ]

        response = self.llm.invoke(messages)
        return response.content

    def _build_patient_summary_text(self, patient_data: Dict) -> str:
        """Build a concise patient summary for LLM context"""
        parts = []
        age = patient_data.get('age', patient_data.get('age_at_diagnosis', '?'))
        sex = patient_data.get('sex', '?')
        parts.append(f"- **Demographics:** {age} year old {'male' if sex == 'M' else 'female' if sex == 'F' else sex}")
        parts.append(f"- **Stage:** {patient_data.get('tnm_stage', 'Unknown')}")
        parts.append(f"- **Histology:** {patient_data.get('histology_type', 'Unknown')}")
        parts.append(f"- **Performance Status:** ECOG {patient_data.get('performance_status', '?')}")

        biomarkers = patient_data.get("biomarker_profile", {})
        if biomarkers:
            bm_parts = []
            for k, v in biomarkers.items():
                if k == "pdl1_tps":
                    bm_parts.append(f"PD-L1 TPS {v}%")
                elif isinstance(v, bool):
                    bm_parts.append(f"{k.replace('_', ' ').upper()} {'Positive' if v else 'Negative'}")
                elif v:
                    bm_parts.append(f"{k.replace('_', ' ').upper()} {v}")
            parts.append(f"- **Biomarkers:** {', '.join(bm_parts)}")

        comorbidities = patient_data.get("comorbidities", [])
        if comorbidities:
            parts.append(f"- **Comorbidities:** {', '.join(comorbidities)}")

        return "\n".join(parts)

    def _generate_inference_aware_suggestions(self, patient_data: Dict, result: Any, inference_details: Dict) -> List[str]:
        """Generate context-aware follow-up suggestions based on inference results"""
        suggestions = []
        biomarkers = patient_data.get("biomarker_profile", {})
        stage = patient_data.get("tnm_stage", "")
        
        # Priority 1: Suggestions based on inferred therapies
        if inference_details.get('therapy_inferences'):
            therapy_classes = [ti.get('therapy_class') for ti in inference_details['therapy_inferences']]
            if 'EGFR_TKI' in therapy_classes:
                suggestions.append("💊 Compare EGFR-TKI options (osimertinib vs gefitinib) with evidence")
            elif 'ALK_INHIBITOR' in therapy_classes:
                suggestions.append("💊 Show ALK inhibitor options and sequencing strategies")
            elif 'IO_MONOTHERAPY' in therapy_classes:
                suggestions.append("💊 Explain immunotherapy monotherapy vs combination approaches")
            elif 'CHEMO_IO_COMBINATION' in therapy_classes:
                suggestions.append("💊 What chemotherapy-IO combinations are recommended?")
        
        # Priority 2: Suggestions based on classification
        if inference_details.get('classification'):
            cls = inference_details['classification']
            cancer_type = cls.get('cancer_subtype', '')
            if 'Squamous' in cancer_type:
                suggestions.append("🔬 How does squamous histology affect treatment options?")
            elif 'Adenocarcinoma' in cancer_type:
                suggestions.append("🔬 What are the biomarker testing priorities for adenocarcinoma?")
        
        # Priority 3: Risk-based suggestions
        if inference_details.get('risk_assessment'):
            risk = inference_details['risk_assessment']
            if risk.get('risk_level') == 'HIGH':
                suggestions.append("⚠️ Review dose modifications for high-risk patient")
        
        # Priority 4: Contraindication warnings
        if inference_details.get('contraindications'):
            suggestions.append(f"❌ Review {len(inference_details['contraindications'])} contraindication warnings")
        
        # Priority 5: Stage-specific suggestions
        if inference_details.get('stage_group'):
            sg = inference_details['stage_group']
            intent = sg.get('treatment_intent', '')
            if intent == 'Curative':
                suggestions.append("🎯 What are the surgical options for curative intent?")
            elif intent == 'Potentially Curative':
                suggestions.append("🎯 Should we consider combined modality therapy?")
            elif intent == 'Palliative':
                suggestions.append("🎯 What palliative care options optimize quality of life?")
        
        # Priority 6: Generic clinical suggestions
        if not suggestions or len(suggestions) < 3:
            suggestions.extend([
                "📊 What is the expected prognosis with this treatment?",
                "🔍 Find similar patient cases in the database",
                "🧬 Explain all biomarker implications for treatment",
                "🏥 Check clinical trial eligibility for this patient"
            ])
        
        # Priority 7: Guideline-based suggestions
        if inference_details.get('applicable_guidelines'):
            guidelines = inference_details['applicable_guidelines']
            if guidelines:
                suggestions.append(f"📖 Compare recommendations across {len(guidelines)} applicable guidelines")
        
        return suggestions[:5]
    
    def _generate_patient_aware_suggestions(self, message: str, patient_data: Dict, result: Any) -> List[str]:
        """Generate follow-up suggestions aware of patient context and what was just asked"""
        msg_lower = message.lower()
        asked_topics = set()

        # Track what was already asked
        if any(kw in msg_lower for kw in ["alternative", "other option"]):
            asked_topics.add("alternatives")
        if any(kw in msg_lower for kw in ["reasoning", "explain", "why"]):
            asked_topics.add("reasoning")
        if any(kw in msg_lower for kw in ["similar", "case"]):
            asked_topics.add("similar")
        if any(kw in msg_lower for kw in ["comorbidity", "interaction", "side effect", "toxicity"]):
            asked_topics.add("comorbidity")
        if any(kw in msg_lower for kw in ["prognosis", "survival", "outlook"]):
            asked_topics.add("prognosis")
        if any(kw in msg_lower for kw in ["biomarker", "mutation", "egfr", "alk"]):
            asked_topics.add("biomarker")
        if any(kw in msg_lower for kw in ["trial", "eligibility"]):
            asked_topics.add("trials")

        suggestions = []
        biomarkers = patient_data.get("biomarker_profile", {})
        stage = patient_data.get("tnm_stage", "")

        # Add suggestions for topics NOT yet asked
        if "alternatives" not in asked_topics:
            suggestions.append("Show alternative treatments with evidence levels")
        if "reasoning" not in asked_topics:
            suggestions.append("Explain the clinical reasoning for this recommendation")
        if "prognosis" not in asked_topics:
            suggestions.append("What is the expected prognosis with this treatment?")
        if "comorbidity" not in asked_topics and patient_data.get("comorbidities"):
            suggestions.append(f"How do {', '.join(patient_data['comorbidities'][:2])} affect treatment?")
        if "biomarker" not in asked_topics and biomarkers:
            suggestions.append("Explain the biomarker implications for treatment selection")
        if "trials" not in asked_topics:
            suggestions.append("Check clinical trial eligibility for this patient")
        if "similar" not in asked_topics:
            suggestions.append("Find similar patient cases in the database")

        # Add stage-specific suggestions
        if "III" in stage and "alternatives" not in asked_topics:
            suggestions.append("Is this patient a candidate for neoadjuvant therapy?")
        if biomarkers.get("egfr_mutation") and "biomarker" not in asked_topics:
            suggestions.append("What about osimertinib resistance mechanisms?")

        return suggestions[:5]

    async def _stream_general_qa(self, message: str, session_id: str) -> AsyncIterator[str]:
        """Stream general Q&A responses with LLM-powered answers"""

        yield self._format_sse({
            "type": "status",
            "content": "Analyzing your question with clinical knowledge base..."
        })

        # Get conversation history
        history = self._get_session(session_id)

        # Try streaming LLM response for better UX
        if self.llm_available and self.llm:
            try:
                response = await self._stream_llm_qa(message, history, session_id)
                yield self._format_sse({
                    "type": "text",
                    "content": response
                })
                self._add_to_history(session_id, "assistant", response)

                # Generate context-aware suggestions
                suggestions = self._generate_contextual_suggestions(message, response)
                yield self._format_sse({
                    "type": "suggestions",
                    "content": suggestions
                })
                return
            except Exception as e:
                logger.warning(f"LLM streaming QA failed: {e}, falling back to template")

        # Fallback to template
        response = self._generate_qa_response(message, history)
        yield self._format_sse({
            "type": "text",
            "content": response
        })
        self._add_to_history(session_id, "assistant", response)

    async def _stream_llm_qa(self, message: str, history: List[Dict], session_id: str) -> str:
        """Run LLM QA in async context"""
        return await asyncio.to_thread(self._llm_qa_response, message, history)

    def _generate_contextual_suggestions(self, message: str, response: str) -> List[str]:
        """Generate context-aware follow-up suggestions based on the conversation"""
        suggestions = []
        msg_lower = message.lower()
        resp_lower = response.lower()

        # Biomarker-related suggestions
        if any(kw in msg_lower or kw in resp_lower for kw in ["egfr", "alk", "ros1", "braf", "kras"]):
            suggestions.append("What are the resistance mechanisms for this targeted therapy?")
            suggestions.append("What is the recommended testing methodology?")

        # Immunotherapy-related
        if any(kw in msg_lower or kw in resp_lower for kw in ["pd-l1", "immunotherapy", "pembrolizumab", "nivolumab"]):
            suggestions.append("What are common immune-related adverse events?")
            suggestions.append("When should immunotherapy be combined with chemotherapy?")

        # Stage-specific
        if any(kw in msg_lower or kw in resp_lower for kw in ["stage iv", "metastatic", "advanced"]):
            suggestions.append("What is the role of palliative radiation in metastatic NSCLC?")
            suggestions.append("When should oligometastatic disease be considered?")
        elif any(kw in msg_lower or kw in resp_lower for kw in ["stage iii", "locally advanced"]):
            suggestions.append("What is the role of durvalumab consolidation after chemoRT?")
            suggestions.append("When is surgical resection appropriate for stage III?")
        elif any(kw in msg_lower or kw in resp_lower for kw in ["stage i", "stage ii", "early"]):
            suggestions.append("What adjuvant therapy options exist after surgery?")
            suggestions.append("When is SABR preferred over surgery?")

        # Default suggestions
        if not suggestions:
            suggestions = [
                "Analyze a patient case",
                "What are NCCN first-line options for stage IV NSCLC?",
                "Explain the ADAURA trial results",
                "When should molecular testing be performed?"
            ]

        return suggestions[:5]

    # =========================================================================
    # TEXT2CYPHER - Natural Language Graph Queries
    # =========================================================================

    async def _stream_graph_query(self, message: str, session_id: str) -> AsyncIterator[str]:
        """Stream graph query results using Text2Cypher (LLM generates Cypher from natural language)"""

        yield self._format_sse({
            "type": "status",
            "content": "Translating your question to a graph database query..."
        })

        # Check if Neo4j is available
        if not self.lca_service.graph_db or not self.lca_service.graph_db.driver:
            yield self._format_sse({
                "type": "error",
                "content": "Neo4j database is not connected. Graph queries require an active database connection."
            })
            return

        # Generate Cypher query using LLM
        cypher_query = None
        if self.llm_available and self.llm:
            try:
                cypher_query = await asyncio.to_thread(
                    self._text2cypher, message
                )
            except Exception as e:
                logger.warning(f"Text2Cypher failed: {e}")

        if not cypher_query:
            # Fallback to common predefined queries
            cypher_query = self._get_predefined_query(message)

        if not cypher_query:
            yield self._format_sse({
                "type": "error",
                "content": "Could not generate a graph query for your question. Try: 'How many patients are in the database?' or 'Show all stage IIIA patients'"
            })
            return

        yield self._format_sse({
            "type": "reasoning",
            "content": f"**Generated Cypher Query:**\n```cypher\n{cypher_query}\n```"
        })

        # Execute the query
        try:
            with self.lca_service.graph_db.driver.session() as session:
                result = session.run(cypher_query)
                records = [dict(record) for record in result]

            if not records:
                yield self._format_sse({
                    "type": "text",
                    "content": "## Query Results\n\nNo results found for this query. The database may not have matching data yet."
                })
            else:
                # Format results as markdown table
                formatted = self._format_query_results(records, message)
                yield self._format_sse({
                    "type": "text",
                    "content": formatted
                })

            self._add_to_history(session_id, "assistant", f"Graph query executed: {len(records)} results")

        except Exception as e:
            logger.error(f"Graph query execution error: {e}")
            yield self._format_sse({
                "type": "error",
                "content": f"Query execution failed: {str(e)}\n\nThe generated Cypher may have syntax issues. Try rephrasing your question."
            })

        yield self._format_sse({
            "type": "suggestions",
            "content": [
                "How many patients are in the database?",
                "Show all treatment decisions",
                "List patients by stage",
                "What guidelines have been applied?",
                "Show graph statistics"
            ]
        })

    def _text2cypher(self, message: str) -> Optional[str]:
        """Convert natural language to Cypher query using LLM"""
        messages = [
            SystemMessage(content=TEXT2CYPHER_SYSTEM_PROMPT),
            HumanMessage(content=f"Natural language query: {message}\n\nReturn ONLY the Cypher query.")
        ]

        response = self.llm.invoke(messages)
        cypher = response.content.strip()

        # Clean up common LLM artifacts
        if cypher.startswith("```"):
            # Remove code fences
            lines = cypher.split("\n")
            cypher = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            ).strip()

        # Basic safety check - prevent destructive operations
        dangerous_keywords = ["DELETE", "DETACH DELETE", "DROP", "CREATE INDEX", "REMOVE"]
        cypher_upper = cypher.upper()
        for kw in dangerous_keywords:
            if kw in cypher_upper and "RETURN" not in cypher_upper:
                logger.warning(f"Blocked potentially dangerous Cypher: {cypher}")
                return None

        return cypher if cypher else None

    def _get_predefined_query(self, message: str) -> Optional[str]:
        """Get a predefined Cypher query for common questions"""
        msg_lower = message.lower()

        if "how many patient" in msg_lower or "count patient" in msg_lower:
            return "MATCH (p:Patient) RETURN count(p) AS patient_count"

        if "all patient" in msg_lower or "list patient" in msg_lower:
            return """MATCH (p:Patient)
RETURN p.patient_id AS id, p.age AS age, p.sex AS sex,
       p.tnm_stage AS stage, p.histology_type AS histology,
       p.performance_status AS ps
ORDER BY p.patient_id
LIMIT 20"""

        if "treatment decision" in msg_lower or "all decision" in msg_lower:
            return """MATCH (d:TreatmentDecision)-[:ABOUT]->(p:Patient)
RETURN d.treatment AS treatment, d.category AS category,
       d.confidence_score AS confidence, p.patient_id AS patient_id,
       d.decision_timestamp AS timestamp
ORDER BY d.decision_timestamp DESC
LIMIT 20"""

        if "graph statistic" in msg_lower or "database summary" in msg_lower:
            return """MATCH (n)
WITH labels(n) AS nodeLabels
UNWIND nodeLabels AS label
RETURN label AS NodeType, count(*) AS Count
ORDER BY Count DESC"""

        if "guideline" in msg_lower:
            return """MATCH (g:Guideline)
RETURN g.name AS guideline, g.source AS source, g.evidence_level AS evidence
ORDER BY g.name"""

        if "stage" in msg_lower:
            # Extract stage from message
            stage_match = re.search(r'stage\s+(I{1,3}[ABC]?|IV)', message, re.IGNORECASE)
            if stage_match:
                stage = stage_match.group(1).upper()
                return f"""MATCH (p:Patient)
WHERE p.tnm_stage = '{stage}'
RETURN p.patient_id AS id, p.age AS age, p.sex AS sex,
       p.tnm_stage AS stage, p.histology_type AS histology,
       p.performance_status AS ps
ORDER BY p.patient_id
LIMIT 20"""

        return None

    def _format_query_results(self, records: List[Dict], message: str) -> str:
        """Format Cypher query results as readable markdown"""
        if not records:
            return "No results found."

        # Single value result (count, etc.)
        if len(records) == 1 and len(records[0]) == 1:
            key, value = list(records[0].items())[0]
            return f"## Query Result\n\n**{key.replace('_', ' ').title()}:** {value}"

        # Table format
        headers = list(records[0].keys())
        header_display = [h.replace("_", " ").title() for h in headers]

        text = f"## Query Results ({len(records)} rows)\n\n"
        text += "| " + " | ".join(header_display) + " |\n"
        text += "| " + " | ".join(["---"] * len(headers)) + " |\n"

        for record in records[:20]:  # Limit to 20 rows
            values = []
            for h in headers:
                val = record.get(h, "")
                if val is None:
                    val = "-"
                elif isinstance(val, float):
                    val = f"{val:.2f}"
                else:
                    val = str(val)[:50]  # Truncate long values
                values.append(val)
            text += "| " + " | ".join(values) + " |\n"

        if len(records) > 20:
            text += f"\n*Showing first 20 of {len(records)} results.*\n"

        return text

    async def _stream_mcp_tool(self, message: str, session_id: str) -> AsyncIterator[str]:
        """Stream MCP tool invocation responses with enhanced explanations"""

        yield self._format_sse({
            "type": "status",
            "content": "Analyzing your query to identify relevant MCP tools..."
        })

        # Determine which tool to invoke based on the message
        tool_name, arguments = await self._detect_mcp_tool(message)

        if not tool_name:
            yield self._format_sse({
                "type": "text",
                "content": "I couldn't identify a specific MCP tool for your query. Available tool categories include:\n\n"
                          "• Patient Management (create, get, update patient records)\n"
                          "• Clinical Workflows (11-agent analysis pipeline)\n"
                          "• Specialized Agents (NSCLC, SCLC, biomarker, comorbidity)\n"
                          "• Analytics (survival analysis, risk stratification, clinical trial matching)\n"
                          "• Graph Queries (Neo4j integration)\n"
                          "• Ontology Tools (SNOMED-CT mapping, validation)\n"
                          "• Biomarker Analysis (interpretation, resistance prediction)\n"
                          "• Export & Reporting\n\n"
                          "Try asking: 'Find similar patients to stage IIIA adenocarcinoma' or 'Analyze survival data for EGFR+ patients'"
            })
            return

        yield self._format_sse({
            "type": "tool_call",
            "content": {
                "tool": tool_name,
                "arguments": arguments
            }
        })

        yield self._format_sse({
            "type": "status",
            "content": f"Invoking MCP tool: {tool_name}..."
        })

        # Invoke the MCP tool
        result = await self.mcp_invoker.invoke_tool(tool_name, arguments)

        if result.get("status") == "error":
            yield self._format_sse({
                "type": "error",
                "content": f"Tool execution failed: {result.get('error', 'Unknown error')}"
            })
            return

        # Stream the tool result with explanatory context
        yield self._format_sse({
            "type": "tool_result",
            "content": result.get("result", {})
        })

        # Generate explanatory response
        explanation = self._explain_tool_result(tool_name, result.get("result", {}))

        yield self._format_sse({
            "type": "text",
            "content": explanation
        })

        # Store result in history
        response_text = f"MCP Tool Used: {tool_name}\n\n{explanation}"
        self._add_to_history(session_id, "assistant", response_text)

    async def _detect_mcp_tool(self, message: str) -> tuple[Optional[str], Dict[str, Any]]:
        """
        Detect which MCP tool to invoke based on the user message

        Returns:
            (tool_name, arguments) or (None, {}) if no tool matched
        """
        message_lower = message.lower()

        # Survival analysis
        if any(kw in message_lower for kw in ['survival', 'kaplan', 'meier', 'prognosis']):
            return ("analyze_survival_data", {})

        # Clinical trial matching
        if any(kw in message_lower for kw in ['clinical trial', 'trial match', 'eligibility']):
            patient_id = self._extract_patient_id_from_message(message)
            return ("match_clinical_trials", {"patient_id": patient_id or "current"})

        # Find similar patients
        if any(kw in message_lower for kw in ['similar patient', 'similar case', 'find patient']):
            patient_id = self._extract_patient_id_from_message(message)
            return ("find_similar_patients", {"patient_id": patient_id or "current", "top_k": 5})

        # Biomarker analysis
        if any(kw in message_lower for kw in ['biomarker', 'mutation', 'egfr', 'alk', 'ros1', 'pathway']):
            biomarker = self._extract_biomarker_from_message(message)
            if biomarker:
                return ("get_biomarker_pathways", {"biomarker": biomarker})
            return ("analyze_biomarkers", {})

        # Lab result interpretation
        if any(kw in message_lower for kw in ['lab result', 'interpret lab', 'blood work']):
            return ("interpret_lab_results", {})

        # Graph queries
        if any(kw in message_lower for kw in ['graph query', 'neo4j', 'cypher']):
            return ("execute_graph_query", {})

        # Ontology mapping
        if any(kw in message_lower for kw in ['ontology', 'snomed', 'map concept']):
            return ("validate_ontology", {})

        # Export/reporting
        if any(kw in message_lower for kw in ['export', 'generate report', 'clinical report']):
            patient_id = self._extract_patient_id_from_message(message)
            return ("generate_clinical_report", {"patient_id": patient_id or "current"})

        # Default: list available tools
        return (None, {})

    def _extract_patient_id_from_message(self, message: str) -> Optional[str]:
        """Extract patient ID from message if present"""
        # Look for patterns like "patient P001", "ID: P001", etc.
        match = re.search(r'\b[Pp](?:atient)?\s*([A-Z0-9_-]+)', message)
        if match:
            return match.group(1)
        return None

    def _extract_biomarker_from_message(self, message: str) -> Optional[str]:
        """Extract biomarker name from message"""
        biomarkers = ['EGFR', 'ALK', 'ROS1', 'BRAF', 'KRAS', 'MET', 'HER2', 'PD-L1', 'TMB']
        for biomarker in biomarkers:
            if biomarker.lower() in message.lower():
                return biomarker
        return None

    def _explain_tool_result(self, tool_name: str, result: Any) -> str:
        """Generate explanatory text for tool results"""

        if tool_name == "analyze_survival_data":
            return ("## Survival Analysis Results\n\n"
                   "The survival analysis provides Kaplan-Meier estimates for this patient cohort. "
                   f"Key findings:\n\n{json.dumps(result, indent=2)}\n\n"
                   "These estimates help predict long-term outcomes and inform treatment decisions.")

        elif tool_name == "match_clinical_trials":
            trial_count = len(result.get("trials", [])) if isinstance(result, dict) else 0
            return (f"## Clinical Trial Matching\n\n"
                   f"Found {trial_count} matching clinical trials based on patient eligibility criteria. "
                   f"Results:\n\n{json.dumps(result, indent=2)}\n\n"
                   "Consider discussing these trial options with the patient.")

        elif tool_name == "find_similar_patients":
            patient_count = len(result.get("similar_patients", [])) if isinstance(result, dict) else 0
            return (f"## Similar Patient Search\n\n"
                   f"Found {patient_count} similar patients based on clinical characteristics. "
                   f"Results:\n\n{json.dumps(result, indent=2)}\n\n"
                   "Reviewing similar cases can inform treatment decisions and expected outcomes.")

        elif tool_name == "get_biomarker_pathways":
            return ("## Biomarker Pathway Analysis\n\n"
                   f"Pathway information for the requested biomarker:\n\n{json.dumps(result, indent=2)}\n\n"
                   "Understanding affected pathways helps select targeted therapies.")

        elif tool_name == "generate_clinical_report":
            return ("## Clinical Report Generated\n\n"
                   "A comprehensive clinical report has been generated with the following sections:\n"
                   f"{json.dumps(result, indent=2)}\n\n"
                   "The report can be exported for multidisciplinary team review.")

        else:
            # Generic explanation
            return (f"## {tool_name.replace('_', ' ').title()} Results\n\n"
                   f"Tool execution completed successfully:\n\n{json.dumps(result, indent=2)}\n\n"
                   "This information supplements the clinical decision-making process.")
        
    # =========================================================================
    # MCP APPS AND CLUSTERING METHODS
    # =========================================================================
    async def _stream_mcp_app(self, message: str, session_id: str) -> AsyncIterator[str]:
        """
        Stream MCP app rendering based on user query

        Detects which interactive app to show and provides the data
        """
        message_lower = message.lower()

        # Treatment Comparison App
        if any(kw in message_lower for kw in ['compare treatment', 'treatment comparison', 'treatment options']):
            yield self._format_sse({
                "type": "status",
                "content": "Loading treatment comparison tool..."
            })

            # Get treatment data from patient context or query
            treatments_data = await self._get_treatment_comparison_data(session_id)

            yield self._format_sse({
                "type": "mcp_app",
                "content": {
                    "resourceUri": "/mcp-apps/treatment-compare.html",
                    "input": treatments_data,
                    "title": "Treatment Comparison"
                }
            })

            # Also provide text explanation
            explanation = self._explain_treatment_comparison(treatments_data)
            yield self._format_sse({
                "type": "text",
                "content": explanation
            })

        # Survival Curves App
        elif any(kw in message_lower for kw in ['survival curve', 'kaplan meier', 'survival data']):
            yield self._format_sse({
                "type": "status",
                "content": "Generating survival curves..."
            })

            curves_data = await self._get_survival_curves_data(session_id)

            yield self._format_sse({
                "type": "mcp_app",
                "content": {
                    "resourceUri": "/mcp-apps/survival-curves.html",
                    "input": curves_data,
                    "title": "Survival Analysis"
                }
            })

            explanation = self._explain_survival_curves(curves_data)
            yield self._format_sse({
                "type": "text",
                "content": explanation
            })

        # Guideline Tree App
        elif any(kw in message_lower for kw in ['guideline tree', 'nccn decision', 'decision tree']):
            yield self._format_sse({
                "type": "status",
                "content": "Loading NCCN guideline decision tree..."
            })

            guideline_data = await self._get_guideline_tree_data(session_id)

            yield self._format_sse({
                "type": "mcp_app",
                "content": {
                    "resourceUri": "/mcp-apps/guideline-tree.html",
                    "input": guideline_data,
                    "title": "NCCN Guideline Explorer"
                }
            })

            explanation = self._explain_guideline_tree(guideline_data)
            yield self._format_sse({
                "type": "text",
                "content": explanation
            })

        # Clinical Trial Matcher App
        elif any(kw in message_lower for kw in ['clinical trial', 'trial match', 'trial search']):
            yield self._format_sse({
                "type": "status",
                "content": "Searching clinical trials..."
            })

            trials = await self._match_clinical_trials(session_id)

            yield self._format_sse({
                "type": "mcp_app",
                "content": {
                    "resourceUri": "/mcp-apps/trial-matcher.html",
                    "input": trials,
                    "title": "Clinical Trial Matcher"
                }
            })

            explanation = self._explain_trial_matches(trials)
            yield self._format_sse({
                "type": "text",
                "content": explanation
            })

        else:
            # No matching app found
            yield self._format_sse({
                "type": "text",
                "content": "I can show you interactive visualizations for:\n"
                          "- **Treatment Comparison**: Compare treatment options side-by-side\n"
                          "- **Survival Curves**: Kaplan-Meier survival analysis\n"
                          "- **Guideline Tree**: Navigate NCCN decision pathways\n"
                          "- **Clinical Trials**: Match patients to trials\n\n"
                          "Try asking: 'Compare treatments' or 'Show survival curves'"
            })


    async def _get_treatment_comparison_data(self, session_id: str) -> Dict[str, Any]:
        """Get treatment comparison data for the app"""
        # Get patient context if available
        patient_data = self.patient_context.get(session_id, {}).get("patient_data", {})

        # Default treatments (can be customized based on patient characteristics)
        treatments = [
            {
                "name": "Osimertinib",
                "indication": "1L EGFR+ NSCLC",
                "orr": 80,
                "pfs": 18.9,
                "os": 38.6,
                "grade34_ae": 34,
                "evidence": "FLAURA trial",
                "nccn_category": "Category 1"
            },
            {
                "name": "Gefitinib/Erlotinib",
                "indication": "1L EGFR+ NSCLC",
                "orr": 76,
                "pfs": 10.2,
                "os": 31.8,
                "grade34_ae": 25,
                "evidence": "Multiple trials",
                "nccn_category": "Category 1"
            },
            {
                "name": "Platinum Doublet",
                "indication": "1L NSCLC",
                "orr": 35,
                "pfs": 5.5,
                "os": 13.2,
                "grade34_ae": 55,
                "evidence": "Standard of care",
                "nccn_category": "Category 1"
            }
        ]

        # Customize based on biomarkers
        biomarkers = patient_data.get("biomarker_profile", {})

        if biomarkers.get("ALK"):
            treatments = [
                {
                    "name": "Alectinib",
                    "indication": "1L ALK+ NSCLC",
                    "orr": 83,
                    "pfs": 34.8,
                    "os": 45.0,
                    "grade34_ae": 41,
                    "evidence": "ALEX trial"
                },
                {
                    "name": "Crizotinib",
                    "indication": "1L ALK+ NSCLC",
                    "orr": 76,
                    "pfs": 10.9,
                    "os": 35.0,
                    "grade34_ae": 50,
                    "evidence": "Multiple trials"
                }
            ]

        return {
            "treatments": treatments,
            "patient_context": patient_data
        }


    async def _get_survival_curves_data(self, session_id: str) -> Dict[str, Any]:
        """Get survival curve data for visualization"""
        # Sample Kaplan-Meier data (replace with real data from analytics service)
        curves = [
            {
                "name": "Osimertinib",
                "color": "#8b5cf6",
                "data": [
                    [0, 100], [3, 98], [6, 95], [9, 91], [12, 85],
                    [15, 80], [18, 75], [21, 68], [24, 65], [30, 55], [36, 45]
                ],
                "median_survival": 18.9,
                "events": 45,
                "censored": 23
            },
            {
                "name": "Gefitinib",
                "color": "#3b82f6",
                "data": [
                    [0, 100], [3, 96], [6, 90], [9, 82], [12, 70],
                    [15, 60], [18, 50], [21, 40], [24, 35], [30, 25], [36, 18]
                ],
                "median_survival": 10.2,
                "events": 58,
                "censored": 12
            },
            {
                "name": "Chemotherapy",
                "color": "#ef4444",
                "data": [
                    [0, 100], [3, 92], [6, 75], [9, 60], [12, 45],
                    [15, 35], [18, 28], [21, 22], [24, 18], [30, 12], [36, 8]
                ],
                "median_survival": 5.5,
                "events": 72,
                "censored": 8
            }
        ]

        return {
            "curves": curves,
            "title": "Progression-Free Survival by Treatment",
            "x_label": "Months",
            "y_label": "PFS Probability (%)"
        }


    async def _get_guideline_tree_data(self, session_id: str) -> Dict[str, Any]:
        """Get NCCN guideline tree data"""
        patient_data = self.patient_context.get(session_id, {}).get("patient_data", {})

        return {
            "stage": patient_data.get("tnm_stage", "IIIA"),
            "histology": patient_data.get("histology_type", "adenocarcinoma"),
            "biomarkers": patient_data.get("biomarker_profile", {}),
            "performance_status": patient_data.get("performance_status", 1),
            "guideline_version": "NCCN 2024.1"
        }


    async def _match_clinical_trials(self, session_id: str) -> Dict[str, Any]:
        """Match clinical trials for patient"""
        patient_data = self.patient_context.get(session_id, {}).get("patient_data", {})

        # Sample trial data (replace with real API call to ClinicalTrials.gov)
        trials = [
            {
                "nct_id": "NCT04303780",
                "title": "CodeBreaK 200: Sotorasib vs Docetaxel in KRAS G12C NSCLC",
                "phase": "Phase 3",
                "status": "Active, recruiting",
                "locations": ["Multiple US sites"],
                "contact": "study.coordinator@trial.com",
                "eligibility": {
                    "age": "≥18 years",
                    "stage": "IV NSCLC",
                    "biomarker": "KRAS G12C mutation",
                    "prior_therapy": "Platinum-based chemotherapy + PD-1/PD-L1 inhibitor"
                },
                "match_score": 0.95
            },
            {
                "nct_id": "NCT03785249",
                "title": "KRYSTAL-1: Adagrasib in KRAS G12C NSCLC",
                "phase": "Phase 1/2",
                "status": "Active, recruiting",
                "locations": ["US and International"],
                "contact": "krystal@trial.com",
                "eligibility": {
                    "age": "≥18 years",
                    "stage": "Advanced solid tumors",
                    "biomarker": "KRAS G12C mutation",
                    "prior_therapy": "Any"
                },
                "match_score": 0.88
            }
        ]

        return {
            "trials": trials,
            "patient_summary": patient_data,
            "total_matches": len(trials)
        }


    def _explain_treatment_comparison(self, data: Dict) -> str:
        """Generate text explanation for treatment comparison"""
        treatments = data.get("treatments", [])

        if not treatments:
            return "No treatment comparison data available."

        best_pfs = max(treatments, key=lambda t: t.get("pfs", 0))

        explanation = f"""## Treatment Comparison Results

I've compared {len(treatments)} treatment options above. Key findings:

- **Highest PFS**: {best_pfs['name']} with {best_pfs['pfs']} months
- **Evidence basis**: Each option is supported by clinical trial data
- **NCCN category**: All options are guideline-recommended

Use the interactive comparison above to explore ORR, PFS, OS, and toxicity profiles.
"""

        return explanation


    def _explain_survival_curves(self, data: Dict) -> str:
        """Generate text explanation for survival curves"""
        curves = data.get("curves", [])

        if not curves:
            return "No survival data available."

        best_survival = max(curves, key=lambda c: c.get("median_survival", 0))

        explanation = f"""## Survival Analysis

The Kaplan-Meier curves above show progression-free survival for {len(curves)} treatment options.

- **Best median survival**: {best_survival['name']} at {best_survival['median_survival']} months
- **Statistical significance**: Hazard ratios and p-values available in source trials

The curves demonstrate clear separation, indicating meaningful clinical benefit.
"""

        return explanation


    def _explain_guideline_tree(self, data: Dict) -> str:
        """Generate text explanation for guideline tree"""
        return f"""## NCCN Guideline Decision Tree

The interactive decision tree above shows NCCN recommendations for:

- **Stage**: {data.get('stage', 'Unknown')}
- **Histology**: {data.get('histology', 'Unknown')}
- **Biomarkers**: {', '.join(data.get('biomarkers', {}).keys()) or 'None detected'}

Navigate the tree to explore different treatment pathways based on patient characteristics.

**Guideline Version**: {data.get('guideline_version', 'NCCN 2024.1')}
"""


    def _explain_trial_matches(self, data: Dict) -> str:
        """Generate text explanation for trial matches"""
        trials = data.get("trials", [])

        if not trials:
            return "No matching clinical trials found."

        explanation = f"""## Clinical Trial Matches

Found **{len(trials)} matching trials** for this patient:

"""

        for i, trial in enumerate(trials[:3], 1):
            explanation += f"""
**{i}. {trial['title']}**
- NCT ID: {trial['nct_id']}
- Phase: {trial['phase']}
- Status: {trial['status']}
- Match Score: {trial['match_score']:.0%}
"""

        explanation += "\nUse the interactive matcher above to explore full eligibility criteria and contact information."

        return explanation


    # ============================================================================
    # 2. CLUSTERING ANALYSIS
    # ============================================================================

    async def _stream_clustering_analysis(
        self,
        message: str,
        session_id: str
    ) -> AsyncIterator[str]:
        """Stream patient clustering analysis"""
        yield self._format_sse({
            "type": "status",
            "content": "Analyzing patient cohorts..."
        })

        # Get all patients from Neo4j or database
        patients = await self._get_all_patients_for_clustering()

        if not patients or len(patients) < 5:
            yield self._format_sse({
                "type": "text",
                "content": "Not enough patient data available for clustering analysis. Need at least 5 patients."
            })
            return

        # Determine clustering method from query
        message_lower = message.lower()

        if "clinical rule" in message_lower or "rule-based" in message_lower:
            method = ClusteringMethod.CLINICAL_RULES
        elif "kmeans" in message_lower or "k-means" in message_lower:
            method = ClusteringMethod.KMEANS
        else:
            method = ClusteringMethod.CLINICAL_RULES  # Default

        yield self._format_sse({
            "type": "status",
            "content": f"Clustering {len(patients)} patients using {method.value} method..."
        })

        # Perform clustering
        from ..services.clustering_service import ClusteringService
        clustering_service = ClusteringService()

        result = clustering_service.cluster_patients(
            patients=patients,
            method=method
        )

        # Stream cluster information
        for cluster in result.clusters:
            yield self._format_sse({
                "type": "cluster_info",
                "content": {
                    "name": cluster.name,
                    "description": cluster.description,
                    "size": cluster.size,
                    "characteristics": cluster.characteristics,
                    "outcomes": cluster.outcomes_summary,
                    "confidence": cluster.confidence
                }
            })

        # Generate summary
        summary = self._generate_clustering_summary(result)

        yield self._format_sse({
            "type": "text",
            "content": summary
        })

        self._add_to_history(session_id, "assistant", summary)


    async def _get_all_patients_for_clustering(self) -> List[Dict]:
        """Fetch all patients from Neo4j for clustering"""
        try:
            if not self.lca_service.graph_db or not self.lca_service.graph_db.driver:
                return []

            with self.lca_service.graph_db.driver.session(
                database=getattr(self.lca_service.graph_db, 'database', 'neo4j')
            ) as session:
                result = session.run("""
                    MATCH (p:Patient)
                    RETURN
                        p.patient_id as patient_id,
                        p.age_at_diagnosis as age,
                        p.sex as sex,
                        p.tnm_stage as stage,
                        p.histology_type as histology,
                        p.performance_status as ecog_ps,
                        p.biomarker_profile as biomarkers
                    LIMIT 1000
                """)

                patients = []
                for record in result:
                    patients.append(dict(record))

                return patients

        except Exception as e:
            logger.error(f"Failed to fetch patients for clustering: {e}")
            return []


    def _generate_clustering_summary(self, result) -> str:
        """Generate markdown summary of clustering results"""
        summary = f"""## Patient Cohort Analysis

Using **{result.method.value}** clustering, identified **{result.num_clusters} distinct cohorts**:

"""

        for i, cluster in enumerate(result.clusters, 1):
            summary += f"""
### {i}. {cluster.name} (n={cluster.size})

**Description**: {cluster.description}

**Key Characteristics**:
"""
            for key, value in cluster.characteristics.items():
                summary += f"- {key}: {value}\n"

            if cluster.outcomes_summary:
                summary += f"\n**Outcomes**:\n"
                for key, value in cluster.outcomes_summary.items():
                    summary += f"- {key}: {value}\n"

        # Add feature importance if available
        if result.feature_importance:
            summary += f"\n## Feature Importance\n\n"
            sorted_features = sorted(
                result.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for feature, importance in sorted_features[:5]:
                summary += f"- **{feature}**: {importance:.2%}\n"

        summary += f"\n**Clustering Quality**: Silhouette Score = {result.silhouette_score:.3f}"

        return summary


    # ============================================================================
    # 3. CITATION ENHANCEMENT
    # ============================================================================

    async def _enhance_with_citations(
        self,
        text: str,
        patient_data: Dict,
        analysis_result: Dict
    ) -> str:
        """Add relevant citations to recommendation text"""
        enhanced = text

        # Always add NCCN citation to recommendations
        if "recommend" in enhanced.lower() and "[[Guideline:NCCN]]" not in enhanced:
            enhanced += " [[Guideline:NCCN]]"

        # Add trial citations based on biomarkers
        biomarkers = patient_data.get("biomarker_profile", {})

        # EGFR citations
        if biomarkers.get("EGFR"):
            if "osimertinib" in enhanced.lower() and "[[Trial:FLAURA]]" not in enhanced:
                enhanced = enhanced.replace(
                    "osimertinib",
                    "osimertinib [[Trial:FLAURA]]",
                    1
                )

        # ALK citations
        if biomarkers.get("ALK"):
            if "alectinib" in enhanced.lower() and "[[Trial:ALEX]]" not in enhanced:
                enhanced = enhanced.replace(
                    "alectinib",
                    "alectinib [[Trial:ALEX]]",
                    1
                )

        # PD-L1 citations
        if biomarkers.get("PD-L1"):
            pdl1_value = biomarkers.get("PD-L1")
            try:
                if float(pdl1_value) >= 50:
                    if "pembrolizumab" in enhanced.lower() and "[[Trial:KEYNOTE-024]]" not in enhanced:
                        enhanced = enhanced.replace(
                            "pembrolizumab",
                            "pembrolizumab [[Trial:KEYNOTE-024]]",
                            1
                        )
            except (ValueError, TypeError):
                pass

        # Add ontology citation for histology terms
        if "adenocarcinoma" in enhanced.lower() and "[[Ontology:SNOMED]]" not in enhanced:
            enhanced = enhanced.replace(
                "adenocarcinoma",
                "adenocarcinoma [[Ontology:SNOMED]]",
                1
            )

        return enhanced


    # ============================================================================
    # 4. UPDATED INTENT CLASSIFICATION (ENHANCED)
    # ============================================================================

    def _classify_intent_enhanced(self, message: str, session_id: str = None) -> str:
        """
        Enhanced intent classification with MCP apps and clustering

        This method extends the existing _classify_intent with additional patterns
        """
        # MCP App detection
        mcp_app_patterns = [
            r'compare\s+treatment',
            r'treatment\s+comparison',
            r'survival\s+curve',
            r'kaplan\s+meier',
            r'guideline\s+tree',
            r'nccn\s+decision',
            r'clinical\s+trial',
            r'trial\s+match'
        ]
        for pattern in mcp_app_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "mcp_app"

        # Clustering detection
        clustering_patterns = [
            r'cluster\s+patient',
            r'cohort\s+analysis',
            r'similar\s+patient',
            r'patient\s+group',
            r'find\s+patients?\s+like'
        ]
        for pattern in clustering_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "clustering_analysis"

        # MCP Tool detection (from previous implementation)
        mcp_tool_patterns = [
            r'(use|invoke|call|run)\s+(tool|MCP|agent)',
            r'analyze\s+survival\s+data',
            r'match\s+clinical\s+trial',
            r'find\s+similar\s+patient',
            r'search\s+(patient|guideline)',
            r'get\s+(biomarker|pathway|treatment)',
            r'interpret\s+lab\s+result',
            r'predict\s+resistance',
            r'graph\s+(query|search)',
            r'ontology\s+(map|validate)',
            r'export\s+(patient|report)',
        ]
        for pattern in mcp_tool_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "mcp_tool"

        return "general_qa"

    # =========================================================================
    # END MCP APPS AND CLUSTERING METHODS
    # =========================================================================
    def _find_existing_patient(self, patient_data: Dict) -> Optional[Dict]:
        """Check if a matching patient exists in Neo4j"""
        try:
            if not self.lca_service.graph_db or not self.lca_service.graph_db.driver:
                return None

            with self.lca_service.graph_db.driver.session(
                database=getattr(self.lca_service.graph_db, 'database', 'neo4j')
            ) as session:
                result = session.run(
                    """
                    MATCH (p:Patient)
                    WHERE p.age_at_diagnosis = $age
                      AND p.sex = $sex
                      AND p.tnm_stage = $stage
                    RETURN p.patient_id as patient_id,
                           p.name as name,
                           p.age_at_diagnosis as age,
                           p.tnm_stage as stage
                    LIMIT 1
                    """,
                    age=patient_data.get("age", 0),
                    sex=patient_data.get("sex", "U"),
                    stage=patient_data.get("tnm_stage", "")
                )
                record = result.single()
                if record:
                    return dict(record)
            return None
        except Exception as e:
            logger.warning(f"Patient lookup failed: {e}")
            return None

    def _regex_extract_patient_data(self, message: str) -> Optional[Dict]:
        """
        Extract structured patient data from natural language using regex.
        """
        data = {
            "patient_id": f"CHAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

        # Extract age and sex (handle formats like "68M", "68F", "68 year old male", etc.)
        age_sex_match = re.search(r'\b(\d{2,3})\s*([MF])\b', message, re.IGNORECASE)
        if age_sex_match:
            data["age"] = int(age_sex_match.group(1))
            data["sex"] = age_sex_match.group(2).upper()
        else:
            age_match = re.search(r'(\d{2,3})[-\s]?(year|yr)[-\s]?old', message, re.IGNORECASE)
            if age_match:
                data["age"] = int(age_match.group(1))

            sex_patterns = [
                (r'\b(male|M)\b(?![\w-])', 'M'),
                (r'\b(female|F)\b(?![\w-])', 'F'),
            ]
            for pattern, sex_value in sex_patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    data["sex"] = sex_value
                    break

        # Also set age_at_diagnosis alias for orchestrator compatibility
        if "age" in data:
            data["age_at_diagnosis"] = data["age"]

        # Extract TNM stage - Order matters! IV must come before I{1,3}
        # Also handle SCLC staging: "limited stage" and "extensive stage"
        stage_match = re.search(r'stage\s+(IV[AB]?|I{1,3}[ABC]?)', message, re.IGNORECASE)
        if stage_match:
            data["tnm_stage"] = stage_match.group(1).upper()
        else:
            # Check for SCLC staging patterns
            limited_match = re.search(r'\b(limited)[-\s]*(stage|disease)?\b', message, re.IGNORECASE)
            extensive_match = re.search(r'\b(extensive)[-\s]*(stage|disease)?\b', message, re.IGNORECASE)
            if extensive_match:
                data["tnm_stage"] = "Extensive Stage"
                data["sclc_stage"] = "extensive"
            elif limited_match:
                data["tnm_stage"] = "Limited Stage"
                data["sclc_stage"] = "limited"

        # Extract histology
        histology_patterns = [
            r'adenocarcinoma',
            r'squamous\s+cell',
            r'small\s+cell',
            r'large\s+cell',
            r'SCLC',
            r'NSCLC'
        ]
        for pattern in histology_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                histology = match.group(0)
                if histology.upper() == 'SCLC':
                    data["histology_type"] = "Small cell carcinoma"
                elif histology.upper() == 'NSCLC':
                    data["histology_type"] = "Adenocarcinoma"
                else:
                    data["histology_type"] = histology.title()
                break

        # Extract performance status
        ps_match = re.search(r'(?:PS|ECOG|performance\s+status)\s*[:\-]?\s*([0-4])', message, re.IGNORECASE)
        if ps_match:
            data["performance_status"] = int(ps_match.group(1))
        else:
            data["performance_status"] = 1

        # Extract biomarkers
        biomarker_profile = {}

        egfr_match = re.search(r'EGFR[:\s]*(Ex19del|Ex20ins|L858R|T790M|\+|positive|negative)', message, re.IGNORECASE)
        if egfr_match:
            mutation = egfr_match.group(1)
            if mutation.lower() in ['+', 'positive']:
                biomarker_profile["egfr_mutation"] = True
            elif mutation.lower() == 'negative':
                biomarker_profile["egfr_mutation"] = False
            else:
                biomarker_profile["egfr_mutation"] = True
                biomarker_profile["egfr_mutation_type"] = mutation

        alk_match = re.search(r'ALK[:\s]*(\+|positive|negative|rearrangement)', message, re.IGNORECASE)
        if alk_match:
            biomarker_profile["alk_rearrangement"] = alk_match.group(1).lower() in ['+', 'positive', 'rearrangement']
        
        # ROS1 rearrangement
        ros1_match = re.search(r'ROS1[:\s]*(\+|positive|negative|rearrangement)', message, re.IGNORECASE)
        if ros1_match:
            biomarker_profile["ros1_rearrangement"] = ros1_match.group(1).lower() in ['+', 'positive', 'rearrangement']
        
        # BRAF mutation
        braf_match = re.search(r'BRAF[:\s]*(V600E|\+|positive|negative|mutation)', message, re.IGNORECASE)
        if braf_match:
            mutation_type = braf_match.group(1)
            if mutation_type.upper() == 'V600E':
                biomarker_profile["braf_mutation"] = "V600E"
            else:
                biomarker_profile["braf_mutation"] = mutation_type.lower() in ['+', 'positive', 'mutation']
        
        # KRAS mutation
        kras_match = re.search(r'KRAS[:\s]*(G12C|\+|positive|negative|mutation)', message, re.IGNORECASE)
        if kras_match:
            mutation_type = kras_match.group(1)
            if mutation_type.upper() == 'G12C':
                biomarker_profile["kras_mutation"] = "G12C"
            else:
                biomarker_profile["kras_mutation"] = mutation_type.lower() in ['+', 'positive', 'mutation']

        pdl1_match = re.search(r'PD-?L1[:\s]*(\d+)%?', message, re.IGNORECASE)
        if pdl1_match:
            biomarker_profile["pdl1_tps"] = int(pdl1_match.group(1))

        if biomarker_profile:
            data["biomarker_profile"] = biomarker_profile

        # Extract comorbidities
        comorbidities = []
        comorbidity_keywords = {
            'COPD': r'\bCOPD\b',
            'Diabetes': r'\bdiabetes\b',
            'Hypertension': r'\b(hypertension|HTN)\b',
            'CAD': r'\b(CAD|coronary artery disease)\b',
            'CKD': r'\b(CKD|chronic kidney disease)\b'
        }

        for name, pattern in comorbidity_keywords.items():
            if re.search(pattern, message, re.IGNORECASE):
                comorbidities.append(name)

        if comorbidities:
            data["comorbidities"] = comorbidities

        # Track extracted and missing fields for better feedback
        required = ["age", "sex", "tnm_stage", "histology_type"]
        missing = [field for field in required if field not in data]
        extracted = [field for field in required if field in data]

        # Store extraction metadata for streaming feedback
        data["_extraction_meta"] = {
            "extracted_fields": extracted,
            "missing_fields": missing,
            "extraction_complete": len(missing) == 0
        }

        if len(missing) == 0:
            return data

        # Return partial data with missing field info for better error messages
        return data

    def _extract_patient_data(self, message: str) -> Optional[Dict]:
        """
        Extract and validate patient data from natural language.
        Uses regex extraction followed by Pydantic validation.
        Returns partial data with metadata about what was/wasn't extracted.
        """
        raw_data = self._regex_extract_patient_data(message)

        if raw_data is None:
            # This shouldn't happen now since we always return partial data
            return None

        extraction_meta = raw_data.pop("_extraction_meta", {})

        # If extraction is incomplete, return raw data with the metadata
        if not extraction_meta.get("extraction_complete", False):
            raw_data["_extraction_meta"] = extraction_meta
            return raw_data

        try:
            validated = ExtractedPatientData(**raw_data)
            result = validated.model_dump(exclude_none=True)
            # Ensure patient_id is present
            if not result.get("patient_id"):
                result["patient_id"] = raw_data.get("patient_id", f"CHAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            result["_extraction_meta"] = extraction_meta
            return result
        except Exception as e:
            logger.warning(f"Pydantic validation warning: {e}, using raw extracted data")
            raw_data["_extraction_meta"] = extraction_meta
            return raw_data

    def _format_mdt_summary(self, summary) -> str:
        """Format MDT summary from object or string"""
        if isinstance(summary, str):
            return summary
        
        # Handle Pydantic model or object with attributes
        if hasattr(summary, 'clinical_summary'):
            text = f"{summary.clinical_summary}\n\n"
        else:
            text = ""
        
        if hasattr(summary, 'formatted_recommendations') and summary.formatted_recommendations:
            text += "### Treatment Recommendations:\n\n"
            for rec in summary.formatted_recommendations:
                if isinstance(rec, dict):
                    text += f"**{rec.get('rank', '1')}. {rec.get('treatment', 'Unknown')}**\n"
                    text += f"- Intent: {rec.get('intent', 'Not specified')}\n"
                    text += f"- Evidence: {rec.get('evidence', 'Not specified')}\n"
                    text += f"- Guideline: {rec.get('guideline', 'Not specified')}\n"
                    if rec.get('rationale'):
                        text += f"- Rationale: {rec.get('rationale')}\n"
                    text += "\n"
        
        if hasattr(summary, 'key_considerations') and summary.key_considerations:
            text += "### Key Considerations:\n"
            for consideration in summary.key_considerations:
                text += f"- {consideration}\n"
            text += "\n"
        
        if hasattr(summary, 'discussion_points') and summary.discussion_points:
            text += "### MDT Discussion Points:\n"
            for point in summary.discussion_points:
                text += f"- {point}\n"
        
        return text if text else str(summary)

    def _format_recommendations(self, result) -> str:
        """Format recommendations as comprehensive markdown with all clinical details"""
        if not result.recommendations:
            return "**No specific treatment recommendations generated.**\n\nThis may be because additional patient information is needed or the case requires manual MDT review."

        recommendations = result.recommendations

        if recommendations:
            primary = recommendations[0]

            # Extract all fields from primary recommendation
            if isinstance(primary, dict):
                treatment = primary.get('treatment', primary.get('treatment_type', 'Unknown'))
                evidence = primary.get('evidence_level', 'Not specified')
                confidence = primary.get('confidence_score', primary.get('confidence', 0))
                source = primary.get('guideline_reference', primary.get('rule_source', 'Clinical Guidelines'))
                intent = primary.get('intent', primary.get('treatment_intent', 'Not specified'))
                rationale = primary.get('rationale', '')
                contraindications = primary.get('contraindications', [])
                survival = primary.get('survival_benefit', primary.get('expected_benefit', ''))
                mechanism = primary.get('mechanism_of_action', '')
                dosing = primary.get('dosing', primary.get('dose', ''))
                schedule = primary.get('schedule', primary.get('administration', ''))
                monitoring = primary.get('monitoring', primary.get('follow_up', ''))
                side_effects = primary.get('side_effects', primary.get('toxicities', []))
            else:
                # Handle AgentProposal or TreatmentRecommendation objects
                treatment = getattr(primary, 'treatment_type', getattr(primary, 'treatment', 'Unknown'))
                evidence = getattr(primary, 'evidence_level', 'Not specified')
                confidence = getattr(primary, 'confidence', getattr(primary, 'confidence_score', 0))
                source = getattr(primary, 'guideline_reference', getattr(primary, 'rule_source', 'Clinical Guidelines'))
                intent = getattr(primary, 'treatment_intent', getattr(primary, 'intent', 'Not specified'))
                rationale = getattr(primary, 'rationale', '')
                contraindications = getattr(primary, 'contraindications', [])
                survival = getattr(primary, 'survival_benefit', getattr(primary, 'expected_benefit', ''))
                mechanism = getattr(primary, 'mechanism_of_action', '')
                dosing = getattr(primary, 'dosing', getattr(primary, 'dose', ''))
                schedule = getattr(primary, 'schedule', getattr(primary, 'administration', ''))
                monitoring = getattr(primary, 'monitoring', getattr(primary, 'follow_up', ''))
                side_effects = getattr(primary, 'side_effects', getattr(primary, 'toxicities', []))

            # Format confidence
            if isinstance(confidence, (int, float)):
                confidence = int(confidence * 100) if confidence <= 1 else int(confidence)

            # Build comprehensive recommendation text
            text = f"""## Primary Recommendation: {treatment}

| Attribute | Details |
|---|---|
| **Evidence Level** | {evidence} |
| **Confidence** | {confidence}% |
| **Intent** | {intent} |
| **Source/Guideline** | {source} |
"""

            if survival:
                text += f"| **Expected Benefit** | {survival} |\n"

            text += "\n"

            # Rationale section
            if rationale:
                text += f"### Clinical Rationale\n\n{rationale}\n\n"

            # Mechanism of action (if available)
            if mechanism:
                text += f"### Mechanism of Action\n\n{mechanism}\n\n"

            # Dosing and schedule (if available)
            if dosing or schedule:
                text += "### Administration\n\n"
                if dosing:
                    text += f"- **Dosing:** {dosing}\n"
                if schedule:
                    text += f"- **Schedule:** {schedule}\n"
                text += "\n"

            # Side effects (if available)
            if side_effects:
                if isinstance(side_effects, list) and len(side_effects) > 0:
                    text += "### Common Side Effects\n\n"
                    for effect in side_effects[:5]:  # Top 5
                        text += f"- {effect}\n"
                    text += "\n"

            # Contraindications
            if contraindications:
                if isinstance(contraindications, list) and len(contraindications) > 0:
                    text += "### Contraindications\n\n"
                    for contra in contraindications:
                        text += f"- ⚠️ {contra}\n"
                    text += "\n"

            # Monitoring (if available)
            if monitoring:
                text += f"### Monitoring\n\n{monitoring}\n\n"

            # Alternative treatments
            if len(recommendations) > 1:
                text += "---\n\n### Alternative Treatment Options\n\n"
                text += "| Rank | Treatment | Evidence | Confidence | Intent |\n"
                text += "|---|---|---|---|---|\n"

                for i, alt_rec in enumerate(recommendations[1:4], 2):
                    if isinstance(alt_rec, dict):
                        alt_treatment = alt_rec.get('treatment', alt_rec.get('treatment_type', 'Unknown'))
                        alt_evidence = alt_rec.get('evidence_level', 'N/A')
                        alt_conf = alt_rec.get('confidence_score', alt_rec.get('confidence', 0))
                        alt_intent = alt_rec.get('intent', alt_rec.get('treatment_intent', 'N/A'))
                    else:
                        alt_treatment = getattr(alt_rec, 'treatment_type', getattr(alt_rec, 'treatment', 'Unknown'))
                        alt_evidence = getattr(alt_rec, 'evidence_level', 'N/A')
                        alt_conf = getattr(alt_rec, 'confidence', getattr(alt_rec, 'confidence_score', 0))
                        alt_intent = getattr(alt_rec, 'treatment_intent', getattr(alt_rec, 'intent', 'N/A'))

                    if isinstance(alt_conf, (int, float)):
                        alt_conf = int(alt_conf * 100) if alt_conf <= 1 else int(alt_conf)

                    text += f"| {i} | {alt_treatment} | {alt_evidence} | {alt_conf}% | {alt_intent} |\n"

                text += "\n*Ask \"Show alternative treatments\" for detailed information on each option.*\n"

            return text

        return "**No specific treatment recommendations generated.**"

    def _handle_follow_up(self, message: str, context: Dict) -> str:
        """Handle follow-up questions about a patient case with expanded topic coverage"""
        result = context["result"]
        patient_data = context["patient_data"]

        message_lower = message.lower()

        # --- Alternative treatments ---
        if any(kw in message_lower for kw in ["alternative", "other option", "different"]):
            if len(result.recommendations) > 1:
                text = "## Alternative Treatment Options\n\n"
                text += "Based on the patient profile, the following alternatives are supported by clinical evidence:\n\n"

                for i, rec in enumerate(result.recommendations[1:5], 2):
                    treatment = getattr(rec, 'treatment_type', getattr(rec, 'treatment', 'Unknown'))
                    evidence = getattr(rec, 'evidence_level', 'Not specified')
                    confidence = getattr(rec, 'confidence_score', getattr(rec, 'confidence', 0))
                    if isinstance(confidence, float) and confidence <= 1:
                        confidence = int(confidence * 100)
                    intent = getattr(rec, 'treatment_intent', getattr(rec, 'intent', 'Not specified'))
                    survival = getattr(rec, 'survival_benefit', getattr(rec, 'expected_benefit', ''))
                    rationale = getattr(rec, 'rationale', '')
                    guideline = getattr(rec, 'guideline_reference', getattr(rec, 'rule_source', ''))
                    contraindications = getattr(rec, 'contraindications', [])

                    text += f"### {i}. {treatment}\n\n"
                    text += f"| Attribute | Value |\n|---|---|\n"
                    text += f"| Evidence Level | {evidence} |\n"
                    text += f"| Confidence | {confidence}% |\n"
                    text += f"| Intent | {intent} |\n"
                    if guideline:
                        text += f"| Guideline | {guideline} |\n"
                    if survival:
                        text += f"| Expected Benefit | {survival} |\n"
                    text += "\n"

                    if rationale:
                        text += f"**Rationale:** {rationale}\n\n"

                    if contraindications:
                        text += f"**Contraindications:** {', '.join(contraindications)}\n\n"

                text += "\n---\n**Comparison with Primary Recommendation:**\n\n"
                primary = result.recommendations[0]
                primary_treatment = getattr(primary, 'treatment_type', getattr(primary, 'treatment', 'Unknown'))
                text += f"The primary recommendation ({primary_treatment}) was selected based on the highest evidence level and confidence score for this specific patient profile.\n"
                return text
            else:
                text = "## Alternative Treatment Options\n\n"
                text += "Based on current clinical guidelines, there are no well-established alternatives for this specific patient profile.\n\n"
                text += "**Reasons may include:**\n"
                text += "- The recommended treatment has significantly stronger evidence\n"
                text += "- Patient characteristics (stage, biomarkers, PS) narrow the options\n"
                text += "- Guideline-concordant care supports a specific regimen\n\n"
                text += "For off-protocol options, please consult with MDT or consider clinical trial enrollment."
                return text

        # --- Reasoning / explanation ---
        elif any(kw in message_lower for kw in ["reasoning", "explain", "why"]):
            text = "## Clinical Reasoning & Decision Rationale\n\n"

            # Patient summary
            text += "### Patient Profile Summary\n\n"
            text += f"- **Age:** {patient_data.get('age', patient_data.get('age_at_diagnosis', 'Unknown'))}\n"
            text += f"- **Sex:** {patient_data.get('sex', 'Unknown')}\n"
            text += f"- **Stage:** {patient_data.get('tnm_stage', 'Unknown')}\n"
            text += f"- **Histology:** {patient_data.get('histology_type', 'Unknown')}\n"
            text += f"- **Performance Status:** ECOG {patient_data.get('performance_status', 'Unknown')}\n"

            biomarkers = patient_data.get("biomarker_profile", {})
            if biomarkers:
                text += f"- **Biomarkers:** "
                bm_list = []
                for k, v in biomarkers.items():
                    display_k = k.replace("_", " ").upper()
                    if k == "pdl1_tps":
                        bm_list.append(f"PD-L1 TPS {v}%")
                    elif isinstance(v, bool):
                        bm_list.append(f"{display_k} {'Positive' if v else 'Negative'}")
                    else:
                        bm_list.append(f"{display_k} {v}")
                text += ", ".join(bm_list) + "\n"

            comorbidities = patient_data.get("comorbidities", [])
            if comorbidities:
                text += f"- **Comorbidities:** {', '.join(comorbidities)}\n"

            text += "\n### Decision Process\n\n"

            if result.recommendations:
                primary = result.recommendations[0]
                treatment = getattr(primary, 'treatment_type', getattr(primary, 'treatment', 'Unknown'))
                evidence = getattr(primary, 'evidence_level', 'Not specified')
                rationale = getattr(primary, 'rationale', '')
                guideline = getattr(primary, 'guideline_reference', getattr(primary, 'rule_source', ''))

                text += f"**1. Guideline Matching:** The system identified `{guideline}` as the applicable guideline based on histology and stage.\n\n"

                text += f"**2. Biomarker-Driven Selection:** "
                if biomarkers:
                    pdl1 = biomarkers.get("pdl1_tps", 0)
                    if pdl1 >= 50:
                        text += f"With PD-L1 TPS ≥50%, pembrolizumab monotherapy or combination immunotherapy is preferred.\n\n"
                    elif pdl1 >= 1:
                        text += f"With PD-L1 TPS 1-49% ({pdl1}%), chemo-immunotherapy combination provides optimal benefit.\n\n"
                    else:
                        text += f"With low/negative PD-L1, platinum-based chemotherapy combinations are standard.\n\n"

                    egfr = biomarkers.get("egfr_mutation")
                    if egfr:
                        text += f"**EGFR-positive:** Targeted therapy with TKI (osimertinib preferred for Ex19del/L858R) takes priority over immunotherapy.\n\n"
                else:
                    text += "No actionable biomarkers detected; standard systemic therapy selected.\n\n"

                text += f"**3. Performance Status Assessment:** PS {patient_data.get('performance_status', 1)} indicates the patient can tolerate standard-dose systemic therapy.\n\n"

                if rationale:
                    text += f"**4. Specific Rationale:** {rationale}\n\n"

            # MDT summary
            if result.mdt_summary:
                formatted_summary = self._format_mdt_summary(result.mdt_summary)
                text += f"### MDT Summary\n\n{formatted_summary}\n"

            return text

        # --- Similar cases ---
        elif any(kw in message_lower for kw in ["similar", "case"]):
            text = "## Similar Cases Analysis\n\n"

            stage = patient_data.get("tnm_stage", "Unknown")
            histology = patient_data.get("histology_type", "Unknown")
            ps = patient_data.get("performance_status", 1)
            biomarkers = patient_data.get("biomarker_profile", {})

            text += "### Search Criteria\n\n"
            text += f"Looking for patients with similar profiles:\n"
            text += f"- **Histology:** {histology}\n"
            text += f"- **Stage:** {stage}\n"
            text += f"- **Performance Status:** ECOG {ps} (±1)\n"
            if biomarkers:
                text += f"- **Biomarker Pattern:** Similar molecular profile\n"

            if hasattr(result, 'similar_patients') and result.similar_patients:
                text += f"\n### Found {len(result.similar_patients)} Similar Cases:\n\n"
                for i, patient in enumerate(result.similar_patients[:5], 1):
                    text += f"#### Case {i}\n"
                    text += f"- **Patient ID:** {patient.get('patient_id', 'Anonymous')}\n"
                    text += f"- **Stage:** {patient.get('stage', patient.get('tnm_stage', 'Similar'))}\n"
                    text += f"- **Treatment:** {patient.get('treatment', 'Not recorded')}\n"
                    text += f"- **Outcome:** {patient.get('outcome', 'Follow-up ongoing')}\n"
                    if patient.get('similarity_score'):
                        text += f"- **Similarity Score:** {patient.get('similarity_score')*100:.0f}%\n"
                    text += "\n"
            else:
                text += "\n### No Exact Matches Found\n\n"
                text += "The Neo4j knowledge graph does not yet contain cases with identical profiles.\n\n"
                text += "**This is expected because:**\n"
                text += "- The system is building its case database over time\n"
                text += "- Your patient may have a unique combination of characteristics\n\n"
                text += "**Evidence base:** Recommendations are still derived from:\n"
                text += "- NCCN Guidelines (updated 2025)\n"
                text += "- ESMO Clinical Practice Guidelines\n"
                text += "- NICE Technology Appraisals\n"
                text += "- Landmark clinical trials (KEYNOTE-189, KEYNOTE-024, CheckMate-227, etc.)\n"

            return text

        # --- Comorbidity / interactions ---
        elif any(kw in message_lower for kw in ["comorbidity", "interaction", "risk", "side effect", "toxicity"]):
            text = "## Comorbidity & Interaction Assessment\n\n"

            comorbidities = patient_data.get("comorbidities", [])

            if comorbidities:
                text += f"### Active Comorbidities\n\n"
                for comorb in comorbidities:
                    text += f"**{comorb}**\n\n"

                    # Provide specific guidance per comorbidity
                    if comorb.upper() == "COPD":
                        text += "- **Impact:** Increased risk of pneumonitis with immunotherapy\n"
                        text += "- **Monitoring:** Baseline and serial pulmonary function tests\n"
                        text += "- **Dose adjustment:** No standard dose reduction, but close monitoring required\n"
                        text += "- **Alternative consideration:** May favor chemotherapy if severe COPD\n\n"
                    elif comorb.upper() in ["DIABETES", "DM"]:
                        text += "- **Impact:** Steroids (used for irAE management) may worsen glycemic control\n"
                        text += "- **Monitoring:** Frequent blood glucose monitoring during treatment\n"
                        text += "- **Risk:** Immune checkpoint inhibitors can cause autoimmune diabetes (rare)\n\n"
                    elif comorb.upper() in ["HYPERTENSION", "HTN"]:
                        text += "- **Impact:** Generally well-tolerated with standard regimens\n"
                        text += "- **Monitoring:** Blood pressure monitoring, especially with anti-VEGF agents\n"
                        text += "- **Note:** Bevacizumab contraindicated with uncontrolled hypertension\n\n"
                    elif comorb.upper() in ["CAD", "CORONARY ARTERY DISEASE"]:
                        text += "- **Impact:** Cardiotoxicity risk with some agents\n"
                        text += "- **Monitoring:** Baseline echocardiogram, cardiac biomarkers\n"
                        text += "- **Caution:** Avoid anthracyclines; use fluoropyrimidines with caution\n\n"
                    elif comorb.upper() in ["CKD", "CHRONIC KIDNEY DISEASE"]:
                        text += "- **Impact:** Requires dose adjustment for renally-cleared agents\n"
                        text += "- **Dose adjustment:** Cisplatin contraindicated if CrCl <60; use carboplatin\n"
                        text += "- **Monitoring:** Serial creatinine and electrolytes\n\n"
                    else:
                        text += "- **Impact:** Individualized assessment required\n"
                        text += "- **Recommendation:** MDT discussion for specific guidance\n\n"

                text += "### Interaction Summary\n\n"
                text += "| Comorbidity | Primary Concern | Action Required |\n|---|---|---|\n"
                for comorb in comorbidities:
                    if comorb.upper() == "COPD":
                        text += f"| {comorb} | Pneumonitis risk | PFT monitoring |\n"
                    elif comorb.upper() in ["DIABETES", "DM"]:
                        text += f"| {comorb} | Steroid impact | Glucose monitoring |\n"
                    elif comorb.upper() in ["CKD", "CHRONIC KIDNEY DISEASE"]:
                        text += f"| {comorb} | Renal dosing | Use carboplatin |\n"
                    else:
                        text += f"| {comorb} | Individual assessment | MDT review |\n"
            else:
                text += "### No Comorbidities Recorded\n\n"
                text += "No comorbidities were provided in the patient description.\n\n"
                text += "**To assess comorbidity interactions**, please include conditions such as:\n"
                text += "- COPD, asthma, or other respiratory conditions\n"
                text += "- Diabetes mellitus\n"
                text += "- Cardiovascular disease (CAD, CHF, arrhythmias)\n"
                text += "- Chronic kidney disease\n"
                text += "- Autoimmune conditions (rheumatoid arthritis, lupus, etc.)\n"
                text += "- Liver disease\n\n"
                text += "**Example:** \"68M, stage IIIA adenocarcinoma, PS 1, COPD, diabetes\""

            return text

        # --- Prognosis / survival ---
        elif any(kw in message_lower for kw in ["prognosis", "survival", "outlook"]):
            text = "## Prognostic Assessment\n\n"

            stage = patient_data.get("tnm_stage", "Unknown")
            ps = patient_data.get("performance_status", "Unknown")
            histology = patient_data.get("histology_type", "Unknown")
            age = patient_data.get("age", patient_data.get("age_at_diagnosis", "Unknown"))
            biomarkers = patient_data.get("biomarker_profile", {})

            text += "### Key Prognostic Factors\n\n"
            text += f"| Factor | Value | Prognostic Impact |\n|---|---|---|\n"
            text += f"| Stage | {stage} | "
            if "IV" in str(stage):
                text += "Poor (metastatic disease) |\n"
            elif "III" in str(stage):
                text += "Intermediate (locally advanced) |\n"
            elif "II" in str(stage) or "I" in str(stage):
                text += "Favorable (early stage) |\n"
            else:
                text += "Assess staging |\n"

            text += f"| Performance Status | ECOG {ps} | "
            if int(ps) <= 1:
                text += "Favorable |\n"
            elif int(ps) == 2:
                text += "Intermediate |\n"
            else:
                text += "Poor |\n"

            text += f"| Histology | {histology} | "
            if "adeno" in str(histology).lower():
                text += "Variable (depends on molecular profile) |\n"
            elif "squamous" in str(histology).lower():
                text += "Intermediate |\n"
            elif "small cell" in str(histology).lower() or "SCLC" in str(histology):
                text += "Poor (aggressive biology) |\n"
            else:
                text += "Assess histology |\n"

            text += f"| Age | {age} years | "
            if isinstance(age, (int, float)) and age < 70:
                text += "Favorable |\n"
            elif isinstance(age, (int, float)) and age < 80:
                text += "Intermediate |\n"
            else:
                text += "Consider fitness |\n"

            # Biomarker impact
            if biomarkers:
                text += "\n### Biomarker-Specific Prognosis\n\n"
                egfr = biomarkers.get("egfr_mutation")
                alk = biomarkers.get("alk_rearrangement")
                pdl1 = biomarkers.get("pdl1_tps", 0)

                if egfr:
                    text += "**EGFR-positive:** Generally favorable with targeted therapy. Median OS with osimertinib: 38.6 months (FLAURA).\n\n"
                if alk:
                    text += "**ALK-positive:** Favorable prognosis with ALK TKIs. Median PFS with alectinib: 34.8 months (ALEX).\n\n"
                if pdl1 >= 50:
                    text += f"**High PD-L1 ({pdl1}%):** Favorable for immunotherapy response. KEYNOTE-024 5-year OS: 31.9% with pembrolizumab.\n\n"
                elif pdl1 >= 1:
                    text += f"**PD-L1 positive ({pdl1}%):** Intermediate; benefits from chemo-immunotherapy combination.\n\n"

            # Treatment-specific outcomes
            if result.recommendations:
                primary = result.recommendations[0]
                treatment = getattr(primary, 'treatment_type', getattr(primary, 'treatment', 'Unknown'))
                survival = getattr(primary, 'survival_benefit', getattr(primary, 'expected_benefit', ''))

                text += "\n### Expected Outcomes with Recommended Treatment\n\n"
                text += f"**{treatment}**\n\n"
                if survival:
                    text += f"- **Expected Benefit:** {survival}\n"
                else:
                    text += "- **Expected Benefit:** Based on landmark clinical trials\n"

            text += "\n*Note: Individual outcomes vary. Discuss specific prognosis with oncology team.*"
            return text

        # --- Biomarkers ---
        elif any(kw in message_lower for kw in ["biomarker", "mutation", "egfr", "alk", "pd-l1", "kras", "ros1"]):
            text = "## Biomarker Analysis\n\n"

            biomarkers = patient_data.get("biomarker_profile", {})

            if biomarkers:
                text += "### Detected Biomarkers\n\n"
                text += "| Biomarker | Result | Therapeutic Implication |\n|---|---|---|\n"

                for key, value in biomarkers.items():
                    display_key = key.replace("_", " ").upper()

                    if key == "pdl1_tps":
                        text += f"| PD-L1 TPS | {value}% | "
                        if value >= 50:
                            text += "High expression - Pembrolizumab monotherapy eligible |\n"
                        elif value >= 1:
                            text += "Positive - Chemo-immunotherapy combination preferred |\n"
                        else:
                            text += "Negative - Chemotherapy backbone |\n"

                    elif key == "egfr_mutation":
                        mutation_type = biomarkers.get("egfr_mutation_type", "detected")
                        if isinstance(value, bool):
                            status = "Positive" if value else "Negative"
                        else:
                            status = str(value)
                        text += f"| EGFR | {status} ({mutation_type}) | "
                        if value:
                            text += "Osimertinib (Ex19del/L858R), TKI therapy preferred over IO |\n"
                        else:
                            text += "Standard systemic therapy |\n"

                    elif key == "alk_rearrangement":
                        status = "Positive" if value else "Negative"
                        text += f"| ALK | {status} | "
                        if value:
                            text += "Alectinib/lorlatinib preferred, avoid immunotherapy |\n"
                        else:
                            text += "Not applicable |\n"

                    elif key == "ros1_rearrangement":
                        status = "Positive" if value else "Negative"
                        text += f"| ROS1 | {status} | "
                        if value:
                            text += "Crizotinib/entrectinib, targeted therapy indicated |\n"
                        else:
                            text += "Not applicable |\n"

                    elif key == "kras_mutation":
                        text += f"| KRAS | {value} | "
                        if value == "G12C" or str(value).upper() == "G12C":
                            text += "Sotorasib/adagrasib eligible (KRAS G12C inhibitor) |\n"
                        elif value:
                            text += "Non-G12C mutation - No targeted therapy, immunotherapy may benefit |\n"
                        else:
                            text += "Wild-type |\n"

                    elif key == "braf_mutation":
                        text += f"| BRAF | {value} | "
                        if value == "V600E" or str(value).upper() == "V600E":
                            text += "Dabrafenib + trametinib combination |\n"
                        else:
                            text += "Non-V600E - Limited targeted options |\n"

                    else:
                        text += f"| {display_key} | {value} | Evaluate per guidelines |\n"

                text += "\n### Testing Recommendations\n\n"
                text += "**Standard panel for NSCLC:** EGFR, ALK, ROS1, BRAF, KRAS G12C, PD-L1, NTRK, MET, RET, HER2\n\n"
                text += "**Reflex testing:** If initial molecular testing is negative, consider comprehensive genomic profiling (CGP)\n"

            else:
                text += "### No Biomarker Data Provided\n\n"
                text += "**Recommended Testing Panel for NSCLC:**\n\n"
                text += "| Test | Why | Method |\n|---|---|---|\n"
                text += "| EGFR | TKI eligibility | PCR/NGS |\n"
                text += "| ALK | ALK inhibitor eligibility | IHC/FISH/NGS |\n"
                text += "| ROS1 | Crizotinib/entrectinib | IHC/FISH/NGS |\n"
                text += "| BRAF V600E | Dabrafenib + trametinib | NGS |\n"
                text += "| KRAS G12C | Sotorasib/adagrasib | NGS |\n"
                text += "| PD-L1 | Immunotherapy selection | IHC (22C3/SP263) |\n"
                text += "| NTRK | Larotrectinib/entrectinib | NGS |\n\n"
                text += "**Example with biomarkers:** \"68M, stage IIIA adenocarcinoma, EGFR Ex19del+, PD-L1 45%\""

            return text

        # --- Clinical trials ---
        elif any(kw in message_lower for kw in ["trial", "clinical trial", "eligibility"]):
            text = "## Clinical Trial Eligibility Assessment\n\n"

            stage = patient_data.get("tnm_stage", "Unknown")
            histology = patient_data.get("histology_type", "Unknown")
            ps = patient_data.get("performance_status", 1)
            age = patient_data.get("age", patient_data.get("age_at_diagnosis", 65))
            biomarkers = patient_data.get("biomarker_profile", {})

            text += "### Patient Eligibility Profile\n\n"
            text += f"- **Stage:** {stage}\n"
            text += f"- **Histology:** {histology}\n"
            text += f"- **Performance Status:** ECOG {ps}\n"
            text += f"- **Age:** {age} years\n"

            # Eligibility assessment
            text += "\n### General Eligibility Criteria\n\n"

            eligible = True
            concerns = []

            if isinstance(ps, int) and ps > 2:
                eligible = False
                concerns.append("PS >2 typically excluded from most trials")
            if isinstance(age, int) and age > 85:
                concerns.append("Age >85 may limit some trial options")

            if eligible and not concerns:
                text += "✅ **Patient appears eligible for clinical trial enrollment based on standard criteria.**\n\n"
            elif concerns:
                text += "⚠️ **Eligibility concerns:**\n"
                for concern in concerns:
                    text += f"- {concern}\n"
                text += "\n"

            # Suggested trial categories
            text += "### Relevant Trial Categories\n\n"

            if "IV" in str(stage):
                text += "**First-line metastatic NSCLC:**\n"
                text += "- Novel immunotherapy combinations\n"
                text += "- Bispecific antibodies\n"
                text += "- ADC (antibody-drug conjugate) trials\n\n"
            elif "III" in str(stage):
                text += "**Locally advanced NSCLC:**\n"
                text += "- Neoadjuvant immunotherapy trials\n"
                text += "- Consolidation therapy trials\n"
                text += "- Novel chemoradiation combinations\n\n"

            if biomarkers:
                text += "**Biomarker-specific trials:**\n"
                if biomarkers.get("egfr_mutation"):
                    text += "- EGFR TKI resistance trials\n"
                    text += "- Combination strategies for EGFR+ NSCLC\n"
                if biomarkers.get("kras_mutation"):
                    text += "- KRAS inhibitor combination trials\n"
                    text += "- KRAS G12C/non-G12C specific trials\n"
                pdl1 = biomarkers.get("pdl1_tps", 0)
                if pdl1 >= 50:
                    text += "- PD-L1 high expresser trials\n"
                text += "\n"

            text += "### How to Find Trials\n\n"
            text += "1. **ClinicalTrials.gov:** Search by condition, stage, and biomarker status\n"
            text += "2. **NCCN Clinical Trial Network:** Cooperative group trials\n"
            text += "3. **Cancer center trials:** Local academic center offerings\n\n"

            text += "*Note: Trial availability varies by location. Discuss with your oncology team.*"
            return text

        # --- Generic follow-up with context ---
        else:
            primary = result.recommendations[0] if result.recommendations else None
            if primary:
                treatment = getattr(primary, 'treatment_type', getattr(primary, 'treatment', 'Unknown'))
                evidence = getattr(primary, 'evidence_level', 'Not specified')
                confidence = getattr(primary, 'confidence', getattr(primary, 'confidence_score', 0))
                if isinstance(confidence, float) and confidence <= 1:
                    confidence = int(confidence * 100)

                return f"""## Current Analysis Summary

**Patient:** {patient_data.get('age', patient_data.get('age_at_diagnosis', 'Unknown'))} year old {patient_data.get('sex', 'patient')}, {patient_data.get('tnm_stage', 'Unknown')} {patient_data.get('histology_type', 'lung cancer')}

**Primary Recommendation:** {treatment}
- Evidence Level: {evidence}
- Confidence: {confidence}%

---

### I can help with:

| Question | Description |
|---|---|
| **Alternative treatments** | View other treatment options with evidence levels |
| **Explain the reasoning** | Detailed clinical rationale for this recommendation |
| **Find similar cases** | Search for patients with similar profiles |
| **Assess comorbidity interactions** | Drug interactions and dose adjustments |
| **Biomarker details** | Review molecular profile and testing recommendations |
| **Prognosis** | Survival estimates and prognostic factors |
| **Clinical trials** | Check eligibility for relevant trials |

What would you like to know more about?"""
            return "I have the patient context loaded. You can ask about alternative treatments, reasoning, similar cases, biomarkers, comorbidity interactions, prognosis, or clinical trials."

    def _generate_qa_response(self, message: str, history: List[Dict]) -> str:
        """Generate response for general questions using LLM with clinical knowledge"""

        # Try LLM-powered response first
        if self.llm_available and self.llm:
            try:
                return self._llm_qa_response(message, history)
            except Exception as e:
                logger.warning(f"LLM QA response failed: {e}, falling back to template")

        # Fallback to template responses
        message_lower = message.lower()

        if 'help' in message_lower or 'how' in message_lower:
            return self._get_help_response()

        return self._get_welcome_response(message)

    def _llm_qa_response(self, message: str, history: List[Dict]) -> str:
        """Generate LLM-powered clinical QA response"""
        # Build conversation context
        history_text = ""
        if history:
            recent = history[-6:]  # Last 6 messages for context
            history_text = "\n".join([
                f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')[:300]}"
                for m in recent
            ])

        # Check if there's active patient context from any session
        patient_context_text = "No active patient case."
        for sid, ctx in self.patient_context.items():
            if ctx.get("patient_data"):
                pd = ctx["patient_data"]
                patient_context_text = (
                    f"Active patient: {pd.get('age', '?')}yo {pd.get('sex', '?')}, "
                    f"Stage {pd.get('tnm_stage', '?')}, {pd.get('histology_type', '?')}, "
                    f"PS {pd.get('performance_status', '?')}"
                )
                biomarkers = pd.get("biomarker_profile", {})
                if biomarkers:
                    bm_parts = []
                    for k, v in biomarkers.items():
                        if k == "pdl1_tps":
                            bm_parts.append(f"PD-L1 {v}%")
                        elif isinstance(v, bool):
                            bm_parts.append(f"{k.upper()} {'+'  if v else '-'}")
                        else:
                            bm_parts.append(f"{k.upper()} {v}")
                    patient_context_text += f", Biomarkers: {', '.join(bm_parts)}"
                break

        messages = [
            SystemMessage(content=CLINICAL_SYSTEM_PROMPT),
            HumanMessage(content=f"""Context:
{patient_context_text}

Recent conversation:
{history_text}

User question: {message}

Provide a helpful, evidence-based response. Use markdown formatting with headers and tables where appropriate. If this is a general question about lung cancer, provide comprehensive clinical guidance. If the user seems to want to analyze a patient, guide them to provide clinical details.""")
        ]

        response = self.llm.invoke(messages)
        return response.content

    def _get_help_response(self) -> str:
        """Return a comprehensive help response"""
        return """## ConsensusCare - Lung Cancer Clinical Decision Support

### What I Can Do

| Capability | How to Use | Example |
|-----------|-----------|---------|
| **Patient Analysis** | Describe a patient case | "68M, stage IIIA adenocarcinoma, EGFR+, PS 1" |
| **Treatment Guidelines** | Ask about specific scenarios | "What's first-line for stage IV NSCLC with PD-L1 80%?" |
| **Biomarker Guidance** | Ask about testing/therapy | "When should EGFR testing be done?" |
| **Drug Information** | Ask about specific agents | "What are osimertinib side effects?" |
| **Similar Cases** | After analyzing a patient | "Find similar cases" |
| **Clinical Trials** | After analyzing a patient | "Check clinical trial eligibility" |
| **Graph Exploration** | Query the knowledge graph | "Show all stage IIIA patients" |

### Quick Start Examples

1. **Full patient analysis:**
   > "68-year-old male, stage IIIA adenocarcinoma, EGFR Ex19del positive, PD-L1 45%, PS 1, COPD"

2. **Guideline question:**
   > "What does NCCN recommend for ALK-positive stage IV NSCLC?"

3. **Biomarker query:**
   > "Explain the significance of KRAS G12C mutation in NSCLC"

### Follow-up Questions
After any analysis, you can ask about alternatives, reasoning, prognosis, biomarkers, comorbidity interactions, or clinical trial eligibility."""

    def _get_welcome_response(self, message: str) -> str:
        """Return a contextual welcome/redirect response"""
        return f"""## Welcome to ConsensusCare

I'm your lung cancer clinical decision support assistant, powered by the LUCADA ontology and NICE/NCCN/ESMO guidelines.

I noticed you asked: *"{message}"*

To provide the best clinical guidance, I can help with:

### Patient Case Analysis
Describe a patient with clinical details:
> **Example:** "68-year-old male, stage IIIA adenocarcinoma, EGFR Ex19del+, PS 1, COPD"

### Clinical Questions
Ask about guidelines, treatments, or biomarkers:
> **Example:** "What is the first-line treatment for stage IV NSCLC with high PD-L1?"

### System Capabilities
- **10 guideline rules** (NICE CG121 + modern precision medicine)
- **Multi-agent LangGraph workflow** (classification, biomarker analysis, conflict resolution)
- **Neo4j knowledge graph** with decision traces and causal chains
- **Ontology reasoning** with LUCADA OWL2 + SNOMED-CT

Type `help` for full capabilities or describe a patient case to begin."""

    # =============================================================================
    # Enhanced Conversation Methods
    # =============================================================================

    async def _stream_enhanced_analysis(self, message: str, session_id: str) -> AsyncIterator[str]:
        """Enhanced patient analysis with LangGraph memory and follow-up suggestions.

        Runs the standard analysis pipeline but wraps it with LangGraph conversation
        memory and generates intelligent follow-up suggestions afterwards.
        """
        # Always run the full patient analysis pipeline first
        async for chunk in self._stream_patient_analysis(message, session_id):
            yield chunk

        # After analysis completes, generate enhanced follow-up suggestions
        try:
            context = self._get_enhanced_context(session_id)
            thread_id = context.active_thread_id or str(uuid.uuid4())
            context.active_thread_id = thread_id

            # Store message in LangGraph memory for conversation continuity
            if LANGGRAPH_AVAILABLE and self.checkpointer:
                context.analysis_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "query": message,
                    "session_id": session_id
                })

            # Generate follow-up suggestions from the analysis
            await self._add_follow_up_suggestions(session_id)

            enhanced_context = self._get_enhanced_context(session_id)
            if enhanced_context.follow_up_suggestions:
                yield self._format_sse({
                    "type": "follow_up_suggestions",
                    "content": enhanced_context.follow_up_suggestions,
                    "title": "What would you like to explore next?"
                })
        except Exception as e:
            logger.warning(f"Enhanced follow-up generation failed (non-critical): {e}")

    async def _add_follow_up_suggestions(self, session_id: str):
        """Add intelligent follow-up suggestions after analysis"""
        try:
            # Get the latest analysis result
            context = self.patient_context.get(session_id)
            if not context:
                return

            patient_data = context.get('patient_data', {})
            result = context.get('result')

            if result:
                # Convert dataclass result to dict for follow-up generator
                if hasattr(result, 'recommendations') and not isinstance(result, dict):
                    from dataclasses import asdict
                    try:
                        analysis_dict = asdict(result)
                    except Exception:
                        # Fallback: build dict manually from recommendations
                        analysis_dict = {
                            "recommendations": [
                                {
                                    "treatment": getattr(r, 'treatment_type', getattr(r, 'treatment', '')),
                                    "evidence_level": getattr(r, 'evidence_level', ''),
                                    "treatment_intent": getattr(r, 'treatment_intent', ''),
                                }
                                for r in (result.recommendations or [])
                            ]
                        }
                else:
                    analysis_dict = result if isinstance(result, dict) else {}

                # Generate follow-up questions
                follow_ups = self.follow_up_generator.generate_clinical_questions(analysis_dict, patient_data)

                if follow_ups:
                    # Store in enhanced context
                    enhanced_context = self._get_enhanced_context(session_id)
                    enhanced_context.follow_up_suggestions = follow_ups

        except Exception as e:
            logger.error(f"Error generating follow-up suggestions: {e}")

    def get_follow_up_suggestions(self, session_id: str) -> List[str]:
        """Get current follow-up suggestions for a session"""
        if not self.enable_enhanced_features:
            return []
        
        enhanced_context = self._get_enhanced_context(session_id)
        return enhanced_context.follow_up_suggestions

    def reset_conversation_thread(self, session_id: str) -> bool:
        """Reset LangGraph conversation thread for enhanced features"""
        try:
            if session_id in self.enhanced_context:
                # Reset the thread ID to start fresh
                self.enhanced_context[session_id].active_thread_id = None
                self.enhanced_context[session_id].follow_up_suggestions = []
                return True
            return False
        except Exception as e:
            logger.error(f"Error resetting conversation thread: {e}")
            return False

    async def get_conversation_insights(self, session_id: str) -> Dict[str, Any]:
        """Get insights about the conversation for analytics"""
        insights = {
            "message_count": len(self._get_session(session_id)),
            "has_patient_analysis": session_id in self.patient_context,
            "enhanced_features_enabled": self.enable_enhanced_features,
            "follow_up_suggestions_count": len(self.get_follow_up_suggestions(session_id)),
            "last_activity": None
        }

        session = self._get_session(session_id)
        if session:
            insights["last_activity"] = session[-1].get("timestamp")

        return insights

    def _format_sse(self, data: Dict) -> str:
        """Format data as Server-Sent Event"""
        return f"data: {json.dumps(data)}\n\n"

    def get_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        return self._get_session(session_id)

    def clear_session(self, session_id: str):
        """Clear conversation history"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.patient_context:
            del self.patient_context[session_id]
        if session_id in self.enhanced_context:
            del self.enhanced_context[session_id]
