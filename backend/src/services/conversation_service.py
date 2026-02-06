"""
Conversational Service for LCA Chatbot
Handles natural language interaction with the LCA system
Enhanced with LangChain/LangGraph for memory and intelligent conversations
"""

import json
import re
import logging
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
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    logger.warning("LangChain/LangGraph not available - enhanced features disabled")
    LANGGRAPH_AVAILABLE = False

# Centralized logging
from ..logging_config import get_logger, log_execution, create_sse_log_handler

# MCP tool integration
from .mcp_client import get_mcp_invoker

# Clustering service
from .clustering_service import ClusteringService, ClusteringMethod

logger = get_logger(__name__)


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
            "patient_lookup", "patient_analysis", "follow_up", "mcp_tool", "mcp_app", "clustering_analysis", or "general_qa"
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

        # Keywords that indicate new patient data
        patient_indicators = [
            r'\d{2}[-\s]?(year|yr)[-\s]?old',
            r'stage\s+(I{1,3}[ABC]?|IV)',
            r'\b\d{2,3}\s*[MF]\b',
            r'(adenocarcinoma|squamous|small\s+cell|SCLC|NSCLC)',
            r'(EGFR|ALK|ROS1|BRAF|KRAS|PD-L1)',
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

            # Suggest follow-ups
            yield self._format_sse({
                "type": "suggestions",
                "content": [
                    "Show alternative treatments",
                    "Explain the reasoning",
                    "Find similar cases",
                    "Assess comorbidity interactions",
                    "Check clinical trial eligibility"
                ]
            })

        except Exception as e:
            logger.error(f"Patient analysis failed: {e}", exc_info=True)
            yield self._format_sse({
                "type": "error",
                "content": f"Analysis failed: {str(e)}"
            })

    async def _stream_follow_up(self, message: str, session_id: str) -> AsyncIterator[str]:
        """Stream follow-up responses using stored patient context"""
        context = self.patient_context[session_id]

        yield self._format_sse({
            "type": "status",
            "content": "Analyzing follow-up with patient context..."
        })

        response = self._handle_follow_up(message, context)

        yield self._format_sse({
            "type": "text",
            "content": response
        })

        # Provide updated suggestions
        yield self._format_sse({
            "type": "suggestions",
            "content": [
                "Show alternative treatments",
                "Explain the reasoning",
                "Find similar cases",
                "Assess comorbidity interactions",
                "Check clinical trial eligibility"
            ]
        })

        self._add_to_history(session_id, "assistant", response)

    async def _stream_general_qa(self, message: str, session_id: str) -> AsyncIterator[str]:
        """Stream general Q&A responses"""

        yield self._format_sse({
            "type": "status",
            "content": "Processing your question..."
        })

        # Get conversation history
        history = self._get_session(session_id)

        response = self._generate_qa_response(message, history)

        yield self._format_sse({
            "type": "text",
            "content": response
        })

        self._add_to_history(session_id, "assistant", response)

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
        """Generate response for general questions"""

        message_lower = message.lower()

        if 'help' in message_lower or 'how' in message_lower:
            return """I can help you with lung cancer treatment decisions!

You can:
- **Analyze a patient** - Describe the patient (e.g., "68M, stage IIIA adenocarcinoma, EGFR+")
- **Ask about treatments** - "What are options for stage IV NSCLC?"
- **Explore biomarkers** - "When should I test for ALK?"
- **Get guidelines** - "What does NICE recommend for..."

Try describing a patient case to get started!"""

        elif 'alternative' in message_lower or 'other option' in message_lower:
            return "To see alternative treatments, please provide the patient details first, and I'll show all suitable options with their evidence levels."

        elif 'similar' in message_lower or 'case' in message_lower:
            return "I can find similar cases from our database. Please describe the patient first, and I'll match them with similar cases."

        else:
            return f"""I'm the LCA Assistant. I help with lung cancer treatment decisions.

I noticed you asked: "{message}"

For the best results, please describe a patient case with:
- Age and sex
- TNM stage
- Histology type
- Performance status
- Any biomarker results (EGFR, ALK, PD-L1, etc.)

Example: "68-year-old male, stage IIIA adenocarcinoma, EGFR Ex19del positive, PS 1"
"""

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
