"""
Conversational Service for LCA Chatbot
Handles natural language interaction with the LCA system
"""

import json
import re
import logging
from typing import Dict, List, Optional, AsyncIterator
from datetime import datetime
import asyncio

from pydantic import BaseModel, Field, field_validator

# Centralized logging
from ..logging_config import get_logger, log_execution, create_sse_log_handler

logger = get_logger(__name__)


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
    - Session-based conversation history
    - Intent classification (patient analysis, follow-up, general Q&A)
    - Context-aware follow-up handling
    """

    def __init__(self, lca_service):
        self.lca_service = lca_service
        self.sessions: Dict[str, List[Dict]] = {}  # session_id -> message history
        self.patient_context: Dict[str, Dict] = {}  # session_id -> {patient_data, result}

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
        message: str
    ) -> AsyncIterator[str]:
        """
        Stream conversational responses

        Args:
            session_id: Unique session identifier
            message: User message

        Yields:
            SSE-formatted chunks
        """
        try:
            # Add user message to history
            self._add_to_history(session_id, "user", message)

            # Classify intent with session context
            intent = self._classify_intent(message, session_id)

            if intent == "patient_analysis":
                async for chunk in self._stream_patient_analysis(message, session_id):
                    yield chunk
            elif intent == "follow_up":
                async for chunk in self._stream_follow_up(message, session_id):
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
            "patient_analysis", "follow_up", or "general_qa"
        """
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

    async def _stream_patient_analysis(self, message: str, session_id: str = None) -> AsyncIterator[str]:
        """Stream patient analysis workflow with enhanced visibility"""

        # Step 1: Extract patient data with detailed progress
        yield self._format_sse({
            "type": "status",
            "content": "Extracting patient data from your message..."
        })

        patient_data = self._extract_patient_data(message)

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

        try:
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
                # Execute with streaming updates and AI workflow enabled
                processing_task = asyncio.create_task(
                    self.lca_service.process_patient(
                        patient_data=patient_data,
                        use_ai_workflow=True,
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
            logger.info(f"Result has {len(result.recommendations)} recommendations")
            if result.recommendations:
                logger.info(f"First recommendation: {result.recommendations[0]}")

            # Format recommendations
            recommendations_text = self._format_recommendations(result)

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
