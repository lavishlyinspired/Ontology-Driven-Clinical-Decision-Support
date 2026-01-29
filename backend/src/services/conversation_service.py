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

logger = logging.getLogger(__name__)


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

            # Create async queue for real-time progress streaming
            import asyncio
            progress_queue = asyncio.Queue()

            async def stream_progress(message: str):
                """Stream progress messages in real-time"""
                await progress_queue.put(message)

            # Start processing in background
            import concurrent.futures
            loop = asyncio.get_event_loop()

            # Execute with streaming updates and AI workflow enabled
            processing_task = asyncio.create_task(
                self.lca_service.process_patient(
                    patient_data=patient_data,
                    use_ai_workflow=True,
                    force_advanced=use_advanced,
                    progress_callback=stream_progress
                )
            )

            # Stream progress as it comes in (with timeout)
            start_time = datetime.now()
            streamed_progress = 0
            while not processing_task.done():
                try:
                    msg = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                    streamed_progress += 1
                    yield self._format_sse({
                        "type": "progress",
                        "content": f"[Agent {streamed_progress}] {msg}"
                    })
                except asyncio.TimeoutError:
                    # Check if still processing
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed > 2 and streamed_progress == 0:
                        yield self._format_sse({
                            "type": "reasoning",
                            "content": f"⏳ Processing ({int(elapsed)}s elapsed)... executing guideline matching"
                        })
                    continue

            # Get result
            result = await processing_task

            # Stream any remaining progress messages
            while not progress_queue.empty():
                msg = await progress_queue.get()
                streamed_progress += 1
                yield self._format_sse({
                    "type": "progress",
                    "content": f"[Agent {streamed_progress}] {msg}"
                })

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
        """Format recommendations as markdown"""
        if not result.recommendations:
            return "**No specific treatment recommendations generated.**\n\nThis may be because additional patient information is needed or the case requires manual MDT review."

        recommendations = result.recommendations

        if recommendations:
            primary = recommendations[0]
            
            if isinstance(primary, dict):
                treatment = primary.get('treatment', 'Unknown')
                evidence = primary.get('evidence_level', 'Not specified')
                confidence = primary.get('confidence_score', 0)
                if isinstance(confidence, (int, float)):
                    confidence = int(confidence * 100) if confidence <= 1 else int(confidence)
                source = primary.get('guideline_reference', primary.get('rule_source', 'Clinical Guidelines'))
                intent = primary.get('intent', 'Not specified')
                rationale = primary.get('rationale', '')
                contraindications = primary.get('contraindications', [])
                survival = primary.get('survival_benefit', '')
                expected_benefit = primary.get('expected_benefit', '')
            else:
                # Handle AgentProposal or TreatmentRecommendation objects
                treatment = getattr(primary, 'treatment_type', getattr(primary, 'treatment', 'Unknown'))
                evidence = getattr(primary, 'evidence_level', 'Not specified')
                confidence = getattr(primary, 'confidence', getattr(primary, 'confidence_score', 0))
                if isinstance(confidence, (int, float)):
                    confidence = int(confidence * 100) if confidence <= 1 else int(confidence)
                source = getattr(primary, 'guideline_reference', getattr(primary, 'rule_source', 'Clinical Guidelines'))
                intent = getattr(primary, 'treatment_intent', getattr(primary, 'intent', 'Not specified'))
                rationale = getattr(primary, 'rationale', '')
                contraindications = getattr(primary, 'contraindications', [])
                survival = getattr(primary, 'survival_benefit', '')
                expected_benefit = getattr(primary, 'expected_benefit', '')

            text = f"""## Primary Recommendation: {treatment}

- **Evidence Level:** {evidence}
- **Confidence:** {confidence}%
- **Source:** {source}
- **Intent:** {intent}
"""

            if rationale:
                text += f"\n**Rationale:** {rationale}\n"

            if survival:
                text += f"- **Expected Benefit:** {survival}\n"

            if contraindications:
                text += f"\n**Contraindications:**\n"
                for contra in contraindications:
                    text += f"- {contra}\n"

            if len(recommendations) > 1:
                text += f"\n### Alternative Options:\n"
                for i, alt_rec in enumerate(recommendations[1:3], 2):
                    if isinstance(alt_rec, dict):
                        alt_treatment = alt_rec.get('treatment', 'Unknown')
                        alt_evidence = alt_rec.get('evidence_level', 'Not specified')
                    else:
                        alt_treatment = getattr(alt_rec, 'treatment_type', getattr(alt_rec, 'treatment', 'Unknown'))
                        alt_evidence = getattr(alt_rec, 'evidence_level', 'Not specified')
                    text += f"{i}. {alt_treatment} (Evidence: {alt_evidence})\n"

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
                text = "### Alternative Treatment Options:\n\n"
                for i, rec in enumerate(result.recommendations[1:4], 2):
                    treatment = getattr(rec, 'treatment_type', getattr(rec, 'treatment', 'Unknown'))
                    evidence = getattr(rec, 'evidence_level', 'Not specified')
                    confidence = getattr(rec, 'confidence_score', 0)
                    if isinstance(confidence, float) and confidence <= 1:
                        confidence = int(confidence * 100)
                    intent = getattr(rec, 'treatment_intent', 'Not specified')
                    survival = getattr(rec, 'survival_benefit', '')
                    text += f"**{i}. {treatment}**\n"
                    text += f"- Evidence: {evidence}\n"
                    text += f"- Confidence: {confidence}%\n"
                    text += f"- Intent: {intent}\n"
                    if survival:
                        text += f"- Expected Benefit: {survival}\n"
                    text += "\n"
                return text
            return "No alternative treatments are available based on current guidelines for this patient profile."

        # --- Reasoning / explanation ---
        elif any(kw in message_lower for kw in ["reasoning", "explain", "why"]):
            summary = result.mdt_summary
            if summary:
                return f"### Clinical Reasoning\n\n{summary}"
            return "No detailed reasoning is available for this case. The recommendation was based on guideline matching."

        # --- Similar cases ---
        elif any(kw in message_lower for kw in ["similar", "case"]):
            if result.similar_patients:
                text = f"### Found {len(result.similar_patients)} Similar Cases:\n\n"
                for i, patient in enumerate(result.similar_patients[:3], 1):
                    text += f"{i}. {patient.get('summary', patient.get('patient_id', 'Similar patient'))}\n"
                return text
            return "No similar cases found. The Neo4j patient database may not contain matching records yet."

        # --- Comorbidity / interactions ---
        elif any(kw in message_lower for kw in ["comorbidity", "interaction", "risk", "side effect", "toxicity"]):
            comorbidities = patient_data.get("comorbidities", [])
            if comorbidities:
                text = "### Comorbidity Assessment\n\n"
                text += f"**Active comorbidities:** {', '.join(comorbidities)}\n\n"
                text += "These comorbidities may affect treatment selection and dosing. "
                text += "The ComorbidityAgent evaluates drug interactions, contraindications, "
                text += "and dose adjustments for each active condition.\n\n"
                text += "Consult the full MDT summary for detailed interaction analysis."
                return text
            return "No comorbidities were recorded for this patient. If comorbidities exist, please include them in the patient description for safety assessment."

        # --- Prognosis / survival ---
        elif any(kw in message_lower for kw in ["prognosis", "survival", "outlook"]):
            stage = patient_data.get("tnm_stage", "Unknown")
            ps = patient_data.get("performance_status", "Unknown")
            histology = patient_data.get("histology_type", "Unknown")
            text = "### Prognostic Factors\n\n"
            text += f"- **Stage:** {stage}\n"
            text += f"- **Performance Status:** WHO {ps}\n"
            text += f"- **Histology:** {histology}\n\n"

            primary = result.recommendations[0] if result.recommendations else None
            if primary:
                survival = getattr(primary, 'survival_benefit', '')
                if survival:
                    text += f"**Expected benefit with recommended treatment:** {survival}\n\n"

            text += "For detailed survival analysis with Kaplan-Meier curves, use the Analytics module."
            return text

        # --- Biomarkers ---
        elif any(kw in message_lower for kw in ["biomarker", "mutation", "egfr", "alk", "pd-l1"]):
            biomarkers = patient_data.get("biomarker_profile", {})
            if biomarkers:
                text = "### Biomarker Profile\n\n"
                for key, value in biomarkers.items():
                    display_key = key.replace("_", " ").title()
                    text += f"- **{display_key}:** {value}\n"
                text += "\nBiomarker status drives targeted therapy selection through the BiomarkerAgent. "
                text += "10 actionable molecular pathways are evaluated."
                return text
            return "No biomarker data was provided. Consider testing for EGFR, ALK, ROS1, PD-L1, and KRAS for targeted therapy eligibility."

        # --- Clinical trials ---
        elif any(kw in message_lower for kw in ["trial", "clinical trial", "eligibility"]):
            stage = patient_data.get("tnm_stage", "Unknown")
            histology = patient_data.get("histology_type", "Unknown")
            text = "### Clinical Trial Eligibility\n\n"
            text += f"Based on **{stage} {histology}**:\n\n"
            text += "The ClinicalTrialMatcher queries ClinicalTrials.gov for eligible studies. "
            text += "Matching criteria include histology (30%), stage (25%), biomarkers (25%), "
            text += "performance status (10%), and age (10%).\n\n"
            text += "For complex/critical cases, trial matching is run automatically as part of the analytics suite."
            return text

        # --- Generic follow-up with context ---
        else:
            primary = result.recommendations[0] if result.recommendations else None
            if primary:
                treatment = getattr(primary, 'treatment_type', getattr(primary, 'treatment', 'Unknown'))
                return f"""Based on the current analysis for this patient:

**Primary Recommendation:** {treatment}
**Stage:** {patient_data.get('tnm_stage', 'Unknown')}
**Histology:** {patient_data.get('histology_type', 'Unknown')}

I can help with:
- **Alternative treatments** - Show other treatment options
- **Reasoning** - Explain why this was recommended
- **Similar cases** - Find patients with similar profiles
- **Comorbidity risks** - Assess treatment interactions
- **Biomarker details** - Review mutation/marker status
- **Prognosis** - Survival and outlook information
- **Clinical trials** - Check eligibility

What would you like to know more about?"""
            return "I have the patient context loaded. You can ask about alternative treatments, reasoning, similar cases, biomarkers, comorbidity interactions, or clinical trials."

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
