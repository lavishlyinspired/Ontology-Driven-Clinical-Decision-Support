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

logger = logging.getLogger(__name__)


class ConversationService:
    """
    Manages conversational interactions with LCA
    
    Features:
    - Natural language patient data extraction
    - Streaming responses via SSE
    - Session-based conversation history
    - Intent classification (patient analysis vs Q&A)
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
            
            # Classify intent
            intent = self._classify_intent(message)
            
            if intent == "patient_analysis":
                async for chunk in self._stream_patient_analysis(message, session_id):
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
    
    def _classify_intent(self, message: str) -> str:
        """
        Classify user intent
        
        Returns:
            "patient_analysis" or "general_qa"
        """
        # Keywords that indicate patient data
        patient_indicators = [
            r'\d{2}[-\s]?(year|yr)[-\s]?old',
            r'stage\s+(I{1,3}[ABC]?|IV)',
            r'(male|female|M|F)',
            r'(adenocarcinoma|squamous|small\s+cell|SCLC|NSCLC)',
            r'(EGFR|ALK|ROS1|BRAF|KRAS|PD-L1)',
            r'(PS|performance\s+status|ECOG)\s*[0-4]',
            r'T\d+N\d+M\d+',
            r'comorbid',
        ]
        
        for pattern in patient_indicators:
            if re.search(pattern, message, re.IGNORECASE):
                return "patient_analysis"
        
        return "general_qa"
    
    async def _stream_patient_analysis(self, message: str, session_id: str = None) -> AsyncIterator[str]:
        """Stream patient analysis workflow"""
        
        # Step 1: Extract patient data
        yield self._format_sse({
            "type": "status",
            "content": "🔍 Extracting patient data from your message..."
        })
        
        patient_data = self._extract_patient_data(message)
        
        if not patient_data:
            yield self._format_sse({
                "type": "error",
                "content": "Could not extract patient data. Please provide: age, sex, stage, histology type."
            })
            return
        
        yield self._format_sse({
            "type": "patient_data",
            "content": patient_data
        })
        
        # Step 2: Assess complexity
        yield self._format_sse({
            "type": "status",
            "content": "📊 Assessing case complexity..."
        })
        
        try:
            complexity = await self.lca_service.assess_complexity(patient_data)
            
            yield self._format_sse({
                "type": "complexity",
                "content": {
                    "level": complexity["complexity"],
                    "workflow": complexity["recommended_workflow"]
                }
            })
            
            # Step 3: Run workflow
            use_advanced = complexity["recommended_workflow"] == "integrated"
            
            yield self._format_sse({
                "type": "status",
                "content": f"⚙️ Running {complexity['recommended_workflow']} workflow..."
            })
            
            # Create a queue for progress messages
            progress_messages = []
            
            async def capture_progress(message: str):
                """Capture progress messages to yield later"""
                progress_messages.append(message)
            
            # Execute with streaming updates and AI workflow enabled
            result = await self.lca_service.process_patient(
                patient_data=patient_data,
                use_ai_workflow=True,  # Enabled for comprehensive clinical argumentation
                force_advanced=use_advanced,
                progress_callback=capture_progress
            )
            
            # Stream collected progress messages
            for msg in progress_messages:
                yield self._format_sse({
                    "type": "progress",
                    "content": msg
                })
            
            # Show matched rules summary
            if result.recommendations:
                rules_summary = ", ".join([f"{r.rule_id} ({r.treatment_type})" for r in result.recommendations[:3]])
                yield self._format_sse({
                    "type": "progress",
                    "content": f"✅ Matched guidelines: {rules_summary}"
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
                "content": f"✅ Analysis complete ({result.execution_time_ms}ms)"
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
            yield self._format_sse({
                "type": "text",
                "content": f"\n\n**Clinical Summary:**\n{result.mdt_summary}"
            })
            
            # Suggest follow-ups
            yield self._format_sse({
                "type": "suggestions",
                "content": [
                    "Show alternative treatments",
                    "Assess comorbidity interactions",
                    "Find similar cases",
                    "Explain the reasoning"
                ]
            })
            
        except Exception as e:
            logger.error(f"Patient analysis failed: {e}", exc_info=True)
            yield self._format_sse({
                "type": "error",
                "content": f"Analysis failed: {str(e)}"
            })
    
    async def _stream_general_qa(self, message: str, session_id: str) -> AsyncIterator[str]:
        """Stream general Q&A responses"""
        
        yield self._format_sse({
            "type": "status",
            "content": "💭 Thinking..."
        })
        
        # Check if this is a follow-up question about a patient
        follow_up_keywords = [
            r'alternative.*treatment',
            r'other.*option',
            r'different.*therapy',
            r'explain.*reasoning',
            r'similar.*case',
            r'comorbidity.*interaction',
            r'side.*effect'
        ]
        
        is_follow_up = any(re.search(pattern, message, re.IGNORECASE) for pattern in follow_up_keywords)
        
        if is_follow_up and session_id in self.patient_context:
            # Handle follow-up questions using stored context
            context = self.patient_context[session_id]
            response = self._handle_follow_up(message, context)
            
            yield self._format_sse({
                "type": "text",
                "content": response
            })
        else:
            # Get conversation history
            history = self._get_session(session_id)
            
            # Simple response for now (can be enhanced with LLM)
            response = self._generate_qa_response(message, history)
            
            yield self._format_sse({
                "type": "text",
                "content": response
            })
        
        self._add_to_history(session_id, "assistant", response)
    
    def _extract_patient_data(self, message: str) -> Optional[Dict]:
        """
        Extract structured patient data from natural language
        
        Uses regex patterns to extract clinical information
        """
        data = {
            "patient_id": f"CHAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Extract age and sex (handle formats like "68M", "68F", "68 year old male", etc.)
        # Try compact format first (e.g., "68M", "72F")
        age_sex_match = re.search(r'\b(\d{2,3})\s*([MF])\b', message, re.IGNORECASE)
        if age_sex_match:
            data["age"] = int(age_sex_match.group(1))
            data["sex"] = age_sex_match.group(2).upper()
        else:
            # Extract age from verbose format
            age_match = re.search(r'(\d{2,3})[-\s]?(year|yr)[-\s]?old', message, re.IGNORECASE)
            if age_match:
                data["age"] = int(age_match.group(1))
            
            # Extract sex separately
            sex_patterns = [
                (r'\b(male|M)\b(?![\w-])', 'M'),
                (r'\b(female|F)\b(?![\w-])', 'F'),
            ]
            for pattern, sex_value in sex_patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    data["sex"] = sex_value
                    break
        
        # Extract TNM stage
        stage_match = re.search(r'stage\s+(I{1,3}[ABC]?|IV)', message, re.IGNORECASE)
        if stage_match:
            data["tnm_stage"] = stage_match.group(1).upper()
        
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
                    data["histology_type"] = "Adenocarcinoma"  # Default for NSCLC
                else:
                    data["histology_type"] = histology.title()
                break
        
        # Extract performance status
        ps_match = re.search(r'(?:PS|ECOG|performance\s+status)\s*[:\-]?\s*([0-4])', message, re.IGNORECASE)
        if ps_match:
            data["performance_status"] = int(ps_match.group(1))
        else:
            data["performance_status"] = 1  # Default
        
        # Extract biomarkers
        biomarker_profile = {}
        
        # EGFR
        egfr_match = re.search(r'EGFR[:\s]*(Ex19del|Ex20ins|L858R|T790M|\+|positive|negative)', message, re.IGNORECASE)
        if egfr_match:
            mutation = egfr_match.group(1)
            if mutation in ['+', 'positive']:
                biomarker_profile["egfr_mutation"] = True
            elif mutation == 'negative':
                biomarker_profile["egfr_mutation"] = False
            else:
                biomarker_profile["egfr_mutation"] = True
                biomarker_profile["egfr_mutation_type"] = mutation
        
        # ALK
        alk_match = re.search(r'ALK[:\s]*(\+|positive|negative|rearrangement)', message, re.IGNORECASE)
        if alk_match:
            biomarker_profile["alk_rearrangement"] = alk_match.group(1).lower() in ['+', 'positive', 'rearrangement']
        
        # PD-L1
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
        
        # Validate minimum required fields
        required = ["age", "sex", "tnm_stage", "histology_type"]
        if all(field in data for field in required):
            return data
        
        return None
    
    def _format_recommendations(self, result) -> str:
        """Format recommendations as markdown"""
        if not result.recommendations:
            return "ℹ️ **No specific treatment recommendations generated.**\n\nThis may be because additional patient information is needed or the case requires manual MDT review."
        
        # Handle both TreatmentRecommendation objects and dicts
        recommendations = result.recommendations
        
        # Convert first recommendation to dict if needed
        if recommendations:
            primary = recommendations[0]
            if isinstance(primary, dict):
                # Already a dict
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
            else:
                # TreatmentRecommendation object
                treatment = getattr(primary, 'treatment_type', getattr(primary, 'treatment', 'Unknown'))
                evidence = getattr(primary, 'evidence_level', 'Not specified')
                confidence = getattr(primary, 'confidence_score', 0)
                if isinstance(confidence, (int, float)):
                    confidence = int(confidence * 100) if confidence <= 1 else int(confidence)
                source = getattr(primary, 'rule_source', getattr(primary, 'guideline_reference', 'Clinical Guidelines'))
                intent = getattr(primary, 'treatment_intent', getattr(primary, 'intent', 'Not specified'))
                rationale = getattr(primary, 'rationale', '')
                contraindications = getattr(primary, 'contraindications', [])
                survival = getattr(primary, 'survival_benefit', '')
            
            text = f"""
## 🎯 Primary Recommendation: {treatment}

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
                text += f"\n**⚠️ Contraindications:**\n"
                for contra in contraindications:
                    text += f"- {contra}\n"
            
            # Add alternative options if available
            if len(recommendations) > 1:
                text += f"\n### Alternative Options:\n"
                for i, alt_rec in enumerate(recommendations[1:3], 2):  # Show up to 2 alternatives
                    if isinstance(alt_rec, dict):
                        alt_treatment = alt_rec.get('treatment', 'Unknown')
                        alt_evidence = alt_rec.get('evidence_level', 'Not specified')
                    else:
                        alt_treatment = getattr(alt_rec, 'treatment_type', getattr(alt_rec, 'treatment', 'Unknown'))
                        alt_evidence = getattr(alt_rec, 'evidence_level', 'Not specified')
                    text += f"{i}. {alt_treatment} (Evidence: {alt_evidence})\n"
            
            return text
        
        return "ℹ️ **No specific treatment recommendations generated.**"
    
    def _handle_follow_up(self, message: str, context: Dict) -> str:
        """Handle follow-up questions about a patient case"""
        result = context["result"]
        patient_data = context["patient_data"]
        
        message_lower = message.lower()
        
        if "alternative" in message_lower or "other" in message_lower:
            # Show alternative treatments
            if len(result.recommendations) > 1:
                text = "### Alternative Treatment Options:\n\n"
                for i, rec in enumerate(result.recommendations[1:4], 2):
                    text += f"**{i}. {rec.treatment_type}**\n"
                    text += f"- Evidence: {rec.evidence_level}\n"
                    text += f"- Confidence: {int(rec.confidence_score * 100)}%\n"
                    text += f"- Intent: {rec.treatment_intent}\n"
                    if rec.survival_benefit:
                        text += f"- Expected Benefit: {rec.survival_benefit}\n"
                    text += "\n"
                return text
            else:
                return "No alternative treatments are available based on current guidelines."
        
        elif "reasoning" in message_lower or "explain" in message_lower:
            return result.mdt_summary
        
        elif "similar" in message_lower:
            if result.similar_patients:
                text = f"### Found {len(result.similar_patients)} Similar Cases:\n\n"
                for i, patient in enumerate(result.similar_patients[:3], 1):
                    text += f"{i}. {patient.get('summary', 'Similar patient case')}\n"
                return text
            else:
                return "No similar cases found in the database."
        
        else:
            return "I can help with: alternative treatments, reasoning explanation, or similar cases. What would you like to know?"
    
    def _generate_qa_response(self, message: str, history: List[Dict]) -> str:
        """Generate response for general questions"""
        
        # Simple pattern matching (can be enhanced with LLM)
        message_lower = message.lower()
        
        if 'help' in message_lower or 'how' in message_lower:
            return """I can help you with lung cancer treatment decisions! 

You can:
- **Analyze a patient** - Just describe the patient (e.g., "68M, stage IIIA adenocarcinoma, EGFR+")
- **Ask about treatments** - "What are options for stage IV NSCLC?"
- **Explore biomarkers** - "When should I test for ALK?"
- **Get guidelines** - "What does NCCN recommend for..."

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
