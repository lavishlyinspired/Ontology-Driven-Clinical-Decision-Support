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
                async for chunk in self._stream_patient_analysis(message):
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
    
    async def _stream_patient_analysis(self, message: str) -> AsyncIterator[str]:
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
            
            # Execute with streaming updates
            result = await self.lca_service.process_patient(
                patient_data=patient_data,
                use_ai_workflow=True,
                force_advanced=use_advanced
            )
            
            # Step 4: Stream results
            yield self._format_sse({
                "type": "status",
                "content": f"✅ Analysis complete ({result.execution_time_ms}ms)"
            })
            
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
        
        # Extract age
        age_match = re.search(r'(\d{2})[-\s]?(year|yr)[-\s]?old', message, re.IGNORECASE)
        if age_match:
            data["age"] = int(age_match.group(1))
        
        # Extract sex
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
            return "No recommendations available."
        
        primary = result.recommendations[0]
        
        text = f"""
## Primary Recommendation: {primary.treatment_type}

- **Evidence Level:** {primary.evidence_level}
- **Confidence:** {int(primary.confidence_score * 100)}%
- **Source:** {primary.rule_source}
- **Intent:** {primary.treatment_intent}
"""
        
        if primary.survival_benefit:
            text += f"- **Expected Benefit:** {primary.survival_benefit}\n"
        
        if primary.contraindications:
            text += f"\n**Contraindications:** {', '.join(primary.contraindications)}\n"
        
        # Add alternatives
        if len(result.recommendations) > 1:
            text += "\n### Alternative Options:\n"
            for rec in result.recommendations[1:3]:  # Top 2 alternatives
                text += f"- {rec.treatment_type} (Evidence: {rec.evidence_level}, Confidence: {int(rec.confidence_score * 100)}%)\n"
        
        return text
    
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
