"""
LLM-Based Patient Data Extraction Service

Replaces regex patterns with Ollama LLM for intelligent extraction
from clinical text, conversations, and medical documents.
"""

import json
from typing import Dict, Any, Optional, List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Centralized logging
from ..logging_config import get_logger, log_execution

logger = get_logger(__name__)


class PatientData(BaseModel):
    """Structured patient data schema."""
    patient_id: Optional[str] = Field(None, description="Medical record number")
    demographics: Dict[str, Any] = Field(default_factory=dict, description="Age, sex, etc.")
    diagnosis: Dict[str, Any] = Field(default_factory=dict, description="Cancer type, stage, histology")
    biomarkers: Dict[str, Any] = Field(default_factory=dict, description="EGFR, ALK, PD-L1, etc.")
    comorbidities: List[str] = Field(default_factory=list, description="Existing conditions")
    performance_status: Optional[str] = Field(None, description="ECOG PS")


class LLMExtractor:
    """Extract patient data using Ollama LLM."""
    
    EXTRACTION_PROMPT = """You are a clinical data extraction specialist. Extract structured patient information from the provided text.

Extract the following information:
1. Demographics: age, sex/gender
2. Diagnosis: cancer type (NSCLC/SCLC), stage (I-IV, Limited/Extensive), histology (adenocarcinoma, squamous, etc.)
3. Biomarkers: EGFR (Ex19del, L858R, etc.), ALK, ROS1, PD-L1 (%), KRAS, BRAF
4. Comorbidities: COPD, diabetes, hypertension, heart disease, kidney disease
5. Performance status: ECOG 0-4

Text to analyze:
{text}

Return ONLY valid JSON matching this schema:
{{
  "patient_id": "MRN if available, null otherwise",
  "demographics": {{"age": number, "sex": "M/F"}},
  "diagnosis": {{"cancer_type": "NSCLC/SCLC", "stage": "I-IV or Limited/Extensive", "histology": "type"}},
  "biomarkers": {{"egfr": "status", "alk": "status", "pdl1": "percentage"}},
  "comorbidities": ["condition1", "condition2"],
  "performance_status": "0-4"
}}

If information is not mentioned, use null or empty values. Be precise and conservative."""

    CONVERSATION_PROMPT = """You are analyzing a conversational message about a lung cancer patient. Extract any mentioned clinical information.

User message:
{message}

Previous context (if any):
{context}

Extract patient data if mentioned. Return JSON matching this schema:
{{
  "demographics": {{"age": number or null, "sex": "M/F" or null}},
  "diagnosis": {{"cancer_type": "NSCLC/SCLC" or null, "stage": "stage" or null, "histology": "type" or null}},
  "biomarkers": {{"egfr": "status" or null, "alk": "status" or null, "pdl1": "%" or null}},
  "comorbidities": ["conditions"],
  "performance_status": "0-4" or null
}}

If no new clinical information is mentioned, return empty/null values."""

    QA_PROMPT = """You are a lung cancer clinical decision support assistant. Answer the user's question based on:
1. The patient data provided (if any)
2. NCCN guidelines for lung cancer
3. Current best practices

Patient Data:
{patient_data}

Conversation History:
{history}

User Question:
{question}

Provide a clear, evidence-based answer. If you mention treatment options, cite relevant NCCN guidelines or clinical evidence.
Keep responses concise (3-5 sentences) unless more detail is requested."""

    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """
        Initialize LLM extractor.
        
        Args:
            model_name: Ollama model to use (llama3.2, mistral, etc.)
            base_url: Ollama server URL
        """
        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.0,  # Deterministic for extraction
            format="json"  # Force JSON output
        )
        
        self.qa_llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.3  # Slightly creative for Q&A
        )
        
        self.parser = JsonOutputParser(pydantic_object=PatientData)
    
    async def extract_from_document(self, text: str) -> Dict[str, Any]:
        """
        Extract patient data from clinical document using LLM.
        
        Args:
            text: Clinical document text
            
        Returns:
            Extracted patient data dictionary
        """
        prompt = ChatPromptTemplate.from_template(self.EXTRACTION_PROMPT)
        
        chain = prompt | self.llm | self.parser
        
        try:
            result = await chain.ainvoke({"text": text})
            return {
                "extracted_data": result,
                "extraction_method": "llm",
                "confidence": "high"  # Could add confidence scoring
            }
        except Exception as e:
            return {
                "extracted_data": None,
                "extraction_method": "llm",
                "error": str(e),
                "confidence": "failed"
            }
    
    async def extract_from_conversation(
        self, 
        message: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract patient data from conversational message.
        
        Args:
            message: User's conversational message
            context: Previous conversation context
            
        Returns:
            Extracted patient data
        """
        context_str = json.dumps(context or {}, indent=2)
        
        prompt = ChatPromptTemplate.from_template(self.CONVERSATION_PROMPT)
        chain = prompt | self.llm | self.parser
        
        try:
            result = await chain.ainvoke({
                "message": message,
                "context": context_str
            })
            return result
        except Exception as e:
            return {
                "demographics": {},
                "diagnosis": {},
                "biomarkers": {},
                "comorbidities": [],
                "error": str(e)
            }
    
    async def answer_question(
        self,
        question: str,
        patient_data: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Answer clinical question using LLM.
        
        Args:
            question: User's question
            patient_data: Current patient data
            history: Conversation history
            
        Returns:
            LLM-generated answer
        """
        patient_str = json.dumps(patient_data or {}, indent=2)
        history_str = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in (history or [])
        ])
        
        prompt = ChatPromptTemplate.from_template(self.QA_PROMPT)
        chain = prompt | self.qa_llm
        
        try:
            response = await chain.ainvoke({
                "patient_data": patient_str,
                "history": history_str,
                "question": question
            })
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def validate_extraction(self, extracted_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate LLM-extracted data.
        
        Returns:
            Validation results with errors/warnings
        """
        errors = []
        warnings = []
        
        demographics = extracted_data.get('demographics', {})
        diagnosis = extracted_data.get('diagnosis', {})
        
        # Check required fields
        if not demographics.get('age'):
            warnings.append("Missing age")
        
        if not demographics.get('sex'):
            warnings.append("Missing sex/gender")
        
        if not diagnosis.get('cancer_type'):
            errors.append("Missing cancer type (NSCLC/SCLC)")
        
        if not diagnosis.get('stage'):
            warnings.append("Missing disease stage")
        
        # Validate values
        cancer_type = diagnosis.get('cancer_type', '')
        if cancer_type and cancer_type not in ['NSCLC', 'SCLC']:
            errors.append(f"Invalid cancer type: {cancer_type} (must be NSCLC or SCLC)")
        
        age = demographics.get('age')
        if age and (age < 0 or age > 120):
            errors.append(f"Invalid age: {age}")
        
        sex = demographics.get('sex', '')
        if sex and sex not in ['M', 'F']:
            errors.append(f"Invalid sex: {sex} (must be M or F)")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'is_valid': len(errors) == 0
        }
    
    def merge_with_existing(
        self, 
        extracted: Dict[str, Any], 
        existing: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge newly extracted data with existing patient data.
        
        New data takes precedence but existing data is preserved if new data is null.
        """
        merged = existing.copy()
        
        # Merge demographics
        if 'demographics' in extracted:
            merged.setdefault('demographics', {})
            for key, value in extracted['demographics'].items():
                if value is not None:
                    merged['demographics'][key] = value
        
        # Merge diagnosis
        if 'diagnosis' in extracted:
            merged.setdefault('diagnosis', {})
            for key, value in extracted['diagnosis'].items():
                if value is not None:
                    merged['diagnosis'][key] = value
        
        # Merge biomarkers
        if 'biomarkers' in extracted:
            merged.setdefault('biomarkers', {})
            for key, value in extracted['biomarkers'].items():
                if value is not None:
                    merged['biomarkers'][key] = value
        
        # Merge comorbidities (combine lists)
        if 'comorbidities' in extracted:
            existing_comorbidities = set(merged.get('comorbidities', []))
            new_comorbidities = set(extracted['comorbidities'])
            merged['comorbidities'] = list(existing_comorbidities | new_comorbidities)
        
        # Performance status (new overwrites)
        if extracted.get('performance_status'):
            merged['performance_status'] = extracted['performance_status']
        
        return merged


class HybridExtractor:
    """Hybrid extraction using both regex (fast) and LLM (accurate)."""
    
    def __init__(self, llm_extractor: LLMExtractor, file_processor):
        """
        Initialize hybrid extractor.
        
        Args:
            llm_extractor: LLM extraction service
            file_processor: Regex-based file processor
        """
        self.llm_extractor = llm_extractor
        self.file_processor = file_processor
    
    async def extract_with_fallback(self, text: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        Extract data with regex first, then enhance with LLM.
        
        Args:
            text: Text to extract from
            use_llm: Whether to use LLM enhancement
            
        Returns:
            Extracted data (regex + LLM)
        """
        # Fast regex extraction
        regex_result = self.file_processor._extract_patient_data(text)
        
        if not use_llm:
            return {
                'extracted_data': regex_result,
                'method': 'regex',
                'confidence': 'medium'
            }
        
        # LLM enhancement
        llm_result = await self.llm_extractor.extract_from_document(text)
        
        # Merge results (LLM takes precedence for conflicts)
        if llm_result.get('extracted_data'):
            merged = self.llm_extractor.merge_with_existing(
                llm_result['extracted_data'],
                regex_result
            )
            return {
                'extracted_data': merged,
                'method': 'hybrid',
                'regex_result': regex_result,
                'llm_result': llm_result.get('extracted_data'),
                'confidence': 'high'
            }
        
        # Fallback to regex if LLM fails
        return {
            'extracted_data': regex_result,
            'method': 'regex_fallback',
            'llm_error': llm_result.get('error'),
            'confidence': 'medium'
        }
