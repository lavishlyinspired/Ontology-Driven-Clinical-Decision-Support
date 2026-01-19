"""
Enhanced Chat Routes with File Upload, LLM Extraction, and Export

Adds:
- File upload endpoint (PDF, DOCX, TXT)
- LLM-based extraction
- Agent transparency streaming
- PDF/FHIR export endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from typing import Optional, Dict, Any
from pydantic import BaseModel
import json
from datetime import datetime
import tempfile
from pathlib import Path

from ..services.conversation_service import ConversationService
from ..services.lca_service import LungCancerAssistantService
from ..services.file_processor import FileProcessor, ClinicalNoteParser
from ..services.llm_extractor import LLMExtractor, HybridExtractor
from ..services.transparency_service import TransparencyService
from ..services.export_service import PDFReportGenerator, FHIRExporter


router = APIRouter(prefix="/chat", tags=["chat"])

# Service initialization
lca_service = LungCancerAssistantService()
conversation_service = ConversationService(lca_service)
file_processor = FileProcessor()
clinical_parser = ClinicalNoteParser()
llm_extractor = LLMExtractor()
transparency_service = TransparencyService()
pdf_generator = PDFReportGenerator() if PDFReportGenerator else None
fhir_exporter = FHIRExporter()


class ChatMessage(BaseModel):
    """Chat message request."""
    message: str
    session_id: Optional[str] = None
    use_llm_extraction: bool = False


class FileUploadResponse(BaseModel):
    """File upload response."""
    success: bool
    file_info: Dict[str, Any]
    extracted_data: Dict[str, Any]
    validation: Dict[str, Any]
    lca_format: Dict[str, Any]


@router.post("/stream")
async def chat_stream(chat_request: ChatMessage):
    """
    Stream chat responses with SSE.
    
    Enhanced with optional LLM extraction.
    """
    async def event_generator():
        try:
            # Set extraction method
            conversation_service.use_llm = chat_request.use_llm_extraction
            
            async for event in conversation_service.chat_stream(
                chat_request.message,
                chat_request.session_id
            ):
                yield event
        except Exception as e:
            error_event = conversation_service._format_sse({
                "type": "error",
                "content": f"Stream error: {str(e)}"
            })
            yield error_event
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    use_llm_extraction: bool = Query(False, description="Use LLM for extraction")
):
    """
    Upload and process clinical document (PDF, DOCX, TXT).
    
    Extracts patient data automatically and validates.
    """
    try:
        # Read file
        file_content = await file.read()
        
        # Process with regex-based extraction
        result = file_processor.process_file(
            BytesIO(file_content),
            file.filename
        )
        
        # Optional LLM enhancement
        if use_llm_extraction:
            llm_result = await llm_extractor.extract_from_document(result['raw_text'])
            if llm_result.get('extracted_data'):
                # Merge LLM with regex results
                result['extracted_data'] = llm_extractor.merge_with_existing(
                    llm_result['extracted_data'],
                    result['extracted_data']
                )
                result['extraction_method'] = 'hybrid'
        
        # Validate extracted data
        validation = file_processor.validate_extracted_data(result)
        
        # Format for LCA
        lca_format = file_processor.format_for_lca(result['extracted_data'])
        
        return FileUploadResponse(
            success=True,
            file_info=result['file_info'],
            extracted_data=result['extracted_data'],
            validation=validation,
            lca_format=lca_format
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")


@router.post("/analyze-file")
async def analyze_uploaded_file(
    file: UploadFile = File(...),
    use_llm: bool = Query(False)
):
    """
    Upload file and immediately run LCA analysis.
    
    Combines upload + extraction + analysis in one step.
    """
    try:
        # Upload and extract
        file_content = await file.read()
        from io import BytesIO
        result = file_processor.process_file(BytesIO(file_content), file.filename)
        
        # Optional LLM extraction
        if use_llm:
            llm_result = await llm_extractor.extract_from_document(result['raw_text'])
            if llm_result.get('extracted_data'):
                result['extracted_data'] = llm_extractor.merge_with_existing(
                    llm_result['extracted_data'],
                    result['extracted_data']
                )
        
        # Format for LCA
        patient_data = file_processor.format_for_lca(result['extracted_data'])
        
        # Validate
        validation = file_processor.validate_extracted_data(result)
        if not validation['is_valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid patient data: {validation['errors']}"
            )
        
        # Run LCA analysis
        analysis_result = await lca_service.analyze_patient(patient_data)
        
        return {
            'success': True,
            'file_info': result['file_info'],
            'extracted_data': result['extracted_data'],
            'analysis': analysis_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    history = conversation_service.sessions.get(session_id, {}).get('history', [])
    
    return {
        "session_id": session_id,
        "message_count": len(history),
        "history": history
    }


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation session."""
    if session_id in conversation_service.sessions:
        del conversation_service.sessions[session_id]
    
    return {"success": True, "session_id": session_id}


@router.get("/sessions/{session_id}/transparency")
async def stream_agent_transparency(session_id: str):
    """
    Stream real-time agent execution updates.
    
    Shows live agent status, confidence scores, and execution graph.
    """
    workflow_graph = transparency_service.get_workflow_graph(session_id)
    
    if not workflow_graph:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    async def event_generator():
        async for update in transparency_service.stream_agent_updates(session_id):
            yield f"data: {json.dumps(update)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@router.post("/sessions/{session_id}/export/pdf")
async def export_conversation_pdf(session_id: str):
    """
    Export conversation as PDF.
    
    Downloads conversation transcript with patient data.
    """
    if not pdf_generator:
        raise HTTPException(
            status_code=501,
            detail="PDF export not available. Install reportlab: pip install reportlab"
        )
    
    session = conversation_service.sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    history = session.get('history', [])
    patient_data = session.get('patient_data')
    
    # Generate PDF
    pdf_buffer = pdf_generator.generate_conversation_pdf(
        history,
        patient_data
    )
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(pdf_buffer.getvalue())
        tmp_path = tmp.name
    
    return FileResponse(
        tmp_path,
        media_type='application/pdf',
        filename=f'conversation_{session_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    )


@router.post("/sessions/{session_id}/export/clinical-report")
async def export_clinical_report_pdf(session_id: str):
    """
    Export full clinical decision support report as PDF.
    
    Includes patient data, analysis results, and recommendations.
    """
    if not pdf_generator:
        raise HTTPException(
            status_code=501,
            detail="PDF export not available. Install reportlab: pip install reportlab"
        )
    
    session = conversation_service.sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    patient_data = session.get('patient_data', {})
    analysis_results = session.get('analysis_results', {})
    
    # Extract recommendations
    recommendations = []
    if 'recommendation' in analysis_results:
        rec = analysis_results['recommendation']
        recommendations.append({
            'treatment': rec.get('treatment', 'N/A'),
            'rationale': rec.get('rationale', 'N/A'),
            'evidence': rec.get('evidence_level', 'N/A'),
            'confidence': f"{rec.get('confidence', 0):.1%}"
        })
    
    # Generate PDF
    pdf_buffer = pdf_generator.generate_clinical_report(
        patient_data,
        analysis_results,
        recommendations
    )
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(pdf_buffer.getvalue())
        tmp_path = tmp.name
    
    return FileResponse(
        tmp_path,
        media_type='application/pdf',
        filename=f'clinical_report_{session_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    )


@router.post("/sessions/{session_id}/export/fhir")
async def export_fhir_bundle(session_id: str):
    """
    Export patient data and results as FHIR R4 Bundle.
    
    For EHR integration (Epic, Cerner, etc.)
    """
    session = conversation_service.sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    patient_data = session.get('patient_data', {})
    analysis_results = session.get('analysis_results', {})
    
    # Build FHIR Bundle
    fhir_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": datetime.now().isoformat(),
        "entry": []
    }
    
    # Add Patient resource
    fhir_patient = fhir_exporter.export_patient(patient_data)
    fhir_bundle['entry'].append({
        "fullUrl": f"Patient/{patient_data.get('patient_id', 'unknown')}",
        "resource": fhir_patient
    })
    
    # Add Condition resource
    fhir_condition = fhir_exporter.export_condition(patient_data)
    fhir_bundle['entry'].append({
        "fullUrl": f"Condition/{patient_data.get('patient_id', 'unknown')}-lung-cancer",
        "resource": fhir_condition
    })
    
    # Add Observation resources for biomarkers
    biomarkers = patient_data.get('biomarkers', {})
    for biomarker, value in biomarkers.items():
        fhir_obs = fhir_exporter.export_observation(
            patient_data.get('patient_id', 'unknown'),
            biomarker,
            value
        )
        fhir_bundle['entry'].append({
            "fullUrl": f"Observation/{patient_data.get('patient_id', 'unknown')}-{biomarker}",
            "resource": fhir_obs
        })
    
    # Add CarePlan for recommendations
    if analysis_results.get('recommendation'):
        recommendations = [analysis_results['recommendation']]
        fhir_careplan = fhir_exporter.export_care_plan(
            patient_data.get('patient_id', 'unknown'),
            recommendations
        )
        fhir_bundle['entry'].append({
            "fullUrl": f"CarePlan/{patient_data.get('patient_id', 'unknown')}-treatment-plan",
            "resource": fhir_careplan
        })
    
    return fhir_bundle
