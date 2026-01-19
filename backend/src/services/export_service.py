"""
Export Service for Clinical Reports

Generates PDF reports and EHR-compatible formats (FHIR, HL7)
for clinical documentation and system integration.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from io import BytesIO
import json

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class PDFReportGenerator:
    """Generate PDF clinical decision support reports."""
    
    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab not installed. Run: pip install reportlab")
        
        self.styles = getSampleStyleSheet()
        
        # Custom styles
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#3b82f6'),
            spaceBefore=20,
            spaceAfter=10
        ))
        
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['BodyText'],
            fontSize=11,
            leading=14,
            alignment=TA_JUSTIFY
        ))
    
    def generate_clinical_report(
        self,
        patient_data: Dict[str, Any],
        analysis_results: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> BytesIO:
        """
        Generate comprehensive clinical decision support report.
        
        Args:
            patient_data: Patient demographics and clinical data
            analysis_results: Results from all agents
            recommendations: Treatment recommendations
            output_path: Optional file path to save PDF
            
        Returns:
            BytesIO buffer with PDF content
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer if not output_path else output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        # Title
        story.append(Paragraph(
            "Lung Cancer Clinical Decision Support Report",
            self.styles['CustomTitle']
        ))
        story.append(Spacer(1, 0.2 * inch))
        
        # Report metadata
        metadata_text = f"""
        <b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Patient ID:</b> {patient_data.get('patient_id', 'N/A')}<br/>
        <b>Analysis ID:</b> {analysis_results.get('analysis_id', 'N/A')}
        """
        story.append(Paragraph(metadata_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.3 * inch))
        
        # Patient Demographics
        story.append(Paragraph("Patient Demographics", self.styles['SectionHeading']))
        demographics = patient_data.get('demographics', {})
        demo_data = [
            ['Age', str(demographics.get('age', 'N/A'))],
            ['Sex', demographics.get('sex', 'N/A')],
            ['Performance Status', patient_data.get('performance_status', 'N/A')],
        ]
        demo_table = Table(demo_data, colWidths=[2*inch, 4*inch])
        demo_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e0e7ff')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f9fafb')])
        ]))
        story.append(demo_table)
        story.append(Spacer(1, 0.3 * inch))
        
        # Clinical Diagnosis
        story.append(Paragraph("Clinical Diagnosis", self.styles['SectionHeading']))
        diagnosis = patient_data.get('diagnosis', {})
        diag_data = [
            ['Cancer Type', diagnosis.get('cancer_type', 'N/A')],
            ['Stage', diagnosis.get('stage', 'N/A')],
            ['Histology', diagnosis.get('histology', 'N/A')],
        ]
        diag_table = Table(diag_data, colWidths=[2*inch, 4*inch])
        diag_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e0e7ff')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f9fafb')])
        ]))
        story.append(diag_table)
        story.append(Spacer(1, 0.3 * inch))
        
        # Biomarkers
        biomarkers = patient_data.get('biomarkers', {})
        if biomarkers:
            story.append(Paragraph("Biomarker Profile", self.styles['SectionHeading']))
            bio_data = [[biomarker.upper(), status] for biomarker, status in biomarkers.items()]
            if bio_data:
                bio_table = Table(bio_data, colWidths=[2*inch, 4*inch])
                bio_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#dcfce7')),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f0fdf4')])
                ]))
                story.append(bio_table)
                story.append(Spacer(1, 0.3 * inch))
        
        # Treatment Recommendations
        story.append(Paragraph("Treatment Recommendations", self.styles['SectionHeading']))
        for i, rec in enumerate(recommendations, 1):
            rec_text = f"""
            <b>{i}. {rec.get('treatment', 'N/A')}</b><br/>
            <i>Rationale:</i> {rec.get('rationale', 'N/A')}<br/>
            <i>Evidence:</i> {rec.get('evidence', 'N/A')}<br/>
            <i>Confidence:</i> {rec.get('confidence', 'N/A')}
            """
            story.append(Paragraph(rec_text, self.styles['BodyText']))
            story.append(Spacer(1, 0.2 * inch))
        
        # Analysis Summary
        if analysis_results.get('summary'):
            story.append(Paragraph("Clinical Summary", self.styles['SectionHeading']))
            summary_text = analysis_results['summary']
            story.append(Paragraph(summary_text, self.styles['BodyText']))
            story.append(Spacer(1, 0.3 * inch))
        
        # Confidence Scores
        if analysis_results.get('confidence_scores'):
            story.append(Paragraph("Agent Confidence Scores", self.styles['SectionHeading']))
            conf_data = [
                [agent, f"{score:.1%}"]
                for agent, score in analysis_results['confidence_scores'].items()
            ]
            if conf_data:
                conf_table = Table(conf_data, colWidths=[3*inch, 2*inch])
                conf_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#fef3c7')),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#fffbeb')])
                ]))
                story.append(conf_table)
        
        # Build PDF
        doc.build(story)
        
        if not output_path:
            buffer.seek(0)
            return buffer
        
        return buffer
    
    def generate_conversation_pdf(
        self,
        conversation_history: List[Dict[str, str]],
        patient_data: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None
    ) -> BytesIO:
        """
        Generate PDF from conversation history.
        
        Args:
            conversation_history: List of messages
            patient_data: Optional patient data
            output_path: Optional file path
            
        Returns:
            BytesIO buffer with PDF
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer if not output_path else output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        # Title
        story.append(Paragraph(
            "Conversation Transcript",
            self.styles['CustomTitle']
        ))
        story.append(Spacer(1, 0.2 * inch))
        
        # Metadata
        metadata_text = f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Paragraph(metadata_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.3 * inch))
        
        # Conversation
        story.append(Paragraph("Conversation", self.styles['SectionHeading']))
        
        for msg in conversation_history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')
            
            role_style = 'Heading3' if role == 'user' else 'Heading4'
            role_text = f"<b>{'User' if role == 'user' else 'Assistant'}</b> {timestamp}"
            
            story.append(Paragraph(role_text, self.styles[role_style]))
            story.append(Paragraph(content, self.styles['BodyText']))
            story.append(Spacer(1, 0.2 * inch))
        
        # Build PDF
        doc.build(story)
        
        if not output_path:
            buffer.seek(0)
            return buffer
        
        return buffer


class FHIRExporter:
    """Export to FHIR R4 format for EHR integration."""
    
    def export_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export patient data as FHIR Patient resource.
        
        FHIR Specification: http://hl7.org/fhir/patient.html
        """
        demographics = patient_data.get('demographics', {})
        
        fhir_patient = {
            "resourceType": "Patient",
            "id": patient_data.get('patient_id', 'unknown'),
            "identifier": [{
                "system": "http://hospital.example.org/mrn",
                "value": patient_data.get('patient_id', '')
            }],
            "gender": self._map_gender(demographics.get('sex', '')),
            "birthDate": self._calculate_birth_date(demographics.get('age'))
        }
        
        return fhir_patient
    
    def export_condition(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export diagnosis as FHIR Condition resource.
        
        FHIR Specification: http://hl7.org/fhir/condition.html
        """
        diagnosis = patient_data.get('diagnosis', {})
        
        fhir_condition = {
            "resourceType": "Condition",
            "id": f"{patient_data.get('patient_id', 'unknown')}-lung-cancer",
            "subject": {
                "reference": f"Patient/{patient_data.get('patient_id', 'unknown')}"
            },
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": self._map_snomed_code(diagnosis.get('cancer_type')),
                    "display": f"{diagnosis.get('cancer_type', 'Lung cancer')}"
                }],
                "text": diagnosis.get('histology', 'Lung cancer')
            },
            "clinicalStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active",
                    "display": "Active"
                }]
            },
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                    "code": "encounter-diagnosis",
                    "display": "Encounter Diagnosis"
                }]
            }],
            "stage": [{
                "summary": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/cancer-staging",
                        "code": diagnosis.get('stage', ''),
                        "display": f"Stage {diagnosis.get('stage', '')}"
                    }]
                }
            }] if diagnosis.get('stage') else []
        }
        
        return fhir_condition
    
    def export_observation(
        self, 
        patient_id: str,
        biomarker: str,
        value: str
    ) -> Dict[str, Any]:
        """
        Export biomarker result as FHIR Observation.
        
        FHIR Specification: http://hl7.org/fhir/observation.html
        """
        fhir_observation = {
            "resourceType": "Observation",
            "id": f"{patient_id}-{biomarker}",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "laboratory",
                    "display": "Laboratory"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": self._map_loinc_code(biomarker),
                    "display": biomarker.upper()
                }],
                "text": f"{biomarker.upper()} test"
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "valueCodeableConcept": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": self._map_result_code(value),
                    "display": value
                }],
                "text": value
            }
        }
        
        return fhir_observation
    
    def export_care_plan(
        self,
        patient_id: str,
        recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Export recommendations as FHIR CarePlan.
        
        FHIR Specification: http://hl7.org/fhir/careplan.html
        """
        activities = []
        
        for rec in recommendations:
            activities.append({
                "detail": {
                    "kind": "MedicationRequest",
                    "code": {
                        "text": rec.get('treatment', '')
                    },
                    "status": "not-started",
                    "description": rec.get('rationale', '')
                }
            })
        
        fhir_careplan = {
            "resourceType": "CarePlan",
            "id": f"{patient_id}-treatment-plan",
            "status": "active",
            "intent": "plan",
            "category": [{
                "coding": [{
                    "system": "http://hl7.org/fhir/us/core/CodeSystem/careplan-category",
                    "code": "assess-plan",
                    "display": "Assessment and Plan of Treatment"
                }]
            }],
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "activity": activities
        }
        
        return fhir_careplan
    
    def _map_gender(self, sex: str) -> str:
        """Map sex to FHIR gender."""
        sex_upper = sex.upper()
        if sex_upper in ['M', 'MALE']:
            return 'male'
        elif sex_upper in ['F', 'FEMALE']:
            return 'female'
        return 'unknown'
    
    def _calculate_birth_date(self, age: Optional[int]) -> Optional[str]:
        """Estimate birth date from age."""
        if not age:
            return None
        birth_year = datetime.now().year - age
        return f"{birth_year}-01-01"
    
    def _map_snomed_code(self, cancer_type: str) -> str:
        """Map cancer type to SNOMED CT code."""
        mapping = {
            'NSCLC': '254637007',  # Non-small cell lung cancer
            'SCLC': '254632001'    # Small cell lung cancer
        }
        return mapping.get(cancer_type, '363358000')  # Generic lung cancer
    
    def _map_loinc_code(self, biomarker: str) -> str:
        """Map biomarker to LOINC code."""
        mapping = {
            'egfr': '81695-5',    # EGFR gene mutation
            'alk': '85337-4',     # ALK gene rearrangement
            'ros1': '85147-7',    # ROS1 gene rearrangement
            'pdl1': '85147-7',    # PD-L1
            'kras': '81692-2'     # KRAS gene mutation
        }
        return mapping.get(biomarker.lower(), '85337-4')
    
    def _map_result_code(self, value: str) -> str:
        """Map result value to SNOMED code."""
        value_lower = value.lower()
        if 'positive' in value_lower or 'detected' in value_lower:
            return '10828004'  # Positive
        elif 'negative' in value_lower or 'not detected' in value_lower:
            return '260385009'  # Negative
        return '82334004'  # Indeterminate
