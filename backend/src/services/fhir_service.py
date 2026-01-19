"""
Enhanced FHIR Integration Service

Full FHIR R4 integration for importing/exporting patient data and observations.
Supports FHIR resources: Patient, Condition, Observation, MedicationStatement, etc.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date
from pydantic import BaseModel
from enum import Enum
import json


class FHIRResourceType(str, Enum):
    """FHIR R4 resource types."""
    PATIENT = "Patient"
    CONDITION = "Condition"
    OBSERVATION = "Observation"
    MEDICATION_STATEMENT = "MedicationStatement"
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    PROCEDURE = "Procedure"
    CARE_PLAN = "CarePlan"
    COMPOSITION = "Composition"
    BUNDLE = "Bundle"


class FHIRCodingSystem(str, Enum):
    """Common FHIR coding systems."""
    SNOMED_CT = "http://snomed.info/sct"
    LOINC = "http://loinc.org"
    ICD_10 = "http://hl7.org/fhir/sid/icd-10"
    RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"


class FHIRPatient(BaseModel):
    """FHIR Patient resource."""
    resourceType: str = "Patient"
    id: str
    
    # Identifiers
    identifier: List[Dict[str, Any]] = []
    
    # Name
    name: List[Dict[str, Any]] = []
    
    # Demographics
    gender: Optional[str] = None  # male, female, other, unknown
    birthDate: Optional[str] = None  # YYYY-MM-DD
    
    # Contact
    telecom: List[Dict[str, Any]] = []
    address: List[Dict[str, Any]] = []
    
    # Active status
    active: bool = True


class FHIRCondition(BaseModel):
    """FHIR Condition resource."""
    resourceType: str = "Condition"
    id: str
    
    # Clinical status
    clinicalStatus: Dict[str, Any]
    verificationStatus: Dict[str, Any]
    
    # Category
    category: List[Dict[str, Any]] = []
    
    # Diagnosis code
    code: Dict[str, Any]
    
    # Subject (patient reference)
    subject: Dict[str, str]
    
    # Onset
    onsetDateTime: Optional[str] = None
    
    # Stage
    stage: List[Dict[str, Any]] = []
    
    # Evidence
    evidence: List[Dict[str, Any]] = []


class FHIRObservation(BaseModel):
    """FHIR Observation resource."""
    resourceType: str = "Observation"
    id: str
    
    # Status
    status: str  # registered, preliminary, final, amended
    
    # Category
    category: List[Dict[str, Any]] = []
    
    # What was observed
    code: Dict[str, Any]
    
    # Subject (patient reference)
    subject: Dict[str, str]
    
    # When observed
    effectiveDateTime: Optional[str] = None
    
    # Result value
    valueQuantity: Optional[Dict[str, Any]] = None
    valueCodeableConcept: Optional[Dict[str, Any]] = None
    valueString: Optional[str] = None
    
    # Interpretation
    interpretation: List[Dict[str, Any]] = []
    
    # Reference range
    referenceRange: List[Dict[str, Any]] = []


class FHIRService:
    """Service for FHIR R4 integration."""
    
    def __init__(self):
        """Initialize FHIR service."""
        pass
    
    # ============= FHIR Import (Parse) =============
    
    def import_patient(self, fhir_patient: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import FHIR Patient resource into LCA format.
        
        Args:
            fhir_patient: FHIR Patient resource (dict)
        
        Returns:
            Patient data in LCA internal format
        """
        patient_data = {
            'patient_id': fhir_patient.get('id'),
            'external_ids': []
        }
        
        # Extract identifiers
        for identifier in fhir_patient.get('identifier', []):
            patient_data['external_ids'].append({
                'system': identifier.get('system'),
                'value': identifier.get('value')
            })
        
        # Extract name
        names = fhir_patient.get('name', [])
        if names:
            name = names[0]
            given_names = name.get('given', [])
            family_name = name.get('family', '')
            patient_data['name'] = f"{' '.join(given_names)} {family_name}".strip()
        
        # Demographics
        patient_data['gender'] = fhir_patient.get('gender')
        
        birth_date_str = fhir_patient.get('birthDate')
        if birth_date_str:
            birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d').date()
            patient_data['birth_date'] = birth_date_str
            patient_data['age'] = self._calculate_age(birth_date)
        
        # Contact info
        telecom = fhir_patient.get('telecom', [])
        for contact in telecom:
            if contact.get('system') == 'phone':
                patient_data['phone'] = contact.get('value')
            elif contact.get('system') == 'email':
                patient_data['email'] = contact.get('value')
        
        return patient_data
    
    def import_condition(self, fhir_condition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import FHIR Condition resource.
        
        Extracts diagnosis, stage, and clinical status.
        """
        condition = {
            'condition_id': fhir_condition.get('id')
        }
        
        # Get diagnosis code
        code = fhir_condition.get('code', {})
        codings = code.get('coding', [])
        
        for coding in codings:
            system = coding.get('system')
            
            if system == FHIRCodingSystem.SNOMED_CT.value:
                condition['snomed_code'] = coding.get('code')
                condition['diagnosis'] = coding.get('display')
            elif system == FHIRCodingSystem.ICD_10.value:
                condition['icd10_code'] = coding.get('code')
        
        # Get stage
        stages = fhir_condition.get('stage', [])
        if stages:
            stage_summary = stages[0].get('summary', {})
            stage_coding = stage_summary.get('coding', [])
            if stage_coding:
                condition['stage'] = stage_coding[0].get('code')
        
        # Clinical status
        clinical_status = fhir_condition.get('clinicalStatus', {})
        status_coding = clinical_status.get('coding', [])
        if status_coding:
            condition['clinical_status'] = status_coding[0].get('code')
        
        # Onset date
        condition['onset_date'] = fhir_condition.get('onsetDateTime')
        
        return condition
    
    def import_observation(self, fhir_observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import FHIR Observation resource.
        
        Handles lab results, biomarkers, vital signs, etc.
        """
        observation = {
            'observation_id': fhir_observation.get('id'),
            'status': fhir_observation.get('status')
        }
        
        # Get observation code (what was measured)
        code = fhir_observation.get('code', {})
        codings = code.get('coding', [])
        
        for coding in codings:
            system = coding.get('system')
            
            if system == FHIRCodingSystem.LOINC.value:
                observation['loinc_code'] = coding.get('code')
                observation['observation_name'] = coding.get('display')
            elif system == FHIRCodingSystem.SNOMED_CT.value:
                observation['snomed_code'] = coding.get('code')
        
        # Get value
        if 'valueQuantity' in fhir_observation:
            value_qty = fhir_observation['valueQuantity']
            observation['value'] = value_qty.get('value')
            observation['unit'] = value_qty.get('unit')
        
        elif 'valueCodeableConcept' in fhir_observation:
            value_concept = fhir_observation['valueCodeableConcept']
            value_coding = value_concept.get('coding', [])
            if value_coding:
                observation['value_code'] = value_coding[0].get('code')
                observation['value_display'] = value_coding[0].get('display')
        
        elif 'valueString' in fhir_observation:
            observation['value'] = fhir_observation['valueString']
        
        # Effective date
        observation['effective_date'] = fhir_observation.get('effectiveDateTime')
        
        # Interpretation
        interpretations = fhir_observation.get('interpretation', [])
        if interpretations:
            interp_coding = interpretations[0].get('coding', [])
            if interp_coding:
                observation['interpretation'] = interp_coding[0].get('code')
        
        return observation
    
    def import_medication(self, fhir_medication_statement: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import FHIR MedicationStatement resource.
        
        Extracts medication information including dosage and timing.
        """
        medication = {
            'medication_id': fhir_medication_statement.get('id'),
            'status': fhir_medication_statement.get('status')
        }
        
        # Get medication code
        medication_code = fhir_medication_statement.get('medicationCodeableConcept', {})
        codings = medication_code.get('coding', [])
        
        for coding in codings:
            system = coding.get('system')
            
            if system == FHIRCodingSystem.RXNORM.value:
                medication['rxnorm_code'] = coding.get('code')
                medication['medication_name'] = coding.get('display')
            elif system == FHIRCodingSystem.SNOMED_CT.value:
                medication['snomed_code'] = coding.get('code')
        
        # Get dosage
        dosage = fhir_medication_statement.get('dosage', [])
        if dosage:
            dose_quantity = dosage[0].get('doseAndRate', [{}])[0].get('doseQuantity', {})
            medication['dosage'] = {
                'value': dose_quantity.get('value'),
                'unit': dose_quantity.get('unit'),
                'timing': dosage[0].get('timing', {}).get('code')
            }
        
        # Effective period
        effective_period = fhir_medication_statement.get('effectivePeriod', {})
        medication['start_date'] = effective_period.get('start')
        medication['end_date'] = effective_period.get('end')
        
        return medication
    
    def import_bundle(self, fhir_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import FHIR Bundle containing multiple resources.
        
        Returns organized patient data with all resources.
        Enhanced to handle more resource types.
        """
        patient_data = {
            'conditions': [],
            'observations': [],
            'medications': [],
            'procedures': []
        }
        
        entries = fhir_bundle.get('entry', [])
        
        for entry in entries:
            resource = entry.get('resource', {})
            resource_type = resource.get('resourceType')
            
            if resource_type == 'Patient':
                patient_data.update(self.import_patient(resource))
            
            elif resource_type == 'Condition':
                patient_data['conditions'].append(self.import_condition(resource))
            
            elif resource_type == 'Observation':
                patient_data['observations'].append(self.import_observation(resource))
            
            elif resource_type == 'MedicationStatement':
                patient_data['medications'].append(self.import_medication(resource))
            
            # Add more resource types as needed
        
        return patient_data
    
    # ============= FHIR Export (Generate) =============
    
    def export_patient(self, patient_data: Dict[str, Any]) -> FHIRPatient:
        """
        Export patient data to FHIR Patient resource.
        """
        patient = FHIRPatient(
            id=patient_data.get('patient_id', 'unknown'),
            identifier=[],
            name=[],
            active=True
        )
        
        # Add identifier
        if 'patient_id' in patient_data:
            patient.identifier.append({
                'system': 'urn:lca:patient-id',
                'value': patient_data['patient_id']
            })
        
        # Add name
        if 'name' in patient_data:
            name_parts = patient_data['name'].split()
            patient.name.append({
                'use': 'official',
                'family': name_parts[-1] if name_parts else '',
                'given': name_parts[:-1] if len(name_parts) > 1 else []
            })
        
        # Demographics
        patient.gender = patient_data.get('gender', 'unknown')
        patient.birthDate = patient_data.get('birth_date')
        
        # Contact
        if 'phone' in patient_data:
            patient.telecom.append({
                'system': 'phone',
                'value': patient_data['phone'],
                'use': 'mobile'
            })
        
        if 'email' in patient_data:
            patient.telecom.append({
                'system': 'email',
                'value': patient_data['email']
            })
        
        return patient
    
    def export_condition(
        self,
        patient_id: str,
        diagnosis: str,
        stage: Optional[str] = None,
        snomed_code: Optional[str] = None,
        icd10_code: Optional[str] = None,
        onset_date: Optional[str] = None
    ) -> FHIRCondition:
        """Export diagnosis as FHIR Condition resource."""
        condition = FHIRCondition(
            id=f"condition-{patient_id}-lung-cancer",
            clinicalStatus={
                'coding': [{
                    'system': 'http://terminology.hl7.org/CodeSystem/condition-clinical',
                    'code': 'active'
                }]
            },
            verificationStatus={
                'coding': [{
                    'system': 'http://terminology.hl7.org/CodeSystem/condition-ver-status',
                    'code': 'confirmed'
                }]
            },
            category=[{
                'coding': [{
                    'system': 'http://terminology.hl7.org/CodeSystem/condition-category',
                    'code': 'encounter-diagnosis'
                }]
            }],
            code={
                'coding': [],
                'text': diagnosis
            },
            subject={
                'reference': f"Patient/{patient_id}"
            }
        )
        
        # Add SNOMED code
        if snomed_code:
            condition.code['coding'].append({
                'system': FHIRCodingSystem.SNOMED_CT.value,
                'code': snomed_code,
                'display': diagnosis
            })
        
        # Add ICD-10 code
        if icd10_code:
            condition.code['coding'].append({
                'system': FHIRCodingSystem.ICD_10.value,
                'code': icd10_code,
                'display': diagnosis
            })
        
        # Add stage
        if stage:
            condition.stage.append({
                'summary': {
                    'coding': [{
                        'system': 'http://cancerstaging.org',
                        'code': stage,
                        'display': f'Stage {stage}'
                    }]
                }
            })
        
        # Onset date
        if onset_date:
            condition.onsetDateTime = onset_date
        
        return condition
    
    def export_observation(
        self,
        patient_id: str,
        observation_name: str,
        value: Any,
        unit: Optional[str] = None,
        loinc_code: Optional[str] = None,
        effective_date: Optional[str] = None
    ) -> FHIRObservation:
        """Export observation/biomarker as FHIR Observation resource."""
        observation = FHIRObservation(
            id=f"obs-{patient_id}-{observation_name.lower().replace(' ', '-')}",
            status='final',
            category=[{
                'coding': [{
                    'system': 'http://terminology.hl7.org/CodeSystem/observation-category',
                    'code': 'laboratory'
                }]
            }],
            code={
                'coding': [],
                'text': observation_name
            },
            subject={
                'reference': f"Patient/{patient_id}"
            }
        )
        
        # Add LOINC code
        if loinc_code:
            observation.code['coding'].append({
                'system': FHIRCodingSystem.LOINC.value,
                'code': loinc_code,
                'display': observation_name
            })
        
        # Add value
        if isinstance(value, (int, float)):
            observation.valueQuantity = {
                'value': value,
                'unit': unit or '',
                'system': 'http://unitsofmeasure.org',
                'code': unit or ''
            }
        elif isinstance(value, str):
            observation.valueString = value
        
        # Effective date
        if effective_date:
            observation.effectiveDateTime = effective_date
        else:
            observation.effectiveDateTime = datetime.now().isoformat()
        
        return observation
    
    def export_bundle(
        self,
        patient_data: Dict[str, Any],
        analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Export complete patient data and analysis as FHIR Bundle.
        
        Includes Patient, Conditions, Observations, and CarePlan.
        Consolidated export logic for all FHIR resources.
        """
        bundle = {
            'resourceType': 'Bundle',
            'type': 'collection',
            'entry': []
        }
        
        patient_id = patient_data.get('patient_id', 'unknown')
        
        # Add Patient resource
        patient = self.export_patient(patient_data)
        bundle['entry'].append({
            'resource': patient.dict(exclude_none=True)
        })
        
        # Add primary diagnosis (lung cancer condition)
        diagnosis = patient_data.get('diagnosis', 'Lung cancer')
        stage = patient_data.get('stage') or patient_data.get('tnm_stage')
        snomed_code = patient_data.get('snomed_code')
        icd10_code = patient_data.get('icd10_code')
        onset_date = patient_data.get('diagnosis_date')
        
        if diagnosis or stage:
            condition = self.export_condition(
                patient_id=patient_id,
                diagnosis=diagnosis,
                stage=stage,
                snomed_code=snomed_code,
                icd10_code=icd10_code,
                onset_date=onset_date
            )
            bundle['entry'].append({
                'resource': condition.dict(exclude_none=True)
            })
        
        # Add comorbidities as additional conditions
        for comorbidity in patient_data.get('comorbidities', []):
            comorbidity_condition = self.export_condition(
                patient_id=patient_id,
                diagnosis=comorbidity,
                snomed_code=None,  # Could be enhanced with SNOMED mapping
                icd10_code=None,
                onset_date=None
            )
            bundle['entry'].append({
                'resource': comorbidity_condition.dict(exclude_none=True)
            })
        
        # Add biomarker observations
        biomarkers = patient_data.get('biomarkers', {})
        for biomarker_name, biomarker_data in biomarkers.items():
            if isinstance(biomarker_data, dict):
                value = biomarker_data.get('value')
                unit = biomarker_data.get('unit')
                loinc_code = biomarker_data.get('loinc_code')
                effective_date = biomarker_data.get('date')
            else:
                value = biomarker_data
                unit = None
                loinc_code = None
                effective_date = None
            
            observation = self.export_observation(
                patient_id=patient_id,
                observation_name=f'{biomarker_name.upper()} biomarker',
                value=value,
                unit=unit,
                loinc_code=loinc_code,
                effective_date=effective_date
            )
            bundle['entry'].append({
                'resource': observation.dict(exclude_none=True)
            })
        
        # Add clinical observations (performance status, FEV1, etc.)
        clinical_obs = {
            'performance_status': {
                'value': patient_data.get('performance_status'),
                'unit': None,
                'loinc_code': '89247-1',  # ECOG Performance Status
                'name': 'ECOG Performance Status'
            },
            'fev1_percent': {
                'value': patient_data.get('fev1_percent'),
                'unit': '%',
                'loinc_code': '19926-5',  # FEV1 measured/predicted
                'name': 'FEV1 % predicted'
            },
            'age_at_diagnosis': {
                'value': patient_data.get('age'),
                'unit': 'years',
                'loinc_code': '21612-7',  # Age at specimen collection
                'name': 'Age at diagnosis'
            }
        }
        
        for obs_key, obs_config in clinical_obs.items():
            if patient_data.get(obs_key) is not None:
                observation = self.export_observation(
                    patient_id=patient_id,
                    observation_name=obs_config['name'],
                    value=obs_config['value'],
                    unit=obs_config['unit'],
                    loinc_code=obs_config['loinc_code']
                )
                bundle['entry'].append({
                    'resource': observation.dict(exclude_none=True)
                })
        
        # Add recommended treatment as CarePlan
        if analysis_result and 'recommendation' in analysis_result:
            care_plan = self._create_care_plan(patient_id, analysis_result['recommendation'])
            bundle['entry'].append({
                'resource': care_plan
            })
        
        # Add Composition resource for clinical summary
        if analysis_result:
            composition = self._create_composition(patient_id, patient_data, analysis_result)
            bundle['entry'].append({
                'resource': composition
            })
        
        return bundle
    
    def _create_care_plan(
        self,
        patient_id: str,
        recommendation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create FHIR CarePlan resource from treatment recommendation."""
        care_plan = {
            'resourceType': 'CarePlan',
            'id': f'careplan-{patient_id}',
            'status': 'active',
            'intent': 'plan',
            'subject': {
                'reference': f'Patient/{patient_id}'
            },
            'title': 'Lung Cancer Treatment Plan',
            'description': recommendation.get('treatment_plan', ''),
            'activity': []
        }
        
        # Add treatment activity
        if 'treatment' in recommendation:
            care_plan['activity'].append({
                'detail': {
                    'kind': 'MedicationRequest',
                    'code': {
                        'text': recommendation['treatment']
                    },
                    'status': 'not-started',
                    'description': recommendation.get('rationale', '')
                }
            })
        
        return care_plan
    
    def _create_composition(
        self,
        patient_id: str,
        patient_data: Dict[str, Any],
        analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create FHIR Composition resource for clinical summary."""
        composition = {
            'resourceType': 'Composition',
            'id': f'composition-{patient_id}',
            'status': 'final',
            'type': {
                'coding': [{
                    'system': 'http://loinc.org',
                    'code': '11503-0',  # Medical records
                    'display': 'Medical Records'
                }]
            },
            'subject': {
                'reference': f'Patient/{patient_id}'
            },
            'date': datetime.now().isoformat(),
            'author': [{
                'display': 'Lung Cancer Assistant AI System'
            }],
            'title': f'Clinical Decision Support Summary - {patient_data.get("name", patient_id)}',
            'section': []
        }
        
        # Patient demographics section
        composition['section'].append({
            'title': 'Patient Demographics',
            'code': {
                'coding': [{
                    'system': 'http://loinc.org',
                    'code': '10160-0',  # History of Present illness
                    'display': 'History of Present illness'
                }]
            },
            'text': {
                'status': 'generated',
                'div': f"""
                <div>
                    <p><strong>Patient:</strong> {patient_data.get('name', 'Unknown')}</p>
                    <p><strong>Age:</strong> {patient_data.get('age', 'Unknown')}</p>
                    <p><strong>Gender:</strong> {patient_data.get('gender', 'Unknown')}</p>
                    <p><strong>Diagnosis:</strong> {patient_data.get('diagnosis', 'Lung cancer')}</p>
                    <p><strong>Stage:</strong> {patient_data.get('stage') or patient_data.get('tnm_stage', 'Unknown')}</p>
                </div>
                """
            }
        })
        
        # Clinical findings section
        composition['section'].append({
            'title': 'Clinical Findings',
            'code': {
                'coding': [{
                    'system': 'http://loinc.org',
                    'code': '11493-4',  # Healthcare provider comment
                    'display': 'Healthcare provider comment'
                }]
            },
            'text': {
                'status': 'generated',
                'div': f"""
                <div>
                    <p><strong>Performance Status:</strong> ECOG {patient_data.get('performance_status', 'Unknown')}</p>
                    <p><strong>Histology:</strong> {patient_data.get('histology_type', 'Unknown')}</p>
                    <p><strong>Laterality:</strong> {patient_data.get('laterality', 'Unknown')}</p>
                    <p><strong>Comorbidities:</strong> {', '.join(patient_data.get('comorbidities', [])) or 'None'}</p>
                </div>
                """
            }
        })
        
        # Treatment recommendation section
        if 'recommendation' in analysis_result:
            rec = analysis_result['recommendation']
            composition['section'].append({
                'title': 'Treatment Recommendations',
                'code': {
                    'coding': [{
                        'system': 'http://loinc.org',
                        'code': '18776-5',  # Plan of care note
                        'display': 'Plan of care note'
                    }]
                },
                'text': {
                    'status': 'generated',
                    'div': f"""
                    <div>
                        <p><strong>Recommended Treatment:</strong> {rec.get('treatment', 'Unknown')}</p>
                        <p><strong>Rationale:</strong> {rec.get('rationale', 'AI-generated recommendation')}</p>
                        <p><strong>Confidence:</strong> {rec.get('confidence', 'Unknown')}%</p>
                        <p><strong>Evidence Level:</strong> {rec.get('evidence_level', 'Unknown')}</p>
                    </div>
                    """
                }
            })
        
        return composition
    
    def _calculate_age(self, birth_date: date) -> int:
        """Calculate age from birth date."""
        today = date.today()
        age = today.year - birth_date.year
        
        # Adjust if birthday hasn't occurred this year
        if (today.month, today.day) < (birth_date.month, birth_date.day):
            age -= 1
        
        return age


# Global FHIR service instance
fhir_service = FHIRService()
