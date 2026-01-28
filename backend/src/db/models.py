"""
Neo4j Data Models for LUCADA
Pydantic models matching the LUCADA ontology structure (Figure 1)
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


# ========================================
# Enums
# ========================================

class Sex(str, Enum):
    MALE = "M"
    FEMALE = "F"
    UNKNOWN = "U"


class TNMStage(str, Enum):
    IA = "IA"
    IB = "IB"
    IIA = "IIA"
    IIB = "IIB"
    IIIA = "IIIA"
    IIIB = "IIIB"
    IV = "IV"


class HistologyType(str, Enum):
    ADENOCARCINOMA = "Adenocarcinoma"
    SQUAMOUS_CELL = "SquamousCellCarcinoma"
    LARGE_CELL = "LargeCellCarcinoma"
    SMALL_CELL = "SmallCellCarcinoma"
    CARCINOSARCOMA = "Carcinosarcoma"
    NSCLC_NOS = "NonSmallCellCarcinoma_NOS"


class PerformanceStatus(int, Enum):
    WHO_0 = 0
    WHO_1 = 1
    WHO_2 = 2
    WHO_3 = 3
    WHO_4 = 4


class Laterality(str, Enum):
    RIGHT = "Right"
    LEFT = "Left"
    BILATERAL = "Bilateral"


class InferenceStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    OBSOLETE = "obsolete"


class EvidenceLevel(str, Enum):
    GRADE_A = "Grade A"
    GRADE_B = "Grade B"
    GRADE_C = "Grade C"
    GRADE_D = "Grade D"
    EXPERT_OPINION = "Expert Opinion"


class TreatmentIntent(str, Enum):
    CURATIVE = "Curative"
    PALLIATIVE = "Palliative"
    ADJUVANT = "Adjuvant"
    NEOADJUVANT = "Neoadjuvant"
    SUPPORTIVE = "Supportive"
    UNKNOWN = "Unknown"


# ========================================
# Core Models (matching LUCADA Figure 1)
# ========================================

class PatientFact(BaseModel):
    """
    Patient data model with ALL 10 LUCADA data properties from Figure 1.
    Used by IngestionAgent for validation.
    """
    patient_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8].upper())
    db_identifier: Optional[str] = None
    name: str
    sex: Sex
    age_at_diagnosis: int = Field(ge=0, le=120)
    tnm_stage: TNMStage
    histology_type: HistologyType
    performance_status: PerformanceStatus
    laterality: Laterality = Laterality.RIGHT
    fev1_percent: Optional[float] = Field(default=None, ge=0, le=200)
    fev1_absolute_amount: Optional[float] = None
    survival_days: Optional[int] = None
    diagnosis: str = "Malignant Neoplasm of Lung"
    diagnosis_site_code: Optional[str] = None
    basis_of_diagnosis: Optional[str] = "Clinical"
    comorbidities: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True


class PatientFactWithCodes(PatientFact):
    """
    Extended PatientFact with SNOMED-CT codes.
    Output of SemanticMappingAgent.
    """
    snomed_diagnosis_code: Optional[str] = None
    snomed_histology_code: Optional[str] = None
    snomed_stage_code: Optional[str] = None
    snomed_ps_code: Optional[str] = None
    snomed_laterality_code: Optional[str] = None
    mapping_confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ClinicalFinding(BaseModel):
    """Clinical Finding node model"""
    finding_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    diagnosis: str
    tnm_stage: str
    diagnosis_site_code: Optional[str] = None
    basis_of_diagnosis: str = "Clinical"
    snomed_code: Optional[str] = None


class Histology(BaseModel):
    """Histology node model"""
    type: HistologyType
    snomed_code: Optional[str] = None
    
    class Config:
        use_enum_values = True


class TreatmentPlan(BaseModel):
    """Treatment Plan node model"""
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    type: str
    intent: TreatmentIntent
    snomed_code: Optional[str] = None
    decision_date: Optional[datetime] = None
    start_date: Optional[datetime] = None
    
    class Config:
        use_enum_values = True


# ========================================
# Classification Models
# ========================================

class PatientScenario(BaseModel):
    """
    Patient Scenario classification result.
    Matches paper's Figure 3 pattern.
    """
    scenario_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    rule_id: str
    rule_name: str
    source: str = "NICE CG121"
    description: str
    recommended_treatment: str
    treatment_intent: TreatmentIntent
    evidence_level: EvidenceLevel
    priority_score: int = Field(ge=0, le=100)
    matching_criteria: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class ClassificationResult(BaseModel):
    """Output of ClassificationAgent"""
    patient_id: str
    scenario: str
    scenario_confidence: float = Field(ge=0.0, le=1.0)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)  # List of recommendation dicts
    reasoning_chain: List[str] = Field(default_factory=list)
    ontology_concepts_matched: List[str] = Field(default_factory=list)
    guideline_refs: List[str] = Field(default_factory=list)
    classification_time: datetime = Field(default_factory=datetime.now)
    reasoner_used: str = "Python-based"
    contraindications: List[str] = Field(default_factory=list)


# ========================================
# Treatment Recommendation Models
# ========================================

class TreatmentArgument(BaseModel):
    """Argumentation model (from paper's Argument class)"""
    argument_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    claim: str
    evidence: str
    strength: str = Field(description="Strong, Moderate, or Weak")
    argument_type: str = Field(description="Supporting or Opposing")
    source: Optional[str] = None


class TreatmentRecommendation(BaseModel):
    """
    Final treatment recommendation.
    Output of ConflictResolutionAgent.
    """
    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    patient_id: str
    primary_treatment: str
    treatment_intent: TreatmentIntent
    evidence_level: EvidenceLevel
    confidence_score: float = Field(ge=0.0, le=1.0)
    supporting_arguments: List[TreatmentArgument] = Field(default_factory=list)
    opposing_arguments: List[TreatmentArgument] = Field(default_factory=list)
    alternative_treatments: List[str] = Field(default_factory=list)
    rationale: str
    guideline_references: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True


# ========================================
# Inference & Audit Models
# ========================================

class InferenceRecord(BaseModel):
    """
    Complete inference audit trail.
    Stored by PersistenceAgent.
    """
    inference_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    status: InferenceStatus = InferenceStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Agent outputs
    ingestion_result: Optional[Dict[str, Any]] = None
    semantic_mapping_result: Optional[Dict[str, Any]] = None
    classification_result: Optional[Dict[str, Any]] = None
    conflict_resolution_result: Optional[Dict[str, Any]] = None
    explanation_result: Optional[Dict[str, Any]] = None
    
    # Metadata
    agent_version: str = "1.0.0"
    workflow_duration_ms: Optional[int] = None
    error_message: Optional[str] = None
    
    class Config:
        use_enum_values = True


class WriteReceipt(BaseModel):
    """
    Confirmation of Neo4j write operation.
    Returned by PersistenceAgent write methods.
    """
    receipt_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    patient_id: Optional[str] = None
    inference_id: Optional[str] = None
    timestamp: str
    success: bool = True
    entities_written: List[str] = Field(default_factory=list)
    relationships_written: List[str] = Field(default_factory=list)
    agent_version: Optional[str] = None
    write_sequence: int = 0
    error_message: Optional[str] = None
    
    # Legacy fields for backward compatibility
    operation: Optional[str] = None
    node_type: Optional[str] = None
    node_id: Optional[str] = None
    message: Optional[str] = None
    properties_written: int = 0


# ========================================
# Cohort & Similar Patient Models
# ========================================

class SimilarPatient(BaseModel):
    """Similar patient for cohort comparison"""
    patient_id: str
    name: str = "Unknown"
    similarity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    tnm_stage: str = "Unknown"
    histology_type: str = "Unknown"
    treatment_received: Optional[str] = None
    outcome: Optional[str] = None
    survival_days: Optional[int] = None


class CohortStats(BaseModel):
    """Aggregated cohort statistics"""
    total_patients: int
    stage_distribution: Dict[str, int] = Field(default_factory=dict)
    histology_distribution: Dict[str, int] = Field(default_factory=dict)
    treatment_distribution: Dict[str, int] = Field(default_factory=dict)
    avg_survival_days: Optional[float] = None
    median_survival_days: Optional[float] = None
    response_rates: Dict[str, float] = Field(default_factory=dict)


# ========================================
# MDT Summary Models
# ========================================

class MDTSummary(BaseModel):
    """
    Human-readable MDT summary.
    Output of ExplanationAgent.
    """
    summary_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    patient_id: str
    generated_at: datetime = Field(default_factory=datetime.now)
    inference_id: Optional[str] = None
    clinical_summary: str
    classification_scenario: str
    scenario_confidence: float
    formatted_recommendations: List[Dict[str, str]] = Field(default_factory=list)
    reasoning_explanation: str
    key_considerations: List[str] = Field(default_factory=list)
    discussion_points: List[str] = Field(default_factory=list)
    guideline_references: List[str] = Field(default_factory=list)
    snomed_mappings: Dict[str, str] = Field(default_factory=dict)
    disclaimer: str = ""


# ========================================
# API Response Models
# ========================================

class InferenceResult(BaseModel):
    """Complete workflow result returned by API"""
    inference_id: str
    patient_id: str
    patient_fact: PatientFact
    patient_with_codes: Optional[PatientFactWithCodes] = None
    classification: Optional[ClassificationResult] = None
    recommendation: Optional[TreatmentRecommendation] = None
    mdt_summary: Optional[MDTSummary] = None
    write_receipts: List[WriteReceipt] = Field(default_factory=list)
    workflow_duration_ms: int
    status: InferenceStatus = InferenceStatus.COMPLETED
    
    class Config:
        use_enum_values = True


class DecisionSupportResponse(BaseModel):
    """Full workflow response for decision support"""
    patient_id: Optional[str] = None
    success: bool = True
    workflow_status: str
    agent_chain: List[str] = Field(default_factory=list)
    
    # Classification results
    scenario: Optional[str] = None
    scenario_confidence: Optional[float] = None
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_chain: List[str] = Field(default_factory=list)
    
    # SNOMED mappings
    snomed_mappings: Dict[str, Optional[str]] = Field(default_factory=dict)
    mapping_confidence: float = 0.0
    
    # Persistence results
    inference_id: Optional[str] = None
    persisted: bool = False
    
    # MDT summary
    mdt_summary: Optional[str] = None
    key_considerations: List[str] = Field(default_factory=list)
    discussion_points: List[str] = Field(default_factory=list)
    
    # Metadata
    processing_time_seconds: float = 0.0
    errors: List[str] = Field(default_factory=list)
    guideline_refs: List[str] = Field(default_factory=list)
