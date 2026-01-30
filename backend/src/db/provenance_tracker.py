"""
Enhanced Provenance Tracking System for LCA

Provides comprehensive lineage tracking for:
- Data transformations and sources
- Model versions and parameters
- Agent execution chains
- Workflow routing decisions
- Temporal evolution of recommendations
- Audit trails for compliance

Implements W3C PROV-DM (Provenance Data Model) patterns.
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json

# Centralized logging
from ..logging_config import get_logger, log_execution

logger = get_logger(__name__)


class ProvenanceType(Enum):
    """Types of provenance entities"""
    ENTITY = "entity"  # Data items (patient, recommendation)
    ACTIVITY = "activity"  # Processes (agent execution, workflow)
    AGENT = "agent"  # Software agents or human actors


class ProvenanceRelation(Enum):
    """W3C PROV relationships"""
    WAS_DERIVED_FROM = "wasDerivedFrom"
    WAS_GENERATED_BY = "wasGeneratedBy"
    USED = "used"
    WAS_ATTRIBUTED_TO = "wasAttributedTo"
    WAS_INFORMED_BY = "wasInformedBy"
    WAS_INFLUENCED_BY = "wasInfluencedBy"


@dataclass
class ProvenanceEntity:
    """
    Entity in provenance graph (data/artifacts)
    
    Examples:
    - Patient data
    - Inference results
    - Treatment recommendations
    - MDT summaries
    """
    id: str
    type: ProvenanceType
    label: str
    timestamp: datetime
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Lineage relationships
    derived_from: List[str] = field(default_factory=list)  # Parent entity IDs
    generated_by: Optional[str] = None  # Activity ID that created this
    
    # Versioning
    version: str = "1.0.0"
    checksum: Optional[str] = None
    
    def compute_checksum(self) -> str:
        """Generate content hash for integrity verification"""
        content = json.dumps(self.attributes, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['type'] = self.type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ProvenanceActivity:
    """
    Activity in provenance graph (processes/transformations)
    
    Examples:
    - Agent execution
    - Workflow step
    - Data validation
    - Model inference
    """
    id: str
    type: ProvenanceType = ProvenanceType.ACTIVITY
    label: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    # Execution details
    agent_name: str = ""
    agent_version: str = ""
    workflow_type: str = ""  # "basic", "6-agent", "integrated"
    complexity_level: Optional[str] = None
    
    # Inputs and outputs
    used_entities: List[str] = field(default_factory=list)  # Input entity IDs
    generated_entities: List[str] = field(default_factory=list)  # Output entity IDs
    
    # Configuration and parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # Status tracking
    status: str = "started"  # started, completed, failed
    error_message: Optional[str] = None
    
    def complete(self, status: str = "completed"):
        """Mark activity as completed"""
        self.end_time = datetime.utcnow()
        self.status = status
    
    def duration_ms(self) -> Optional[int]:
        """Calculate duration in milliseconds"""
        if self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['type'] = self.type.value
        data['start_time'] = self.start_time.isoformat()
        data['end_time'] = self.end_time.isoformat() if self.end_time else None
        return data


@dataclass
class ProvenanceAgent:
    """
    Agent in provenance graph (software/human actors)
    
    Examples:
    - LLM models (Ollama)
    - Software agents (IngestionAgent, ClassificationAgent)
    - Human reviewers
    """
    id: str
    type: ProvenanceType = ProvenanceType.AGENT
    label: str = ""
    
    # Agent details
    agent_type: str = ""  # "llm", "software_agent", "human"
    version: str = "1.0.0"
    
    # For LLM models
    model_name: Optional[str] = None
    model_provider: Optional[str] = None
    temperature: Optional[float] = None
    
    # For software agents
    code_version: Optional[str] = None
    dependencies: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['type'] = self.type.value
        return data


@dataclass
class ProvenanceRecord:
    """
    Complete provenance record for a clinical decision
    
    Tracks full lineage from patient data to recommendations
    """
    record_id: str
    patient_id: str
    workflow_session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Provenance graph components
    entities: Dict[str, ProvenanceEntity] = field(default_factory=dict)
    activities: Dict[str, ProvenanceActivity] = field(default_factory=dict)
    agents: Dict[str, ProvenanceAgent] = field(default_factory=dict)
    
    # Workflow metadata
    workflow_type: str = ""  # "basic", "6-agent", "integrated"
    complexity_routing: Optional[str] = None
    
    # Data sources
    data_sources: List[Dict[str, Any]] = field(default_factory=list)
    ontology_versions: Dict[str, str] = field(default_factory=dict)
    
    def add_entity(self, entity: ProvenanceEntity):
        """Add entity to provenance graph"""
        self.entities[entity.id] = entity
    
    def add_activity(self, activity: ProvenanceActivity):
        """Add activity to provenance graph"""
        self.activities[activity.id] = activity
    
    def add_agent(self, agent: ProvenanceAgent):
        """Add agent to provenance graph"""
        self.agents[agent.id] = agent
    
    def get_lineage(self, entity_id: str) -> List[str]:
        """
        Get complete lineage chain for an entity
        
        Returns list of entity IDs from source to target
        """
        if entity_id not in self.entities:
            return []
        
        lineage = [entity_id]
        entity = self.entities[entity_id]
        
        # Recursively trace back through derived_from relationships
        for parent_id in entity.derived_from:
            lineage.extend(self.get_lineage(parent_id))
        
        return lineage
    
    def get_execution_chain(self) -> List[str]:
        """Get ordered list of activities (agent execution chain)"""
        # Sort activities by start time
        sorted_activities = sorted(
            self.activities.values(),
            key=lambda a: a.start_time
        )
        return [a.label for a in sorted_activities]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "record_id": self.record_id,
            "patient_id": self.patient_id,
            "workflow_session_id": self.workflow_session_id,
            "created_at": self.created_at.isoformat(),
            "workflow_type": self.workflow_type,
            "complexity_routing": self.complexity_routing,
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "activities": {k: v.to_dict() for k, v in self.activities.items()},
            "agents": {k: v.to_dict() for k, v in self.agents.items()},
            "data_sources": self.data_sources,
            "ontology_versions": self.ontology_versions,
            "execution_chain": self.get_execution_chain()
        }


class ProvenanceTracker:
    """
    Main provenance tracking system
    
    Manages creation and querying of provenance records
    """
    
    def __init__(self):
        self.current_record: Optional[ProvenanceRecord] = None
        self.records_history: Dict[str, ProvenanceRecord] = {}
    
    def start_session(
        self,
        patient_id: str,
        workflow_type: str,
        complexity_level: Optional[str] = None
    ) -> str:
        """
        Start a new provenance tracking session
        
        Returns session ID
        """
        import uuid
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        record_id = f"prov_{patient_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_record = ProvenanceRecord(
            record_id=record_id,
            patient_id=patient_id,
            workflow_session_id=session_id,
            workflow_type=workflow_type,
            complexity_routing=complexity_level
        )
        
        logger.info(f"[Provenance] Started session {session_id} for patient {patient_id}")
        return session_id
    
    def track_data_ingestion(
        self,
        source: str,
        data: Dict[str, Any],
        activity_id: Optional[str] = None
    ) -> str:
        """Track data ingestion from external sources"""
        if not self.current_record:
            raise ValueError("No active provenance session")
        
        # Create entity for ingested data
        entity_id = f"entity_ingestion_{datetime.utcnow().strftime('%H%M%S%f')}"
        entity = ProvenanceEntity(
            id=entity_id,
            type=ProvenanceType.ENTITY,
            label=f"Patient Data from {source}",
            timestamp=datetime.utcnow(),
            attributes={"source": source, "data_preview": str(data)[:200]},
            version="1.0.0"
        )
        entity.checksum = entity.compute_checksum()
        
        self.current_record.add_entity(entity)
        self.current_record.data_sources.append({
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
            "entity_id": entity_id
        })
        
        logger.debug(f"[Provenance] Tracked data ingestion from {source}")
        return entity_id
    
    def track_agent_execution(
        self,
        agent_name: str,
        agent_version: str,
        input_entity_ids: List[str],
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Track start of agent execution
        
        Returns activity ID
        """
        if not self.current_record:
            raise ValueError("No active provenance session")
        
        activity_id = f"activity_{agent_name}_{datetime.utcnow().strftime('%H%M%S%f')}"
        activity = ProvenanceActivity(
            id=activity_id,
            label=f"{agent_name} Execution",
            agent_name=agent_name,
            agent_version=agent_version,
            workflow_type=self.current_record.workflow_type,
            complexity_level=self.current_record.complexity_routing,
            used_entities=input_entity_ids,
            parameters=parameters or {},
            status="started"
        )
        
        self.current_record.add_activity(activity)
        logger.debug(f"[Provenance] Tracking {agent_name} execution")
        return activity_id
    
    def track_agent_completion(
        self,
        activity_id: str,
        output_entity_id: str,
        status: str = "completed",
        error: Optional[str] = None
    ):
        """Mark agent execution as completed"""
        if not self.current_record or activity_id not in self.current_record.activities:
            return
        
        activity = self.current_record.activities[activity_id]
        activity.complete(status)
        activity.generated_entities.append(output_entity_id)
        
        if error:
            activity.error_message = error
        
        # Update output entity to link to this activity
        if output_entity_id in self.current_record.entities:
            self.current_record.entities[output_entity_id].generated_by = activity_id
        
        duration = activity.duration_ms()
        logger.debug(f"[Provenance] {activity.label} completed in {duration}ms")
    
    def track_transformation(
        self,
        input_entity_id: str,
        output_data: Dict[str, Any],
        transformation_type: str,
        activity_id: str
    ) -> str:
        """Track data transformation"""
        if not self.current_record:
            raise ValueError("No active provenance session")
        
        entity_id = f"entity_{transformation_type}_{datetime.utcnow().strftime('%H%M%S%f')}"
        entity = ProvenanceEntity(
            id=entity_id,
            type=ProvenanceType.ENTITY,
            label=f"{transformation_type} Output",
            timestamp=datetime.utcnow(),
            attributes=output_data,
            derived_from=[input_entity_id],
            generated_by=activity_id
        )
        entity.checksum = entity.compute_checksum()
        
        self.current_record.add_entity(entity)
        return entity_id
    
    def track_model(
        self,
        model_name: str,
        model_provider: str,
        version: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Track LLM model usage"""
        if not self.current_record:
            raise ValueError("No active provenance session")
        
        agent_id = f"agent_llm_{model_name}"
        if agent_id not in self.current_record.agents:
            agent = ProvenanceAgent(
                id=agent_id,
                label=f"LLM: {model_name}",
                agent_type="llm",
                version=version,
                model_name=model_name,
                model_provider=model_provider,
                temperature=config.get("temperature") if config else None
            )
            self.current_record.add_agent(agent)
        
        return agent_id
    
    def track_ontology_version(self, ontology_name: str, version: str):
        """Track ontology version used"""
        if not self.current_record:
            return
        
        self.current_record.ontology_versions[ontology_name] = version
        logger.debug(f"[Provenance] Using {ontology_name} version {version}")
    
    def end_session(self) -> ProvenanceRecord:
        """
        End current provenance session and return record
        """
        if not self.current_record:
            raise ValueError("No active provenance session")
        
        record = self.current_record
        self.records_history[record.record_id] = record
        self.current_record = None
        
        logger.info(f"[Provenance] Ended session {record.workflow_session_id}")
        return record
    
    def get_record(self, record_id: str) -> Optional[ProvenanceRecord]:
        """Retrieve historical provenance record"""
        return self.records_history.get(record_id)
    
    def query_by_patient(self, patient_id: str) -> List[ProvenanceRecord]:
        """Get all provenance records for a patient"""
        return [
            record for record in self.records_history.values()
            if record.patient_id == patient_id
        ]
    
    def export_record(self, record_id: str) -> Dict[str, Any]:
        """Export provenance record for audit/compliance"""
        record = self.get_record(record_id)
        if not record:
            return {}
        
        return record.to_dict()


# Global provenance tracker instance
_global_tracker: Optional[ProvenanceTracker] = None


def get_provenance_tracker() -> ProvenanceTracker:
    """Get or create global provenance tracker"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ProvenanceTracker()
    return _global_tracker
