"""
Agent Transparency Service

Provides real-time visibility into agent execution with confidence scores,
execution graphs, and detailed status updates.
"""

import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
from enum import Enum
import json


class AgentStatus(Enum):
    """Agent execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentExecutionNode:
    """Represents a single agent execution in the workflow."""
    
    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        description: str,
        dependencies: List[str] = None
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.description = description
        self.dependencies = dependencies or []
        
        self.status = AgentStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.confidence_score: Optional[float] = None
        self.output_summary: Optional[str] = None
        self.error: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    def start(self):
        """Mark agent as started."""
        self.status = AgentStatus.RUNNING
        self.start_time = datetime.now()
    
    def complete(self, confidence: float, output_summary: str, metadata: Dict[str, Any] = None):
        """Mark agent as completed with results."""
        self.status = AgentStatus.COMPLETED
        self.end_time = datetime.now()
        self.confidence_score = confidence
        self.output_summary = output_summary
        if metadata:
            self.metadata.update(metadata)
    
    def fail(self, error: str):
        """Mark agent as failed."""
        self.status = AgentStatus.FAILED
        self.end_time = datetime.now()
        self.error = error
    
    def skip(self, reason: str):
        """Mark agent as skipped."""
        self.status = AgentStatus.SKIPPED
        self.metadata['skip_reason'] = reason
    
    def duration_ms(self) -> Optional[int]:
        """Calculate execution duration in milliseconds."""
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'description': self.description,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_ms': self.duration_ms(),
            'confidence_score': self.confidence_score,
            'output_summary': self.output_summary,
            'error': self.error,
            'dependencies': self.dependencies,
            'metadata': self.metadata
        }


class WorkflowExecutionGraph:
    """Tracks the entire workflow execution graph."""
    
    def __init__(self, workflow_id: str, complexity: str):
        self.workflow_id = workflow_id
        self.complexity = complexity
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        
        self.nodes: Dict[str, AgentExecutionNode] = {}
        self.execution_order: List[str] = []
        self.current_agent: Optional[str] = None
    
    def add_agent(self, node: AgentExecutionNode):
        """Add agent to execution graph."""
        self.nodes[node.agent_id] = node
    
    def start_agent(self, agent_id: str):
        """Start executing an agent."""
        if agent_id in self.nodes:
            self.nodes[agent_id].start()
            self.current_agent = agent_id
            self.execution_order.append(agent_id)
    
    def complete_agent(
        self, 
        agent_id: str, 
        confidence: float, 
        output_summary: str,
        metadata: Dict[str, Any] = None
    ):
        """Complete agent execution."""
        if agent_id in self.nodes:
            self.nodes[agent_id].complete(confidence, output_summary, metadata)
    
    def fail_agent(self, agent_id: str, error: str):
        """Mark agent as failed."""
        if agent_id in self.nodes:
            self.nodes[agent_id].fail(error)
    
    def skip_agent(self, agent_id: str, reason: str):
        """Skip agent execution."""
        if agent_id in self.nodes:
            self.nodes[agent_id].skip(reason)
    
    def complete_workflow(self):
        """Mark entire workflow as completed."""
        self.end_time = datetime.now()
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current workflow progress."""
        total = len(self.nodes)
        completed = sum(1 for n in self.nodes.values() if n.status == AgentStatus.COMPLETED)
        failed = sum(1 for n in self.nodes.values() if n.status == AgentStatus.FAILED)
        running = sum(1 for n in self.nodes.values() if n.status == AgentStatus.RUNNING)
        
        return {
            'total_agents': total,
            'completed': completed,
            'failed': failed,
            'running': running,
            'pending': total - completed - failed - running,
            'progress_percent': int((completed / total) * 100) if total > 0 else 0
        }
    
    def get_confidence_scores(self) -> Dict[str, float]:
        """Get confidence scores for all completed agents."""
        return {
            agent_id: node.confidence_score
            for agent_id, node in self.nodes.items()
            if node.confidence_score is not None
        }
    
    def get_overall_confidence(self) -> Optional[float]:
        """Calculate overall workflow confidence (weighted average)."""
        scores = [
            node.confidence_score 
            for node in self.nodes.values() 
            if node.confidence_score is not None
        ]
        
        if not scores:
            return None
        
        return sum(scores) / len(scores)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'workflow_id': self.workflow_id,
            'complexity': self.complexity,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_ms': self._total_duration_ms(),
            'progress': self.get_progress(),
            'overall_confidence': self.get_overall_confidence(),
            'agents': {
                agent_id: node.to_dict()
                for agent_id, node in self.nodes.items()
            },
            'execution_order': self.execution_order,
            'current_agent': self.current_agent
        }
    
    def _total_duration_ms(self) -> Optional[int]:
        """Calculate total workflow duration."""
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return None


class TransparencyService:
    """Service for streaming agent execution transparency."""
    
    def __init__(self):
        self.active_workflows: Dict[str, WorkflowExecutionGraph] = {}
    
    def create_workflow_graph(
        self, 
        workflow_id: str, 
        complexity: str,
        agent_definitions: List[Dict[str, Any]]
    ) -> WorkflowExecutionGraph:
        """
        Create a new workflow execution graph.
        
        Args:
            workflow_id: Unique workflow identifier
            complexity: Workflow complexity level
            agent_definitions: List of agent definitions with name, description, dependencies
            
        Returns:
            Created workflow graph
        """
        graph = WorkflowExecutionGraph(workflow_id, complexity)
        
        # Add all agents to graph
        for agent_def in agent_definitions:
            node = AgentExecutionNode(
                agent_id=agent_def['id'],
                agent_name=agent_def['name'],
                description=agent_def.get('description', ''),
                dependencies=agent_def.get('dependencies', [])
            )
            graph.add_agent(node)
        
        self.active_workflows[workflow_id] = graph
        return graph
    
    def get_workflow_graph(self, workflow_id: str) -> Optional[WorkflowExecutionGraph]:
        """Get existing workflow graph."""
        return self.active_workflows.get(workflow_id)
    
    async def stream_agent_updates(
        self, 
        workflow_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream real-time agent execution updates.
        
        Args:
            workflow_id: Workflow to stream updates for
            
        Yields:
            SSE-formatted agent status updates
        """
        graph = self.active_workflows.get(workflow_id)
        if not graph:
            yield {'error': 'Workflow not found'}
            return
        
        last_state = None
        
        # Stream updates until workflow completes
        while graph.end_time is None:
            current_state = graph.to_dict()
            
            # Only send if state changed
            if current_state != last_state:
                yield {
                    'type': 'workflow_update',
                    'data': current_state
                }
                last_state = current_state
            
            await asyncio.sleep(0.5)  # 500ms polling
        
        # Send final state
        yield {
            'type': 'workflow_complete',
            'data': graph.to_dict()
        }
    
    def cleanup_workflow(self, workflow_id: str):
        """Remove completed workflow from active tracking."""
        if workflow_id in self.active_workflows:
            del self.active_workflows[workflow_id]


# Agent confidence calculation utilities

class ConfidenceCalculator:
    """Calculate confidence scores for agent outputs."""
    
    @staticmethod
    def calculate_extraction_confidence(
        extracted_data: Dict[str, Any],
        source_text: str
    ) -> float:
        """
        Calculate confidence for data extraction.
        
        Based on:
        - Completeness of extracted fields
        - Presence of validation keywords
        - Data consistency
        """
        score = 0.0
        
        # Completeness (40%)
        required_fields = ['demographics', 'diagnosis', 'biomarkers']
        found_fields = sum(1 for field in required_fields if extracted_data.get(field))
        completeness = (found_fields / len(required_fields)) * 0.4
        score += completeness
        
        # Demographics confidence (20%)
        demographics = extracted_data.get('demographics', {})
        if demographics.get('age') and demographics.get('sex'):
            score += 0.2
        elif demographics.get('age') or demographics.get('sex'):
            score += 0.1
        
        # Diagnosis confidence (20%)
        diagnosis = extracted_data.get('diagnosis', {})
        if diagnosis.get('cancer_type') and diagnosis.get('stage'):
            score += 0.2
        elif diagnosis.get('cancer_type') or diagnosis.get('stage'):
            score += 0.1
        
        # Biomarker confidence (20%)
        biomarkers = extracted_data.get('biomarkers', {})
        if len(biomarkers) >= 2:
            score += 0.2
        elif len(biomarkers) == 1:
            score += 0.1
        
        return min(score, 1.0)
    
    @staticmethod
    def calculate_classification_confidence(
        classification: str,
        evidence: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence for NSCLC/SCLC classification.
        
        Based on evidence strength and consistency.
        """
        base_confidence = 0.5
        
        # Histology evidence (+30%)
        if evidence.get('histology'):
            histology = evidence['histology'].lower()
            if any(term in histology for term in ['adenocarcinoma', 'squamous', 'large cell']):
                base_confidence += 0.3
            elif 'small cell' in histology:
                base_confidence += 0.3
        
        # Biomarker evidence (+20%)
        biomarkers = evidence.get('biomarkers', {})
        if classification == 'NSCLC' and any(biomarkers.get(b) for b in ['egfr', 'alk', 'ros1']):
            base_confidence += 0.2
        
        # Stage consistency (+10%)
        stage = evidence.get('stage', '')
        if (classification == 'SCLC' and 'limited' in stage.lower()) or \
           (classification == 'SCLC' and 'extensive' in stage.lower()):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    @staticmethod
    def calculate_biomarker_confidence(
        biomarker_results: Dict[str, Any],
        test_methods: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate confidence for each biomarker result.
        
        Based on test method and result clarity.
        """
        confidences = {}
        
        high_confidence_methods = ['ngs', 'next generation sequencing', 'fish', 'ihc']
        
        for biomarker, value in biomarker_results.items():
            base = 0.6
            
            # Test method quality
            method = test_methods.get(biomarker, '').lower()
            if any(m in method for m in high_confidence_methods):
                base += 0.2
            
            # Result clarity
            if value in ['positive', 'negative', 'detected', 'not detected']:
                base += 0.2
            elif isinstance(value, (int, float)):  # Quantitative
                base += 0.15
            
            confidences[biomarker] = min(base, 1.0)
        
        return confidences
    
    @staticmethod
    def calculate_overall_recommendation_confidence(
        agent_confidences: Dict[str, float],
        conflicts: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall recommendation confidence.
        
        Penalized by conflicts and low individual agent scores.
        """
        if not agent_confidences:
            return 0.5
        
        # Average agent confidence
        avg_confidence = sum(agent_confidences.values()) / len(agent_confidences)
        
        # Penalize for conflicts (-10% per conflict)
        conflict_penalty = min(len(conflicts) * 0.1, 0.5)
        
        # Penalize for low-confidence agents
        low_confidence_agents = sum(1 for conf in agent_confidences.values() if conf < 0.6)
        low_conf_penalty = (low_confidence_agents / len(agent_confidences)) * 0.2
        
        final_confidence = avg_confidence - conflict_penalty - low_conf_penalty
        
        return max(final_confidence, 0.1)  # Never go below 10%
