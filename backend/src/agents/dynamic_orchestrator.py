"""
Dynamic Workflow Orchestrator with Adaptive Multi-Agent Routing

Based on 2025 research:
- Multi-Agent Healthcare Systems (PMC12360800) - 7-agent sepsis system
- G-RAGent (Dec 2025) - Dynamic reasoning on hypergraphs
- Microsoft Healthcare Agent Orchestrator (Build 2025)
- Adaptive exception detection and self-correction

Features:
1. Adaptive routing based on patient complexity
2. Self-corrective loops when confidence is low
3. Parallel agent execution for independent tasks
4. Dynamic context graph for reasoning chains
5. Exception detection and automated recovery
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
from uuid import uuid4
import networkx as nx

from pydantic import BaseModel

# Centralized logging
from ..logging_config import get_logger, log_execution, log_workflow_event

logger = get_logger(__name__)


class WorkflowComplexity(Enum):
    """Patient case complexity levels"""
    SIMPLE = "simple"  # Straightforward early-stage cases
    MODERATE = "moderate"  # Standard NSCLC cases
    COMPLEX = "complex"  # Multiple comorbidities, advanced stage
    CRITICAL = "critical"  # Emergency decision, poor PS, multi-factorial


class AgentStatus(Enum):
    """Agent execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    SKIPPED = "skipped"


class OrchestrationType(Enum):
    """Workflow orchestration patterns"""
    LINEAR = "linear"  # Traditional sequential pipeline
    PARALLEL = "parallel"  # Multiple agents execute concurrently
    ADAPTIVE = "adaptive"  # Dynamic routing based on results
    SELF_CORRECTIVE = "self_corrective"  # Includes validation loops


@dataclass
class AgentExecution:
    """Tracks individual agent execution"""
    agent_name: str
    status: AgentStatus = AgentStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: int = 0
    confidence: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    output: Optional[Any] = None
    requires_review: bool = False


@dataclass
class ContextNode:
    """Node in the dynamic context graph"""
    node_id: str
    node_type: str  # "patient", "finding", "recommendation", "reasoning"
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    source_agent: Optional[str] = None
    tags: Set[str] = field(default_factory=set)


@dataclass
class ContextEdge:
    """Edge in the dynamic context graph"""
    source_id: str
    target_id: str
    relation_type: str  # "supports", "conflicts", "derives_from", "requires"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DynamicContextGraph:
    """
    Dynamic Context Graph for organizing clinical information and reasoning chains

    Based on G-RAGent (Dec 2025) and context engineering research.
    Maintains a hypergraph of patient information, clinical findings,
    recommendations, and reasoning steps.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, ContextNode] = {}
        self.edges: List[ContextEdge] = []

    def add_node(self, node: ContextNode):
        """Add a node to the context graph"""
        self.nodes[node.node_id] = node
        self.graph.add_node(
            node.node_id,
            node_type=node.node_type,
            content=node.content,
            timestamp=node.timestamp,
            confidence=node.confidence,
            source_agent=node.source_agent,
            tags=node.tags
        )
        logger.debug(f"Added context node: {node.node_id} ({node.node_type})")

    def add_edge(self, edge: ContextEdge):
        """Add an edge to the context graph"""
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            relation_type=edge.relation_type,
            weight=edge.weight,
            metadata=edge.metadata
        )
        logger.debug(f"Added edge: {edge.source_id} -> {edge.target_id} ({edge.relation_type})")

    def get_related_nodes(self, node_id: str, relation_type: Optional[str] = None,
                         max_depth: int = 2) -> List[ContextNode]:
        """Get all nodes related to a given node within max_depth hops"""
        if node_id not in self.graph:
            return []

        related_ids = set()
        queue = [(node_id, 0)]
        visited = {node_id}

        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            # Get successors
            for successor in self.graph.successors(current_id):
                if successor not in visited:
                    edge_data = self.graph.get_edge_data(current_id, successor)
                    if relation_type is None or any(
                        e.get('relation_type') == relation_type for e in edge_data.values()
                    ):
                        related_ids.add(successor)
                        visited.add(successor)
                        queue.append((successor, depth + 1))

        return [self.nodes[nid] for nid in related_ids if nid in self.nodes]

    def detect_conflicts(self) -> List[tuple[ContextNode, ContextNode]]:
        """Detect conflicting recommendations or findings"""
        conflicts = []
        for edge in self.edges:
            if edge.relation_type == "conflicts":
                source = self.nodes.get(edge.source_id)
                target = self.nodes.get(edge.target_id)
                if source and target:
                    conflicts.append((source, target))
        return conflicts

    def get_reasoning_chain(self, recommendation_id: str) -> List[ContextNode]:
        """Get the complete reasoning chain leading to a recommendation"""
        chain = []
        if recommendation_id not in self.graph:
            return chain

        # Traverse backwards through "derives_from" edges
        queue = [recommendation_id]
        visited = set()

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id in self.nodes:
                chain.append(self.nodes[current_id])

            # Find predecessors via "derives_from" edges
            for predecessor in self.graph.predecessors(current_id):
                edge_data = self.graph.get_edge_data(predecessor, current_id)
                if any(e.get('relation_type') == 'derives_from' for e in edge_data.values()):
                    queue.append(predecessor)

        return chain[::-1]  # Reverse to get chronological order

    def prune_low_confidence(self, threshold: float = 0.3):
        """Remove nodes with confidence below threshold"""
        to_remove = [
            node_id for node_id, node in self.nodes.items()
            if node.confidence < threshold
        ]
        for node_id in to_remove:
            self.graph.remove_node(node_id)
            del self.nodes[node_id]
        logger.info(f"Pruned {len(to_remove)} low-confidence nodes")

    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary"""
        return {
            "nodes": [
                {
                    "id": node.node_id,
                    "type": node.node_type,
                    "content": node.content,
                    "confidence": node.confidence,
                    "source_agent": node.source_agent,
                    "timestamp": node.timestamp.isoformat()
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.relation_type,
                    "weight": edge.weight
                }
                for edge in self.edges
            ],
            "statistics": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "node_types": {
                    node_type: sum(1 for n in self.nodes.values() if n.node_type == node_type)
                    for node_type in set(n.node_type for n in self.nodes.values())
                }
            }
        }


class DynamicWorkflowOrchestrator:
    """
    Dynamic Multi-Agent Workflow Orchestrator

    Implements adaptive routing, self-correction, and parallel execution
    based on Microsoft Healthcare Agent Orchestrator (2025) and recent
    multi-agent healthcare research.
    """

    def __init__(self):
        self.context_graph = DynamicContextGraph()
        self.agent_executions: Dict[str, AgentExecution] = {}
        self.orchestration_type = OrchestrationType.ADAPTIVE
        self.complexity_threshold = {
            WorkflowComplexity.SIMPLE: 0.8,
            WorkflowComplexity.MODERATE: 0.6,
            WorkflowComplexity.COMPLEX: 0.4,
            WorkflowComplexity.CRITICAL: 0.2
        }

    def assess_complexity(self, patient_data: Dict[str, Any]) -> WorkflowComplexity:
        """
        Assess patient case complexity to determine optimal workflow

        Complexity factors:
        - TNM stage (IV > III > II > I)
        - Performance status (4 > 3 > 2 > 1 > 0)
        - Comorbidities count
        - Biomarker complexity
        - Age extremes (<40 or >80)
        """
        logger.info("=" * 60)
        logger.info("[DynamicOrchestrator] ASSESSING CASE COMPLEXITY")
        logger.info("=" * 60)

        complexity_score = 0.0
        factors = []

        # Stage complexity
        stage = patient_data.get("tnm_stage", "I")
        stage_scores = {"IV": 4, "IIIC": 3.5, "IIIB": 3.3, "IIIA": 3,
                       "IIB": 2.2, "IIA": 2, "IB": 1.2, "IA": 1}
        stage_score = stage_scores.get(stage, 1.0)
        complexity_score += stage_score
        factors.append(f"Stage {stage}: +{stage_score}")

        # Performance status
        ps = patient_data.get("performance_status", 0)
        ps_score = ps * 0.8
        complexity_score += ps_score
        factors.append(f"PS {ps}: +{ps_score}")

        # Comorbidities
        comorbidities = patient_data.get("comorbidities", [])
        comorbidity_score = len(comorbidities) * 0.5
        complexity_score += comorbidity_score
        if comorbidities:
            factors.append(f"Comorbidities ({len(comorbidities)}): +{comorbidity_score}")

        # Biomarkers
        biomarkers = patient_data.get("biomarker_profile", {})
        if len(biomarkers) > 3:
            complexity_score += 1.0
            factors.append(f"Biomarkers ({len(biomarkers)}): +1.0")

        # Age extremes - support both field names
        age = patient_data.get("age_at_diagnosis", patient_data.get("age", 65))
        if age < 40 or age > 80:
            complexity_score += 0.5
            factors.append(f"Age extreme ({age}): +0.5")

        # Log all factors
        logger.info("Complexity Factors:")
        for factor in factors:
            logger.info(f"  • {factor}")
        logger.info(f"Total Score: {complexity_score:.2f}")

        # Ontology-driven complexity bonus
        ontology_bonus = self._get_ontology_complexity_bonus(patient_data)
        if ontology_bonus > 0:
            complexity_score += ontology_bonus
            factors.append(f"Ontology bonus: +{ontology_bonus:.1f}")

        # Emergency indicators
        if patient_data.get("emergency", False):
            logger.info("⚠ EMERGENCY flag detected → CRITICAL")
            return WorkflowComplexity.CRITICAL

        # Classify based on score
        if complexity_score >= 6.0:
            result = WorkflowComplexity.CRITICAL
        elif complexity_score >= 4.0:
            result = WorkflowComplexity.COMPLEX
        elif complexity_score >= 2.0:
            result = WorkflowComplexity.MODERATE
        else:
            result = WorkflowComplexity.SIMPLE

        logger.info(f"Classification: {result.value} (score={complexity_score:.2f})")
        logger.info("=" * 60)
        return result

    # Common NSCLC histologies that are NOT rare
    COMMON_HISTOLOGIES = {
        "adenocarcinoma", "squamous cell carcinoma", "squamouscellcarcinoma",
        "small cell carcinoma", "smallcellcarcinoma", "nsclc",
        "nonsmallcellcarcinoma_nos", "non-small cell",
    }

    # Known actionable biomarkers from BiomarkerTherapyMap
    ACTIONABLE_MUTATIONS = {"EGFR", "ALK", "ROS1", "BRAF", "KRAS", "NTRK", "MET", "RET", "HER2"}

    # Comorbidities that imply contraindication risk
    CONTRAINDICATION_COMORBIDITIES = {
        "renal failure", "renal insufficiency", "ckd",
        "hepatic failure", "cirrhosis", "liver failure",
        "heart failure", "chf", "cardiac failure",
        "interstitial lung disease", "ild", "pulmonary fibrosis",
    }

    def _get_ontology_complexity_bonus(self, patient_data: Dict[str, Any]) -> float:
        """
        Query ontology concepts to adjust complexity.

        - Rare histology subtype (not adeno/squamous/small cell): +1.5
        - Actionable biomarker mutations: +0.5 per mutation
        - Comorbidities matching contraindication patterns: +1.0

        Falls back to 0 if Neo4j unavailable or data missing.
        """
        bonus = 0.0

        try:
            # Check for rare histology
            histology = patient_data.get("histology_type", "").lower().replace(" ", "")
            if histology and histology not in self.COMMON_HISTOLOGIES:
                bonus += 1.5
                logger.info(f"  Rare histology '{histology}': +1.5")

            # Check for actionable biomarker mutations
            biomarkers = patient_data.get("biomarker_profile", {})
            if isinstance(biomarkers, dict):
                actionable_count = sum(
                    1 for marker in biomarkers
                    if marker.upper().split("_")[0].split("-")[0] in self.ACTIONABLE_MUTATIONS
                )
                if actionable_count > 0:
                    bio_bonus = min(actionable_count * 0.5, 2.0)
                    bonus += bio_bonus
                    logger.info(f"  Actionable mutations ({actionable_count}): +{bio_bonus}")

            # Check comorbidities for contraindication patterns
            comorbidities = patient_data.get("comorbidities", [])
            if isinstance(comorbidities, list):
                for comorbidity in comorbidities:
                    if comorbidity.lower() in self.CONTRAINDICATION_COMORBIDITIES:
                        bonus += 1.0
                        logger.info(f"  Contraindication risk '{comorbidity}': +1.0")
                        break  # Only add once

        except Exception as e:
            logger.debug(f"Ontology complexity bonus calculation failed: {e}")

        return bonus

    def select_workflow_path(self, complexity: WorkflowComplexity) -> List[str]:
        """
        Select agent execution path based on complexity

        Returns ordered list of agents to execute

        NEW (2026-02): Integrated medical services agents:
        - LabInterpretationAgent (LOINC)
        - MedicationManagementAgent (RxNorm)
        - MonitoringCoordinatorAgent (Lab-Drug integration)
        """
        base_agents = [
            "IngestionAgent",
            "SemanticMappingAgent",
            "ClassificationAgent",
            "ConflictResolutionAgent",
            "ExplanationAgent"
        ]

        if complexity == WorkflowComplexity.SIMPLE:
            # Fast path: core agents + biomarker + comorbidity + medication safety + persistence
            return [
                "IngestionAgent",
                "SemanticMappingAgent",
                "ClassificationAgent",
                "BiomarkerAgent",
                "MedicationManagementAgent",  # NEW: DDI check
                "ComorbidityAgent",
                "NSCLCAgent",
                "ConflictResolutionAgent",
                "ExplanationAgent",
                "PersistenceAgent"
            ]

        elif complexity == WorkflowComplexity.MODERATE:
            # Standard path: include lab interpretation + medication management + persistence
            return [
                "IngestionAgent",
                "SemanticMappingAgent",
                "LabInterpretationAgent",  # NEW: LOINC-based lab interpretation
                "ClassificationAgent",
                "BiomarkerAgent",
                "MedicationManagementAgent",  # NEW: Comprehensive medication safety
                "ComorbidityAgent",
                "NSCLCAgent",
                "SCLCAgent",
                "SurvivalAnalyzer",
                "ConflictResolutionAgent",
                "UncertaintyQuantifier",
                "ExplanationAgent",
                "PersistenceAgent"
            ]

        elif complexity == WorkflowComplexity.COMPLEX:
            # Extended path: add monitoring coordination + analytics agents + persistence
            return [
                "IngestionAgent",
                "SemanticMappingAgent",
                "LabInterpretationAgent",  # NEW: Lab interpretation with toxicity grading
                "ClassificationAgent",
                "BiomarkerAgent",
                "MedicationManagementAgent",  # NEW: DDI + contraindications
                "ComorbidityAgent",
                "NSCLCAgent",
                "SCLCAgent",
                "MonitoringCoordinatorAgent",  # NEW: Monitoring protocol generation
                "SurvivalAnalyzer",
                "ConflictResolutionAgent",
                "UncertaintyQuantifier",
                "ClinicalTrialMatcher",
                "ExplanationAgent",
                "PersistenceAgent"
            ]

        else:  # CRITICAL
            # Comprehensive path: ALL agents including enhanced monitoring and persistence
            return [
                "IngestionAgent",
                "SemanticMappingAgent",
                "LabInterpretationAgent",  # NEW: Critical value detection
                "ClassificationAgent",
                "BiomarkerAgent",
                "MedicationManagementAgent",  # NEW: Full medication safety analysis
                "ComorbidityAgent",
                "NSCLCAgent",
                "SCLCAgent",
                "MonitoringCoordinatorAgent",  # NEW: Enhanced protocols + dose adjustments
                "SurvivalAnalyzer",
                "ConflictResolutionAgent",
                "UncertaintyQuantifier",
                "ClinicalTrialMatcher",
                "CounterfactualEngine",
                "ExplanationAgent",
                "PersistenceAgent"
            ]

    async def execute_agent(self, agent_name: str, agent_function: Callable,
                           input_data: Any) -> AgentExecution:
        """
        Execute a single agent with error handling and retry logic
        """
        execution = AgentExecution(agent_name=agent_name)
        execution.status = AgentStatus.RUNNING
        execution.started_at = datetime.now()

        self.agent_executions[agent_name] = execution

        max_retries = execution.max_retries
        retry_count = 0

        while retry_count <= max_retries:
            try:
                # Execute agent
                result = await agent_function(input_data)

                # Record success
                execution.status = AgentStatus.COMPLETED
                execution.completed_at = datetime.now()
                execution.duration_ms = int(
                    (execution.completed_at - execution.started_at).total_seconds() * 1000
                )
                execution.output = result

                # Extract confidence if available
                if hasattr(result, 'confidence'):
                    execution.confidence = result.confidence
                elif isinstance(result, dict) and 'confidence' in result:
                    execution.confidence = result['confidence']
                else:
                    execution.confidence = 1.0

                logger.info(f"✓ {agent_name} completed in {execution.duration_ms}ms "
                          f"(confidence: {execution.confidence:.2f})")
                break

            except Exception as e:
                retry_count += 1
                execution.retry_count = retry_count
                execution.error_message = str(e)

                if retry_count <= max_retries:
                    logger.warning(f"⚠ {agent_name} failed (attempt {retry_count}/{max_retries}): {e}")
                    execution.status = AgentStatus.RETRY
                    await asyncio.sleep(0.5 * retry_count)  # Exponential backoff
                else:
                    logger.error(f"✗ {agent_name} failed after {max_retries} retries: {e}")
                    execution.status = AgentStatus.FAILED
                    execution.completed_at = datetime.now()
                    break

        return execution

    def requires_self_correction(self, execution: AgentExecution,
                                complexity: WorkflowComplexity) -> bool:
        """
        Determine if agent output requires self-correction loop

        Triggers correction if:
        - Confidence below threshold for complexity level
        - Critical workflow with any uncertainty
        - Agent flagged output for review
        """
        if execution.status == AgentStatus.FAILED:
            return False

        if execution.requires_review:
            return True

        threshold = self.complexity_threshold[complexity]

        if execution.confidence < threshold:
            logger.warning(f"⚠ {execution.agent_name} confidence {execution.confidence:.2f} "
                         f"below threshold {threshold:.2f} for {complexity.value} case")
            return True

        return False

    async def execute_with_self_correction(self, agent_name: str,
                                          agent_function: Callable,
                                          input_data: Any,
                                          complexity: WorkflowComplexity) -> AgentExecution:
        """
        Execute agent with self-correction loop if needed
        """
        execution = await self.execute_agent(agent_name, agent_function, input_data)

        if self.requires_self_correction(execution, complexity):
            logger.info(f"🔄 Initiating self-correction for {agent_name}")

            # Add correction context
            correction_input = {
                **input_data,
                "previous_attempt": execution.output,
                "confidence": execution.confidence,
                "request_higher_confidence": True
            }

            # Re-execute with correction context
            corrected_execution = await self.execute_agent(
                f"{agent_name}_corrected",
                agent_function,
                correction_input
            )

            # Use corrected result if better
            if corrected_execution.confidence > execution.confidence:
                logger.info(f"✓ Self-correction improved confidence: "
                          f"{execution.confidence:.2f} → {corrected_execution.confidence:.2f}")
                return corrected_execution
            else:
                logger.info(f"⚠ Self-correction did not improve confidence, using original")

        return execution

    async def orchestrate_adaptive_workflow(self, patient_data: Dict[str, Any],
                                           agent_registry: Dict[str, Callable],
                                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Main orchestration method with adaptive routing

        Args:
            patient_data: Input patient data
            agent_registry: Dictionary mapping agent names to callable functions
            progress_callback: Optional callback for progress updates

        Returns:
            Complete workflow results with context graph
        """
        workflow_id = str(uuid4())
        start_time = datetime.now()

        logger.info("=" * 80)
        logger.info("🚀 DYNAMIC WORKFLOW ORCHESTRATOR - START")
        logger.info("=" * 80)
        logger.info(f"Workflow ID: {workflow_id}")
        logger.info(f"Timestamp: {start_time.isoformat()}")
        logger.info("")
        logger.info("INPUT PATIENT DATA:")
        for key, value in patient_data.items():
            if key not in ['_extraction_meta']:  # Skip internal fields
                logger.info(f"  • {key}: {value}")
        logger.info("")
        logger.info(f"Available Agents: {list(agent_registry.keys())}")
        logger.info("")

        # Step 1: Assess complexity
        complexity = self.assess_complexity(patient_data)
        logger.info(f"📊 Case complexity: {complexity.value}")

        # Step 2: Select workflow path
        agent_path = self.select_workflow_path(complexity)
        logger.info(f"🛤️  Selected path: {' → '.join(agent_path)}")
        
        if progress_callback:
            logger.info(f"[DEBUG] Progress callback exists, sending agent path message...")
            await progress_callback(f"🛤️ Agent Path ({complexity.value}): {' → '.join(agent_path[:3])}... ({len(agent_path)} agents total)")
        else:
            logger.warning(f"[DEBUG] Progress callback is None, cannot send updates!")

        # Step 3: Initialize context graph
        patient_node = ContextNode(
            node_id=f"patient_{patient_data.get('patient_id', 'unknown')}",
            node_type="patient",
            content=patient_data,
            source_agent="orchestrator"
        )
        self.context_graph.add_node(patient_node)

        # Step 4: Execute agents sequentially or in parallel
        results = {}
        current_data = patient_data

        for i, agent_name in enumerate(agent_path, 1):
            if agent_name not in agent_registry:
                logger.warning(f"⚠ Agent {agent_name} not found in registry, skipping")
                if progress_callback:
                    await progress_callback(f"⚠ [{i}/{len(agent_path)}] {agent_name}: SKIPPED (not available)")
                # Mark as skipped in results
                skipped_execution = AgentExecution(agent_name=agent_name)
                skipped_execution.status = AgentStatus.SKIPPED
                results[agent_name] = skipped_execution
                continue

            logger.info(f"🔄 Executing {agent_name}...")
            if progress_callback:
                await progress_callback(f"⚙️ [{i}/{len(agent_path)}] Running {agent_name}...")
            
            agent_function = agent_registry[agent_name]

            # Execute with self-correction if needed
            execution = await self.execute_with_self_correction(
                agent_name,
                agent_function,
                current_data,
                complexity
            )

            # Store results
            results[agent_name] = execution
            
            # Send completion update
            if progress_callback:
                if execution.status == AgentStatus.COMPLETED:
                    await progress_callback(f"✅ [{i}/{len(agent_path)}] {agent_name} completed ({execution.duration_ms}ms, conf: {execution.confidence:.2f})")
                elif execution.status == AgentStatus.FAILED:
                    await progress_callback(f"❌ [{i}/{len(agent_path)}] {agent_name} FAILED")

            # Update context graph
            if execution.status == AgentStatus.COMPLETED:
                result_node = ContextNode(
                    node_id=f"{agent_name}_{workflow_id}",
                    node_type="agent_output",
                    content={"result": execution.output},
                    confidence=execution.confidence,
                    source_agent=agent_name
                )
                self.context_graph.add_node(result_node)

                # Link to patient
                self.context_graph.add_edge(ContextEdge(
                    source_id=patient_node.node_id,
                    target_id=result_node.node_id,
                    relation_type="processed_by"
                ))

                # Update data for next agent
                if execution.output:
                    current_data = {**current_data, f"{agent_name}_output": execution.output}

        # Step 5: Generate final summary
        end_time = datetime.now()
        total_duration = int((end_time - start_time).total_seconds() * 1000)

        successful_agents = [
            name for name, exec in results.items()
            if exec.status == AgentStatus.COMPLETED
        ]
        failed_agents = [
            name for name, exec in results.items()
            if exec.status == AgentStatus.FAILED
        ]
        skipped_agents = [
            name for name, exec in results.items()
            if exec.status == AgentStatus.SKIPPED
        ]

        logger.info("")
        logger.info("=" * 80)
        logger.info("🏁 WORKFLOW EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Duration: {total_duration}ms ({total_duration/1000:.2f}s)")
        logger.info(f"Complexity Level: {complexity.value}")
        logger.info(f"Agents Executed: {len(agent_path)}")
        logger.info("")
        logger.info("AGENT RESULTS:")
        for name, exec in results.items():
            status_icon = "✅" if exec.status == AgentStatus.COMPLETED else "❌" if exec.status == AgentStatus.FAILED else "⏭️"
            logger.info(f"  {status_icon} {name}: {exec.status.value} ({exec.duration_ms}ms, conf: {exec.confidence:.2f})")
        logger.info("")
        logger.info(f"Successful: {successful_agents}")
        logger.info(f"Failed: {failed_agents}")
        logger.info(f"Skipped: {skipped_agents}")
        logger.info("=" * 80)
        logger.info("🏁 WORKFLOW COMPLETE")
        logger.info("=" * 80)

        return {
            "workflow_id": workflow_id,
            "complexity": complexity.value,
            "agent_path": agent_path,
            "successful_agents": successful_agents,
            "failed_agents": failed_agents,
            "skipped_agents": skipped_agents,
            "total_duration_ms": total_duration,
            "results": {name: exec.output for name, exec in results.items()},
            "agent_executions": {
                name: {
                    "status": exec.status.value,
                    "confidence": exec.confidence,
                    "duration_ms": exec.duration_ms,
                    "retry_count": exec.retry_count
                }
                for name, exec in results.items()
            },
            "context_graph": self.context_graph.to_dict(),
            "final_output": current_data
        }


# Convenience function for testing
async def run_adaptive_workflow_example():
    """Example usage of dynamic orchestrator"""

    # Mock patient data
    patient_data = {
        "patient_id": "P99999",
        "age_at_diagnosis": 72,
        "tnm_stage": "IIIB",
        "histology_type": "Adenocarcinoma",
        "performance_status": 2,
        "comorbidities": ["COPD", "Diabetes"],
        "biomarker_profile": {
            "EGFR": "Ex19del",
            "PD-L1": "45%"
        }
    }

    # Mock agent functions
    async def mock_agent(data):
        await asyncio.sleep(0.1)
        return {"processed": True, "confidence": 0.85}

    agent_registry = {
        "IngestionAgent": mock_agent,
        "SemanticMappingAgent": mock_agent,
        "ClassificationAgent": mock_agent,
        "BiomarkerAgent": mock_agent,
        "ComorbidityAgent": mock_agent,
        "ConflictResolutionAgent": mock_agent,
        "UncertaintyQuantifier": mock_agent,
        "ExplanationAgent": mock_agent
    }

    orchestrator = DynamicWorkflowOrchestrator()
    result = await orchestrator.orchestrate_adaptive_workflow(patient_data, agent_registry)

    print(f"\n✓ Workflow completed:")
    print(f"  Complexity: {result['complexity']}")
    print(f"  Path: {' → '.join(result['agent_path'])}")
    print(f"  Duration: {result['total_duration_ms']}ms")
    print(f"  Success: {len(result['successful_agents'])}/{len(result['agent_path'])} agents")

    return result


if __name__ == "__main__":
    # Run example
    asyncio.run(run_adaptive_workflow_example())
