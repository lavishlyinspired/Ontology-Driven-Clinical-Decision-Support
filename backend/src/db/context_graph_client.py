"""
Lung Cancer Context Graph Client
================================

Neo4j client for context graph operations specific to lung cancer clinical decisions.
Handles patients, treatment decisions, biomarkers, and causal relationships.

Based on the Context Graph architecture from context-graph-demo, adapted for
oncology clinical decision support.

Key Concepts:
- Decision Traces: Full reasoning chain for treatment recommendations
- Causal Chains: Links between decisions (e.g., biomarker result → targeted therapy)
- Precedent Search: Find similar past cases using FastRP + semantic embeddings
"""

import uuid
from datetime import date, datetime
from typing import Any, Optional, List, Dict
import os
import logging

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from ..logging_config import get_logger

logger = get_logger(__name__)


def convert_neo4j_value(value: Any) -> Any:
    """Convert Neo4j types to JSON-serializable Python types."""
    from neo4j.time import Date as Neo4jDate, DateTime as Neo4jDateTime

    if isinstance(value, Neo4jDateTime):
        return value.isoformat()
    elif isinstance(value, Neo4jDate):
        return value.isoformat()
    elif isinstance(value, (datetime, date)):
        return value.isoformat()
    elif isinstance(value, list):
        return [convert_neo4j_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: convert_neo4j_value(v) for k, v in value.items()}
    return value


def convert_node_properties(props: dict) -> dict:
    """Convert all properties in a node to JSON-serializable types."""
    return {k: convert_neo4j_value(v) for k, v in props.items()}


# Pydantic models for type safety
from pydantic import BaseModel
from typing import List, Optional


class GraphNode(BaseModel):
    """Node in the context graph"""
    id: str
    labels: List[str]
    properties: Dict[str, Any]


class GraphRelationship(BaseModel):
    """Relationship in the context graph"""
    id: str
    type: str
    startNodeId: str
    endNodeId: str
    properties: Dict[str, Any] = {}


class GraphData(BaseModel):
    """Graph data for visualization"""
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]


class TreatmentDecision(BaseModel):
    """Treatment decision with full context"""
    id: str
    decision_type: str  # "treatment_recommendation", "biomarker_test", "staging", etc.
    category: str  # "NSCLC", "SCLC", "immunotherapy", "targeted_therapy"
    status: str  # "recommended", "approved", "administered", "declined"
    reasoning: str
    confidence_score: float
    risk_factors: List[str] = []
    guideline_references: List[str] = []
    decision_timestamp: Optional[str] = None


class LCAContextGraphClient:
    """
    Neo4j client for Lung Cancer Assistant context graph operations.

    Manages:
    - Patient nodes and their clinical history
    - Treatment decision traces with full reasoning
    - Biomarker findings and their implications
    - Causal relationships between decisions
    - Guideline/policy references
    """

    def __init__(self, uri: str = None, username: str = None, password: str = None, database: str = None):
        """Initialize the context graph client."""
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")

        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )
            logger.info(f"[ContextGraphClient] Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"[ContextGraphClient] Failed to connect: {e}")
            self.driver = None

    def close(self):
        """Close the driver connection."""
        if self.driver:
            self.driver.close()

    def verify_connectivity(self) -> bool:
        """Verify connection to Neo4j."""
        if not self.driver:
            return False
        try:
            self.driver.verify_connectivity()
            return True
        except ServiceUnavailable:
            return False

    # ============================================
    # PATIENT OPERATIONS
    # ============================================

    def search_patients(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for patients by name, ID, or clinical characteristics."""
        if not self.driver:
            return []

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (p:Patient)
                WHERE toLower(p.name) CONTAINS toLower($query)
                   OR p.patient_id CONTAINS $query
                   OR toLower(p.tnm_stage) CONTAINS toLower($query)
                   OR toLower(p.histology_type) CONTAINS toLower($query)
                OPTIONAL MATCH (d:TreatmentDecision)-[:ABOUT]->(p)
                RETURN p.patient_id AS id,
                       p.name AS name,
                       p.age_at_diagnosis AS age,
                       p.tnm_stage AS stage,
                       p.histology_type AS histology,
                       p.performance_status AS ps,
                       count(DISTINCT d) AS decision_count
                ORDER BY decision_count DESC
                LIMIT $limit
                """,
                {"query": query, "limit": limit},
            )
            return [dict(record) for record in result]

    def get_patient(self, patient_id: str) -> Optional[Dict]:
        """Get a patient by ID with related entities."""
        if not self.driver:
            return None

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (p:Patient {patient_id: $patient_id})
                WITH p LIMIT 1
                OPTIONAL MATCH (p)-[:HAS_BIOMARKER]->(b:Biomarker)
                OPTIONAL MATCH (p)-[:HAS_COMORBIDITY]->(c:Comorbidity)
                RETURN p {
                    .*,
                    biomarkers: collect(DISTINCT b {.*}),
                    comorbidities: collect(DISTINCT c {.*})
                } AS patient
                """,
                {"patient_id": patient_id},
            )
            record = result.single()
            return convert_node_properties(record["patient"]) if record else None

    def get_patient_decisions(
        self,
        patient_id: str,
        decision_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Get all treatment decisions made for a patient."""
        if not self.driver:
            return []

        type_filter = "AND d.decision_type = $decision_type" if decision_type else ""

        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"""
                MATCH (d:TreatmentDecision)-[:ABOUT]->(p:Patient {{patient_id: $patient_id}})
                WHERE true {type_filter}
                OPTIONAL MATCH (d)-[:APPLIED_GUIDELINE]->(g:Guideline)
                OPTIONAL MATCH (d)-[:BASED_ON]->(b:Biomarker)
                WITH d, collect(DISTINCT g.name) AS guidelines, collect(DISTINCT b.marker_type) AS biomarkers
                RETURN d {{
                    .*,
                    guidelines_applied: guidelines,
                    biomarkers_considered: biomarkers
                }} AS decision
                ORDER BY d.decision_timestamp DESC
                LIMIT $limit
                """,
                {
                    "patient_id": patient_id,
                    "decision_type": decision_type,
                    "limit": limit,
                },
            )
            return [convert_node_properties(record["decision"]) for record in result]

    # ============================================
    # TREATMENT DECISION OPERATIONS
    # ============================================

    def record_decision(
        self,
        decision_type: str,
        category: str,
        reasoning: str,
        patient_id: Optional[str] = None,
        treatment: Optional[str] = None,
        biomarker_ids: List[str] = None,
        guideline_ids: List[str] = None,
        precedent_ids: List[str] = None,
        risk_factors: List[str] = None,
        confidence_score: float = 0.8,
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> str:
        """
        Record a new treatment decision with full context.

        This creates a Decision Trace - capturing the full reasoning,
        linked biomarkers, applicable guidelines, and similar precedents.
        """
        if not self.driver:
            logger.error("[ContextGraphClient] Cannot record decision - no driver")
            return ""

        decision_id = str(uuid.uuid4())
        risk_factors = risk_factors or []
        biomarker_ids = biomarker_ids or []
        guideline_ids = guideline_ids or []
        precedent_ids = precedent_ids or []

        logger.info(f"[ContextGraphClient] Recording decision {decision_id}")
        logger.info(f"  • Type: {decision_type}")
        logger.info(f"  • Category: {category}")
        logger.info(f"  • Treatment: {treatment}")
        logger.info(f"  • Confidence: {confidence_score}")

        with self.driver.session(database=self.database) as session:
            # Create the decision node
            session.run(
                """
                CREATE (d:TreatmentDecision {
                    id: $decision_id,
                    decision_type: $decision_type,
                    category: $category,
                    status: 'recommended',
                    decision_timestamp: datetime(),
                    reasoning: $reasoning,
                    reasoning_summary: $reasoning_summary,
                    treatment: $treatment,
                    confidence_score: $confidence_score,
                    risk_factors: $risk_factors,
                    session_id: $session_id,
                    agent_name: $agent_name,
                    created_at: datetime()
                })
                """,
                {
                    "decision_id": decision_id,
                    "decision_type": decision_type,
                    "category": category,
                    "reasoning": reasoning,
                    "reasoning_summary": reasoning[:200] + "..." if len(reasoning) > 200 else reasoning,
                    "treatment": treatment,
                    "confidence_score": confidence_score,
                    "risk_factors": risk_factors,
                    "session_id": session_id,
                    "agent_name": agent_name,
                },
            )

            # Link to patient
            if patient_id:
                session.run(
                    """
                    MATCH (d:TreatmentDecision {id: $decision_id})
                    MATCH (p:Patient {patient_id: $patient_id})
                    MERGE (d)-[:ABOUT]->(p)
                    """,
                    {"decision_id": decision_id, "patient_id": patient_id},
                )

            # Link to biomarkers
            for biomarker_id in biomarker_ids:
                session.run(
                    """
                    MATCH (d:TreatmentDecision {id: $decision_id})
                    MATCH (b:Biomarker {id: $biomarker_id})
                    MERGE (d)-[:BASED_ON]->(b)
                    """,
                    {"decision_id": decision_id, "biomarker_id": biomarker_id},
                )

            # Link to guidelines
            for guideline_id in guideline_ids:
                session.run(
                    """
                    MATCH (d:TreatmentDecision {id: $decision_id})
                    MATCH (g:Guideline {id: $guideline_id})
                    MERGE (d)-[:APPLIED_GUIDELINE]->(g)
                    """,
                    {"decision_id": decision_id, "guideline_id": guideline_id},
                )

            # Link to precedents (similar past decisions)
            for precedent_id in precedent_ids:
                session.run(
                    """
                    MATCH (d:TreatmentDecision {id: $decision_id})
                    MATCH (p:TreatmentDecision {id: $precedent_id})
                    MERGE (d)-[:FOLLOWED_PRECEDENT]->(p)
                    """,
                    {"decision_id": decision_id, "precedent_id": precedent_id},
                )

            # Create PROV-O provenance record
            prov_id = str(uuid.uuid4())
            session.run(
                """
                CREATE (prov:ProvenanceRecord {
                    id: $prov_id,
                    activity: $activity,
                    agent: $agent_name,
                    entity: $decision_id,
                    started_at: datetime(),
                    ended_at: datetime()
                })
                WITH prov
                MATCH (d:TreatmentDecision {id: $decision_id})
                SET d.prov_generated_at = datetime(),
                    d.prov_was_derived_from = $patient_id,
                    d.prov_was_attributed_to = $agent_name
                MERGE (d)-[:PROV_WAS_GENERATED_BY]->(prov)
                """,
                {
                    "prov_id": prov_id,
                    "activity": f"decision_{decision_type}",
                    "agent_name": agent_name or "system",
                    "decision_id": decision_id,
                    "patient_id": patient_id or "",
                },
            )

        # Ontology enrichment: link patient to ontology class if available
        if patient_id:
            self._enrich_with_ontology(patient_id, "Patient")

        logger.info(f"[ContextGraphClient] Decision {decision_id} recorded with PROV-O provenance")
        return decision_id

    def get_decision(self, decision_id: str) -> Optional[Dict]:
        """Get a decision by ID with full context."""
        if not self.driver:
            return None

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (d:TreatmentDecision {id: $decision_id})
                WITH d LIMIT 1
                OPTIONAL MATCH (d)-[:ABOUT]->(patient:Patient)
                OPTIONAL MATCH (d)-[:APPLIED_GUIDELINE]->(guideline:Guideline)
                OPTIONAL MATCH (d)-[:BASED_ON]->(biomarker:Biomarker)
                OPTIONAL MATCH (d)-[:FOLLOWED_PRECEDENT]->(precedent:TreatmentDecision)
                OPTIONAL MATCH (d)-[:CAUSED]->(effect:TreatmentDecision)
                RETURN d {
                    .*,
                    patient: patient {.*},
                    guidelines: collect(DISTINCT guideline {.*}),
                    biomarkers: collect(DISTINCT biomarker {.*}),
                    precedents: collect(DISTINCT precedent {.id, .decision_type, .treatment, .confidence_score}),
                    effects: collect(DISTINCT effect {.id, .decision_type, .treatment})
                } AS decision
                """,
                {"decision_id": decision_id},
            )
            record = result.single()
            return convert_node_properties(record["decision"]) if record else None

    def get_provenance_chain(self, decision_id: str) -> Dict:
        """
        Get the PROV-O provenance chain for a treatment decision.

        Returns provenance records linked via PROV_WAS_GENERATED_BY,
        plus the decision's prov_was_derived_from and prov_was_attributed_to.
        """
        if not self.driver:
            return {"decision_id": decision_id, "provenance": []}

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (d:TreatmentDecision {id: $decision_id})
                OPTIONAL MATCH (d)-[:PROV_WAS_GENERATED_BY]->(prov:ProvenanceRecord)
                RETURN d {
                    .id, .decision_type, .treatment,
                    .prov_generated_at, .prov_was_derived_from, .prov_was_attributed_to
                } AS decision,
                collect(prov {.*}) AS provenance_records
                """,
                {"decision_id": decision_id},
            )
            record = result.single()
            if not record:
                return {"decision_id": decision_id, "provenance": []}

            return {
                "decision_id": decision_id,
                "decision": convert_node_properties(record["decision"]),
                "provenance_records": [
                    convert_node_properties(p) for p in record["provenance_records"]
                ],
            }

    def get_causal_chain(
        self,
        decision_id: str,
        direction: str = "both",
        depth: int = 3,
    ) -> Dict:
        """
        Trace the causal chain of a treatment decision.

        Returns causes (what led to this decision) and effects (what this decision led to).
        """
        if not self.driver:
            return {"decision_id": decision_id, "causes": [], "effects": [], "depth": depth}

        with self.driver.session(database=self.database) as session:
            causes = []
            effects = []

            if direction in ("both", "causes"):
                result = session.run(
                    f"""
                    MATCH (d:TreatmentDecision {{id: $decision_id}})
                    MATCH path = (cause:TreatmentDecision)-[:CAUSED|INFLUENCED*1..{depth}]->(d)
                    WITH cause, length(path) AS distance
                    RETURN cause {{.*, distance: distance}} AS decision
                    ORDER BY distance
                    """,
                    {"decision_id": decision_id},
                )
                causes = [convert_node_properties(record["decision"]) for record in result]

            if direction in ("both", "effects"):
                result = session.run(
                    f"""
                    MATCH (d:TreatmentDecision {{id: $decision_id}})
                    MATCH path = (d)-[:CAUSED|INFLUENCED*1..{depth}]->(effect:TreatmentDecision)
                    WITH effect, length(path) AS distance
                    RETURN effect {{.*, distance: distance}} AS decision
                    ORDER BY distance
                    """,
                    {"decision_id": decision_id},
                )
                effects = [convert_node_properties(record["decision"]) for record in result]

            return {
                "decision_id": decision_id,
                "causes": causes,
                "effects": effects,
                "depth": depth,
            }

    def find_similar_decisions(
        self,
        decision_id: str,
        limit: int = 5,
    ) -> List[Dict]:
        """
        Find similar past treatment decisions using graph structure.

        Uses pattern matching on:
        - Same cancer type/stage
        - Similar biomarker profile
        - Similar patient characteristics
        """
        if not self.driver:
            return []

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (d:TreatmentDecision {id: $decision_id})-[:ABOUT]->(p:Patient)
                MATCH (similar:TreatmentDecision)-[:ABOUT]->(sp:Patient)
                WHERE similar.id <> d.id
                  AND sp.tnm_stage = p.tnm_stage
                  AND sp.histology_type = p.histology_type
                WITH similar, sp,
                     CASE WHEN sp.performance_status = p.performance_status THEN 1 ELSE 0 END AS ps_match,
                     CASE WHEN abs(sp.age_at_diagnosis - p.age_at_diagnosis) <= 10 THEN 1 ELSE 0 END AS age_match
                RETURN similar {
                    .*,
                    patient_stage: sp.tnm_stage,
                    patient_histology: sp.histology_type,
                    similarity_score: ps_match + age_match
                } AS decision
                ORDER BY decision.similarity_score DESC, decision.decision_timestamp DESC
                LIMIT $limit
                """,
                {"decision_id": decision_id, "limit": limit},
            )
            return [convert_node_properties(record["decision"]) for record in result]

    # ============================================
    # ONTOLOGY ENRICHMENT
    # ============================================

    def _enrich_with_ontology(self, node_id: str, node_label: str) -> bool:
        """
        Link a Patient/Drug/Biomarker node to its ontology class counterpart.

        Looks for matching SNOMEDConcept or NCItConcept nodes by label/name
        and creates [:HAS_ONTOLOGY_CLASS] relationships.
        Sets ontology_class property on the node.

        Args:
            node_id: The id/patient_id of the node to enrich.
            node_label: The Neo4j label (e.g., "Patient", "Drug", "Biomarker").

        Returns:
            True if enrichment was applied, False otherwise.
        """
        if not self.driver:
            return False

        try:
            with self.driver.session(database=self.database) as session:
                if node_label == "Patient":
                    # Link patient to SNOMED concept based on histology_type
                    result = session.run(
                        """
                        MATCH (p:Patient {patient_id: $node_id})
                        WHERE p.histology_type IS NOT NULL
                        OPTIONAL MATCH (sc:SNOMEDConcept)
                            WHERE toLower(sc.fsn) CONTAINS toLower(p.histology_type)
                        WITH p, sc LIMIT 1
                        WHERE sc IS NOT NULL
                        SET p.ontology_class = sc.sctid
                        MERGE (p)-[:HAS_ONTOLOGY_CLASS]->(sc)
                        RETURN sc.sctid AS matched_sctid
                        """,
                        {"node_id": node_id},
                    )
                    record = result.single()
                    if record:
                        logger.debug(f"Patient {node_id} linked to SNOMED {record['matched_sctid']}")
                        return True

                elif node_label == "Drug":
                    # Link drug to NCIt concept by name
                    result = session.run(
                        """
                        MATCH (d:Drug {id: $node_id})
                        WHERE d.name IS NOT NULL
                        OPTIONAL MATCH (nc:NCItConcept)
                            WHERE toLower(nc.label) CONTAINS toLower(d.name)
                        WITH d, nc LIMIT 1
                        WHERE nc IS NOT NULL
                        SET d.ontology_class = nc.ncit_code
                        MERGE (d)-[:HAS_ONTOLOGY_CLASS]->(nc)
                        RETURN nc.ncit_code AS matched_code
                        """,
                        {"node_id": node_id},
                    )
                    record = result.single()
                    if record:
                        logger.debug(f"Drug {node_id} linked to NCIt {record['matched_code']}")
                        return True

                elif node_label == "Biomarker":
                    # Link biomarker to NCIt Gene concept
                    result = session.run(
                        """
                        MATCH (b:Biomarker {id: $node_id})
                        WHERE b.marker_type IS NOT NULL
                        OPTIONAL MATCH (nc:NCItConcept)
                            WHERE toLower(nc.label) = toLower(b.marker_type)
                        WITH b, nc LIMIT 1
                        WHERE nc IS NOT NULL
                        SET b.ontology_class = nc.ncit_code
                        MERGE (b)-[:HAS_ONTOLOGY_CLASS]->(nc)
                        RETURN nc.ncit_code AS matched_code
                        """,
                        {"node_id": node_id},
                    )
                    record = result.single()
                    if record:
                        logger.debug(f"Biomarker {node_id} linked to NCIt {record['matched_code']}")
                        return True

        except Exception as e:
            logger.warning(f"Ontology enrichment failed for {node_label} {node_id}: {e}")

        return False

    # ============================================
    # GUIDELINE OPERATIONS
    # ============================================

    def get_guidelines(self, category: Optional[str] = None) -> List[Dict]:
        """Get clinical guidelines, optionally filtered by category."""
        if not self.driver:
            return []

        category_filter = "WHERE g.category = $category" if category else ""

        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"""
                MATCH (g:Guideline)
                {category_filter}
                RETURN g {{.*}} AS guideline
                ORDER BY g.name
                """,
                {"category": category},
            )
            return [convert_node_properties(record["guideline"]) for record in result]

    # ============================================
    # GRAPH VISUALIZATION
    # ============================================

    def get_graph_data(
        self,
        center_node_id: Optional[str] = None,
        center_node_type: Optional[str] = None,
        depth: int = 2,
        limit: int = 100,
    ) -> GraphData:
        """
        Get graph data for NVL visualization.

        Returns nodes and relationships centered on a specific node,
        or a sample of the graph if no center is specified.
        """
        if not self.driver:
            return GraphData(nodes=[], relationships=[])

        with self.driver.session(database=self.database) as session:
            if center_node_id:
                # Get subgraph centered on a specific node
                result = session.run(
                    """
                    MATCH (center)
                    WHERE center.patient_id = $center_id
                       OR center.id = $center_id
                       OR elementId(center) = $center_id
                    OPTIONAL MATCH (center)-[r1]-(n1)
                    OPTIONAL MATCH (n1)-[r2]-(n2) WHERE n2 <> center
                    WITH center,
                         collect(DISTINCT n1) + collect(DISTINCT n2) AS connectedNodes,
                         collect(DISTINCT r1) + collect(DISTINCT r2) AS allRels
                    WITH [center] + connectedNodes[0..$limit] AS nodes, allRels AS relationships
                    RETURN nodes, relationships
                    """,
                    {"center_id": center_node_id, "limit": limit},
                )
            else:
                # Get a sample of the graph - ALL nodes (not filtered by type)
                result = session.run(
                    """
                    MATCH (n)
                    WITH n LIMIT $limit
                    OPTIONAL MATCH (n)-[r]-(m)
                    WITH collect(DISTINCT n) + collect(DISTINCT m) AS nodes,
                         collect(DISTINCT r) AS relationships
                    RETURN nodes[0..$limit] AS nodes, relationships
                    """,
                    {"limit": limit},
                )

            record = result.single()
            if not record:
                return GraphData(nodes=[], relationships=[])

            nodes = []
            seen_node_ids = set()
            for node in record["nodes"] or []:
                if node and node.element_id not in seen_node_ids:
                    seen_node_ids.add(node.element_id)
                    nodes.append(
                        GraphNode(
                            id=str(node.element_id),
                            labels=list(node.labels),
                            properties=convert_node_properties(dict(node)),
                        )
                    )

            relationships = []
            seen_rel_ids = set()
            for rel in record["relationships"] or []:
                if rel is not None and rel.element_id not in seen_rel_ids:
                    seen_rel_ids.add(rel.element_id)
                    relationships.append(
                        GraphRelationship(
                            id=str(rel.element_id),
                            type=rel.type,
                            startNodeId=str(rel.start_node.element_id),
                            endNodeId=str(rel.end_node.element_id),
                            properties=convert_node_properties(dict(rel)),
                        )
                    )

            return GraphData(nodes=nodes, relationships=relationships)

    def expand_node(self, node_id: str, limit: int = 50) -> GraphData:
        """Get all nodes directly connected to a given node."""
        if not self.driver:
            return GraphData(nodes=[], relationships=[])

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (center)
                WHERE center.patient_id = $node_id
                   OR center.id = $node_id
                   OR elementId(center) = $node_id
                WITH center LIMIT 1
                OPTIONAL MATCH (center)-[r]-(connected)
                WITH center, collect(DISTINCT connected)[0..$limit] AS connectedNodes,
                     collect(DISTINCT r) AS rels
                RETURN [center] + connectedNodes AS nodes, rels AS relationships
                """,
                {"node_id": node_id, "limit": limit},
            )

            record = result.single()
            if not record:
                return GraphData(nodes=[], relationships=[])

            nodes = []
            seen_node_ids = set()
            for node in record["nodes"] or []:
                if node and node.element_id not in seen_node_ids:
                    seen_node_ids.add(node.element_id)
                    nodes.append(
                        GraphNode(
                            id=str(node.element_id),
                            labels=list(node.labels),
                            properties=convert_node_properties(dict(node)),
                        )
                    )

            relationships = []
            seen_rel_ids = set()
            for rel in record["relationships"] or []:
                if rel is not None and rel.element_id not in seen_rel_ids:
                    seen_rel_ids.add(rel.element_id)
                    relationships.append(
                        GraphRelationship(
                            id=str(rel.element_id),
                            type=rel.type,
                            startNodeId=str(rel.start_node.element_id),
                            endNodeId=str(rel.end_node.element_id),
                            properties=convert_node_properties(dict(rel)),
                        )
                    )

            return GraphData(nodes=nodes, relationships=relationships)

    # ============================================
    # STATISTICS
    # ============================================

    def get_statistics(self) -> Dict:
        """Get graph statistics."""
        if not self.driver:
            return {"error": "No database connection"}

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (n)
                WITH labels(n) AS nodeLabels
                UNWIND nodeLabels AS label
                WITH label, count(*) AS count
                RETURN collect({label: label, count: count}) AS node_counts
                """
            )
            record = result.single()
            node_counts = {item["label"]: item["count"] for item in record["node_counts"]}

            result = session.run(
                """
                MATCH ()-[r]->()
                WITH type(r) AS relType
                WITH relType, count(*) AS count
                RETURN collect({type: relType, count: count}) AS rel_counts
                """
            )
            record = result.single()
            rel_counts = {item["type"]: item["count"] for item in record["rel_counts"]}

            return {
                "node_counts": node_counts,
                "relationship_counts": rel_counts,
                "total_nodes": sum(node_counts.values()),
                "total_relationships": sum(rel_counts.values()),
            }

    def get_schema(self) -> Dict[str, Any]:
        """Get the graph database schema."""
        if not self.driver:
            return {"error": "No database connection"}

        with self.driver.session(database=self.database) as session:
            # Get node labels
            labels_result = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
            node_labels = [record["label"] for record in labels_result]

            # Get relationship types
            rel_types_result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType")
            relationship_types = [record["relationshipType"] for record in rel_types_result]

            return {
                "node_labels": node_labels,
                "relationship_types": relationship_types,
            }

    def get_relationships_between(self, node_ids: List[str]) -> List[GraphRelationship]:
        """
        Get all relationships between a set of nodes.

        Used for graph visualization to find missing relationships
        after node expansion.
        """
        if not self.driver or len(node_ids) < 2:
            return []

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                UNWIND $node_ids AS id1
                UNWIND $node_ids AS id2
                WITH id1, id2 WHERE id1 < id2
                MATCH (n1)-[r]-(n2)
                WHERE (n1.patient_id = id1 OR n1.id = id1 OR elementId(n1) = id1)
                  AND (n2.patient_id = id2 OR n2.id = id2 OR elementId(n2) = id2)
                RETURN DISTINCT r, elementId(startNode(r)) AS start_id, elementId(endNode(r)) AS end_id
                """,
                {"node_ids": node_ids},
            )

            relationships = []
            seen_rel_ids = set()
            for record in result:
                rel = record["r"]
                if rel.element_id not in seen_rel_ids:
                    seen_rel_ids.add(rel.element_id)
                    relationships.append(
                        GraphRelationship(
                            id=str(rel.element_id),
                            type=rel.type,
                            startNodeId=str(record["start_id"]),
                            endNodeId=str(record["end_id"]),
                            properties=convert_node_properties(dict(rel)),
                        )
                    )

            return relationships

    def get_decision_with_graph(self, decision_id: str) -> Dict:
        """
        Get a decision with its full context and graph data for visualization.

        Combines get_decision() with get_graph_data() for convenience.
        """
        decision = self.get_decision(decision_id)
        if not decision:
            return {"decision": None, "graph_data": GraphData(nodes=[], relationships=[])}

        graph_data = self.get_graph_data(
            center_node_id=decision_id,
            center_node_type="TreatmentDecision",
            depth=2,
            limit=50,
        )

        return {
            "decision": decision,
            "graph_data": graph_data,
        }

    def search_patients_with_graph(
        self,
        query: str,
        limit: int = 5,
        graph_depth: int = 1,
    ) -> Dict:
        """
        Search for patients and return results with graph data.

        Convenience method that combines search with graph visualization.
        """
        patients = self.search_patients(query, limit=limit)

        if not patients:
            return {
                "patients": [],
                "graph_data": GraphData(nodes=[], relationships=[]),
            }

        # Get graph data for the first patient found
        first_patient_id = patients[0]["id"]
        graph_data = self.get_graph_data(
            center_node_id=first_patient_id,
            center_node_type="Patient",
            depth=graph_depth,
            limit=50,
        )

        return {
            "patients": patients,
            "graph_data": graph_data,
        }


# Singleton instance
_context_graph_client = None

def get_context_graph_client() -> LCAContextGraphClient:
    """Get or create the singleton context graph client."""
    global _context_graph_client
    if _context_graph_client is None:
        _context_graph_client = LCAContextGraphClient()
    return _context_graph_client
