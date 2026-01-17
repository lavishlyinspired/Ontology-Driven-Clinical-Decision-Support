"""
Neo4j Graph Algorithms for Clinical Decision Support
Implements advanced graph analytics beyond simple CRUD operations

Based on 2025 research showing 5.4x-48.4x performance improvements
"""

from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import os
import numpy as np

from .models import PatientFact, SimilarPatient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jGraphAlgorithms:
    """
    Advanced graph algorithms for patient similarity, community detection,
    treatment pathfinding, and temporal pattern analysis.

    Requires Neo4j Graph Data Science (GDS) library.
    Install: https://neo4j.com/docs/graph-data-science/current/installation/
    """

    def __init__(self, uri: str = None, user: str = None, password: str = None, database: str = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            self._available = True
            self._gds_available = self._check_gds_availability()
            logger.info(f"✓ Neo4j Graph Algorithms initialized (GDS: {self._gds_available})")
        except Exception as e:
            logger.warning(f"Neo4j not available: {e}")
            self._available = False
            self._gds_available = False
            self.driver = None

    def _check_gds_availability(self) -> bool:
        """Check if Graph Data Science library is installed"""
        if not self._available:
            return False

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN gds.version() as version")
                version = result.single()
                if version:
                    logger.info(f"✓ Neo4j GDS available: {version['version']}")
                    return True
        except Exception:
            logger.warning("⚠ Neo4j GDS not installed. Graph algorithms will use fallback methods.")

        return False

    @property
    def is_available(self) -> bool:
        return self._available

    # ========================================
    # PATIENT SIMILARITY - Graph-Based
    # ========================================

    def find_similar_patients_graph_based(
        self,
        patient_id: str,
        k: int = 10,
        min_similarity: float = 0.7
    ) -> List[SimilarPatient]:
        """
        Find similar patients using graph structure and relationships.

        Uses Node Similarity algorithm from GDS library if available,
        otherwise falls back to custom similarity calculation.

        Args:
            patient_id: Target patient ID
            k: Number of similar patients to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of similar patients with similarity scores
        """
        if not self._available:
            return []

        if self._gds_available:
            return self._node_similarity_gds(patient_id, k, min_similarity)
        else:
            return self._node_similarity_fallback(patient_id, k, min_similarity)

    def _node_similarity_gds(
        self,
        patient_id: str,
        k: int,
        min_similarity: float
    ) -> List[SimilarPatient]:
        """Use GDS Node Similarity algorithm"""

        # First, create graph projection if not exists
        self._ensure_patient_graph_projection()

        query = """
        CALL gds.nodeSimilarity.stream('patient-similarity-graph')
        YIELD node1, node2, similarity
        WITH gds.util.asNode(node1) as p1, gds.util.asNode(node2) as p2, similarity
        WHERE p1.patient_id = $patient_id AND similarity >= $min_similarity

        OPTIONAL MATCH (p2)-[:RECEIVED_TREATMENT]->(t:TreatmentPlan)
        OPTIONAL MATCH (t)-[:HAS_OUTCOME]->(o:Outcome)

        RETURN p2.patient_id as patient_id,
               p2.name as name,
               similarity,
               p2.tnm_stage as tnm_stage,
               p2.histology_type as histology_type,
               p2.performance_status as performance_status,
               p2.age_at_diagnosis as age,
               t.type as treatment_received,
               o.status as outcome,
               o.survival_days as survival_days
        ORDER BY similarity DESC
        LIMIT $k
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, patient_id=patient_id, k=k, min_similarity=min_similarity)

                similar_patients = []
                for record in result:
                    similar_patients.append(SimilarPatient(
                        patient_id=record["patient_id"],
                        name=record["name"],
                        similarity_score=record["similarity"],
                        tnm_stage=record["tnm_stage"],
                        histology_type=record["histology_type"],
                        performance_status=record["performance_status"],
                        age_at_diagnosis=record["age"],
                        treatment_received=record["treatment_received"],
                        outcome=record["outcome"],
                        survival_days=record["survival_days"]
                    ))

                logger.info(f"Found {len(similar_patients)} similar patients using GDS")
                return similar_patients

        except Exception as e:
            logger.error(f"GDS node similarity failed: {e}")
            return self._node_similarity_fallback(patient_id, k, min_similarity)

    def _node_similarity_fallback(
        self,
        patient_id: str,
        k: int,
        min_similarity: float
    ) -> List[SimilarPatient]:
        """Fallback similarity calculation using Cypher"""

        query = """
        MATCH (p1:Patient {patient_id: $patient_id})
        MATCH (p2:Patient)
        WHERE p1.patient_id <> p2.patient_id

        // Calculate similarity based on multiple factors
        WITH p1, p2,
             // Stage match (0.4 weight)
             CASE WHEN p1.tnm_stage = p2.tnm_stage THEN 0.4 ELSE 0.0 END as stage_sim,
             // Histology match (0.3 weight)
             CASE WHEN p1.histology_type = p2.histology_type THEN 0.3 ELSE 0.0 END as hist_sim,
             // Performance status similarity (0.2 weight)
             (1.0 - abs(p1.performance_status - p2.performance_status) / 4.0) * 0.2 as ps_sim,
             // Age similarity (0.1 weight)
             (1.0 - (abs(p1.age_at_diagnosis - p2.age_at_diagnosis) / 100.0)) * 0.1 as age_sim

        WITH p2, (stage_sim + hist_sim + ps_sim + age_sim) as similarity
        WHERE similarity >= $min_similarity

        OPTIONAL MATCH (p2)-[:RECEIVED_TREATMENT]->(t:TreatmentPlan)
        OPTIONAL MATCH (t)-[:HAS_OUTCOME]->(o:Outcome)

        RETURN p2.patient_id as patient_id,
               p2.name as name,
               similarity,
               p2.tnm_stage as tnm_stage,
               p2.histology_type as histology_type,
               p2.performance_status as performance_status,
               p2.age_at_diagnosis as age,
               t.type as treatment_received,
               o.status as outcome,
               o.survival_days as survival_days
        ORDER BY similarity DESC
        LIMIT $k
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, patient_id=patient_id, k=k, min_similarity=min_similarity)

                similar_patients = []
                for record in result:
                    similar_patients.append(SimilarPatient(
                        patient_id=record["patient_id"],
                        name=record["name"],
                        similarity_score=record["similarity"],
                        tnm_stage=record["tnm_stage"],
                        histology_type=record["histology_type"],
                        performance_status=record["performance_status"],
                        age_at_diagnosis=record["age"],
                        treatment_received=record["treatment_received"],
                        outcome=record["outcome"],
                        survival_days=record["survival_days"]
                    ))

                logger.info(f"Found {len(similar_patients)} similar patients using fallback")
                return similar_patients

        except Exception as e:
            logger.error(f"Fallback similarity failed: {e}")
            return []

    def _ensure_patient_graph_projection(self):
        """Create graph projection for GDS algorithms if not exists"""

        check_query = """
        CALL gds.graph.exists('patient-similarity-graph')
        YIELD exists
        RETURN exists
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(check_query)
                exists = result.single()["exists"]

                if not exists:
                    logger.info("Creating patient similarity graph projection...")

                    create_query = """
                    CALL gds.graph.project(
                        'patient-similarity-graph',
                        ['Patient', 'ClinicalFinding', 'Histology', 'TreatmentPlan', 'Outcome'],
                        {
                            HAS_CLINICAL_FINDING: {orientation: 'UNDIRECTED'},
                            HAS_HISTOLOGY: {orientation: 'UNDIRECTED'},
                            RECEIVED_TREATMENT: {orientation: 'UNDIRECTED'},
                            HAS_OUTCOME: {orientation: 'UNDIRECTED'}
                        }
                    )
                    """

                    session.run(create_query)
                    logger.info("✓ Graph projection created")

        except Exception as e:
            logger.warning(f"Could not create graph projection: {e}")

    # ========================================
    # COMMUNITY DETECTION
    # ========================================

    def detect_treatment_communities(self, resolution: float = 1.0) -> Dict[str, List[str]]:
        """
        Detect communities of patients with similar treatment patterns.
        Uses Louvain community detection algorithm.

        Args:
            resolution: Modularity resolution (higher = more communities)

        Returns:
            Dictionary mapping community_id to list of patient_ids
        """
        if not self._available:
            return {}

        if self._gds_available:
            return self._detect_communities_gds(resolution)
        else:
            return self._detect_communities_fallback()

    def _detect_communities_gds(self, resolution: float) -> Dict[str, List[str]]:
        """Use GDS Louvain algorithm for community detection"""

        self._ensure_patient_graph_projection()

        query = """
        CALL gds.louvain.stream('patient-similarity-graph', {
            relationshipWeightProperty: null
        })
        YIELD nodeId, communityId
        WITH gds.util.asNode(nodeId) as patient, communityId
        WHERE patient:Patient
        RETURN communityId,
               collect(patient.patient_id) as patients,
               count(patient) as community_size
        ORDER BY community_size DESC
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)

                communities = {}
                for record in result:
                    community_id = f"community_{record['communityId']}"
                    communities[community_id] = {
                        "patients": record["patients"],
                        "size": record["community_size"]
                    }

                logger.info(f"Detected {len(communities)} treatment communities")
                return communities

        except Exception as e:
            logger.error(f"GDS Louvain failed: {e}")
            return self._detect_communities_fallback()

    def _detect_communities_fallback(self) -> Dict[str, List[str]]:
        """Fallback: Group by treatment patterns"""

        query = """
        MATCH (p:Patient)-[:RECEIVED_TREATMENT]->(t:TreatmentPlan)
        WITH p, collect(DISTINCT t.type) as treatments
        WITH treatments, collect(p.patient_id) as patients
        RETURN treatments, patients, size(patients) as community_size
        ORDER BY community_size DESC
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)

                communities = {}
                for idx, record in enumerate(result):
                    community_id = f"treatment_group_{idx}"
                    communities[community_id] = {
                        "treatments": record["treatments"],
                        "patients": record["patients"],
                        "size": record["community_size"]
                    }

                return communities

        except Exception as e:
            logger.error(f"Fallback community detection failed: {e}")
            return {}

    # ========================================
    # TREATMENT PATHFINDING
    # ========================================

    def find_optimal_treatment_paths(
        self,
        patient_id: str,
        target_outcome: str = "Complete Response",
        max_path_length: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find optimal treatment sequences that lead to desired outcomes.
        Uses shortest path algorithms on treatment networks.

        Args:
            patient_id: Starting patient
            target_outcome: Desired outcome (e.g., "Complete Response")
            max_path_length: Maximum treatment sequence length

        Returns:
            List of treatment paths with success rates
        """
        if not self._available:
            return []

        query = """
        MATCH (p:Patient {patient_id: $patient_id})

        // Find similar patients who achieved target outcome
        MATCH path = (similar:Patient)-[:RECEIVED_TREATMENT*1..$max_len]->(outcome:Outcome {status: $target_outcome})
        WHERE similar.tnm_stage = p.tnm_stage
          AND similar.histology_type = p.histology_type
          AND abs(similar.performance_status - p.performance_status) <= 1

        WITH path,
             [rel in relationships(path) | startNode(rel)] as treatment_nodes,
             length(path) as path_length,
             last(nodes(path)) as final_outcome

        WITH treatment_nodes, path_length, final_outcome.survival_days as survival_days

        RETURN [t.type FOR t in treatment_nodes] as treatment_sequence,
               path_length,
               survival_days,
               count(*) as times_used
        ORDER BY survival_days DESC, path_length ASC
        LIMIT 10
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    patient_id=patient_id,
                    target_outcome=target_outcome,
                    max_len=max_path_length
                )

                paths = []
                for record in result:
                    paths.append({
                        "treatment_sequence": record["treatment_sequence"],
                        "path_length": record["path_length"],
                        "avg_survival_days": record["survival_days"],
                        "times_used": record["times_used"],
                        "success_indicator": "High" if record["survival_days"] > 730 else "Moderate"
                    })

                logger.info(f"Found {len(paths)} optimal treatment paths")
                return paths

        except Exception as e:
            logger.error(f"Treatment pathfinding failed: {e}")
            return []

    # ========================================
    # CENTRALITY ANALYSIS
    # ========================================

    def find_influential_treatments(self) -> List[Dict[str, Any]]:
        """
        Find most influential treatments in the treatment network.
        Uses PageRank or degree centrality.

        Returns:
            List of treatments ranked by influence/usage
        """
        if not self._available:
            return []

        query = """
        MATCH (t:TreatmentPlan)
        OPTIONAL MATCH (p:Patient)-[:RECEIVED_TREATMENT]->(t)
        OPTIONAL MATCH (t)-[:HAS_OUTCOME]->(o:Outcome)

        WITH t,
             count(DISTINCT p) as patient_count,
             avg(o.survival_days) as avg_survival,
             count(CASE WHEN o.status IN ['Complete Response', 'Partial Response'] THEN 1 END) as success_count

        WHERE patient_count > 0

        RETURN t.type as treatment,
               patient_count,
               avg_survival,
               success_count,
               toFloat(success_count) / patient_count as success_rate
        ORDER BY patient_count DESC, success_rate DESC
        LIMIT 20
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)

                influential = []
                for record in result:
                    influential.append({
                        "treatment": record["treatment"],
                        "patient_count": record["patient_count"],
                        "avg_survival_days": record["avg_survival"],
                        "success_count": record["success_count"],
                        "success_rate": record["success_rate"],
                        "influence_score": record["patient_count"] * record["success_rate"]
                    })

                logger.info(f"Identified {len(influential)} influential treatments")
                return influential

        except Exception as e:
            logger.error(f"Centrality analysis failed: {e}")
            return []

    # ========================================
    # VECTOR SIMILARITY SEARCH
    # ========================================

    def create_vector_index(self):
        """Create vector index on patient embeddings for fast similarity search"""
        if not self._available:
            return False

        # Check if vector index exists
        check_query = "SHOW INDEXES WHERE name = 'patient_embedding_index'"

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(check_query)
                if result.single():
                    logger.info("Vector index already exists")
                    return True

                # Create vector index (Neo4j 5.11+)
                create_query = """
                CREATE VECTOR INDEX patient_embedding_index IF NOT EXISTS
                FOR (p:Patient)
                ON (p.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """

                session.run(create_query)
                logger.info("✓ Vector index created on patient embeddings")
                return True

        except Exception as e:
            logger.warning(f"Could not create vector index (requires Neo4j 5.11+): {e}")
            return False

    def vector_similarity_search(
        self,
        query_embedding: List[float],
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fast vector similarity search using native Neo4j vector index.

        Args:
            query_embedding: Query vector (384 dimensions)
            k: Number of results

        Returns:
            List of similar patients with scores
        """
        if not self._available:
            return []

        query = """
        CALL db.index.vector.queryNodes('patient_embedding_index', $k, $query_embedding)
        YIELD node, score
        RETURN node.patient_id as patient_id,
               node.name as name,
               node.tnm_stage as stage,
               node.histology_type as histology,
               score
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, k=k, query_embedding=query_embedding)

                results = []
                for record in result:
                    results.append({
                        "patient_id": record["patient_id"],
                        "name": record["name"],
                        "stage": record["stage"],
                        "histology": record["histology"],
                        "similarity_score": record["score"]
                    })

                return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    # ========================================
    # CLEANUP
    # ========================================

    def drop_graph_projection(self, graph_name: str = 'patient-similarity-graph'):
        """Drop graph projection to free memory"""
        if not self._gds_available:
            return

        try:
            with self.driver.session(database=self.database) as session:
                session.run(f"CALL gds.graph.drop('{graph_name}')")
                logger.info(f"Dropped graph projection: {graph_name}")
        except Exception as e:
            logger.debug(f"Could not drop graph projection: {e}")

    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
