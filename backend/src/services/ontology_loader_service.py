"""
Ontology Loader Service
=======================

Loads SNOMED-CT RF2 snapshot files and SHACL shapes into Neo4j.
Supports subset-only loading (lung-cancer relevant concepts) or full load.
"""

import os
from typing import Dict, List, Any, Optional, Set
from pathlib import Path

from neo4j import GraphDatabase

from ..config import LCAConfig
from ..logging_config import get_logger

logger = get_logger(__name__)

# Lung-cancer-relevant SNOMED root concepts for subset loading.
# Transitive closure from these roots captures ~5000 relevant concepts.
LUNG_CANCER_ROOT_SCTIDS = {
    "363358000",   # Malignant neoplasm of lung
    "254637007",   # NSCLC
    "254632001",   # Small cell carcinoma of lung
    "387713003",   # Surgical procedure
    "367336001",   # Chemotherapy
    "108290001",   # Radiation oncology
    "24484000",    # Severe
    "35917007",    # Adenocarcinoma
    "59367005",    # Squamous cell carcinoma
    "65399005",    # Large cell carcinoma
    "128462008",   # Metastatic malignant neoplasm
    "426964009",   # EGFR+ NSCLC
    "830151004",   # ALK+ NSCLC
    "723301009",   # Squamous NSCLC
    "373803006",   # ECOG PS 0
    "373804000",   # ECOG PS 1
    "373805004",   # ECOG PS 2
    "399537006",   # Clinical TNM stage
    "258215001",   # Stage 1
    "258219007",   # Stage 2
    "258224005",   # Stage 3
    "258228008",   # Stage 4
}

# IS_A relationship typeId in SNOMED RF2
IS_A_TYPE_ID = "116680003"
# Fully specified name typeId
FSN_TYPE_ID = "900000000000003001"


class OntologyLoaderService:
    """
    Loads SNOMED-CT RF2 files and SHACL shapes into Neo4j.

    Usage:
        loader = OntologyLoaderService()
        status = loader.load_snomed_rf2(subset_only=True)
        loader.close()
    """

    def __init__(self, driver=None, database: str = None):
        self.database = database or LCAConfig.NEO4J_DATABASE
        self._available = False

        if driver:
            self.driver = driver
            self._available = True
        else:
            uri = LCAConfig.NEO4J_URI
            user = LCAConfig.NEO4J_USER
            password = LCAConfig.NEO4J_PASSWORD
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                self.driver.verify_connectivity()
                self._available = True
            except Exception as e:
                logger.warning(f"Neo4j not available: {e}")
                self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    # ========================================
    # SNOMED RF2 LOADING
    # ========================================

    def load_snomed_rf2(self, subset_only: bool = True) -> Dict[str, Any]:
        """
        Load SNOMED-CT RF2 Snapshot into Neo4j as :SNOMEDConcept nodes.

        Args:
            subset_only: If True, only load lung-cancer-relevant subset (~5000 concepts).
                         If False, load all active concepts (~400K).

        Returns:
            Dict with counts of loaded concepts and relationships.
        """
        if not self._available:
            return {"success": False, "message": "Neo4j not available"}

        rf2_path = LCAConfig.SNOMED_RF2_PATH
        if not os.path.isdir(rf2_path):
            return {"success": False, "message": f"RF2 path not found: {rf2_path}"}

        concept_file = os.path.join(rf2_path, "sct2_Concept_Snapshot_INT_20260101.txt")
        desc_file = os.path.join(rf2_path, "sct2_Description_Snapshot-en_INT_20260101.txt")
        rel_file = os.path.join(rf2_path, "sct2_Relationship_Snapshot_INT_20260101.txt")

        for f in [concept_file, desc_file, rel_file]:
            if not os.path.exists(f):
                return {"success": False, "message": f"Missing RF2 file: {f}"}

        logger.info(f"Loading SNOMED RF2 (subset_only={subset_only})...")

        # Step 1: Read all IS_A relationships to build parent map
        logger.info("Reading relationships for IS_A hierarchy...")
        child_to_parents: Dict[str, List[str]] = {}
        parent_to_children: Dict[str, List[str]] = {}
        with open(rel_file, 'r', encoding='utf-8') as f:
            header = f.readline()  # skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 8:
                    continue
                active = parts[2]
                source_id = parts[4]   # child
                dest_id = parts[5]     # parent
                type_id = parts[7]
                if active == "1" and type_id == IS_A_TYPE_ID:
                    child_to_parents.setdefault(source_id, []).append(dest_id)
                    parent_to_children.setdefault(dest_id, []).append(source_id)

        logger.info(f"Read {sum(len(v) for v in child_to_parents.values())} IS_A relationships")

        # Step 2: Determine which concepts to load
        if subset_only:
            target_sctids = self._compute_transitive_closure(
                LUNG_CANCER_ROOT_SCTIDS, parent_to_children
            )
            logger.info(f"Subset: {len(target_sctids)} concepts in lung-cancer closure")
        else:
            target_sctids = None  # load all

        # Step 3: Read active concepts
        logger.info("Reading concepts...")
        active_concepts: Dict[str, bool] = {}
        with open(concept_file, 'r', encoding='utf-8') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                sctid = parts[0]
                active = parts[2]
                if active == "1":
                    if target_sctids is None or sctid in target_sctids:
                        active_concepts[sctid] = True

        logger.info(f"Found {len(active_concepts)} active concepts to load")

        # Step 4: Read FSN descriptions for target concepts
        logger.info("Reading descriptions (FSN labels)...")
        sctid_to_fsn: Dict[str, str] = {}
        with open(desc_file, 'r', encoding='utf-8') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                active = parts[2]
                concept_id = parts[4]
                type_id = parts[6]
                term = parts[7]
                if active == "1" and type_id == FSN_TYPE_ID and concept_id in active_concepts:
                    sctid_to_fsn[concept_id] = term

        logger.info(f"Read {len(sctid_to_fsn)} FSN labels")

        # Step 5: Create indexes
        self._create_snomed_indexes()

        # Step 6: Batch MERGE concepts into Neo4j
        concept_count = self._batch_merge_concepts(active_concepts, sctid_to_fsn)

        # Step 7: Batch MERGE IS_A relationships
        rel_count = self._batch_merge_relationships(
            child_to_parents, active_concepts
        )

        logger.info(f"SNOMED RF2 load complete: {concept_count} concepts, {rel_count} relationships")

        return {
            "success": True,
            "concepts_loaded": concept_count,
            "relationships_loaded": rel_count,
            "subset_only": subset_only,
        }

    def _compute_transitive_closure(
        self,
        roots: Set[str],
        parent_to_children: Dict[str, List[str]],
    ) -> Set[str]:
        """Compute transitive closure of descendants from root SCTIDs."""
        visited: Set[str] = set()
        stack = list(roots)
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for child in parent_to_children.get(node, []):
                if child not in visited:
                    stack.append(child)
        return visited

    def _create_snomed_indexes(self):
        """Create indexes on SNOMEDConcept nodes."""
        index_queries = [
            "CREATE INDEX snomed_sctid IF NOT EXISTS FOR (c:SNOMEDConcept) ON (c.sctid)",
            "CREATE INDEX snomed_fsn IF NOT EXISTS FOR (c:SNOMEDConcept) ON (c.fsn)",
        ]
        try:
            with self.driver.session(database=self.database) as session:
                for q in index_queries:
                    session.run(q)
            logger.info("SNOMED indexes created")
        except Exception as e:
            logger.warning(f"Index creation issue (may already exist): {e}")

    def _batch_merge_concepts(
        self,
        active_concepts: Dict[str, bool],
        sctid_to_fsn: Dict[str, str],
        batch_size: int = 5000,
    ) -> int:
        """Batch MERGE SNOMEDConcept nodes into Neo4j."""
        batch = []
        total = 0

        for sctid in active_concepts:
            batch.append({
                "sctid": sctid,
                "fsn": sctid_to_fsn.get(sctid, ""),
                "active": True,
            })
            if len(batch) >= batch_size:
                total += self._execute_concept_batch(batch)
                batch = []

        if batch:
            total += self._execute_concept_batch(batch)

        return total

    def _execute_concept_batch(self, batch: List[Dict]) -> int:
        query = """
        UNWIND $batch AS row
        MERGE (c:SNOMEDConcept {sctid: row.sctid})
        SET c.fsn = row.fsn, c.active = row.active
        RETURN count(c) AS cnt
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, batch=batch)
                record = result.single()
                return record["cnt"] if record else 0
        except Exception as e:
            logger.error(f"Concept batch failed: {e}")
            return 0

    def _batch_merge_relationships(
        self,
        child_to_parents: Dict[str, List[str]],
        active_concepts: Dict[str, bool],
        batch_size: int = 5000,
    ) -> int:
        """Batch MERGE IS_A relationships between SNOMEDConcept nodes."""
        batch = []
        total = 0

        for child, parents in child_to_parents.items():
            if child not in active_concepts:
                continue
            for parent in parents:
                if parent not in active_concepts:
                    continue
                batch.append({"child": child, "parent": parent})
                if len(batch) >= batch_size:
                    total += self._execute_rel_batch(batch)
                    batch = []

        if batch:
            total += self._execute_rel_batch(batch)

        return total

    def _execute_rel_batch(self, batch: List[Dict]) -> int:
        query = """
        UNWIND $batch AS row
        MATCH (child:SNOMEDConcept {sctid: row.child})
        MATCH (parent:SNOMEDConcept {sctid: row.parent})
        MERGE (child)-[:IS_A]->(parent)
        RETURN count(*) AS cnt
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, batch=batch)
                record = result.single()
                return record["cnt"] if record else 0
        except Exception as e:
            logger.error(f"Relationship batch failed: {e}")
            return 0

    # ========================================
    # N10S-BASED LOADING (ALTERNATIVE)
    # ========================================

    def load_snomed_via_n10s(self, ttl_path: str = None) -> Dict[str, Any]:
        """
        Alternative: Load SNOMED via n10s plugin from Turtle file.

        Args:
            ttl_path: Path to .ttl file. Defaults to LCAConfig.SNOMED_TTL_PATH.
        """
        if not self._available:
            return {"success": False, "message": "Neo4j not available"}

        ttl_path = ttl_path or LCAConfig.SNOMED_TTL_PATH
        if not os.path.exists(ttl_path):
            return {"success": False, "message": f"TTL file not found: {ttl_path}"}

        file_uri = Path(ttl_path).absolute().as_uri()

        try:
            with self.driver.session(database=self.database) as session:
                # Initialize n10s if needed
                try:
                    session.run("CALL n10s.graphconfig.init({handleVocabUris: 'MAP'})")
                except Exception:
                    pass  # may already be initialized

                result = session.run(
                    "CALL n10s.onto.import.fetch($url, 'Turtle')",
                    url=file_uri
                )
                stats = result.single()
                triples = stats.get("triplesLoaded", 0) if stats else 0

            logger.info(f"Loaded {triples} triples via n10s from {ttl_path}")
            return {"success": True, "triples_loaded": triples}

        except Exception as e:
            logger.error(f"n10s SNOMED load failed: {e}")
            return {"success": False, "message": str(e)}

    # ========================================
    # SHACL SHAPES
    # ========================================

    def load_shacl_shapes(self, shapes_path: str = None) -> Dict[str, Any]:
        """
        Load SHACL shapes via n10s for validation.

        Args:
            shapes_path: Path to shapes .ttl file. Defaults to LCAConfig.SHACL_SHAPES_PATH.
        """
        if not self._available:
            return {"success": False, "message": "Neo4j not available"}

        shapes_path = shapes_path or LCAConfig.SHACL_SHAPES_PATH
        if not os.path.exists(shapes_path):
            return {"success": False, "message": f"SHACL file not found: {shapes_path}"}

        file_uri = Path(shapes_path).absolute().as_uri()

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    "CALL n10s.validation.shacl.import.fetch($url, 'Turtle')",
                    url=file_uri
                )
                stats = result.single()

            logger.info(f"Loaded SHACL shapes from {shapes_path}")
            return {"success": True, "shapes_loaded": shapes_path}

        except Exception as e:
            logger.error(f"SHACL load failed (n10s may not be available): {e}")
            return {"success": False, "message": str(e)}

    def validate_with_shacl(self, focus_label: str = None) -> Dict[str, Any]:
        """
        Run SHACL validation on the graph.

        Args:
            focus_label: Optional node label to validate (e.g., "Drug", "Patient").
        """
        if not self._available:
            return {"success": False, "message": "Neo4j not available"}

        try:
            if focus_label:
                query = """
                CALL n10s.validation.shacl.validate()
                YIELD focusNode, resultPath, resultMessage, resultSeverity
                WHERE focusNode CONTAINS $label
                RETURN focusNode, resultPath, resultMessage, resultSeverity
                """
                params = {"label": focus_label}
            else:
                query = """
                CALL n10s.validation.shacl.validate()
                YIELD focusNode, resultPath, resultMessage, resultSeverity
                RETURN focusNode, resultPath, resultMessage, resultSeverity
                """
                params = {}

            with self.driver.session(database=self.database) as session:
                result = session.run(query, params)
                violations = [dict(record) for record in result]

            return {
                "success": True,
                "violations": violations,
                "violation_count": len(violations),
                "valid": len(violations) == 0,
            }

        except Exception as e:
            logger.error(f"SHACL validation failed: {e}")
            return {"success": False, "message": str(e)}

    # ========================================
    # NCI THESAURUS (NCIt) LOADING
    # ========================================

    # NCIt root concepts for subset loading (cancer-relevant)
    NCIT_SUBSET_ROOTS = {
        "C3262",   # Neoplasm
        "C1908",   # Drug, Food, Chemical or Biomedical Material
        "C16612",  # Gene
    }

    def load_ncit_via_n10s(self, owl_path: str = None) -> Dict[str, Any]:
        """
        Load NCIt OWL via n10s plugin (preferred for full ontology).

        Falls back to subset loading if n10s is unavailable.
        """
        if not self._available:
            return {"success": False, "message": "Neo4j not available"}

        owl_path = owl_path or LCAConfig.NCIT_OWL_PATH
        if not os.path.exists(owl_path):
            logger.info(f"NCIt OWL not found at {owl_path}, trying subset load...")
            return self.load_ncit_subset(owl_path)

        file_uri = Path(owl_path).absolute().as_uri()

        try:
            with self.driver.session(database=self.database) as session:
                try:
                    session.run("CALL n10s.graphconfig.init({handleVocabUris: 'MAP'})")
                except Exception:
                    pass

                result = session.run(
                    "CALL n10s.onto.import.fetch($url, 'RDF/XML')",
                    url=file_uri
                )
                stats = result.single()
                triples = stats.get("triplesLoaded", 0) if stats else 0

            logger.info(f"Loaded {triples} NCIt triples via n10s")
            return {"success": True, "triples_loaded": triples, "method": "n10s"}

        except Exception as e:
            logger.warning(f"n10s NCIt load failed ({e}), falling back to subset parse...")
            return self.load_ncit_subset(owl_path)

    def load_ncit_subset(self, owl_path: str = None) -> Dict[str, Any]:
        """
        Stream-parse ncit.owl with iterparse, extracting classes under
        Neoplasm (C3262), Drug (C1908), and Gene (C16612).

        Creates :NCItConcept nodes with [:IS_A] edges. Batch MERGE.
        """
        if not self._available:
            return {"success": False, "message": "Neo4j not available"}

        owl_path = owl_path or LCAConfig.NCIT_OWL_PATH
        if not os.path.exists(owl_path):
            return {"success": False, "message": f"NCIt OWL not found: {owl_path}"}

        try:
            from xml.etree.ElementTree import iterparse
        except ImportError:
            return {"success": False, "message": "xml.etree not available"}

        logger.info(f"Stream-parsing NCIt OWL subset from {owl_path}...")

        OWL_NS = "http://www.w3.org/2002/07/owl#"
        RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
        NCIT_PREFIX = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"

        # First pass: collect all parent→child relationships and labels
        concepts: Dict[str, str] = {}  # code → label
        parents: Dict[str, List[str]] = {}  # child_code → [parent_codes]
        current_class = None
        current_label = None
        current_parents: List[str] = []

        def extract_code(uri: str) -> Optional[str]:
            if uri and NCIT_PREFIX in uri:
                return uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]
            return None

        count = 0
        for event, elem in iterparse(owl_path, events=("start", "end")):
            if event == "start" and elem.tag == f"{{{OWL_NS}}}Class":
                about = elem.get(f"{{{RDF_NS}}}about", "")
                code = extract_code(about)
                if code:
                    current_class = code
                    current_label = None
                    current_parents = []

            elif event == "end":
                if elem.tag == f"{{{RDFS_NS}}}label" and current_class:
                    current_label = elem.text or ""

                elif elem.tag == f"{{{RDFS_NS}}}subClassOf" and current_class:
                    parent_uri = elem.get(f"{{{RDF_NS}}}resource", "")
                    parent_code = extract_code(parent_uri)
                    if parent_code:
                        current_parents.append(parent_code)

                elif elem.tag == f"{{{OWL_NS}}}Class" and current_class:
                    concepts[current_class] = current_label or current_class
                    if current_parents:
                        parents[current_class] = current_parents
                    current_class = None
                    count += 1
                    if count % 50000 == 0:
                        logger.info(f"  Parsed {count} OWL classes...")

                # Free memory
                elem.clear()

        logger.info(f"Parsed {len(concepts)} NCIt classes, {sum(len(v) for v in parents.values())} IS_A rels")

        # Compute transitive closure from subset roots
        parent_to_children: Dict[str, List[str]] = {}
        for child, parent_list in parents.items():
            for p in parent_list:
                parent_to_children.setdefault(p, []).append(child)

        target_codes = self._compute_transitive_closure(
            self.NCIT_SUBSET_ROOTS, parent_to_children
        )
        # Also include the roots themselves
        target_codes.update(self.NCIT_SUBSET_ROOTS)

        logger.info(f"NCIt subset: {len(target_codes)} concepts in closure")

        # Create indexes
        try:
            with self.driver.session(database=self.database) as session:
                session.run("CREATE INDEX ncit_code IF NOT EXISTS FOR (c:NCItConcept) ON (c.ncit_code)")
                session.run("CREATE INDEX ncit_label IF NOT EXISTS FOR (c:NCItConcept) ON (c.label)")
        except Exception as e:
            logger.warning(f"NCIt index creation issue: {e}")

        # Batch MERGE concepts
        batch = []
        total_concepts = 0
        for code in target_codes:
            if code in concepts:
                batch.append({"ncit_code": code, "label": concepts[code]})
                if len(batch) >= 5000:
                    total_concepts += self._execute_ncit_concept_batch(batch)
                    batch = []
        if batch:
            total_concepts += self._execute_ncit_concept_batch(batch)

        # Batch MERGE IS_A relationships
        batch = []
        total_rels = 0
        for child, parent_list in parents.items():
            if child not in target_codes:
                continue
            for parent in parent_list:
                if parent not in target_codes:
                    continue
                batch.append({"child": child, "parent": parent})
                if len(batch) >= 5000:
                    total_rels += self._execute_ncit_rel_batch(batch)
                    batch = []
        if batch:
            total_rels += self._execute_ncit_rel_batch(batch)

        logger.info(f"NCIt subset load complete: {total_concepts} concepts, {total_rels} relationships")
        return {
            "success": True,
            "concepts_loaded": total_concepts,
            "relationships_loaded": total_rels,
            "method": "iterparse_subset",
        }

    def _execute_ncit_concept_batch(self, batch: List[Dict]) -> int:
        query = """
        UNWIND $batch AS row
        MERGE (c:NCItConcept {ncit_code: row.ncit_code})
        SET c.label = row.label
        RETURN count(c) AS cnt
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, batch=batch)
                record = result.single()
                return record["cnt"] if record else 0
        except Exception as e:
            logger.error(f"NCIt concept batch failed: {e}")
            return 0

    def _execute_ncit_rel_batch(self, batch: List[Dict]) -> int:
        query = """
        UNWIND $batch AS row
        MATCH (child:NCItConcept {ncit_code: row.child})
        MATCH (parent:NCItConcept {ncit_code: row.parent})
        MERGE (child)-[:IS_A]->(parent)
        RETURN count(*) AS cnt
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, batch=batch)
                record = result.single()
                return record["cnt"] if record else 0
        except Exception as e:
            logger.error(f"NCIt relationship batch failed: {e}")
            return 0

    # ========================================
    # STATUS
    # ========================================

    def get_load_status(self) -> Dict[str, Any]:
        """Check what ontology data is loaded in Neo4j."""
        if not self._available:
            return {"available": False}

        try:
            with self.driver.session(database=self.database) as session:
                # Count SNOMEDConcept nodes
                r1 = session.run("MATCH (c:SNOMEDConcept) RETURN count(c) AS cnt")
                concept_count = r1.single()["cnt"]

                # Count IS_A relationships
                r2 = session.run(
                    "MATCH (:SNOMEDConcept)-[r:IS_A]->(:SNOMEDConcept) RETURN count(r) AS cnt"
                )
                rel_count = r2.single()["cnt"]

                # Count other ontology nodes (Class from n10s)
                r3 = session.run("MATCH (c:Class) RETURN count(c) AS cnt")
                class_count = r3.single()["cnt"]

                # Count NCIt concepts
                r4 = session.run("MATCH (c:NCItConcept) RETURN count(c) AS cnt")
                ncit_count = r4.single()["cnt"]

                # Count NCIt IS_A relationships
                r5 = session.run(
                    "MATCH (:NCItConcept)-[r:IS_A]->(:NCItConcept) RETURN count(r) AS cnt"
                )
                ncit_rel_count = r5.single()["cnt"]

            return {
                "available": True,
                "snomed_concepts": concept_count,
                "snomed_is_a_relationships": rel_count,
                "n10s_class_nodes": class_count,
                "snomed_loaded": concept_count > 0,
                "ncit_concepts": ncit_count,
                "ncit_is_a_relationships": ncit_rel_count,
                "ncit_loaded": ncit_count > 0,
            }

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {"available": False, "error": str(e)}
