"""
Neosemantics (n10s) Integration for Native Ontology Reasoning in Neo4j

Enables direct OWL/RDF import and SPARQL querying within Neo4j.
Requires Neosemantics plugin: https://neo4j.com/labs/neosemantics/

Based on 2025 research on ontology reasoning with Neo4j.
"""

from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeosemanticsTools:
    """
    Integration with Neosemantics (n10s) plugin for native ontology support.

    Features:
    - Import OWL/RDF ontologies directly into Neo4j
    - SPARQL query support
    - Automatic inference of subclass relationships
    - Semantic validation
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
            self._n10s_available = self._check_n10s_availability()
            logger.info(f"✓ Neosemantics tools initialized (n10s: {self._n10s_available})")
        except Exception as e:
            logger.warning(f"Neo4j not available: {e}")
            self._available = False
            self._n10s_available = False
            self.driver = None

    def _check_n10s_availability(self) -> bool:
        """Check if Neosemantics plugin is installed"""
        if not self._available:
            return False

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("CALL n10s.graphconfig.init() YIELD param")
                if result.peek():
                    logger.info("✓ Neosemantics (n10s) plugin available")
                    return True
        except Exception as e:
            logger.warning(f"⚠ Neosemantics plugin not installed: {e}")
            logger.info("Install from: https://neo4j.com/labs/neosemantics/")

        return False

    @property
    def is_available(self) -> bool:
        return self._available and self._n10s_available

    # ========================================
    # INITIALIZATION
    # ========================================

    def initialize_n10s(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize Neosemantics configuration.

        Args:
            config: Configuration options for n10s
                - handleVocabUris: How to handle vocabulary URIs ('SHORTEN', 'MAP', 'KEEP')
                - handleMultival: How to handle multivalued properties ('OVERWRITE', 'ARRAY')
                - multivalPropList: List of properties that can have multiple values

        Returns:
            True if successful
        """
        if not self.is_available:
            logger.warning("Neosemantics not available")
            return False

        default_config = {
            "handleVocabUris": "MAP",
            "handleMultival": "ARRAY",
            "multivalPropList": ["comorbidities", "treatments", "biomarkers"],
            "keepLangTag": False,
            "keepCustomDataTypes": True
        }

        if config:
            default_config.update(config)

        query = """
        CALL n10s.graphconfig.init($config)
        """

        try:
            with self.driver.session(database=self.database) as session:
                session.run(query, config=default_config)
                logger.info("✓ Neosemantics initialized with config")
                return True

        except Exception as e:
            logger.error(f"Failed to initialize n10s: {e}")
            return False

    # ========================================
    # ONTOLOGY IMPORT
    # ========================================

    def import_owl_ontology(
        self,
        ontology_path: str,
        format: str = "RDF/XML"
    ) -> Dict[str, Any]:
        """
        Import OWL ontology into Neo4j using Neosemantics.

        Args:
            ontology_path: Path to OWL file or URL
            format: RDF format ('RDF/XML', 'Turtle', 'N-Triples')

        Returns:
            Import statistics
        """
        if not self.is_available:
            return {"success": False, "message": "Neosemantics not available"}

        # Determine if path is file or URL
        if ontology_path.startswith("http"):
            query = """
            CALL n10s.onto.import.fetch($url, $format)
            """
            params = {"url": ontology_path, "format": format}
        else:
            # For local files, use file:/// protocol
            file_path = Path(ontology_path).absolute()
            query = """
            CALL n10s.onto.import.fetch($url, $format)
            """
            params = {"url": f"file:///{file_path}", "format": format}

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, **params)
                stats = result.single()

                logger.info(f"✓ Imported ontology from {ontology_path}")
                return {
                    "success": True,
                    "triples_loaded": stats.get("triplesLoaded", 0) if stats else 0,
                    "namespaces": stats.get("namespaces", {}) if stats else {}
                }

        except Exception as e:
            logger.error(f"Failed to import ontology: {e}")
            return {"success": False, "message": str(e)}

    def import_lucada_ontology(self, ontology_file: str = None) -> Dict[str, Any]:
        """
        Import LUCADA ontology specifically.

        Args:
            ontology_file: Path to LUCADA OWL file (auto-detects if not provided)

        Returns:
            Import statistics
        """
        if ontology_file is None:
            # Try to find LUCADA ontology file
            possible_paths = [
                "/home/user/Ontology-Driven-Clinical-Decision-Support/backend/data/lucada_ontology.owl",
                "/home/user/Ontology-Driven-Clinical-Decision-Support/ontologies/lucada.owl",
                "./data/lucada_ontology.owl"
            ]

            for path in possible_paths:
                if Path(path).exists():
                    ontology_file = path
                    break

        if ontology_file is None:
            logger.warning("Could not find LUCADA ontology file")
            return {"success": False, "message": "LUCADA ontology file not found"}

        logger.info(f"Importing LUCADA ontology from {ontology_file}")
        return self.import_owl_ontology(ontology_file, format="RDF/XML")

    def import_snomed_subset(self, subset_file: str) -> Dict[str, Any]:
        """
        Import SNOMED-CT subset into Neo4j.

        Args:
            subset_file: Path to SNOMED subset (RDF format)

        Returns:
            Import statistics
        """
        logger.info(f"Importing SNOMED subset from {subset_file}")
        return self.import_owl_ontology(subset_file, format="RDF/XML")

    # ========================================
    # ONTOLOGY QUERYING
    # ========================================

    def get_subclasses(self, class_uri: str, direct_only: bool = False) -> List[str]:
        """
        Get all subclasses of a given class.

        Args:
            class_uri: URI of the parent class
            direct_only: If True, only direct subclasses

        Returns:
            List of subclass URIs
        """
        if not self.is_available:
            return []

        depth = "1" if direct_only else "*"

        query = f"""
        MATCH (parent:Class {{uri: $class_uri}})<-[:SCO*{depth}]-(subclass:Class)
        RETURN DISTINCT subclass.uri as uri, subclass.name as name
        ORDER BY name
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, class_uri=class_uri)

                subclasses = [record["uri"] for record in result]
                logger.info(f"Found {len(subclasses)} subclasses of {class_uri}")
                return subclasses

        except Exception as e:
            logger.error(f"Failed to get subclasses: {e}")
            return []

    def get_class_properties(self, class_uri: str) -> List[Dict[str, Any]]:
        """
        Get all properties defined for a class.

        Args:
            class_uri: URI of the class

        Returns:
            List of property definitions
        """
        if not self.is_available:
            return []

        query = """
        MATCH (class:Class {uri: $class_uri})
        OPTIONAL MATCH (class)-[:DOMAIN_OF]->(prop:Property)
        OPTIONAL MATCH (prop)-[:RANGE_OF]->(range)

        RETURN prop.uri as property_uri,
               prop.name as property_name,
               prop.type as property_type,
               range.uri as range_uri,
               range.name as range_name
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, class_uri=class_uri)

                properties = []
                for record in result:
                    if record["property_uri"]:
                        properties.append({
                            "uri": record["property_uri"],
                            "name": record["property_name"],
                            "type": record["property_type"],
                            "range": {
                                "uri": record["range_uri"],
                                "name": record["range_name"]
                            }
                        })

                return properties

        except Exception as e:
            logger.error(f"Failed to get class properties: {e}")
            return []

    def check_subsumption(self, class1_uri: str, class2_uri: str) -> bool:
        """
        Check if class1 is a subclass of class2 (subsumption).

        Args:
            class1_uri: URI of potential subclass
            class2_uri: URI of potential superclass

        Returns:
            True if class1 is subclass of class2
        """
        if not self.is_available:
            return False

        query = """
        MATCH (c1:Class {uri: $class1})-[:SCO*]->(c2:Class {uri: $class2})
        RETURN count(*) > 0 as is_subclass
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, class1=class1_uri, class2=class2_uri)
                record = result.single()
                return record["is_subclass"] if record else False

        except Exception as e:
            logger.error(f"Failed to check subsumption: {e}")
            return False

    # ========================================
    # SEMANTIC VALIDATION
    # ========================================

    def validate_patient_against_ontology(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate patient data against LUCADA ontology constraints.

        Args:
            patient_data: Patient data dictionary

        Returns:
            Validation results with errors and warnings
        """
        if not self.is_available:
            return {"valid": False, "message": "Neosemantics not available"}

        errors = []
        warnings = []

        # Check if histology is valid LUCADA class
        histology = patient_data.get("histology_type")
        if histology:
            valid_histology = self._check_valid_histology(histology)
            if not valid_histology:
                errors.append(f"Invalid histology type: {histology}")

        # Check if stage is valid
        stage = patient_data.get("tnm_stage")
        if stage:
            valid_stage = self._check_valid_stage(stage)
            if not valid_stage:
                warnings.append(f"Non-standard TNM stage: {stage}")

        # Check performance status range
        ps = patient_data.get("performance_status")
        if ps is not None and (ps < 0 or ps > 4):
            errors.append(f"Invalid performance status: {ps} (must be 0-4)")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def _check_valid_histology(self, histology: str) -> bool:
        """Check if histology is valid LUCADA class"""
        valid_histologies = [
            "Adenocarcinoma", "SquamousCellCarcinoma", "LargeCellCarcinoma",
            "SmallCellCarcinoma", "NonSmallCellCarcinoma_NOS"
        ]
        return histology in valid_histologies

    def _check_valid_stage(self, stage: str) -> bool:
        """Check if TNM stage is valid"""
        valid_stages = ["I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IIIC", "IV", "IVA", "IVB"]
        return stage in valid_stages

    # ========================================
    # SPARQL SUPPORT (if available)
    # ========================================

    def execute_sparql(self, sparql_query: str) -> List[Dict[str, Any]]:
        """
        Execute SPARQL query using n10s (if supported).

        Args:
            sparql_query: SPARQL query string

        Returns:
            Query results
        """
        if not self.is_available:
            return []

        # Note: n10s SPARQL support may require additional configuration
        query = """
        CALL n10s.rdf.query.sparql($sparql_query)
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, sparql_query=sparql_query)

                results = []
                for record in result:
                    results.append(dict(record))

                return results

        except Exception as e:
            logger.warning(f"SPARQL query failed (may not be supported): {e}")
            return []

    # ========================================
    # ONTOLOGY EXPORT
    # ========================================

    def export_ontology(self, output_format: str = "Turtle") -> str:
        """
        Export current ontology from Neo4j to RDF format.

        Args:
            output_format: RDF format ('Turtle', 'RDF/XML', 'N-Triples')

        Returns:
            RDF string
        """
        if not self.is_available:
            return ""

        query = """
        CALL n10s.rdf.export.cypher($cypher, $format)
        """

        # Export all ontology classes and properties
        cypher = "MATCH (n) WHERE n:Class OR n:Property RETURN n"

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, cypher=cypher, format=output_format)
                rdf_output = result.single()

                return rdf_output["rdf"] if rdf_output else ""

        except Exception as e:
            logger.error(f"Failed to export ontology: {e}")
            return ""

    # ========================================
    # ONTOLOGY MAPPING
    # ========================================

    def map_to_snomed(self, lucada_concept: str) -> Optional[str]:
        """
        Map LUCADA concept to SNOMED-CT code using ontology mappings.

        Args:
            lucada_concept: LUCADA concept name

        Returns:
            SNOMED code or None
        """
        if not self.is_available:
            return None

        query = """
        MATCH (lucada:Class {name: $concept})
        MATCH (lucada)-[:EQUIVALENT_TO|MAPS_TO]->(snomed:Class)
        WHERE snomed.uri CONTAINS 'snomed'
        RETURN snomed.uri as snomed_code, snomed.name as snomed_name
        LIMIT 1
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, concept=lucada_concept)
                record = result.single()

                if record:
                    return record["snomed_code"]

                return None

        except Exception as e:
            logger.error(f"Failed to map to SNOMED: {e}")
            return None

    # ========================================
    # CLEANUP
    # ========================================

    def clear_ontology(self) -> bool:
        """Clear all ontology data from Neo4j"""
        if not self.is_available:
            return False

        query = """
        CALL n10s.graphconfig.drop()
        """

        try:
            with self.driver.session(database=self.database) as session:
                session.run(query)
                logger.info("✓ Cleared ontology data")
                return True

        except Exception as e:
            logger.error(f"Failed to clear ontology: {e}")
            return False

    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()


# ========================================
# UTILITY FUNCTIONS
# ========================================

def setup_neosemantics(
    neo4j_uri: str = None,
    neo4j_user: str = None,
    neo4j_password: str = None
) -> bool:
    """
    Setup Neosemantics and import LUCADA ontology.

    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password

    Returns:
        True if successful
    """
    n10s = NeosemanticsTools(neo4j_uri, neo4j_user, neo4j_password)

    if not n10s.is_available:
        logger.warning("Neosemantics not available - install from: https://neo4j.com/labs/neosemantics/")
        return False

    # Initialize n10s
    if not n10s.initialize_n10s():
        return False

    # Import LUCADA ontology
    result = n10s.import_lucada_ontology()

    n10s.close()

    return result.get("success", False)
