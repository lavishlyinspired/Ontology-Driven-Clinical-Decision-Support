"""
FHIR R4 Terminology Service
============================

Implements FHIR R4 terminology operations backed by loaded ontologies
in Neo4j (SNOMED-CT, LOINC, RxNorm, NCIt) and the UMLS crosswalk.

Supported operations:
- CodeSystem/$lookup — Look up a code in a code system
- ConceptMap/$translate — Translate between code systems (via UMLS CUI)
- ValueSet/$expand — Expand a value set (e.g., lung cancer histologies)
"""

from typing import Dict, List, Optional, Any

from ..logging_config import get_logger
from ..config import LCAConfig

logger = get_logger(__name__)

# FHIR code system URIs to internal SAB mapping
SYSTEM_TO_SAB = {
    "http://snomed.info/sct": "SNOMEDCT_US",
    "http://loinc.org": "LNC",
    "http://www.nlm.nih.gov/research/umls/rxnorm": "RXNORM",
    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl": "NCI",
}

SAB_TO_SYSTEM = {v: k for k, v in SYSTEM_TO_SAB.items()}

# Predefined value sets
VALUE_SETS = {
    "lung-cancer-histologies": {
        "url": "http://coherenceplm.org/fhir/ValueSet/lung-cancer-histologies",
        "name": "LungCancerHistologies",
        "title": "Lung Cancer Histology Types",
        "status": "active",
        "snomed_root": "254637007",  # NSCLC parent concept
    },
    "lung-cancer-biomarkers": {
        "url": "http://coherenceplm.org/fhir/ValueSet/lung-cancer-biomarkers",
        "name": "LungCancerBiomarkers",
        "title": "Lung Cancer Actionable Biomarkers",
        "status": "active",
        "codes": [
            ("EGFR", "Epidermal Growth Factor Receptor"),
            ("ALK", "ALK Receptor Tyrosine Kinase"),
            ("ROS1", "ROS Proto-Oncogene 1"),
            ("BRAF", "B-Raf Proto-Oncogene"),
            ("KRAS", "KRAS Proto-Oncogene"),
            ("NTRK", "Neurotrophic Tyrosine Receptor Kinase"),
            ("MET", "MET Proto-Oncogene"),
            ("RET", "Ret Proto-Oncogene"),
            ("HER2", "Erb-B2 Receptor Tyrosine Kinase 2"),
            ("PD-L1", "Programmed Death-Ligand 1"),
        ],
    },
}


class FHIRTerminologyService:
    """
    FHIR R4 terminology operations backed by Neo4j ontologies and UMLS crosswalk.
    """

    def __init__(self):
        self._neo4j_driver = None
        self._database = LCAConfig.NEO4J_DATABASE

    def _get_driver(self):
        """Lazy-initialize Neo4j driver."""
        if self._neo4j_driver is None:
            try:
                from neo4j import GraphDatabase
                self._neo4j_driver = GraphDatabase.driver(
                    LCAConfig.NEO4J_URI,
                    auth=(LCAConfig.NEO4J_USER, LCAConfig.NEO4J_PASSWORD),
                )
                self._neo4j_driver.verify_connectivity()
            except Exception as e:
                logger.warning(f"Neo4j not available for FHIR service: {e}")
                self._neo4j_driver = None
        return self._neo4j_driver

    def close(self):
        if self._neo4j_driver:
            self._neo4j_driver.close()
            self._neo4j_driver = None

    def lookup(self, system: str, code: str) -> Dict[str, Any]:
        """
        FHIR CodeSystem/$lookup operation.

        Args:
            system: Code system URI (e.g., "http://snomed.info/sct")
            code: Concept code (e.g., "254637007")

        Returns:
            FHIR Parameters-style response with name, display, designation.
        """
        sab = SYSTEM_TO_SAB.get(system)
        if not sab:
            return {"error": f"Unsupported code system: {system}"}

        driver = self._get_driver()
        if not driver:
            return {"error": "Neo4j not available"}

        try:
            with driver.session(database=self._database) as session:
                if sab == "SNOMEDCT_US":
                    result = session.run(
                        "MATCH (c:SNOMEDConcept {sctid: $code}) RETURN c.sctid AS code, c.fsn AS display",
                        {"code": code},
                    )
                elif sab == "NCI":
                    result = session.run(
                        "MATCH (c:NCItConcept {ncit_code: $code}) RETURN c.ncit_code AS code, c.label AS display",
                        {"code": code},
                    )
                else:
                    # Try generic label-based lookup
                    result = session.run(
                        """
                        MATCH (c)
                        WHERE any(l IN labels(c) WHERE l CONTAINS $sab_hint)
                          AND (c.code = $code OR c.loinc_num = $code OR c.rxcui = $code)
                        RETURN c.code AS code, c.name AS display
                        LIMIT 1
                        """,
                        {"code": code, "sab_hint": sab[:4]},
                    )

                record = result.single()
                if not record:
                    return {
                        "resourceType": "Parameters",
                        "parameter": [
                            {"name": "result", "valueBoolean": False},
                            {"name": "message", "valueString": f"Code {code} not found in {system}"},
                        ],
                    }

                return {
                    "resourceType": "Parameters",
                    "parameter": [
                        {"name": "result", "valueBoolean": True},
                        {"name": "name", "valueString": system},
                        {"name": "code", "valueCode": record["code"]},
                        {"name": "display", "valueString": record["display"] or ""},
                        {"name": "version", "valueString": "2026-01"},
                    ],
                }

        except Exception as e:
            logger.error(f"FHIR lookup failed: {e}")
            return {"error": str(e)}

    def translate(
        self, system: str, code: str, target_system: str
    ) -> Dict[str, Any]:
        """
        FHIR ConceptMap/$translate operation via UMLS crosswalk.

        Args:
            system: Source system URI
            code: Source code
            target_system: Target system URI

        Returns:
            FHIR Parameters-style response with translated code.
        """
        source_sab = SYSTEM_TO_SAB.get(system)
        target_sab = SYSTEM_TO_SAB.get(target_system)

        if not source_sab:
            return {"error": f"Unsupported source system: {system}"}
        if not target_sab:
            return {"error": f"Unsupported target system: {target_system}"}

        try:
            from .umls_crosswalk_service import get_crosswalk_service

            svc = get_crosswalk_service()
            if not svc.loaded:
                load_result = svc.load()
                if not load_result.get("success"):
                    return {"error": "UMLS crosswalk not available"}

            target_code = svc.get_crosswalk(source_sab, code, target_sab)

            if not target_code:
                return {
                    "resourceType": "Parameters",
                    "parameter": [
                        {"name": "result", "valueBoolean": False},
                        {"name": "message", "valueString": f"No mapping from {source_sab}:{code} to {target_sab}"},
                    ],
                }

            return {
                "resourceType": "Parameters",
                "parameter": [
                    {"name": "result", "valueBoolean": True},
                    {
                        "name": "match",
                        "part": [
                            {"name": "equivalence", "valueCode": "equivalent"},
                            {
                                "name": "concept",
                                "valueCoding": {
                                    "system": target_system,
                                    "code": target_code,
                                },
                            },
                        ],
                    },
                ],
            }

        except Exception as e:
            logger.error(f"FHIR translate failed: {e}")
            return {"error": str(e)}

    def expand(self, value_set_url: str) -> Dict[str, Any]:
        """
        FHIR ValueSet/$expand operation.

        Expands a predefined or SNOMED-based value set.

        Args:
            value_set_url: Value set URL or short name (e.g., "lung-cancer-histologies")

        Returns:
            FHIR ValueSet expansion with concepts.
        """
        # Resolve short name
        vs_key = value_set_url.split("/")[-1] if "/" in value_set_url else value_set_url
        vs_def = VALUE_SETS.get(vs_key)

        if not vs_def:
            return {"error": f"Unknown value set: {value_set_url}"}

        concepts = []

        # Static code list
        if "codes" in vs_def:
            for code, display in vs_def["codes"]:
                concepts.append({"code": code, "display": display})

        # SNOMED-based expansion from Neo4j
        elif "snomed_root" in vs_def:
            driver = self._get_driver()
            if driver:
                try:
                    with driver.session(database=self._database) as session:
                        result = session.run(
                            """
                            MATCH (root:SNOMEDConcept {sctid: $root_sctid})
                            MATCH (child:SNOMEDConcept)-[:IS_A*1..3]->(root)
                            RETURN child.sctid AS code, child.fsn AS display
                            LIMIT 100
                            """,
                            {"root_sctid": vs_def["snomed_root"]},
                        )
                        for record in result:
                            concepts.append({
                                "system": "http://snomed.info/sct",
                                "code": record["code"],
                                "display": record["display"] or "",
                            })
                except Exception as e:
                    logger.error(f"ValueSet expansion from Neo4j failed: {e}")

        return {
            "resourceType": "ValueSet",
            "url": vs_def.get("url", value_set_url),
            "name": vs_def.get("name", vs_key),
            "title": vs_def.get("title", vs_key),
            "status": vs_def.get("status", "active"),
            "expansion": {
                "timestamp": "2026-02-07",
                "total": len(concepts),
                "contains": concepts,
            },
        }


# Singleton
_fhir_terminology_service: Optional[FHIRTerminologyService] = None


def get_fhir_terminology_service() -> FHIRTerminologyService:
    """Get or create the singleton FHIR terminology service."""
    global _fhir_terminology_service
    if _fhir_terminology_service is None:
        _fhir_terminology_service = FHIRTerminologyService()
    return _fhir_terminology_service
