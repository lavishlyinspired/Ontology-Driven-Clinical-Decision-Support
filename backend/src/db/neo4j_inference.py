"""
Neo4j Inference Engine
======================

Implements ontology-based inference rules using Neo4j APOC triggers and
Cypher-based reasoning. Provides automated classification, relationship
inference, and knowledge derivation.

Based on GoingMeta sessions:
- S3/S11: SHACL validation patterns
- S22-24: Ontology-driven inference
- S34: Ontology-enhanced tool calling
"""

import os
from typing import Dict, List, Any, Optional

from neo4j import GraphDatabase

from ..logging_config import get_logger

logger = get_logger(__name__)


class Neo4jInferenceEngine:
    """
    Ontology-based inference engine for Neo4j.

    Capabilities:
    1. Patient classification based on TNM staging and histology
    2. Biomarker-treatment inference (actionable mutation → therapy)
    3. Guideline applicability inference
    4. Risk stratification based on comorbidities
    5. Treatment contraindication inference
    6. Causal chain inference between decisions
    """

    def __init__(self, driver=None, database: str = "neo4j"):
        self.driver = driver
        self.database = database

        if not self.driver:
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "123456789")
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
            except Exception as e:
                logger.error(f"Neo4j connection failed: {e}")

    def close(self):
        if self.driver:
            self.driver.close()

    def run_all_inferences(self, patient_id: Optional[str] = None) -> Dict[str, int]:
        """Run all inference rules, optionally for a specific patient"""
        results = {}
        results["cancer_type_classification"] = self._infer_cancer_type_classification(patient_id)
        results["biomarker_therapy_inference"] = self._infer_biomarker_therapy_links(patient_id)
        results["guideline_applicability"] = self._infer_guideline_applicability(patient_id)
        results["risk_stratification"] = self._infer_risk_stratification(patient_id)
        results["contraindication_check"] = self._infer_contraindications(patient_id)
        results["stage_group_inference"] = self._infer_stage_groups(patient_id)
        return results

    def _infer_cancer_type_classification(self, patient_id: Optional[str] = None) -> int:
        """Classify patients into NSCLC/SCLC subtypes based on histology"""
        if not self.driver:
            return 0

        patient_filter = "WHERE p.patient_id = $patient_id" if patient_id else ""
        params = {"patient_id": patient_id} if patient_id else {}

        query = f"""
        MATCH (p:Patient)
        {patient_filter}
        WITH p,
             CASE
                WHEN toLower(p.histology_type) CONTAINS 'adenocarcinoma' THEN 'NSCLC_Adenocarcinoma'
                WHEN toLower(p.histology_type) CONTAINS 'squamous' THEN 'NSCLC_Squamous'
                WHEN toLower(p.histology_type) CONTAINS 'large cell' THEN 'NSCLC_LargeCell'
                WHEN toLower(p.histology_type) CONTAINS 'small cell' OR
                     toLower(p.histology_type) CONTAINS 'sclc' THEN 'SCLC'
                WHEN toLower(p.histology_type) CONTAINS 'nsclc' OR
                     toLower(p.histology_type) CONTAINS 'non-small' OR
                     toLower(p.histology_type) CONTAINS 'nonsmall' THEN 'NSCLC_NOS'
                ELSE 'Unknown'
             END AS cancer_subtype,
             CASE
                WHEN toLower(p.histology_type) CONTAINS 'small cell' AND
                     NOT toLower(p.histology_type) CONTAINS 'non' THEN 'SCLC'
                ELSE 'NSCLC'
             END AS cancer_category
        MERGE (cls:Inference:CancerClassification {{patient_id: p.patient_id}})
        SET cls.cancer_subtype = cancer_subtype,
            cls.cancer_category = cancer_category,
            cls.inferred_at = datetime(),
            cls.rule = 'histology_classification'
        MERGE (p)-[:HAS_CLASSIFICATION]->(cls)
        RETURN count(cls) AS inferred
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, params)
            record = result.single()
            return record["inferred"] if record else 0

    def _infer_biomarker_therapy_links(self, patient_id: Optional[str] = None) -> int:
        """Infer recommended therapies based on patient biomarker profile"""
        if not self.driver:
            return 0

        patient_filter = "WHERE p.patient_id = $patient_id" if patient_id else ""
        params = {"patient_id": patient_id} if patient_id else {}

        query = f"""
        MATCH (p:Patient)-[:HAS_BIOMARKER]->(b:Biomarker)
        {patient_filter}
        WITH p, b,
             CASE
                WHEN toLower(b.marker_type) CONTAINS 'egfr' AND
                     (toLower(b.value) CONTAINS 'positive' OR toLower(b.value) = 'true')
                THEN 'EGFR_TKI'
                WHEN toLower(b.marker_type) CONTAINS 'alk' AND
                     (toLower(b.value) CONTAINS 'positive' OR toLower(b.value) = 'true')
                THEN 'ALK_INHIBITOR'
                WHEN toLower(b.marker_type) CONTAINS 'ros1' AND
                     (toLower(b.value) CONTAINS 'positive' OR toLower(b.value) = 'true')
                THEN 'ROS1_INHIBITOR'
                WHEN toLower(b.marker_type) CONTAINS 'pdl1' AND
                     toFloat(replace(b.value, '%', '')) >= 50
                THEN 'IO_MONOTHERAPY'
                WHEN toLower(b.marker_type) CONTAINS 'pdl1' AND
                     toFloat(replace(b.value, '%', '')) >= 1 AND
                     toFloat(replace(b.value, '%', '')) < 50
                THEN 'CHEMO_IO_COMBINATION'
                WHEN toLower(b.marker_type) CONTAINS 'kras' AND
                     toLower(b.value) CONTAINS 'g12c'
                THEN 'KRAS_G12C_INHIBITOR'
                WHEN toLower(b.marker_type) CONTAINS 'braf' AND
                     toLower(b.value) CONTAINS 'v600e'
                THEN 'BRAF_MEK_COMBINATION'
                ELSE NULL
             END AS inferred_therapy_class
        WHERE inferred_therapy_class IS NOT NULL
        MERGE (inf:Inference:TherapyInference {{
            patient_id: p.patient_id,
            biomarker: b.marker_type
        }})
        SET inf.therapy_class = inferred_therapy_class,
            inf.biomarker_value = b.value,
            inf.inferred_at = datetime(),
            inf.rule = 'biomarker_therapy_inference'
        MERGE (b)-[:INDICATES]->(inf)
        MERGE (p)-[:HAS_THERAPY_INFERENCE]->(inf)
        RETURN count(inf) AS inferred
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, params)
            record = result.single()
            return record["inferred"] if record else 0

    def _infer_guideline_applicability(self, patient_id: Optional[str] = None) -> int:
        """Infer which guidelines apply to each patient based on stage/histology/PS"""
        if not self.driver:
            return 0

        patient_filter = "WHERE p.patient_id = $patient_id" if patient_id else ""
        params = {"patient_id": patient_id} if patient_id else {}

        query = f"""
        MATCH (p:Patient), (g:Guideline)
        {patient_filter}
        WHERE p.tnm_stage IN g.stage_applicability
        MERGE (p)-[r:ELIGIBLE_FOR_GUIDELINE]->(g)
        SET r.inferred_at = datetime(),
            r.patient_stage = p.tnm_stage,
            r.patient_ps = p.performance_status,
            r.rule = 'stage_guideline_matching'
        RETURN count(r) AS inferred
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, params)
            record = result.single()
            return record["inferred"] if record else 0

    def _infer_risk_stratification(self, patient_id: Optional[str] = None) -> int:
        """Infer risk levels based on comorbidities and performance status"""
        if not self.driver:
            return 0

        patient_filter = "WHERE p.patient_id = $patient_id" if patient_id else ""
        params = {"patient_id": patient_id} if patient_id else {}

        query = f"""
        MATCH (p:Patient)
        {patient_filter}
        OPTIONAL MATCH (p)-[:HAS_COMORBIDITY]->(c:Comorbidity)
        WITH p, count(c) AS comorbidity_count, collect(c.name) AS comorbidities
        WITH p, comorbidity_count, comorbidities,
             CASE
                WHEN p.performance_status >= 3 THEN 'HIGH_RISK'
                WHEN p.performance_status = 2 AND comorbidity_count >= 2 THEN 'HIGH_RISK'
                WHEN p.performance_status = 2 OR comorbidity_count >= 3 THEN 'MODERATE_RISK'
                WHEN comorbidity_count >= 1 THEN 'LOW_MODERATE_RISK'
                ELSE 'LOW_RISK'
             END AS risk_level,
             CASE
                WHEN p.performance_status >= 3 THEN 'Poor PS limits treatment tolerance'
                WHEN comorbidity_count >= 3 THEN 'Multiple comorbidities increase treatment risk'
                WHEN p.performance_status = 2 THEN 'Borderline PS may limit intensive regimens'
                ELSE 'Standard treatment risk'
             END AS risk_rationale
        MERGE (r:Inference:RiskStratification {{patient_id: p.patient_id}})
        SET r.risk_level = risk_level,
            r.risk_rationale = risk_rationale,
            r.comorbidity_count = comorbidity_count,
            r.comorbidities = comorbidities,
            r.performance_status = p.performance_status,
            r.inferred_at = datetime(),
            r.rule = 'risk_stratification'
        MERGE (p)-[:HAS_RISK_ASSESSMENT]->(r)
        RETURN count(r) AS inferred
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, params)
            record = result.single()
            return record["inferred"] if record else 0

    def _infer_contraindications(self, patient_id: Optional[str] = None) -> int:
        """Infer treatment contraindications based on comorbidities"""
        if not self.driver:
            return 0

        patient_filter = "AND p.patient_id = $patient_id" if patient_id else ""
        params = {"patient_id": patient_id} if patient_id else {}

        # Known contraindication rules
        query = f"""
        MATCH (p:Patient)-[:HAS_COMORBIDITY]->(c:Comorbidity)
        WHERE true {patient_filter}
        WITH p, c,
             CASE
                WHEN toLower(c.name) CONTAINS 'autoimmune' THEN ['Pembrolizumab', 'Nivolumab', 'Atezolizumab', 'Durvalumab']
                WHEN toLower(c.name) CONTAINS 'interstitial lung' OR toLower(c.name) = 'ild' THEN ['Osimertinib', 'Pembrolizumab']
                WHEN toLower(c.name) CONTAINS 'ckd' OR toLower(c.name) CONTAINS 'renal' THEN ['Cisplatin']
                WHEN toLower(c.name) CONTAINS 'heart failure' OR toLower(c.name) = 'chf' THEN ['Bevacizumab']
                WHEN toLower(c.name) CONTAINS 'liver' OR toLower(c.name) CONTAINS 'hepat' THEN ['Osimertinib', 'Alectinib']
                ELSE []
             END AS contraindicated_drugs
        WHERE size(contraindicated_drugs) > 0
        UNWIND contraindicated_drugs AS drug_name
        MERGE (ci:Inference:Contraindication {{
            patient_id: p.patient_id,
            drug: drug_name,
            reason: c.name
        }})
        SET ci.inferred_at = datetime(),
            ci.rule = 'comorbidity_contraindication'
        MERGE (p)-[:HAS_CONTRAINDICATION]->(ci)
        RETURN count(ci) AS inferred
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, params)
            record = result.single()
            return record["inferred"] if record else 0

    def _infer_stage_groups(self, patient_id: Optional[str] = None) -> int:
        """Infer stage grouping (early, locally advanced, metastatic)"""
        if not self.driver:
            return 0

        patient_filter = "WHERE p.patient_id = $patient_id" if patient_id else ""
        params = {"patient_id": patient_id} if patient_id else {}

        query = f"""
        MATCH (p:Patient)
        {patient_filter}
        WITH p,
             CASE
                WHEN p.tnm_stage IN ['IA', 'IB', 'IIA', 'IIB'] THEN 'EARLY_STAGE'
                WHEN p.tnm_stage IN ['IIIA', 'IIIB'] THEN 'LOCALLY_ADVANCED'
                WHEN p.tnm_stage = 'IV' THEN 'METASTATIC'
                WHEN p.tnm_stage IN ['Limited', 'limited'] THEN 'LIMITED_DISEASE'
                WHEN p.tnm_stage IN ['Extensive', 'extensive'] THEN 'EXTENSIVE_DISEASE'
                ELSE 'UNKNOWN_STAGE'
             END AS stage_group,
             CASE
                WHEN p.tnm_stage IN ['IA', 'IB', 'IIA', 'IIB'] THEN 'Curative'
                WHEN p.tnm_stage IN ['IIIA', 'IIIB'] THEN 'Potentially Curative'
                WHEN p.tnm_stage = 'IV' THEN 'Palliative'
                WHEN p.tnm_stage IN ['Limited', 'limited'] THEN 'Potentially Curative'
                WHEN p.tnm_stage IN ['Extensive', 'extensive'] THEN 'Palliative'
                ELSE 'Assess'
             END AS treatment_intent
        MERGE (sg:Inference:StageGroup {{patient_id: p.patient_id}})
        SET sg.stage_group = stage_group,
            sg.treatment_intent = treatment_intent,
            sg.original_stage = p.tnm_stage,
            sg.inferred_at = datetime(),
            sg.rule = 'stage_grouping'
        MERGE (p)-[:HAS_STAGE_GROUP]->(sg)
        RETURN count(sg) AS inferred
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, params)
            record = result.single()
            return record["inferred"] if record else 0

    def get_patient_inferences(self, patient_id: str) -> Dict[str, Any]:
        """Get all inferred knowledge for a patient"""
        if not self.driver:
            return {}

        query = """
        MATCH (p:Patient {patient_id: $patient_id})
        OPTIONAL MATCH (p)-[:HAS_CLASSIFICATION]->(cls:Inference)
        OPTIONAL MATCH (p)-[:HAS_THERAPY_INFERENCE]->(ti:Inference)
        OPTIONAL MATCH (p)-[:HAS_RISK_ASSESSMENT]->(ra:Inference)
        OPTIONAL MATCH (p)-[:HAS_CONTRAINDICATION]->(ci:Inference)
        OPTIONAL MATCH (p)-[:HAS_STAGE_GROUP]->(sg:Inference)
        OPTIONAL MATCH (p)-[:ELIGIBLE_FOR_GUIDELINE]->(g:Guideline)
        RETURN
            cls {.*} AS classification,
            collect(DISTINCT ti {.*}) AS therapy_inferences,
            ra {.*} AS risk_assessment,
            collect(DISTINCT ci {.*}) AS contraindications,
            sg {.*} AS stage_group,
            collect(DISTINCT g.name) AS applicable_guidelines
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, {"patient_id": patient_id})
            record = result.single()
            if not record:
                return {}
            return {
                "classification": dict(record["classification"]) if record["classification"] else None,
                "therapy_inferences": [dict(ti) for ti in record["therapy_inferences"]],
                "risk_assessment": dict(record["risk_assessment"]) if record["risk_assessment"] else None,
                "contraindications": [dict(ci) for ci in record["contraindications"]],
                "stage_group": dict(record["stage_group"]) if record["stage_group"] else None,
                "applicable_guidelines": record["applicable_guidelines"]
            }


def get_inference_engine(driver=None) -> Neo4jInferenceEngine:
    """Factory function"""
    return Neo4jInferenceEngine(driver=driver)
