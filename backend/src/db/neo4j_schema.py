"""
Neo4j Knowledge Graph Integration for LUCADA
Provides graph-based patient data storage and similarity queries
"""

from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration"""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "lucada"


class LUCADAGraphDB:
    """
    Neo4j implementation for LUCADA patient data and relationships.
    Complements OWL ontology for efficient querying and storage.
    """

    SCHEMA_QUERIES = [
        # Constraints
        "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE",
        "CREATE CONSTRAINT rule_id IF NOT EXISTS FOR (r:GuidelineRule) REQUIRE r.rule_id IS UNIQUE",
        "CREATE CONSTRAINT outcome_id IF NOT EXISTS FOR (o:Outcome) REQUIRE o.outcome_id IS UNIQUE",
        "CREATE CONSTRAINT treatment_plan_id IF NOT EXISTS FOR (t:TreatmentPlan) REQUIRE t.plan_id IS UNIQUE",

        # Indexes
        "CREATE INDEX patient_stage IF NOT EXISTS FOR (p:Patient) ON (p.tnm_stage)",
        "CREATE INDEX histology_type IF NOT EXISTS FOR (h:Histology) ON (h.type)",
        "CREATE INDEX outcome_status IF NOT EXISTS FOR (o:Outcome) ON (o.status)",
        "CREATE INDEX treatment_type IF NOT EXISTS FOR (t:TreatmentPlan) ON (t.type)",

        # Full-text search
        """CREATE FULLTEXT INDEX patient_search IF NOT EXISTS
           FOR (p:Patient) ON EACH [p.name, p.notes]""",
    ]

    def __init__(self, config: Neo4jConfig = None):
        """Initialize Neo4j connection"""
        self.config = config or Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database=os.getenv("NEO4J_DATABASE", "lucada")
        )

        try:
            # Suppress Neo4j warnings about non-existent relationships (expected for new patients)
            import warnings
            warnings.filterwarnings('ignore', category=DeprecationWarning, module='neo4j')
            
            # Suppress Neo4j notification warnings for missing relationships/properties
            # These are expected when database is being populated
            logging.getLogger('neo4j.notifications').setLevel(logging.ERROR)
            
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password)
            )
            logger.info(f"✓ Connected to Neo4j at {self.config.uri}")
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}")
            logger.warning("Graph database features will be disabled")
            self.driver = None

    @property
    def is_available(self) -> bool:
        """Check if Neo4j connection is available"""
        return self.driver is not None

    def setup_schema(self):
        """Initialize database schema"""
        if not self.driver:
            logger.warning("Neo4j not available, skipping schema setup")
            return

        with self.driver.session(database=self.config.database) as session:
            for query in self.SCHEMA_QUERIES:
                try:
                    session.run(query)
                    logger.debug(f"Executed: {query[:50]}...")
                except Exception as e:
                    logger.debug(f"Schema query skipped: {e}")

        logger.info("✓ Neo4j schema initialized")

    def create_patient(self, patient_data: Dict[str, Any]) -> str:
        """Create a patient node with all relationships (mirrors Figure 2)"""
        if not self.driver:
            logger.warning("Neo4j not available")
            return patient_data.get("patient_id", "UNKNOWN")

        query = """
        MERGE (p:Patient {patient_id: $patient_id})
        SET p.name = $name,
            p.sex = $sex,
            p.age_at_diagnosis = $age,
            p.tnm_stage = $tnm_stage,
            p.performance_status = $performance_status,
            p.fev1_percent = $fev1_percent,
            p.created_at = datetime()

        MERGE (cf:ClinicalFinding {id: $patient_id + '_finding'})
        SET cf.diagnosis = $diagnosis,
            cf.tnm_stage = $tnm_stage,
            cf.basis_of_diagnosis = 'Clinical'

        MERGE (h:Histology {type: $histology_type})

        MERGE (bs:BodyStructure:Neoplasm {id: $patient_id + '_tumor'})
        SET bs.laterality = $laterality

        MERGE (ps:PerformanceStatus {grade: $performance_status})

        MERGE (p)-[:HAS_CLINICAL_FINDING]->(cf)
        MERGE (cf)-[:HAS_HISTOLOGY]->(h)
        MERGE (cf)-[:AFFECTS]->(bs)
        MERGE (p)-[:HAS_PERFORMANCE_STATUS]->(ps)

        RETURN p.patient_id as patient_id
        """

        params = {
            "patient_id": patient_data.get("patient_id", "UNKNOWN"),
            "name": patient_data.get("name", ""),
            "sex": patient_data.get("sex", "U"),
            "age": patient_data.get("age", 0),
            "tnm_stage": patient_data.get("tnm_stage", ""),
            "performance_status": patient_data.get("performance_status", 0),
            "fev1_percent": patient_data.get("fev1_percent", 0.0),
            "diagnosis": patient_data.get("diagnosis", "Malignant Neoplasm of Lung"),
            "histology_type": patient_data.get("histology_type", ""),
            "laterality": patient_data.get("laterality", "Unknown")
        }

        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run(query, **params)
                record = result.single()
                return record["patient_id"] if record else params["patient_id"]
        except Exception as e:
            logger.error(f"Failed to create patient in Neo4j: {e}")
            return params["patient_id"]

    def find_similar_patients(self, patient_id: str, limit: int = 10) -> List[Dict]:
        """Find patients with similar clinical profiles"""
        if not self.driver:
            return []

        query = """
        MATCH (p:Patient {patient_id: $patient_id})

        WITH p.tnm_stage as stage, p.performance_status as ps

        MATCH (similar:Patient)
        WHERE similar.patient_id <> $patient_id
          AND similar.tnm_stage = stage
          AND abs(similar.performance_status - ps) <= 1

        MATCH (similar)-[:HAS_CLINICAL_FINDING]->(cf)-[:HAS_HISTOLOGY]->(h)

        OPTIONAL MATCH (similar)-[:HAS_TREATMENT_PLAN]->(tp)-[:HAS_OUTCOME]->(o)

        RETURN similar.patient_id as patient_id,
               similar.name as name,
               similar.age_at_diagnosis as age,
               similar.tnm_stage as stage,
               h.type as histology,
               collect(DISTINCT tp.type) as treatments,
               avg(o.survival_days) as avg_survival
        ORDER BY similar.age_at_diagnosis
        LIMIT $limit
        """

        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run(query, patient_id=patient_id, limit=limit)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Failed to find similar patients: {e}")
            return []

    def get_treatment_statistics(self, treatment_type: str) -> Dict:
        """Get outcome statistics for a treatment type"""
        if not self.driver:
            return {}

        query = """
        MATCH (tp:TreatmentPlan {type: $treatment_type})-[:HAS_OUTCOME]->(o:Outcome)
        MATCH (p:Patient)-[:HAS_TREATMENT_PLAN]->(tp)

        RETURN tp.type as treatment,
               count(DISTINCT p) as patient_count,
               avg(o.survival_days) as avg_survival_days,
               percentileDisc(o.survival_days, 0.5) as median_survival_days
        """

        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run(query, treatment_type=treatment_type)
                record = result.single()
                return dict(record) if record else {}
        except Exception as e:
            logger.error(f"Failed to get treatment statistics: {e}")
            return {}

    def store_recommendation(self, patient_id: str, recommendations: List[Dict]) -> None:
        """Store treatment recommendations for a patient"""
        if not self.driver:
            return

        # Filter out recommendations with null/empty treatment types
        valid_recommendations = [
            rec for rec in recommendations 
            if rec.get('treatment') or rec.get('recommended_treatment')
        ]
        
        if not valid_recommendations:
            logger.warning(f"No valid recommendations to store for patient {patient_id}")
            return

        # Normalize treatment field name
        normalized_recs = []
        for rec in valid_recommendations:
            treatment = rec.get('treatment') or rec.get('recommended_treatment')
            normalized_recs.append({
                'treatment': treatment,
                'rule_id': rec.get('rule_id', 'UNKNOWN'),
                'evidence_level': rec.get('evidence_level', 'Grade C'),
                'priority': rec.get('priority', 0)
            })

        query = """
        MATCH (p:Patient {patient_id: $patient_id})

        FOREACH (rec IN $recommendations |
            MERGE (tp:TreatmentPlan {type: rec.treatment})
            SET tp.plan_id = rec.treatment + '_' + p.patient_id,
                tp.intent = rec.treatment_intent,
                tp.evidence_level = rec.evidence_level,
                tp.created_at = datetime()
            
            MERGE (r:GuidelineRule {rule_id: rec.rule_id})
            SET r.evidence_level = rec.evidence_level,
                r.priority = rec.priority
            
            MERGE (p)-[rel:RECOMMENDED_TREATMENT {
                priority: rec.priority,
                timestamp: datetime()
            }]->(tp)
            
            MERGE (r)-[:RECOMMENDS]->(tp)
            
            // Create HAS_TREATMENT_PLAN relationship for compatibility
            MERGE (p)-[:HAS_TREATMENT_PLAN]->(tp)
        )
        """

        try:
            with self.driver.session(database=self.config.database) as session:
                session.run(query, patient_id=patient_id, recommendations=normalized_recs)
                logger.debug(f"Stored {len(normalized_recs)} recommendations for {patient_id}")
        except Exception as e:
            logger.error(f"Failed to store recommendations: {e}")

    def get_patient_history(self, patient_id: str) -> Dict:
        """Get complete patient history including recommendations"""
        if not self.driver:
            return {}

        query = """
        MATCH (p:Patient {patient_id: $patient_id})
        OPTIONAL MATCH (p)-[r:RECOMMENDED_TREATMENT]->(tp:TreatmentPlan)
        OPTIONAL MATCH (rule:GuidelineRule)-[:RECOMMENDS]->(tp)

        RETURN p as patient,
               collect({
                   treatment: tp.type,
                   priority: r.priority,
                   timestamp: r.timestamp,
                   rule_id: rule.rule_id,
                   evidence: rule.evidence_level
               }) as recommendations
        """

        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run(query, patient_id=patient_id)
                record = result.single()
                if record:
                    return {
                        "patient": dict(record["patient"]),
                        "recommendations": record["recommendations"]
                    }
                return {}
        except Exception as e:
            logger.error(f"Failed to get patient history: {e}")
            return {}

    def create_treatment_outcome(self, patient_id: str, outcome_data: Dict[str, Any]) -> str:
        """
        Create treatment outcome record with SNOMED codes.
        
        Args:
            patient_id: Patient identifier
            outcome_data: Dictionary containing:
                - treatment_type: Type of treatment
                - outcome_status: complete_response, partial_response, stable_disease, progressive_disease
                - survival_days: Days of survival/follow-up
                - response_date: Date of outcome assessment
                - snomed_code: SNOMED code for outcome status
                - toxicity_grade: 0-5 (CTCAE grading)
                - quality_of_life_score: Optional QoL score
                
        Returns:
            Outcome ID
        """
        if not self.driver:
            logger.warning("Neo4j not available")
            return ""

        import uuid
        outcome_id = outcome_data.get("outcome_id", str(uuid.uuid4())[:12])

        query = """
        MATCH (p:Patient {patient_id: $patient_id})
        
        MERGE (tp:TreatmentPlan {type: $treatment_type})
        ON CREATE SET tp.plan_id = $plan_id
        
        CREATE (o:Outcome {
            outcome_id: $outcome_id,
            status: $outcome_status,
            snomed_code: $snomed_code,
            survival_days: $survival_days,
            response_date: datetime($response_date),
            toxicity_grade: $toxicity_grade,
            quality_of_life_score: $qol_score,
            created_at: datetime()
        })
        
        MERGE (p)-[:RECEIVED_TREATMENT]->(tp)
        CREATE (tp)-[:HAS_OUTCOME]->(o)
        
        RETURN o.outcome_id as outcome_id
        """

        params = {
            "patient_id": patient_id,
            "outcome_id": outcome_id,
            "plan_id": f"{patient_id}_{outcome_data.get('treatment_type')}",
            "treatment_type": outcome_data.get("treatment_type", "Unknown"),
            "outcome_status": outcome_data.get("outcome_status", "unknown"),
            "snomed_code": outcome_data.get("snomed_code", ""),
            "survival_days": outcome_data.get("survival_days", 0),
            "response_date": outcome_data.get("response_date", "2026-01-17T00:00:00"),
            "toxicity_grade": outcome_data.get("toxicity_grade", 0),
            "qol_score": outcome_data.get("quality_of_life_score")
        }

        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run(query, **params)
                record = result.single()
                logger.info(f"✓ Created outcome record: {outcome_id}")
                return record["outcome_id"] if record else outcome_id
        except Exception as e:
            logger.error(f"Failed to create treatment outcome: {e}")
            return ""

    def get_treatment_outcomes(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Get all treatment outcomes for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            List of outcome records
        """
        if not self.driver:
            return []

        query = """
        MATCH (p:Patient {patient_id: $patient_id})-[:RECEIVED_TREATMENT]->(tp:TreatmentPlan)
        MATCH (tp)-[:HAS_OUTCOME]->(o:Outcome)
        
        RETURN tp.type as treatment_type,
               o.outcome_id as outcome_id,
               o.status as status,
               o.snomed_code as snomed_code,
               o.survival_days as survival_days,
               o.response_date as response_date,
               o.toxicity_grade as toxicity_grade,
               o.quality_of_life_score as qol_score
        ORDER BY o.response_date DESC
        """

        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run(query, patient_id=patient_id)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Failed to get treatment outcomes: {e}")
            return []

    def get_cohort_outcomes(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get aggregated outcome statistics for a patient cohort.
        
        Args:
            criteria: Filtering criteria (stage, histology, treatment, etc.)
            
        Returns:
            Statistical summary of outcomes
        """
        if not self.driver:
            return {}

        # Build dynamic WHERE clause
        where_clauses = []
        params = {}
        
        if criteria.get("tnm_stage"):
            where_clauses.append("p.tnm_stage = $stage")
            params["stage"] = criteria["tnm_stage"]
            
        if criteria.get("histology_type"):
            where_clauses.append("h.type = $histology")
            params["histology"] = criteria["histology_type"]
            
        if criteria.get("treatment_type"):
            where_clauses.append("tp.type = $treatment")
            params["treatment"] = criteria["treatment_type"]

        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
        MATCH (p:Patient)-[:HAS_CLINICAL_FINDING]->(cf)-[:HAS_HISTOLOGY]->(h)
        MATCH (p)-[:RECEIVED_TREATMENT]->(tp:TreatmentPlan)-[:HAS_OUTCOME]->(o:Outcome)
        {where_clause}
        
        RETURN 
            count(DISTINCT p) as total_patients,
            tp.type as treatment_type,
            avg(o.survival_days) as avg_survival_days,
            percentileDisc(o.survival_days, 0.5) as median_survival_days,
            percentileDisc(o.survival_days, 0.25) as q1_survival,
            percentileDisc(o.survival_days, 0.75) as q3_survival,
            max(o.survival_days) as max_survival,
            min(o.survival_days) as min_survival,
            count(CASE WHEN o.status = 'complete_response' THEN 1 END) as complete_responses,
            count(CASE WHEN o.status = 'partial_response' THEN 1 END) as partial_responses,
            count(CASE WHEN o.status = 'stable_disease' THEN 1 END) as stable_disease,
            count(CASE WHEN o.status = 'progressive_disease' THEN 1 END) as progressive_disease,
            avg(o.toxicity_grade) as avg_toxicity,
            avg(o.quality_of_life_score) as avg_qol
        """

        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run(query, **params)
                records = [dict(record) for record in result]
                
                if records:
                    return {
                        "cohort_statistics": records,
                        "criteria": criteria,
                        "timestamp": datetime.now().isoformat()
                    }
                return {}
        except Exception as e:
            logger.error(f"Failed to get cohort outcomes: {e}")
            return {}

    def track_disease_progression(self, patient_id: str, progression_data: Dict[str, Any]) -> str:
        """
        Track disease progression over time.
        
        Args:
            patient_id: Patient identifier
            progression_data: Dictionary containing progression information
            
        Returns:
            Progression record ID
        """
        if not self.driver:
            return ""

        import uuid
        progression_id = str(uuid.uuid4())[:12]

        query = """
        MATCH (p:Patient {patient_id: $patient_id})
        
        CREATE (prog:DiseaseProgression {
            progression_id: $progression_id,
            assessment_date: datetime($assessment_date),
            tnm_stage: $tnm_stage,
            tumor_size_mm: $tumor_size,
            lymph_node_involvement: $lymph_nodes,
            metastasis_sites: $metastasis_sites,
            notes: $notes,
            created_at: datetime()
        })
        
        CREATE (p)-[:HAS_PROGRESSION_RECORD]->(prog)
        
        RETURN prog.progression_id as progression_id
        """

        params = {
            "patient_id": patient_id,
            "progression_id": progression_id,
            "assessment_date": progression_data.get("assessment_date", "2026-01-17T00:00:00"),
            "tnm_stage": progression_data.get("tnm_stage", ""),
            "tumor_size": progression_data.get("tumor_size_mm", 0.0),
            "lymph_nodes": progression_data.get("lymph_node_involvement", False),
            "metastasis_sites": progression_data.get("metastasis_sites", []),
            "notes": progression_data.get("notes", "")
        }

        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run(query, **params)
                record = result.single()
                return record["progression_id"] if record else progression_id
        except Exception as e:
            logger.error(f"Failed to track disease progression: {e}")
            return ""

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("✓ Neo4j connection closed")


# Test function
if __name__ == "__main__":
    print("Testing Neo4j Integration")
    print("=" * 80)

    # Initialize
    db = LUCADAGraphDB()

    if db.driver:
        # Setup schema
        db.setup_schema()

        # Test patient
        test_patient = {
            "patient_id": "NEO4J_TEST_001",
            "name": "Test Patient",
            "age": 68,
            "sex": "M",
            "tnm_stage": "IIIA",
            "histology_type": "Adenocarcinoma",
            "performance_status": 1,
            "fev1_percent": 65.0,
            "laterality": "Right"
        }

        # Create patient
        patient_id = db.create_patient(test_patient)
        print(f"✓ Created patient: {patient_id}")

        # Store recommendation
        recommendations = [
            {
                "treatment": "Chemoradiotherapy",
                "rule_id": "R6",
                "evidence_level": "Grade A",
                "priority": 85
            }
        ]
        db.store_recommendation(patient_id, recommendations)
        print(f"✓ Stored recommendations")

        # Get history
        history = db.get_patient_history(patient_id)
        print(f"✓ Retrieved history: {len(history.get('recommendations', []))} recommendations")

        db.close()
    else:
        print("⚠ Neo4j not available - install and start Neo4j server")
        print("  Docker: docker run -p 7474:7474 -p 7687:7687 neo4j:5.15-community")
