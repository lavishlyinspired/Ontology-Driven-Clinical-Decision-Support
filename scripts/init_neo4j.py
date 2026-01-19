"""
Neo4j Database Initialization Script
Creates indices, constraints, and vector indices for LCA system
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Neo4jInitializer:
    """Initialize Neo4j database schema for LCA system"""
    
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.database = os.getenv("NEO4J_DATABASE", "lucada")
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
    
    def close(self):
        """Close driver connection"""
        self.driver.close()
    
    def create_constraints(self):
        """Create uniqueness constraints"""
        print("\n📋 Creating Constraints...")
        
        constraints = [
            # Patient constraints
            ("CREATE CONSTRAINT patient_id_unique IF NOT EXISTS FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE", 
             "Patient ID uniqueness"),
            
            # Clinical finding constraints
            ("CREATE CONSTRAINT snomed_code_unique IF NOT EXISTS FOR (c:ClinicalFinding) REQUIRE c.snomed_code IS UNIQUE",
             "SNOMED-CT code uniqueness"),
            
            # Treatment plan constraints
            ("CREATE CONSTRAINT treatment_plan_id_unique IF NOT EXISTS FOR (t:TreatmentPlan) REQUIRE t.plan_id IS UNIQUE",
             "Treatment plan ID uniqueness"),
            
            # Procedure constraints
            ("CREATE CONSTRAINT procedure_code_unique IF NOT EXISTS FOR (proc:Procedure) REQUIRE proc.code IS UNIQUE",
             "Procedure code uniqueness"),
            
            # Medication constraints
            ("CREATE CONSTRAINT rxnorm_code_unique IF NOT EXISTS FOR (m:Medication) REQUIRE m.rxnorm_code IS UNIQUE",
             "RxNorm code uniqueness"),
            
            # User constraints (for auth)
            ("CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
             "User ID uniqueness"),
            ("CREATE CONSTRAINT username_unique IF NOT EXISTS FOR (u:User) REQUIRE u.username IS UNIQUE",
             "Username uniqueness"),
            ("CREATE CONSTRAINT email_unique IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE",
             "Email uniqueness"),
            
            # Guideline version constraints
            ("CREATE CONSTRAINT version_id_unique IF NOT EXISTS FOR (v:GuidelineVersion) REQUIRE v.version_id IS UNIQUE",
             "Version ID uniqueness"),
        ]
        
        with self.driver.session(database=self.database) as session:
            for cypher, description in constraints:
                try:
                    session.run(cypher)
                    print(f"   ✓ {description}")
                except Exception as e:
                    print(f"   ⚠ {description} - {str(e)}")
    
    def create_indices(self):
        """Create performance indices"""
        print("\n📋 Creating Indices...")
        
        indices = [
            # Patient indices
            ("CREATE INDEX patient_age_idx IF NOT EXISTS FOR (p:Patient) ON (p.age)",
             "Patient age index"),
            ("CREATE INDEX patient_stage_idx IF NOT EXISTS FOR (p:Patient) ON (p.tnm_stage)",
             "Patient TNM stage index"),
            ("CREATE INDEX patient_histology_idx IF NOT EXISTS FOR (p:Patient) ON (p.histology_type)",
             "Patient histology index"),
            
            # Clinical finding indices
            ("CREATE INDEX finding_type_idx IF NOT EXISTS FOR (c:ClinicalFinding) ON (c.finding_type)",
             "Clinical finding type index"),
            
            # Treatment indices
            ("CREATE INDEX treatment_type_idx IF NOT EXISTS FOR (t:TreatmentPlan) ON (t.treatment_type)",
             "Treatment type index"),
            ("CREATE INDEX treatment_intent_idx IF NOT EXISTS FOR (t:TreatmentPlan) ON (t.intent)",
             "Treatment intent index"),
            
            # Temporal indices
            ("CREATE INDEX patient_diagnosis_date_idx IF NOT EXISTS FOR (p:Patient) ON (p.diagnosis_date)",
             "Patient diagnosis date index"),
            ("CREATE INDEX treatment_start_date_idx IF NOT EXISTS FOR (t:TreatmentPlan) ON (t.start_date)",
             "Treatment start date index"),
            
            # Audit log indices
            ("CREATE INDEX audit_timestamp_idx IF NOT EXISTS FOR (a:AuditLog) ON (a.timestamp)",
             "Audit log timestamp index"),
            ("CREATE INDEX audit_user_idx IF NOT EXISTS FOR (a:AuditLog) ON (a.user_id)",
             "Audit log user index"),
            ("CREATE INDEX audit_action_idx IF NOT EXISTS FOR (a:AuditLog) ON (a.action)",
             "Audit log action index"),
            
            # HITL indices
            ("CREATE INDEX hitl_status_idx IF NOT EXISTS FOR (h:HITLCase) ON (h.status)",
             "HITL case status index"),
            ("CREATE INDEX hitl_priority_idx IF NOT EXISTS FOR (h:HITLCase) ON (h.priority)",
             "HITL case priority index"),
        ]
        
        with self.driver.session(database=self.database) as session:
            for cypher, description in indices:
                try:
                    session.run(cypher)
                    print(f"   ✓ {description}")
                except Exception as e:
                    print(f"   ⚠ {description} - {str(e)}")
    
    def create_vector_indices(self):
        """Create vector indices for similarity search"""
        print("\n📋 Creating Vector Indices...")
        
        vector_index_name = os.getenv("NEO4J_VECTOR_INDEX", "clinical_guidelines_vector")
        vector_dimension = int(os.getenv("VECTOR_DIMENSION", "384"))
        
        # Vector index for guideline embeddings
        vector_index_cypher = f"""
        CREATE VECTOR INDEX {vector_index_name} IF NOT EXISTS
        FOR (g:Guideline) ON (g.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {vector_dimension},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        
        # Vector index for patient similarity
        patient_vector_cypher = f"""
        CREATE VECTOR INDEX patient_similarity_vector IF NOT EXISTS
        FOR (p:Patient) ON (p.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {vector_dimension},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        
        with self.driver.session(database=self.database) as session:
            try:
                session.run(vector_index_cypher)
                print(f"   ✓ Guideline vector index ({vector_dimension}d)")
            except Exception as e:
                print(f"   ⚠ Guideline vector index - {str(e)}")
            
            try:
                session.run(patient_vector_cypher)
                print(f"   ✓ Patient similarity vector index ({vector_dimension}d)")
            except Exception as e:
                print(f"   ⚠ Patient similarity vector index - {str(e)}")
    
    def create_full_text_indices(self):
        """Create full-text search indices"""
        print("\n📋 Creating Full-Text Search Indices...")
        
        fulltext_indices = [
            ("""
            CREATE FULLTEXT INDEX guideline_fulltext IF NOT EXISTS
            FOR (g:Guideline) ON EACH [g.title, g.description, g.text_content]
            """, "Guideline full-text search"),
            
            ("""
            CREATE FULLTEXT INDEX patient_notes_fulltext IF NOT EXISTS
            FOR (p:Patient) ON EACH [p.clinical_notes, p.history]
            """, "Patient notes full-text search"),
        ]
        
        with self.driver.session(database=self.database) as session:
            for cypher, description in fulltext_indices:
                try:
                    session.run(cypher)
                    print(f"   ✓ {description}")
                except Exception as e:
                    print(f"   ⚠ {description} - {str(e)}")
    
    def verify_setup(self):
        """Verify database setup"""
        print("\n📋 Verifying Setup...")
        
        with self.driver.session(database=self.database) as session:
            # Count constraints
            result = session.run("SHOW CONSTRAINTS")
            constraints = list(result)
            print(f"   ✓ Constraints: {len(constraints)}")
            
            # Count indices
            result = session.run("SHOW INDEXES")
            indices = list(result)
            print(f"   ✓ Indices: {len(indices)}")
    
    def initialize_all(self):
        """Run complete initialization"""
        print("=" * 80)
        print("🚀 Initializing Neo4j Database for LCA System")
        print(f"📍 URI: {self.uri}")
        print(f"🗄️ Database: {self.database}")
        print("=" * 80)
        
        try:
            # Test connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS test")
                result.single()
                print("✓ Connection successful\n")
            
            # Create database objects
            self.create_constraints()
            self.create_indices()
            self.create_vector_indices()
            self.create_full_text_indices()
            self.verify_setup()
            
            print("\n" + "=" * 80)
            print("✅ Neo4j database initialized successfully!")
            print("=" * 80)
            
        except Exception as e:
            print("\n" + "=" * 80)
            print(f"❌ Initialization failed: {e}")
            print("=" * 80)
            raise
        
        finally:
            self.close()


def main():
    """Main entry point"""
    initializer = Neo4jInitializer()
    initializer.initialize_all()


if __name__ == "__main__":
    main()
