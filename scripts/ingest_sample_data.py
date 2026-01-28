#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Neo4j Data Ingestion for LCA System
Creates complete graph with all nodes, relationships, and properties
"""

import sys
import random
from datetime import datetime, timedelta
import json

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        pass

try:
    from neo4j import GraphDatabase
except ImportError:
    print("✗ Error: neo4j package not installed")
    print("  Install with: pip install neo4j")
    sys.exit(1)

print("=" * 70)
print("LCA Comprehensive Data Ingestion")
print("=" * 70)
print()

# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "123456789"
NEO4J_DATABASE = "neo4j"

# Data pools
STAGES = ["IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IV"]
HISTOLOGIES = {
    "Adenocarcinoma": "NSCLC",
    "SquamousCellCarcinoma": "NSCLC",
    "LargeCellCarcinoma": "NSCLC",
    "SmallCellCarcinoma": "SCLC"
}
LATERALITY = ["Right", "Left", "Bilateral"]
COMORBIDITIES = ["COPD", "CAD", "Hypertension", "Diabetes", "CKD Stage 3", "AFib", "CHF", "Asthma"]

# Treatment options by category
TREATMENTS = {
    "surgery": ["Lobectomy", "Pneumonectomy", "Wedge Resection", "SABR/SBRT"],
    "chemo": ["Cisplatin + Etoposide", "Carboplatin + Pemetrexed", "Cisplatin + Gemcitabine"],
    "targeted": ["Osimertinib (EGFR TKI)", "Alectinib (ALK inhibitor)", "Crizotinib", "Dabrafenib + Trametinib"],
    "immuno": ["Pembrolizumab", "Nivolumab", "Atezolizumab", "Durvalumab"],
    "radiation": ["Concurrent Chemoradiotherapy", "Sequential Chemoradiotherapy", "Thoracic RT", "PCI"],
    "supportive": ["Best Supportive Care", "Palliative RT", "Symptom Management"]
}

# NICE Guidelines
GUIDELINES = [
    {"id": "R1", "rule": "Early stage operable NSCLC → Surgery", "evidence": "NICE CG121"},
    {"id": "R2", "rule": "Advanced NSCLC with PS 0-1 → Chemotherapy", "evidence": "NICE CG121"},
    {"id": "R3", "rule": "Limited stage SCLC → Concurrent chemoradiotherapy", "evidence": "NICE CG121"},
    {"id": "R4", "rule": "EGFR-mutated NSCLC → Targeted therapy", "evidence": "NICE CG121, FLAURA"},
    {"id": "R5", "rule": "ALK-rearranged NSCLC → ALK inhibitor", "evidence": "NICE CG121, ALEX"},
    {"id": "R6", "rule": "High PD-L1 (≥50%) → Immunotherapy", "evidence": "NICE CG121, KEYNOTE-024"},
    {"id": "R7", "rule": "PS 3-4 → Best supportive care", "evidence": "NICE CG121"}
]

def clear_database(session):
    """Clear all existing data from the database"""
    print("Clearing existing database...")

    # Delete all nodes and relationships
    session.run("MATCH (n) DETACH DELETE n")

    print("✓ Database cleared")

def create_comprehensive_schema(session):
    """Create complete Neo4j schema with indexes and constraints"""

    print("Creating comprehensive Neo4j schema...")

    # Create constraints (unique identifiers)
    constraints = [
        "CREATE CONSTRAINT patient_id_unique IF NOT EXISTS FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE",
        "CREATE CONSTRAINT treatment_id_unique IF NOT EXISTS FOR (t:Treatment) REQUIRE t.treatment_id IS UNIQUE",
        "CREATE CONSTRAINT biomarker_id_unique IF NOT EXISTS FOR (b:Biomarker) REQUIRE b.biomarker_id IS UNIQUE",
        "CREATE CONSTRAINT guideline_id_unique IF NOT EXISTS FOR (g:Guideline) REQUIRE g.guideline_id IS UNIQUE",
    ]

    for constraint in constraints:
        try:
            session.run(constraint)
        except:
            pass  # Constraint may already exist

    # Create indexes for fast lookups
    indexes = [
        "CREATE INDEX patient_stage_idx IF NOT EXISTS FOR (p:Patient) ON (p.stage)",
        "CREATE INDEX patient_histology_idx IF NOT EXISTS FOR (p:Patient) ON (p.histology)",
        "CREATE INDEX patient_ps_idx IF NOT EXISTS FOR (p:Patient) ON (p.ps)",
        "CREATE INDEX treatment_name_idx IF NOT EXISTS FOR (t:Treatment) ON (t.name)",
        "CREATE INDEX biomarker_test_idx IF NOT EXISTS FOR (b:Biomarker) ON (b.test_name)",
    ]

    for index in indexes:
        try:
            session.run(index)
        except:
            pass

    print("✓ Schema created with constraints and indexes")

def create_guideline_nodes(session):
    """Create NICE guideline nodes"""

    for guideline in GUIDELINES:
        session.run("""
            MERGE (g:Guideline {guideline_id: $guideline_id})
            SET g.rule = $rule,
                g.evidence = $evidence,
                g.source = 'NICE CG121'
        """, guideline_id=guideline["id"], rule=guideline["rule"], evidence=guideline["evidence"])

    print(f"✓ Created {len(GUIDELINES)} guideline nodes")

def generate_patient_data(num):
    """Generate comprehensive patient data"""

    age = random.randint(50, 85)
    sex = random.choice(["M", "F"])
    stage = random.choice(STAGES)
    histology = random.choice(list(HISTOLOGIES.keys()))
    cancer_type = HISTOLOGIES[histology]
    ps = random.choices([0, 1, 2, 3], weights=[20, 50, 25, 5])[0]
    laterality = random.choice(LATERALITY)
    fev1 = random.randint(40, 95)

    # Comorbidities
    num_comorbidities = random.choices([0, 1, 2, 3, 4], weights=[30, 35, 20, 10, 5])[0]
    if age > 70: num_comorbidities += 1
    if ps >= 2: num_comorbidities += 1
    comorbidities = random.sample(COMORBIDITIES, min(num_comorbidities, len(COMORBIDITIES)))

    # Biomarkers for advanced NSCLC
    biomarkers = {}
    if cancer_type == "NSCLC" and stage in ["IIIB", "IV"]:
        egfr_status = random.choice(["Negative", "Ex19del", "L858R", "T790M", "Exon 20 insertion"])
        alk_status = random.choice(["Negative", "Positive"])
        ros1_status = random.choice(["Negative", "Positive"]) if alk_status == "Negative" else "Negative"
        pdl1_tps = random.choice([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90])

        biomarkers = {
            "EGFR": egfr_status,
            "ALK": alk_status,
            "ROS1": ros1_status,
            "PD-L1 TPS": pdl1_tps,
            "KRAS": random.choice(["Wild-type", "G12C", "G12V", "G12D"]),
            "BRAF": random.choice(["Wild-type", "V600E"])
        }

    return {
        "patient_id": f"PT{num:03d}",
        "name": f"Patient {num}",
        "age": age,
        "sex": sex,
        "stage": stage,
        "histology": histology,
        "cancer_type": cancer_type,
        "ps": ps,
        "laterality": laterality,
        "fev1": fev1,
        "comorbidities": comorbidities,
        "biomarkers": biomarkers,
        "diagnosis_date": (datetime.now() - timedelta(days=random.randint(30, 730))).isoformat(),
        "created_at": datetime.now().isoformat()
    }

def create_patient_with_relationships(session, patient):
    """Create patient node with ALL relationships and properties"""

    # Create Patient node
    session.run("""
        CREATE (p:Patient {
            patient_id: $patient_id,
            name: $name,
            age: $age,
            sex: $sex,
            stage: $stage,
            histology: $histology,
            cancer_type: $cancer_type,
            ps: $ps,
            laterality: $laterality,
            fev1: $fev1,
            diagnosis_date: datetime($diagnosis_date),
            created_at: datetime($created_at)
        })
    """, **patient)

    # Create Comorbidity nodes and relationships
    for comorbidity in patient["comorbidities"]:
        session.run("""
            MATCH (p:Patient {patient_id: $patient_id})
            MERGE (c:Comorbidity {name: $comorbidity})
            CREATE (p)-[:HAS_COMORBIDITY]->(c)
        """, patient_id=patient["patient_id"], comorbidity=comorbidity)

    # Create Biomarker nodes and relationships
    for test_name, result in patient["biomarkers"].items():
        biomarker_id = f"{patient['patient_id']}_{test_name.replace(' ', '_').replace('-', '')}"
        test_date = (datetime.now() - timedelta(days=random.randint(7, 60))).date().isoformat()
        session.run("""
            MATCH (p:Patient {patient_id: $patient_id})
            CREATE (b:Biomarker {
                biomarker_id: $biomarker_id,
                test_name: $test_name,
                result: $result,
                test_date: date($test_date)
            })
            CREATE (p)-[:HAS_BIOMARKER]->(b)
        """,
            patient_id=patient["patient_id"],
            biomarker_id=biomarker_id,
            test_name=test_name,
            result=result,
            test_date=test_date
        )

    # Determine appropriate treatment
    treatment_name, guideline_id = select_treatment(patient)

    # Create Treatment node and relationship
    treatment_id = f"{patient['patient_id']}_T1"
    start_date = (datetime.now() - timedelta(days=random.randint(7, 180))).date().isoformat()
    response = random.choice(["Complete Response", "Partial Response", "Stable Disease", "Progressive Disease"])

    session.run("""
        MATCH (p:Patient {patient_id: $patient_id})
        CREATE (t:Treatment {
            treatment_id: $treatment_id,
            name: $treatment_name,
            start_date: date($start_date),
            response: $response,
            ongoing: $ongoing
        })
        CREATE (p)-[:RECEIVED_TREATMENT {
            start_date: date($start_date),
            response: $response
        }]->(t)
    """,
        patient_id=patient["patient_id"],
        treatment_id=treatment_id,
        treatment_name=treatment_name,
        start_date=start_date,
        response=response,
        ongoing=response in ["Partial Response", "Stable Disease"]
    )

    # Link to applicable guideline
    if guideline_id:
        session.run("""
            MATCH (p:Patient {patient_id: $patient_id})
            MATCH (g:Guideline {guideline_id: $guideline_id})
            CREATE (p)-[:GUIDED_BY]->(g)
        """, patient_id=patient["patient_id"], guideline_id=guideline_id)

def select_treatment(patient):
    """Select appropriate treatment based on patient characteristics"""

    stage = patient["stage"]
    cancer_type = patient["cancer_type"]
    ps = patient["ps"]
    biomarkers = patient["biomarkers"]

    # PS 3-4 → Best supportive care (R7)
    if ps >= 3:
        return "Best Supportive Care", "R7"

    # SCLC
    if cancer_type == "SCLC":
        if stage in ["IA", "IB", "IIA", "IIB", "IIIA", "IIIB"]:
            return "Concurrent Chemoradiotherapy (Cisplatin + Etoposide)", "R3"
        else:
            return "Chemotherapy (Carboplatin + Etoposide) + Atezolizumab", None

    # NSCLC
    # Check biomarkers for targeted therapy
    if biomarkers.get("EGFR") in ["Ex19del", "L858R"]:
        return "Osimertinib (EGFR TKI)", "R4"

    if biomarkers.get("ALK") == "Positive":
        return "Alectinib (ALK inhibitor)", "R5"

    if biomarkers.get("PD-L1 TPS", 0) >= 50 and ps <= 1:
        return "Pembrolizumab (Immunotherapy)", "R6"

    # Stage-based treatment
    if stage in ["IA", "IB", "IIA", "IIB"]:
        return "Surgery (Lobectomy)", "R1"
    elif stage in ["IIIA", "IIIB"]:
        if ps <= 1:
            return "Concurrent Chemoradiotherapy", None
        else:
            return "Sequential Chemoradiotherapy", None
    else:  # Stage IV
        if ps <= 1:
            return "Chemotherapy (Carboplatin + Pemetrexed)", "R2"
        else:
            return "Best Supportive Care", "R7"

def create_patient_similarity_relationships(session):
    """Create similarity relationships between patients"""

    print("Creating patient similarity relationships...")

    # Similar stage and histology
    session.run("""
        MATCH (p1:Patient), (p2:Patient)
        WHERE p1.patient_id < p2.patient_id
        AND p1.stage = p2.stage
        AND p1.histology = p2.histology
        AND abs(p1.age - p2.age) <= 10
        CREATE (p1)-[:SIMILAR_TO {
            score: 0.85,
            basis: 'stage_histology_age'
        }]->(p2)
    """)

    print("✓ Created similarity relationships")

def ingest_data(num_patients=50, clear_first=True):
    """Main ingestion function"""

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        print(f"✓ Connected to Neo4j at {NEO4J_URI}")
        print()

        with driver.session(database=NEO4J_DATABASE) as session:
            # Clear database if requested
            if clear_first:
                clear_database(session)
                print()

            # Create schema
            create_comprehensive_schema(session)
            print()

            # Create guideline nodes
            create_guideline_nodes(session)
            print()

            # Ingest patients
            print(f"Ingesting {num_patients} patients with complete relationships...")
            print()

            successful = 0
            failed = 0

            for i in range(1, num_patients + 1):
                try:
                    patient = generate_patient_data(i)

                    print(f"[{i}/{num_patients}] {patient['patient_id']}: "
                          f"{patient['age']}{patient['sex']}, "
                          f"Stage {patient['stage']} {patient['histology']}, "
                          f"PS {patient['ps']}")

                    create_patient_with_relationships(session, patient)

                    print(f"  ✓ Created with {len(patient['comorbidities'])} comorbidities, "
                          f"{len(patient['biomarkers'])} biomarkers")
                    successful += 1

                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    failed += 1

            print()

            # Create similarity relationships
            create_patient_similarity_relationships(session)

        driver.close()

        print()
        print("=" * 70)
        print(f"Ingestion Complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print("=" * 70)
        print()

        return successful, failed

    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        print("Make sure Neo4j is running:")
        print("  docker run -d --name neo4j-lca -p 7474:7474 -p 7687:7687 \\")
        print("    -e NEO4J_AUTH=neo4j/123456789 neo4j:5.15")
        return 0, 0

def verify_ingestion(driver):
    """Verify all data and relationships were created"""

    print("Verifying ingestion...")
    print()

    with driver.session(database=NEO4J_DATABASE) as session:
        # Count all nodes
        stats = {}

        for label in ["Patient", "Treatment", "Biomarker", "Comorbidity", "Guideline"]:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
            stats[label] = result.single()["count"]

        print("Node Counts:")
        for label, count in stats.items():
            print(f"  {label}: {count}")

        print()

        # Count relationships
        rel_result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(r) as count
            ORDER BY count DESC
        """)

        print("Relationship Counts:")
        for record in rel_result:
            print(f"  {record['rel_type']}: {record['count']}")

        print()

        # Sample patient with all relationships
        sample = session.run("""
            MATCH (p:Patient)
            OPTIONAL MATCH (p)-[:HAS_COMORBIDITY]->(c:Comorbidity)
            OPTIONAL MATCH (p)-[:HAS_BIOMARKER]->(b:Biomarker)
            OPTIONAL MATCH (p)-[:RECEIVED_TREATMENT]->(t:Treatment)
            OPTIONAL MATCH (p)-[:GUIDED_BY]->(g:Guideline)
            RETURN p.patient_id as id,
                   p.age as age,
                   p.sex as sex,
                   p.stage as stage,
                   p.histology as histology,
                   collect(DISTINCT c.name) as comorbidities,
                   collect(DISTINCT CASE WHEN b.test_name IS NOT NULL THEN b.test_name + ': ' + toString(b.result) ELSE null END) as biomarkers,
                   collect(DISTINCT t.name) as treatments,
                   collect(DISTINCT g.rule) as guidelines
            LIMIT 2
        """)

        print("Sample Patients with Relationships:")
        for record in sample:
            comorbidities_list = [c for c in record['comorbidities'] if c is not None]
            biomarkers_list = [b for b in record['biomarkers'] if b is not None]
            treatments_list = [t for t in record['treatments'] if t is not None]
            guidelines_list = [g for g in record['guidelines'] if g is not None]

            print(f"\n  {record['id']}: {record['age']}{record['sex']}, "
                  f"Stage {record['stage']} {record['histology']}")
            print(f"    Comorbidities: {', '.join(comorbidities_list) if comorbidities_list else 'None'}")
            print(f"    Biomarkers: {len(biomarkers_list)}")
            print(f"    Treatments: {', '.join(treatments_list) if treatments_list else 'None'}")
            print(f"    Guidelines: {guidelines_list[0] if guidelines_list else 'None'}")

        print()
        print("✓ All nodes and relationships created successfully!")
        print()
        print("You can now query through Claude Desktop:")
        print('  - "How many patients are in the database?"')
        print('  - "Find patients with EGFR mutations"')
        print('  - "Show me stage IV patients"')
        print('  - "Find patients similar to: 65M, stage IIIA, PS 1"')

def main():
    """Main execution"""

    # Check for command line argument to skip confirmation
    skip_confirm = len(sys.argv) > 1 and sys.argv[1] in ['-y', '--yes', 'yes']
    clear_first = '--no-clear' not in sys.argv

    if not skip_confirm:
        try:
            response = input("Ingest 50 patients with complete graph structure? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("Cancelled.")
                return 0
        except EOFError:
            print("\nNo input provided. Use '-y' or '--yes' to skip confirmation.")
            print("Example: python scripts/ingest_sample_data.py -y")
            return 1

    print()

    successful, failed = ingest_data(50, clear_first=clear_first)

    if successful > 0:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        verify_ingestion(driver)
        driver.close()

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
