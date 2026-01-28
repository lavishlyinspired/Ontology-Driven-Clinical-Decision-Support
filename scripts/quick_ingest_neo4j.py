#!/usr/bin/env python3
"""
Quick Neo4j Data Ingestion - Direct Database Population
Faster ingestion without running full workflow
"""

import sys
import os
from pathlib import Path
import random
from datetime import datetime, timedelta
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neo4j import GraphDatabase

print("=" * 70)
print("Quick Neo4j Data Ingestion")
print("=" * 70)
print()

# Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "123456789"

# Sample data pools
STAGES = ["IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IV"]
HISTOLOGIES = ["Adenocarcinoma", "SquamousCellCarcinoma", "SmallCellCarcinoma", "LargeCellCarcinoma"]
LATERALITY = ["Right", "Left", "Bilateral"]
COMORBIDITIES = ["COPD", "CAD", "Hypertension", "Diabetes", "CKD Stage 3", "AFib", "CHF"]

# Treatment options by stage
TREATMENTS = {
    "early": ["Surgery (Lobectomy)", "Surgery (Pneumonectomy)", "SABR/SBRT"],
    "local_advanced": ["Concurrent Chemoradiotherapy", "Neoadjuvant Chemo + Surgery", "Sequential Chemoradiotherapy"],
    "advanced": ["Chemotherapy (Platinum-doublet)", "EGFR TKI (Osimertinib)", "ALK Inhibitor (Alectinib)",
                 "Immunotherapy (Pembrolizumab)", "Chemo + Immunotherapy", "Best Supportive Care"],
    "sclc": ["Concurrent Chemoradiotherapy", "Chemotherapy + Atezolizumab", "PCI"]
}

def create_schema(driver):
    """Create Neo4j schema and indexes"""
    with driver.session() as session:
        # Create constraints
        try:
            session.run("CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE")
        except:
            pass

        # Create indexes
        session.run("CREATE INDEX patient_stage IF NOT EXISTS FOR (p:Patient) ON (p.stage)")
        session.run("CREATE INDEX patient_histology IF NOT EXISTS FOR (p:Patient) ON (p.histology)")

    print("✓ Schema created")

def generate_patient(num):
    """Generate patient data"""
    age = random.randint(50, 85)
    sex = random.choice(["M", "F"])
    stage = random.choice(STAGES)
    histology = random.choice(HISTOLOGIES)
    ps = random.choices([0, 1, 2, 3], weights=[20, 50, 25, 5])[0]
    laterality = random.choice(LATERALITY)
    fev1 = random.randint(40, 95)

    num_comorbidities = random.choices([0, 1, 2, 3], weights=[40, 35, 20, 5])[0]
    if age > 70: num_comorbidities += 1
    comorbidities = random.sample(COMORBIDITIES, min(num_comorbidities, len(COMORBIDITIES)))

    # Biomarkers for advanced NSCLC
    biomarkers = {}
    if histology in ["Adenocarcinoma", "LargeCellCarcinoma"] and stage in ["IIIB", "IV"]:
        biomarkers = {
            "egfr": random.choice(["Negative", "Ex19del", "L858R", "T790M"]),
            "alk": random.choice(["Negative", "Positive"]),
            "ros1": random.choice(["Negative", "Positive"]),
            "pdl1_tps": random.choice([0, 5, 15, 30, 50, 70, 90])
        }

    return {
        "patient_id": f"PT{num:03d}",
        "name": f"Patient {num}",
        "age": age,
        "sex": sex,
        "stage": stage,
        "histology": histology,
        "ps": ps,
        "laterality": laterality,
        "fev1": fev1,
        "comorbidities": comorbidities,
        "biomarkers": biomarkers,
        "created_at": datetime.now().isoformat()
    }

def get_treatment_for_patient(patient):
    """Select appropriate treatment based on patient characteristics"""
    stage = patient["stage"]
    histology = patient["histology"]
    ps = patient["ps"]
    biomarkers = patient.get("biomarkers", {})

    if histology == "SmallCellCarcinoma":
        treatment_pool = TREATMENTS["sclc"]
    elif stage in ["IA", "IB", "IIA", "IIB"]:
        treatment_pool = TREATMENTS["early"]
    elif stage in ["IIIA", "IIIB"]:
        treatment_pool = TREATMENTS["local_advanced"]
    else:  # Stage IV
        treatment_pool = TREATMENTS["advanced"]

    # Modify based on biomarkers
    if biomarkers.get("egfr") in ["Ex19del", "L858R"]:
        return "EGFR TKI (Osimertinib)"
    elif biomarkers.get("alk") == "Positive":
        return "ALK Inhibitor (Alectinib)"
    elif biomarkers.get("pdl1_tps", 0) >= 50 and ps <= 1:
        return "Immunotherapy (Pembrolizumab)"
    elif ps >= 3:
        return "Best Supportive Care"

    return random.choice(treatment_pool)

def create_patient_node(session, patient):
    """Create patient node in Neo4j"""
    query = """
    CREATE (p:Patient {
        patient_id: $patient_id,
        name: $name,
        age: $age,
        sex: $sex,
        stage: $stage,
        histology: $histology,
        ps: $ps,
        laterality: $laterality,
        fev1: $fev1,
        comorbidities: $comorbidities,
        biomarkers: $biomarkers,
        created_at: datetime($created_at)
    })
    RETURN p.patient_id as patient_id
    """

    result = session.run(query,
        patient_id=patient["patient_id"],
        name=patient["name"],
        age=patient["age"],
        sex=patient["sex"],
        stage=patient["stage"],
        histology=patient["histology"],
        ps=patient["ps"],
        laterality=patient["laterality"],
        fev1=patient["fev1"],
        comorbidities=patient["comorbidities"],
        biomarkers=json.dumps(patient["biomarkers"]),
        created_at=patient["created_at"]
    )

    return result.single()["patient_id"]

def create_treatment_history(session, patient_id, treatment):
    """Add treatment history"""
    query = """
    MATCH (p:Patient {patient_id: $patient_id})
    CREATE (t:Treatment {
        treatment_id: $treatment_id,
        name: $treatment_name,
        start_date: date($start_date),
        response: $response
    })
    CREATE (p)-[:RECEIVED_TREATMENT]->(t)
    """

    start_date = datetime.now() - timedelta(days=random.randint(30, 365))
    response = random.choice(["CR", "PR", "SD", "PD"])

    session.run(query,
        patient_id=patient_id,
        treatment_id=f"{patient_id}_T1",
        treatment_name=treatment,
        start_date=start_date.isoformat(),
        response=response
    )

def ingest_patients(num_patients=50):
    """Ingest patients directly into Neo4j"""

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        print("✓ Connected to Neo4j")
        print()

        # Create schema
        create_schema(driver)
        print()

        # Ingest patients
        print(f"Ingesting {num_patients} patients...")
        print()

        successful = 0
        failed = 0

        with driver.session() as session:
            for i in range(1, num_patients + 1):
                try:
                    patient = generate_patient(i)

                    print(f"[{i}/{num_patients}] {patient['patient_id']}: "
                          f"{patient['age']}{patient['sex']}, "
                          f"Stage {patient['stage']} {patient['histology']}, "
                          f"PS {patient['ps']}")

                    # Create patient node
                    patient_id = create_patient_node(session, patient)

                    # Add treatment
                    treatment = get_treatment_for_patient(patient)
                    create_treatment_history(session, patient_id, treatment)

                    print(f"  ✓ Created with treatment: {treatment}")
                    successful += 1

                except Exception as e:
                    print(f"  ✗ Failed: {e}")
                    failed += 1

        driver.close()

        print()
        print("=" * 70)
        print(f"Ingestion Complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print("=" * 70)
        print()

        return successful

    except Exception as e:
        print(f"✗ Error connecting to Neo4j: {e}")
        print()
        print("Make sure Neo4j is running:")
        print("  Docker: docker run -d --name neo4j-lca -p 7474:7474 -p 7687:7687 \\")
        print("          -e NEO4J_AUTH=neo4j/123456789 neo4j:5.15")
        return 0

def verify_data():
    """Verify ingested data"""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        with driver.session() as session:
            # Count patients
            result = session.run("MATCH (p:Patient) RETURN count(p) as count")
            patient_count = result.single()["count"]

            # Count treatments
            result = session.run("MATCH (t:Treatment) RETURN count(t) as count")
            treatment_count = result.single()["count"]

            # Sample patient
            result = session.run("""
                MATCH (p:Patient)-[:RECEIVED_TREATMENT]->(t:Treatment)
                RETURN p.patient_id, p.age, p.sex, p.stage, p.histology, t.name
                LIMIT 3
            """)

            print("✓ Verification Complete")
            print(f"  Patients: {patient_count}")
            print(f"  Treatments: {treatment_count}")
            print()
            print("Sample patients:")
            for record in result:
                print(f"  - {record['p.patient_id']}: {record['p.age']}{record['p.sex']}, "
                      f"Stage {record['p.stage']} {record['p.histology']}, "
                      f"Treatment: {record['t.name']}")

            print()
            print("You can now query this data through Claude Desktop!")
            print()
            print("Try these queries:")
            print('  1. "How many patients are in the database?"')
            print('  2. "Show me stage IV patients"')
            print('  3. "Find patients with EGFR mutations"')
            print('  4. "What treatments are most common?"')

        driver.close()

    except Exception as e:
        print(f"⚠ Verification error: {e}")

def main():
    """Main execution"""

    print("This script will ingest sample patient data directly into Neo4j.")
    print()

    response = input("Continue with ingestion of 50 patients? (yes/no): ").strip().lower()

    if response not in ['yes', 'y']:
        print("Cancelled.")
        return 0

    print()

    successful = ingest_patients(50)

    if successful > 0:
        print()
        verify_data()

    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
