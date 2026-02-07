"""
Clinical Data Ingestion Service
================================

Seeds Neo4j with clinical guidelines, sample patient cohorts, and reference data.
Uses ontology-driven entity extraction and APOC/Neosemantics patterns from GoingMeta sessions.

Data Sources:
- NCCN/NICE/ESMO guideline rules (from GuidelineRuleEngine)
- Synthetic patient cohorts for demonstration
- Biomarker-therapy mapping reference data
- Clinical trial reference data
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random

from neo4j import GraphDatabase

from ..logging_config import get_logger

logger = get_logger(__name__)


class ClinicalDataIngestor:
    """
    Ingests clinical reference data into Neo4j knowledge graph.

    Creates:
    - Guideline nodes with evidence levels and treatment mappings
    - Sample patient cohort for similar case matching
    - Biomarker-therapy reference relationships
    - Drug nodes with mechanism of action and side effects
    - Clinical trial reference data
    - Ontology class hierarchy (LUCADA)
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
                logger.info(f"ClinicalDataIngestor connected to Neo4j at {uri}")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def ingest_all(self) -> Dict[str, int]:
        """Run all ingestion steps and return counts"""
        if not self.driver:
            return {"error": "No database connection"}

        counts = {}
        counts["constraints"] = self._create_constraints()
        counts["guidelines"] = self._ingest_guidelines()
        counts["drugs"] = self._ingest_drug_reference_data()
        counts["biomarker_therapy_map"] = self._ingest_biomarker_therapy_map()
        counts["sample_patients"] = self._ingest_sample_patient_cohort()
        counts["clinical_trials"] = self._ingest_clinical_trials()
        counts["ontology_classes"] = self._ingest_ontology_hierarchy()

        logger.info(f"Clinical data ingestion complete: {counts}")
        return counts

    def _create_constraints(self) -> int:
        """Create uniqueness constraints and indexes"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Guideline) REQUIRE g.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:ClinicalTrial) REQUIRE t.nct_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (b:BiomarkerType) REQUIRE b.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (p:Patient) ON (p.tnm_stage)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Patient) ON (p.histology_type)",
            "CREATE INDEX IF NOT EXISTS FOR (d:TreatmentDecision) ON (d.treatment)",
        ]
        count = 0
        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    count += 1
                except Exception as e:
                    logger.debug(f"Constraint/index already exists or error: {e}")
        return count

    def _ingest_guidelines(self) -> int:
        """Ingest clinical guideline nodes"""
        guidelines = [
            {
                "id": "NICE_CG121_R1", "name": "NICE CG121 - Chemo for Stage III-IV NSCLC",
                "source": "NICE", "version": "CG121 2011",
                "category": "NSCLC", "stage_applicability": ["III", "IIIA", "IIIB", "IV"],
                "treatment": "Chemotherapy", "evidence_level": "Grade A",
                "intent": "Palliative", "summary": "Offer chemotherapy to PS 0-1 patients with stage III-IV NSCLC"
            },
            {
                "id": "NICE_CG121_R2", "name": "NICE CG121 - Surgery for Early NSCLC",
                "source": "NICE", "version": "CG121 2011",
                "category": "NSCLC", "stage_applicability": ["IA", "IB", "IIA", "IIB"],
                "treatment": "Surgery", "evidence_level": "Grade A",
                "intent": "Curative", "summary": "Offer surgical resection for stage I-II NSCLC with PS 0-1"
            },
            {
                "id": "NCCN_NSCLC_2025_EGFR", "name": "NCCN 2025 - EGFR TKI for EGFR+ NSCLC",
                "source": "NCCN", "version": "2025.2",
                "category": "NSCLC", "stage_applicability": ["IIIB", "IV"],
                "treatment": "Osimertinib", "evidence_level": "Category 1",
                "intent": "Palliative", "summary": "Preferred first-line for EGFR Ex19del/L858R: osimertinib 80mg daily",
                "key_trial": "FLAURA", "median_pfs": "18.9 months", "median_os": "38.6 months"
            },
            {
                "id": "NCCN_NSCLC_2025_ALK", "name": "NCCN 2025 - ALK Inhibitor for ALK+ NSCLC",
                "source": "NCCN", "version": "2025.2",
                "category": "NSCLC", "stage_applicability": ["IIIB", "IV"],
                "treatment": "Alectinib", "evidence_level": "Category 1",
                "intent": "Palliative", "summary": "Preferred first-line for ALK+: alectinib 600mg BID",
                "key_trial": "ALEX", "median_pfs": "34.8 months"
            },
            {
                "id": "NCCN_NSCLC_2025_PDL1_HIGH", "name": "NCCN 2025 - IO Monotherapy for PD-L1≥50%",
                "source": "NCCN", "version": "2025.2",
                "category": "NSCLC", "stage_applicability": ["IV"],
                "treatment": "Pembrolizumab", "evidence_level": "Category 1",
                "intent": "Palliative", "summary": "Pembrolizumab monotherapy for PD-L1≥50% without driver mutations",
                "key_trial": "KEYNOTE-024", "five_year_os": "31.9%"
            },
            {
                "id": "NCCN_NSCLC_2025_CHEMO_IO", "name": "NCCN 2025 - Chemo-IO for PD-L1 1-49%",
                "source": "NCCN", "version": "2025.2",
                "category": "NSCLC", "stage_applicability": ["IV"],
                "treatment": "Carboplatin/Pemetrexed/Pembrolizumab", "evidence_level": "Category 1",
                "intent": "Palliative", "summary": "Chemo-immunotherapy for PD-L1 1-49% without driver mutations",
                "key_trial": "KEYNOTE-189", "median_os": "22 months"
            },
            {
                "id": "ADAURA_ADJUVANT", "name": "ADAURA - Adjuvant Osimertinib",
                "source": "ASCO/NCCN", "version": "2025",
                "category": "NSCLC", "stage_applicability": ["IB", "II", "IIA", "IIB", "IIIA"],
                "treatment": "Adjuvant Osimertinib", "evidence_level": "Category 1",
                "intent": "Curative", "summary": "Adjuvant osimertinib for resected IB-IIIA EGFR+ NSCLC",
                "key_trial": "ADAURA", "dfs_hr": "0.17", "three_year_dfs": "89%"
            },
            {
                "id": "NCCN_SCLC_2025_LD", "name": "NCCN 2025 - Limited Disease SCLC",
                "source": "NCCN", "version": "2025.1",
                "category": "SCLC", "stage_applicability": ["Limited"],
                "treatment": "Concurrent ChemoRT + PCI", "evidence_level": "Category 1",
                "intent": "Curative", "summary": "Cisplatin/etoposide + concurrent RT for LD-SCLC"
            },
            {
                "id": "NCCN_SCLC_2025_ED", "name": "NCCN 2025 - Extensive Disease SCLC",
                "source": "NCCN", "version": "2025.1",
                "category": "SCLC", "stage_applicability": ["Extensive"],
                "treatment": "Chemo + Atezolizumab", "evidence_level": "Category 1",
                "intent": "Palliative", "summary": "Carboplatin/etoposide + atezolizumab for ED-SCLC",
                "key_trial": "IMpower133", "median_os": "12.3 months"
            },
        ]

        count = 0
        with self.driver.session(database=self.database) as session:
            for g in guidelines:
                session.run("""
                    MERGE (gl:Guideline {id: $id})
                    SET gl += $props,
                        gl.updated_at = datetime()
                """, {"id": g["id"], "props": g})
                count += 1
        return count

    def _ingest_drug_reference_data(self) -> int:
        """Ingest drug nodes with clinical pharmacology data"""
        drugs = [
            {
                "name": "Osimertinib", "brand": "Tagrisso",
                "class": "EGFR TKI (3rd generation)", "mechanism": "Irreversible EGFR inhibitor including T790M",
                "dosing": "80mg oral daily", "indication": "EGFR-mutated NSCLC",
                "common_side_effects": ["Diarrhea", "Rash", "Dry skin", "Paronychia", "Stomatitis"],
                "serious_side_effects": ["Interstitial lung disease", "QTc prolongation", "Cardiomyopathy"],
                "monitoring": ["ECG at baseline and monthly", "LFTs q4 weeks", "PFTs if respiratory symptoms"],
                "rxnorm_code": "1860494"
            },
            {
                "name": "Pembrolizumab", "brand": "Keytruda",
                "class": "Anti-PD-1 checkpoint inhibitor", "mechanism": "Blocks PD-1/PD-L1 interaction, restoring T-cell activity",
                "dosing": "200mg IV q3w or 400mg IV q6w", "indication": "PD-L1+ NSCLC, various solid tumors",
                "common_side_effects": ["Fatigue", "Pruritus", "Rash", "Diarrhea", "Nausea"],
                "serious_side_effects": ["Pneumonitis", "Colitis", "Hepatitis", "Thyroid dysfunction", "Type 1 DM"],
                "monitoring": ["TSH q6 weeks", "LFTs q4 weeks", "Cortisol if symptoms", "Blood glucose"],
                "rxnorm_code": "1547545"
            },
            {
                "name": "Alectinib", "brand": "Alecensa",
                "class": "ALK TKI (2nd generation)", "mechanism": "Selective ALK inhibitor with CNS penetration",
                "dosing": "600mg oral BID with food", "indication": "ALK-rearranged NSCLC",
                "common_side_effects": ["Constipation", "Edema", "Myalgia", "Anemia"],
                "serious_side_effects": ["Hepatotoxicity", "ILD/Pneumonitis", "Bradycardia", "Renal impairment"],
                "monitoring": ["LFTs q2 weeks x 3 months then monthly", "CPK levels", "Heart rate"],
                "rxnorm_code": "1727581"
            },
            {
                "name": "Carboplatin", "brand": "Paraplatin",
                "class": "Platinum-based chemotherapy", "mechanism": "DNA crosslinking, inhibits replication",
                "dosing": "AUC 5-6 IV q3w", "indication": "NSCLC, SCLC, ovarian cancer",
                "common_side_effects": ["Nausea", "Vomiting", "Myelosuppression", "Peripheral neuropathy"],
                "serious_side_effects": ["Severe thrombocytopenia", "Nephrotoxicity", "Anaphylaxis"],
                "monitoring": ["CBC before each cycle", "Renal function", "Audiometry if symptoms"],
                "rxnorm_code": "40048"
            },
            {
                "name": "Durvalumab", "brand": "Imfinzi",
                "class": "Anti-PD-L1 checkpoint inhibitor", "mechanism": "Blocks PD-L1, restoring anti-tumor immunity",
                "dosing": "10mg/kg IV q2w or 1500mg IV q4w", "indication": "Stage III NSCLC post-chemoRT, ED-SCLC",
                "common_side_effects": ["Cough", "Fatigue", "Pneumonitis", "Rash"],
                "serious_side_effects": ["Immune-mediated pneumonitis", "Hepatitis", "Colitis", "Endocrinopathies"],
                "monitoring": ["TSH q4-6 weeks", "LFTs", "PFTs if respiratory symptoms"],
                "rxnorm_code": "1919459"
            },
            {
                "name": "Sotorasib", "brand": "Lumakras",
                "class": "KRAS G12C inhibitor", "mechanism": "Covalent inhibitor of KRAS G12C mutant protein",
                "dosing": "960mg oral daily", "indication": "KRAS G12C-mutated NSCLC",
                "common_side_effects": ["Diarrhea", "Musculoskeletal pain", "Nausea", "Fatigue", "Hepatotoxicity"],
                "serious_side_effects": ["Grade 3-4 hepatotoxicity", "ILD/Pneumonitis"],
                "monitoring": ["LFTs q3 weeks x 3 months", "PFTs if symptoms"],
                "rxnorm_code": "2470937"
            },
        ]

        count = 0
        with self.driver.session(database=self.database) as session:
            for d in drugs:
                session.run("""
                    MERGE (drug:Drug {name: $name})
                    SET drug += $props,
                        drug.updated_at = datetime()
                """, {"name": d["name"], "props": d})
                count += 1
        return count

    def _ingest_biomarker_therapy_map(self) -> int:
        """Ingest biomarker-to-therapy mapping relationships"""
        mappings = [
            ("EGFR", "Osimertinib", "first_line", "Ex19del or L858R mutation", "Category 1"),
            ("EGFR", "Erlotinib", "alternative", "Any sensitizing mutation", "Category 2A"),
            ("ALK", "Alectinib", "first_line", "ALK rearrangement", "Category 1"),
            ("ALK", "Lorlatinib", "first_line", "ALK rearrangement (preferred if CNS mets)", "Category 1"),
            ("ROS1", "Crizotinib", "first_line", "ROS1 fusion", "Category 2A"),
            ("ROS1", "Entrectinib", "first_line", "ROS1 fusion (preferred if CNS mets)", "Category 2A"),
            ("BRAF_V600E", "Dabrafenib+Trametinib", "first_line", "BRAF V600E mutation", "Category 2A"),
            ("KRAS_G12C", "Sotorasib", "second_line", "KRAS G12C mutation (post chemo)", "Category 2A"),
            ("PD-L1_HIGH", "Pembrolizumab", "first_line", "PD-L1 TPS ≥50% without driver", "Category 1"),
            ("PD-L1_LOW", "Chemo+Pembrolizumab", "first_line", "PD-L1 TPS 1-49% without driver", "Category 1"),
            ("MET_EX14", "Capmatinib", "first_line", "MET exon 14 skipping", "Category 2A"),
            ("RET", "Selpercatinib", "first_line", "RET fusion", "Category 2A"),
            ("NTRK", "Larotrectinib", "first_line", "NTRK fusion", "Category 2A"),
            ("HER2", "Trastuzumab-deruxtecan", "second_line", "HER2 mutation", "Category 2A"),
        ]

        count = 0
        with self.driver.session(database=self.database) as session:
            for biomarker, drug, line, criteria, evidence in mappings:
                session.run("""
                    MERGE (b:BiomarkerType {name: $biomarker})
                    MERGE (d:Drug {name: $drug})
                    MERGE (b)-[r:INDICATES_THERAPY]->(d)
                    SET r.line_of_therapy = $line,
                        r.criteria = $criteria,
                        r.evidence_level = $evidence,
                        r.updated_at = datetime()
                """, {
                    "biomarker": biomarker, "drug": drug,
                    "line": line, "criteria": criteria, "evidence": evidence
                })
                count += 1
        return count

    def _ingest_sample_patient_cohort(self) -> int:
        """Ingest a diverse sample patient cohort for similar case matching and demonstration"""
        patients = [
            {
                "patient_id": "COHORT_001", "age": 65, "sex": "M",
                "tnm_stage": "IIIA", "histology_type": "Adenocarcinoma",
                "performance_status": 1, "laterality": "Right",
                "biomarkers": {"egfr_mutation": True, "egfr_mutation_type": "Ex19del", "pdl1_tps": 15},
                "comorbidities": ["COPD", "Hypertension"],
                "treatment_given": "Concurrent chemoRT → Adjuvant osimertinib",
                "outcome": "Partial response, DFS 24+ months"
            },
            {
                "patient_id": "COHORT_002", "age": 58, "sex": "F",
                "tnm_stage": "IV", "histology_type": "Adenocarcinoma",
                "performance_status": 0, "laterality": "Left",
                "biomarkers": {"egfr_mutation": True, "egfr_mutation_type": "L858R", "pdl1_tps": 5},
                "comorbidities": [],
                "treatment_given": "Osimertinib 80mg daily",
                "outcome": "Partial response, PFS 22 months"
            },
            {
                "patient_id": "COHORT_003", "age": 72, "sex": "M",
                "tnm_stage": "IV", "histology_type": "Adenocarcinoma",
                "performance_status": 1, "laterality": "Right",
                "biomarkers": {"alk_rearrangement": True, "pdl1_tps": 30},
                "comorbidities": ["Diabetes", "CAD"],
                "treatment_given": "Alectinib 600mg BID",
                "outcome": "Complete response, PFS 30+ months"
            },
            {
                "patient_id": "COHORT_004", "age": 55, "sex": "F",
                "tnm_stage": "IV", "histology_type": "Adenocarcinoma",
                "performance_status": 0, "laterality": "Left",
                "biomarkers": {"pdl1_tps": 80, "egfr_mutation": False, "alk_rearrangement": False},
                "comorbidities": [],
                "treatment_given": "Pembrolizumab monotherapy",
                "outcome": "Durable response, OS 36+ months"
            },
            {
                "patient_id": "COHORT_005", "age": 68, "sex": "M",
                "tnm_stage": "IIA", "histology_type": "SquamousCellCarcinoma",
                "performance_status": 0, "laterality": "Right",
                "biomarkers": {"pdl1_tps": 40},
                "comorbidities": ["COPD"],
                "treatment_given": "Lobectomy → Adjuvant cisplatin/vinorelbine",
                "outcome": "Disease-free at 18 months"
            },
            {
                "patient_id": "COHORT_006", "age": 63, "sex": "M",
                "tnm_stage": "Extensive", "histology_type": "SmallCellCarcinoma",
                "performance_status": 1, "laterality": "Right",
                "biomarkers": {},
                "comorbidities": ["Hypertension"],
                "treatment_given": "Carboplatin/etoposide + atezolizumab",
                "outcome": "Partial response, OS 14 months"
            },
            {
                "patient_id": "COHORT_007", "age": 45, "sex": "F",
                "tnm_stage": "IV", "histology_type": "Adenocarcinoma",
                "performance_status": 0, "laterality": "Left",
                "biomarkers": {"ros1_rearrangement": True, "pdl1_tps": 10},
                "comorbidities": [],
                "treatment_given": "Entrectinib",
                "outcome": "Complete response including brain metastases, PFS 20+ months"
            },
            {
                "patient_id": "COHORT_008", "age": 70, "sex": "M",
                "tnm_stage": "IIIB", "histology_type": "Adenocarcinoma",
                "performance_status": 1, "laterality": "Right",
                "biomarkers": {"pdl1_tps": 60, "egfr_mutation": False, "alk_rearrangement": False},
                "comorbidities": ["COPD", "Diabetes"],
                "treatment_given": "Concurrent chemoRT → Durvalumab consolidation",
                "outcome": "Partial response, PFS 18 months"
            },
            {
                "patient_id": "COHORT_009", "age": 62, "sex": "F",
                "tnm_stage": "IV", "histology_type": "Adenocarcinoma",
                "performance_status": 1, "laterality": "Left",
                "biomarkers": {"kras_mutation": "G12C", "pdl1_tps": 25},
                "comorbidities": ["Hypertension"],
                "treatment_given": "Carboplatin/pemetrexed/pembrolizumab → Sotorasib (2L)",
                "outcome": "PFS 8 months (1L), PFS 6 months (2L)"
            },
            {
                "patient_id": "COHORT_010", "age": 78, "sex": "M",
                "tnm_stage": "IV", "histology_type": "SquamousCellCarcinoma",
                "performance_status": 2, "laterality": "Right",
                "biomarkers": {"pdl1_tps": 90},
                "comorbidities": ["COPD", "CHF", "CKD"],
                "treatment_given": "Pembrolizumab monotherapy (dose-reduced)",
                "outcome": "Stable disease, OS 12 months"
            },
        ]

        count = 0
        with self.driver.session(database=self.database) as session:
            for p in patients:
                # Create patient node
                biomarkers = p.pop("biomarkers", {})
                comorbidities = p.pop("comorbidities", [])
                treatment = p.pop("treatment_given", "")
                outcome = p.pop("outcome", "")

                session.run("""
                    MERGE (pat:Patient {patient_id: $patient_id})
                    SET pat += $props,
                        pat.treatment_given = $treatment,
                        pat.outcome = $outcome,
                        pat.is_cohort = true,
                        pat.updated_at = datetime()
                """, {
                    "patient_id": p["patient_id"],
                    "props": p,
                    "treatment": treatment,
                    "outcome": outcome
                })

                # Create biomarker nodes
                for bm_name, bm_value in biomarkers.items():
                    display_name = bm_name.replace("_", " ").upper()
                    if isinstance(bm_value, bool):
                        display_value = "Positive" if bm_value else "Negative"
                    elif bm_name == "pdl1_tps":
                        display_value = f"{bm_value}%"
                    else:
                        display_value = str(bm_value)

                    session.run("""
                        MATCH (pat:Patient {patient_id: $patient_id})
                        MERGE (b:Biomarker {marker_type: $marker_type, patient_id: $patient_id})
                        SET b.value = $value, b.display_value = $display_value
                        MERGE (pat)-[:HAS_BIOMARKER]->(b)
                    """, {
                        "patient_id": p["patient_id"],
                        "marker_type": display_name,
                        "value": str(bm_value),
                        "display_value": display_value
                    })

                # Create comorbidity nodes
                for comorb in comorbidities:
                    session.run("""
                        MATCH (pat:Patient {patient_id: $patient_id})
                        MERGE (c:Comorbidity {name: $name})
                        MERGE (pat)-[:HAS_COMORBIDITY]->(c)
                    """, {
                        "patient_id": p["patient_id"],
                        "name": comorb
                    })

                count += 1
        return count

    def _ingest_clinical_trials(self) -> int:
        """Ingest reference clinical trial data"""
        trials = [
            {
                "nct_id": "NCT02296125", "name": "FLAURA",
                "phase": "Phase III", "status": "Completed",
                "condition": "EGFR-mutated advanced NSCLC",
                "intervention": "Osimertinib vs gefitinib/erlotinib",
                "primary_endpoint": "PFS", "result": "PFS 18.9 vs 10.2 months (HR 0.46)",
                "biomarker_requirement": "EGFR Ex19del or L858R"
            },
            {
                "nct_id": "NCT02075840", "name": "ALEX",
                "phase": "Phase III", "status": "Completed",
                "condition": "ALK-positive advanced NSCLC",
                "intervention": "Alectinib vs crizotinib",
                "primary_endpoint": "PFS", "result": "PFS 34.8 vs 10.9 months (HR 0.43)",
                "biomarker_requirement": "ALK rearrangement"
            },
            {
                "nct_id": "NCT02142738", "name": "KEYNOTE-024",
                "phase": "Phase III", "status": "Completed",
                "condition": "PD-L1≥50% advanced NSCLC",
                "intervention": "Pembrolizumab vs platinum chemotherapy",
                "primary_endpoint": "PFS", "result": "5-year OS 31.9% vs 16.3%",
                "biomarker_requirement": "PD-L1 TPS ≥50%"
            },
            {
                "nct_id": "NCT02578680", "name": "KEYNOTE-189",
                "phase": "Phase III", "status": "Completed",
                "condition": "Non-squamous advanced NSCLC",
                "intervention": "Chemo + pembrolizumab vs chemo + placebo",
                "primary_endpoint": "OS, PFS", "result": "Median OS 22.0 vs 10.6 months (HR 0.56)",
                "biomarker_requirement": "Any PD-L1 (benefit across all subgroups)"
            },
            {
                "nct_id": "NCT02511106", "name": "ADAURA",
                "phase": "Phase III", "status": "Completed",
                "condition": "Resected IB-IIIA EGFR+ NSCLC",
                "intervention": "Adjuvant osimertinib vs placebo",
                "primary_endpoint": "DFS", "result": "DFS HR 0.17 (Stage II-IIIA); 3-year DFS 89% vs 53%",
                "biomarker_requirement": "EGFR Ex19del or L858R"
            },
            {
                "nct_id": "NCT03003962", "name": "IMpower133",
                "phase": "Phase III", "status": "Completed",
                "condition": "Extensive-stage SCLC",
                "intervention": "Carboplatin/etoposide + atezolizumab vs chemo alone",
                "primary_endpoint": "OS", "result": "Median OS 12.3 vs 10.3 months (HR 0.70)",
                "biomarker_requirement": "None (all ES-SCLC)"
            },
            {
                "nct_id": "NCT02453282", "name": "PACIFIC",
                "phase": "Phase III", "status": "Completed",
                "condition": "Stage III NSCLC post-chemoRT",
                "intervention": "Durvalumab consolidation vs placebo",
                "primary_endpoint": "PFS", "result": "5-year OS 42.9% vs 33.4%",
                "biomarker_requirement": "Unresectable stage III, no progression post-chemoRT"
            },
        ]

        count = 0
        with self.driver.session(database=self.database) as session:
            for t in trials:
                session.run("""
                    MERGE (trial:ClinicalTrial {nct_id: $nct_id})
                    SET trial += $props,
                        trial.updated_at = datetime()
                """, {"nct_id": t["nct_id"], "props": t})

                # Link to relevant guideline
                session.run("""
                    MATCH (trial:ClinicalTrial {nct_id: $nct_id})
                    MATCH (g:Guideline)
                    WHERE g.key_trial = $name OR g.treatment CONTAINS $intervention_drug
                    MERGE (g)-[:CITES_TRIAL]->(trial)
                """, {
                    "nct_id": t["nct_id"],
                    "name": t["name"],
                    "intervention_drug": t["intervention"].split(" vs ")[0].split("+")[-1].strip().split("/")[-1].strip()
                })

                count += 1
        return count

    def _ingest_ontology_hierarchy(self) -> int:
        """Ingest LUCADA ontology class hierarchy into Neo4j for browsing"""
        classes = [
            ("LUCADAEntity", None),
            ("Patient", "LUCADAEntity"),
            ("Diagnosis", "LUCADAEntity"),
            ("NeoplasticDisease", "Diagnosis"),
            ("NonSmallCellCarcinoma", "NeoplasticDisease"),
            ("Adenocarcinoma", "NonSmallCellCarcinoma"),
            ("SquamousCellCarcinoma", "NonSmallCellCarcinoma"),
            ("LargeCellCarcinoma", "NonSmallCellCarcinoma"),
            ("SmallCellCarcinoma", "NeoplasticDisease"),
            ("TreatmentPlan", "LUCADAEntity"),
            ("Surgery", "TreatmentPlan"),
            ("Chemotherapy", "TreatmentPlan"),
            ("Radiotherapy", "TreatmentPlan"),
            ("Chemoradiotherapy", "TreatmentPlan"),
            ("Immunotherapy", "TreatmentPlan"),
            ("TargetedTherapy", "TreatmentPlan"),
            ("PalliativeCare", "TreatmentPlan"),
            ("Decision", "LUCADAEntity"),
            ("PatientScenario", "LUCADAEntity"),
            ("Biomarker", "LUCADAEntity"),
            ("Comorbidity", "LUCADAEntity"),
            ("PerformanceStatus", "LUCADAEntity"),
            ("WHOPerfStatusGrade0", "PerformanceStatus"),
            ("WHOPerfStatusGrade1", "PerformanceStatus"),
            ("WHOPerfStatusGrade2", "PerformanceStatus"),
            ("WHOPerfStatusGrade3", "PerformanceStatus"),
            ("WHOPerfStatusGrade4", "PerformanceStatus"),
            ("TNMStage", "LUCADAEntity"),
            ("TreatmentIntent", "LUCADAEntity"),
        ]

        count = 0
        with self.driver.session(database=self.database) as session:
            for class_name, parent_name in classes:
                session.run("""
                    MERGE (c:OntologyClass {name: $name})
                    SET c.source = 'LUCADA', c.updated_at = datetime()
                """, {"name": class_name})

                if parent_name:
                    session.run("""
                        MATCH (c:OntologyClass {name: $child})
                        MATCH (p:OntologyClass {name: $parent})
                        MERGE (c)-[:SUBCLASS_OF]->(p)
                    """, {"child": class_name, "parent": parent_name})

                count += 1
        return count


def get_clinical_data_ingestor(driver=None) -> ClinicalDataIngestor:
    """Factory function to create ClinicalDataIngestor"""
    return ClinicalDataIngestor(driver=driver)
