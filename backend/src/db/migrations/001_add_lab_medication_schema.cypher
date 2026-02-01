// Migration 001: Add Lab, Medication, Monitoring, and Clinical Trial Schema
// Created: 2026-02-01
// Purpose: Extend Neo4j schema to support LOINC, RxNorm, Lab-Drug, and Clinical Trials services

// =============================================================================
// STEP 1: CREATE INDEXES FOR PERFORMANCE
// =============================================================================

// LabResult indexes
CREATE INDEX lab_result_loinc IF NOT EXISTS FOR (l:LabResult) ON (l.loinc_code);
CREATE INDEX lab_result_date IF NOT EXISTS FOR (l:LabResult) ON (l.test_date);
CREATE INDEX lab_result_severity IF NOT EXISTS FOR (l:LabResult) ON (l.severity);
CREATE INDEX lab_result_category IF NOT EXISTS FOR (l:LabResult) ON (l.lab_category);

// Medication indexes
CREATE INDEX medication_rxcui IF NOT EXISTS FOR (m:Medication) ON (m.rxcui);
CREATE INDEX medication_name IF NOT EXISTS FOR (m:Medication) ON (m.drug_name);
CREATE INDEX medication_class IF NOT EXISTS FOR (m:Medication) ON (m.drug_class);
CREATE INDEX medication_start_date IF NOT EXISTS FOR (m:Medication) ON (m.start_date);

// DrugInteraction indexes
CREATE INDEX interaction_severity IF NOT EXISTS FOR (i:DrugInteraction) ON (i.severity);

// MonitoringProtocol indexes
CREATE INDEX protocol_regimen IF NOT EXISTS FOR (p:MonitoringProtocol) ON (p.regimen);
CREATE INDEX protocol_date IF NOT EXISTS FOR (p:MonitoringProtocol) ON (p.created_date);

// ClinicalTrial indexes
CREATE INDEX trial_nct IF NOT EXISTS FOR (t:ClinicalTrial) ON (t.nct_id);
CREATE INDEX trial_phase IF NOT EXISTS FOR (t:ClinicalTrial) ON (t.phase);
CREATE INDEX trial_status IF NOT EXISTS FOR (t:ClinicalTrial) ON (t.status);
CREATE INDEX trial_condition IF NOT EXISTS FOR (t:ClinicalTrial) ON (t.condition);

// =============================================================================
// STEP 2: CREATE CONSTRAINTS FOR DATA INTEGRITY
// =============================================================================

// Unique constraints
CREATE CONSTRAINT lab_result_id IF NOT EXISTS FOR (l:LabResult) REQUIRE l.id IS UNIQUE;
CREATE CONSTRAINT medication_id IF NOT EXISTS FOR (m:Medication) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT interaction_id IF NOT EXISTS FOR (i:DrugInteraction) REQUIRE i.id IS UNIQUE;
CREATE CONSTRAINT protocol_id IF NOT EXISTS FOR (p:MonitoringProtocol) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT trial_nct_unique IF NOT EXISTS FOR (t:ClinicalTrial) REQUIRE t.nct_id IS UNIQUE;

// =============================================================================
// NODE TYPE DEFINITIONS (for documentation - will be created by agents)
// =============================================================================

// (:LabResult {
//   id: String (unique),
//   loinc_code: String,
//   loinc_name: String,
//   value: Float,
//   units: String,
//   reference_range: String,
//   interpretation: String,  // "normal", "high", "low", "critical"
//   severity: String,        // "normal", "grade1", "grade2", "grade3", "grade4"
//   test_date: DateTime,
//   lab_category: String,    // "tumor_marker", "hematology", "chemistry", etc.
//   notes: String (optional)
// })

// (:Medication {
//   id: String (unique),
//   rxcui: String,
//   drug_name: String,
//   drug_class: String,      // "EGFR_TKI", "immunotherapy", etc.
//   dose: String,
//   route: String,           // "oral", "IV", etc.
//   frequency: String,       // "once daily", "q3weeks", etc.
//   start_date: DateTime,
//   end_date: DateTime (optional),
//   status: String          // "active", "discontinued", "completed"
// })

// (:DrugInteraction {
//   id: String (unique),
//   drug1_rxcui: String,
//   drug2_rxcui: String,
//   drug1_name: String,
//   drug2_name: String,
//   severity: String,        // "SEVERE", "MODERATE", "MILD"
//   mechanism: String,
//   clinical_effect: String,
//   recommendation: String,
//   evidence_level: String   // "Grade A", "Grade B", "Grade C"
// })

// (:MonitoringProtocol {
//   id: String (unique),
//   protocol_name: String,
//   regimen: String,
//   frequency: String,       // "weekly", "q3weeks", "monthly", etc.
//   duration: String,        // "4 weeks", "indefinite", etc.
//   created_date: DateTime,
//   status: String,         // "active", "completed", "discontinued"
//   lab_tests: List[String] // LOINC codes to monitor
// })

// (:ClinicalTrial {
//   nct_id: String (unique),
//   title: String,
//   phase: String,           // "Phase 1", "Phase 2", "Phase 3", "Phase 4"
//   status: String,          // "Recruiting", "Active", "Completed", etc.
//   condition: String,
//   intervention: String,
//   eligibility_criteria: String,
//   location: String,
//   sponsor: String,
//   start_date: Date (optional),
//   completion_date: Date (optional)
// })

// =============================================================================
// RELATIONSHIP TYPE DEFINITIONS (for documentation)
// =============================================================================

// Lab Result Relationships:
// (Patient)-[:HAS_LAB_RESULT {test_date: DateTime}]->(LabResult)
// (LabResult)-[:INDICATES]->(ClinicalFinding)
// (LabResult)-[:TRIGGERED_BY]->(Medication)
// (LabResult)-[:PROMPTED]->(TreatmentDecision)

// Medication Relationships:
// (Patient)-[:PRESCRIBED {start_date: DateTime, end_date: DateTime}]->(Medication)
// (Medication)-[:TARGETS]->(Biomarker)
// (Medication)-[:INTERACTS_WITH]->(Medication)
// (DrugInteraction)-[:INVOLVES]->(Medication)
// (TreatmentDecision)-[:PRESCRIBED {rationale: String}]->(Medication)

// Monitoring Relationships:
// (Patient)-[:FOLLOWS {start_date: DateTime}]->(MonitoringProtocol)
// (MonitoringProtocol)-[:INCLUDES]->(LabResult)
// (MonitoringProtocol)-[:MONITORS]->(Medication)
// (MonitoringProtocol)-[:BASED_ON]->(Guideline)

// Clinical Trial Relationships:
// (Patient)-[:ELIGIBLE_FOR {eligibility_score: Float}]->(ClinicalTrial)
// (Patient)-[:ENROLLED_IN {enrollment_date: DateTime}]->(ClinicalTrial)
// (ClinicalTrial)-[:INVESTIGATES]->(Biomarker)
// (ClinicalTrial)-[:USES_INTERVENTION]->(Medication)

// =============================================================================
// STEP 3: CREATE SAMPLE DATA (Optional - for testing)
// =============================================================================

// Uncomment the following to create sample test data

// // Sample LabResult
// CREATE (lab:LabResult {
//   id: 'lab_sample_001',
//   loinc_code: '1742-6',
//   loinc_name: 'ALT',
//   value: 45.0,
//   units: 'U/L',
//   reference_range: '7-56 U/L',
//   interpretation: 'normal',
//   severity: 'normal',
//   test_date: datetime('2026-02-01T10:00:00'),
//   lab_category: 'chemistry'
// });

// // Sample Medication
// CREATE (med:Medication {
//   id: 'med_sample_001',
//   rxcui: '2058233',
//   drug_name: 'Osimertinib',
//   drug_class: 'EGFR_TKI',
//   dose: '80 mg',
//   route: 'oral',
//   frequency: 'once daily',
//   start_date: datetime('2026-01-15T00:00:00'),
//   status: 'active'
// });

// // Sample DrugInteraction
// CREATE (interaction:DrugInteraction {
//   id: 'interaction_sample_001',
//   drug1_rxcui: '2058233',
//   drug2_rxcui: '9384',
//   drug1_name: 'Osimertinib',
//   drug2_name: 'Rifampin',
//   severity: 'SEVERE',
//   mechanism: 'CYP3A4 induction',
//   clinical_effect: '78% decreased osimertinib exposure',
//   recommendation: 'Avoid combination. Consider alternative antibiotic.',
//   evidence_level: 'Grade A'
// });

// // Sample MonitoringProtocol
// CREATE (protocol:MonitoringProtocol {
//   id: 'protocol_sample_001',
//   protocol_name: 'Osimertinib Monitoring',
//   regimen: 'Osimertinib 80mg daily',
//   frequency: 'monthly',
//   duration: 'indefinite',
//   created_date: datetime('2026-01-15T00:00:00'),
//   status: 'active',
//   lab_tests: ['1742-6', '1920-8', '2160-0']  // ALT, AST, Creatinine
// });

// // Sample ClinicalTrial
// CREATE (trial:ClinicalTrial {
//   nct_id: 'NCT04487080',
//   title: 'Study of Osimertinib in EGFR-Mutant NSCLC',
//   phase: 'Phase 3',
//   status: 'Recruiting',
//   condition: 'Non-Small Cell Lung Cancer',
//   intervention: 'Osimertinib 80mg',
//   eligibility_criteria: 'EGFR exon 19 deletion or L858R mutation, Stage IV',
//   location: 'Multiple sites',
//   sponsor: 'AstraZeneca'
// });

// =============================================================================
// MIGRATION COMPLETE
// =============================================================================

// Verify indexes
SHOW INDEXES;

// Verify constraints
SHOW CONSTRAINTS;
