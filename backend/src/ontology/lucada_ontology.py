"""
LUCADA Ontology - Lung Cancer Decision Support
Extends SNOMED-CT with domain-specific classes for lung cancer treatment
Based on the paper by Sesen et al., University of Oxford
"""

import os
from typing import Optional, List, Dict, Any
from datetime import datetime as py_datetime
from owlready2 import *
import logging

from .snomed_loader import SNOMEDLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LUCADAOntology:
    """
    LUCADA (Lung Cancer Data) Ontology Implementation

    Implements the ontology structure from Figure 1 of the paper:
    - Patient class with clinical data properties
    - Clinical Finding hierarchy
    - Treatment Plan and Procedure classes
    - Argumentation domain for guideline reasoning
    """

    def __init__(
        self,
        ontology_iri: str = "http://www.ox.ac.uk/lucada",
        snomed_loader: Optional[SNOMEDLoader] = None
    ):
        """
        Initialize LUCADA ontology.

        Args:
            ontology_iri: IRI for the LUCADA ontology
            snomed_loader: Optional pre-loaded SNOMED loader
        """
        self.ontology_iri = ontology_iri
        self.snomed = snomed_loader or SNOMEDLoader()
        self.onto: Optional[Ontology] = None

    def create(self) -> Ontology:
        """
        Create the LUCADA ontology with all classes and properties.

        Returns:
            Created ontology object
        """
        logger.info("Creating LUCADA ontology...")

        # Create new ontology
        self.onto = get_ontology(self.ontology_iri)

        with self.onto:
            # ========================================
            # BASE CLASSES (SNOMED-CT Integration)
            # ========================================

            class SNOMEDConcept(Thing):
                """Base class for SNOMED-CT concepts"""
                pass

            # ========================================
            # PATIENT DOMAIN
            # ========================================

            class Patient(SNOMEDConcept):
                """
                Patient class (SNOMED: 116154003)
                Central class linking all clinical information
                """
                pass

            # Patient Data Properties
            class db_identifier(DataProperty, FunctionalProperty):
                """Database identifier"""
                domain = [Patient]
                range = [str]

            class patient_name(DataProperty, FunctionalProperty):
                """Patient name"""
                domain = [Patient]
                range = [str]

            class sex(DataProperty, FunctionalProperty):
                """Patient sex (M/F)"""
                domain = [Patient]
                range = [str]

            class age_at_diagnosis(DataProperty, FunctionalProperty):
                """Age at diagnosis in years"""
                domain = [Patient]
                range = [int]

            class mdt_discussion_indicator(DataProperty, FunctionalProperty):
                """Whether patient was discussed in MDT"""
                domain = [Patient]
                range = [bool]

            class mdt_discussion_date(DataProperty, FunctionalProperty):
                """Date of MDT discussion"""
                domain = [Patient]
                range = [str]

            class fev1_absolute_amount(DataProperty, FunctionalProperty):
                """FEV1 lung function test (percentage)"""
                domain = [Patient]
                range = [float]

            class clinical_trial_status(DataProperty, FunctionalProperty):
                """Clinical trial participation status"""
                domain = [Patient]
                range = [str]

            class survival_days(DataProperty, FunctionalProperty):
                """Survival in days from diagnosis"""
                domain = [Patient]
                range = [int]

            class survival_cohort(DataProperty, FunctionalProperty):
                """Survival cohort classification"""
                domain = [Patient]
                range = [str]

            # ========================================
            # CLINICAL FINDING DOMAIN
            # ========================================

            class ClinicalFinding(SNOMEDConcept):
                """SNOMED: 404684003 - Clinical finding"""
                pass

            class NeoplasticDisease(ClinicalFinding):
                """SNOMED: 64572001 - Neoplastic disease"""
                pass

            class MalignantNeoplasmOfLung(NeoplasticDisease):
                """SNOMED: 363358000 - Malignant neoplasm of lung"""
                pass

            class Comorbidity(ClinicalFinding):
                """Patient comorbidity"""
                pass

            class Dementia(Comorbidity):
                """Dementia comorbidity"""
                pass

            class CardiovascularDisease(Comorbidity):
                """Cardiovascular disease comorbidity"""
                pass

            class COPD(Comorbidity):
                """Chronic obstructive pulmonary disease"""
                pass

            class Diabetes(Comorbidity):
                """Diabetes mellitus"""
                pass

            # Clinical Finding Data Properties
            class tnm_staging_version(DataProperty):
                """TNM staging version (e.g., 7th edition, 8th edition)"""
                domain = [ClinicalFinding]
                range = [str]

            class diagnosis_site_code(DataProperty):
                """ICD-O-3 topography code"""
                domain = [ClinicalFinding]
                range = [str]

            class basis_of_diagnosis(DataProperty):
                """How diagnosis was made (Clinical/Histological/etc)"""
                domain = [ClinicalFinding]
                range = [str]

            class has_pre_tnm_staging(DataProperty, FunctionalProperty):
                """Pre-treatment TNM stage"""
                domain = [ClinicalFinding]
                range = [str]

            class has_post_tnm_staging(DataProperty):
                """Post-treatment TNM stage"""
                domain = [ClinicalFinding]
                range = [str]

            class investigation_result_date(DataProperty):
                """Date of investigation"""
                domain = [ClinicalFinding]
                range = [str]

            # ========================================
            # HISTOLOGY DOMAIN
            # ========================================

            class Histology(SNOMEDConcept):
                """Tumor histology classification"""
                pass

            class NonSmallCellCarcinoma(Histology):
                """SNOMED: 254637007 - Non-small cell lung cancer"""
                pass

            class SmallCellCarcinoma(Histology):
                """SNOMED: 254632001 - Small cell carcinoma"""
                pass

            class Adenocarcinoma(NonSmallCellCarcinoma):
                """SNOMED: 35917007 - Adenocarcinoma"""
                pass

            class SquamousCellCarcinoma(NonSmallCellCarcinoma):
                """SNOMED: 59367005 - Squamous cell carcinoma"""
                pass

            class LargeCellCarcinoma(NonSmallCellCarcinoma):
                """SNOMED: 67101007 - Large cell carcinoma"""
                pass

            class Carcinosarcoma(Histology):
                """SNOMED: 128885008 - Carcinosarcoma"""
                pass

            # ========================================
            # BIOMARKER DOMAIN (Modern Precision Medicine)
            # ========================================

            class Biomarker(ClinicalFinding):
                """Molecular biomarker finding"""
                pass

            class EGFRMutation(Biomarker):
                """EGFR mutation status - SNOMED: 426964009"""
                pass

            class EGFRPositive(EGFRMutation):
                """EGFR mutation positive"""
                pass

            class EGFRNegative(EGFRMutation):
                """EGFR mutation negative"""
                pass

            class ALKRearrangement(Biomarker):
                """ALK fusion/rearrangement status - SNOMED: 830151004"""
                pass

            class ALKPositive(ALKRearrangement):
                """ALK rearrangement positive"""
                pass

            class ALKNegative(ALKRearrangement):
                """ALK rearrangement negative"""
                pass

            class PDL1Expression(Biomarker):
                """PD-L1 expression status"""
                pass

            class PDL1High(PDL1Expression):
                """PD-L1 expression ≥50%"""
                pass

            class PDL1Low(PDL1Expression):
                """PD-L1 expression 1-49%"""
                pass

            class PDL1Negative(PDL1Expression):
                """PD-L1 expression <1%"""
                pass

            class ROSRearrangement(Biomarker):
                """ROS1 rearrangement status"""
                pass

            class KRASMutation(Biomarker):
                """KRAS mutation status"""
                pass

            # Biomarker Data Properties
            class pdl1_score(DataProperty, FunctionalProperty):
                """PD-L1 tumor proportion score (TPS) percentage"""
                domain = [PDL1Expression]
                range = [float]

            class egfr_mutation_type(DataProperty):
                """Specific EGFR mutation (e.g., exon 19 deletion, L858R)"""
                domain = [EGFRMutation]
                range = [str]

            # ========================================
            # TNM STAGING COMPONENTS
            # ========================================

            class TNMStaging(ClinicalFinding):
                """TNM staging classification"""
                pass

            class TStage(TNMStaging):
                """Tumor stage component"""
                pass

            class NStage(TNMStaging):
                """Node stage component"""
                pass

            class MStage(TNMStaging):
                """Metastasis stage component"""
                pass

            # TNM Staging Data Properties
            class t_component(DataProperty):
                """T stage component (T1a, T1b, T2a, T2b, T3, T4)"""
                domain = [TNMStaging]
                range = [str]

            class n_component(DataProperty):
                """N stage component (N0, N1, N2, N3)"""
                domain = [TNMStaging]
                range = [str]

            class m_component(DataProperty):
                """M stage component (M0, M1a, M1b, M1c)"""
                domain = [TNMStaging]
                range = [str]

            class staging_edition(DataProperty):
                """TNM staging edition (7th, 8th)"""
                domain = [TNMStaging]
                range = [str]

            # ========================================
            # BODY STRUCTURE DOMAIN
            # ========================================

            class BodyStructure(SNOMEDConcept):
                """SNOMED: 123037004 - Body structure"""
                pass

            class Neoplasm(BodyStructure):
                """Tumor body structure"""
                pass

            class Side(Thing):
                """Laterality reference class"""
                pass

            # ========================================
            # PERFORMANCE STATUS
            # ========================================

            class PerformanceStatus(ClinicalFinding):
                """WHO/ECOG Performance Status"""
                pass

            class WHOPerfStatusGrade0(PerformanceStatus):
                """SNOMED: 373803006 - Fully active"""
                pass

            class WHOPerfStatusGrade1(PerformanceStatus):
                """SNOMED: 373804000 - Restricted but ambulatory"""
                pass

            class WHOPerfStatusGrade2(PerformanceStatus):
                """SNOMED: 373805004 - Ambulatory, self-care capable"""
                pass

            class WHOPerfStatusGrade3(PerformanceStatus):
                """SNOMED: 373806003 - Limited self-care"""
                pass

            class WHOPerfStatusGrade4(PerformanceStatus):
                """SNOMED: 373807007 - Completely disabled"""
                pass

            # ========================================
            # PATIENT REFERRAL
            # ========================================

            class PatientReferral(Thing):
                """Patient referral information"""
                pass

            class referral_decision_date(DataProperty):
                """Date referral decision made"""
                domain = [PatientReferral]
                range = [str]

            class first_seen_date(DataProperty):
                """Date patient first seen"""
                domain = [PatientReferral]
                range = [str]

            class referral_date(DataProperty):
                """Date of referral"""
                domain = [PatientReferral]
                range = [str]

            # ========================================
            # PROCEDURE DOMAIN
            # ========================================

            class Procedure(SNOMEDConcept):
                """SNOMED: 71388002 - Procedure"""
                pass

            class EvaluationProcedure(Procedure):
                """Diagnostic/evaluation procedures"""
                pass

            class CTScan(EvaluationProcedure):
                """CT scan"""
                pass

            class PETScan(EvaluationProcedure):
                """PET scan"""
                pass

            class Bronchoscopy(EvaluationProcedure):
                """Bronchoscopy"""
                pass

            class CTGuidedBiopsy(EvaluationProcedure):
                """CT-guided biopsy"""
                pass

            class TherapeuticProcedure(Procedure):
                """Treatment procedures"""
                pass

            class Surgery(TherapeuticProcedure):
                """SNOMED: 387713003 - Surgical procedure"""
                pass

            class Lobectomy(Surgery):
                """SNOMED: 173171007 - Lobectomy of lung"""
                pass

            class Pneumonectomy(Surgery):
                """SNOMED: 49795001 - Pneumonectomy"""
                pass

            class Chemotherapy(TherapeuticProcedure):
                """SNOMED: 367336001 - Chemotherapy"""
                pass

            class Radiotherapy(TherapeuticProcedure):
                """SNOMED: 108290001 - Radiation therapy"""
                pass

            class Chemoradiotherapy(TherapeuticProcedure):
                """Combined chemotherapy and radiotherapy"""
                pass

            class Brachytherapy(TherapeuticProcedure):
                """Internal radiation therapy"""
                pass

            class PalliativeCare(TherapeuticProcedure):
                """SNOMED: 103735009 - Palliative care"""
                pass

            class ActiveMonitoring(TherapeuticProcedure):
                """Active monitoring/surveillance"""
                pass

            class Immunotherapy(TherapeuticProcedure):
                """Immunotherapy treatment"""
                pass

            # Procedure Data Properties
            class surgery_site_code(DataProperty):
                """Site of surgery code"""
                domain = [Surgery]
                range = [str]

            class surgery_decision_date(DataProperty):
                """Date surgery decision made"""
                domain = [Surgery]
                range = [str]

            class decision_to_treat_date(DataProperty):
                """Date treatment decision made"""
                domain = [TherapeuticProcedure]
                range = [str]

            class treatment_start_date(DataProperty):
                """Date treatment started"""
                domain = [TherapeuticProcedure]
                range = [str]

            # ========================================
            # TREATMENT PLAN
            # ========================================

            class TreatmentPlan(Thing):
                """Treatment plan container"""
                pass

            class treatment_plan_type(DataProperty):
                """Type of treatment plan"""
                domain = [TreatmentPlan]
                range = [str]

            # ========================================
            # OUTCOME
            # ========================================

            class Outcome(Thing):
                """Treatment outcome"""
                pass

            class was_death_related_to_treatment(DataProperty):
                """Whether death was treatment-related"""
                domain = [Outcome]
                range = [bool]

            class cancer_morbidity_type(DataProperty):
                """Type of cancer-related morbidity"""
                domain = [Outcome]
                range = [str]

            class original_treatment_plan_carried_out(DataProperty):
                """Whether original plan was completed"""
                domain = [Outcome]
                range = [bool]

            class treatment_failure_reason(DataProperty):
                """Reason for treatment failure"""
                domain = [Outcome]
                range = [str]

            # ========================================
            # ARGUMENTATION DOMAIN (KEY INNOVATION)
            # ========================================

            class Argumentation(Thing):
                """Base argumentation class"""
                pass

            class PatientScenario(Patient, Argumentation):
                """
                Hybrid class: Patient + Argumentation
                Represents hypothetical patient cohort matching guideline criteria
                KEY INNOVATION from the paper - enables guideline-based classification
                """
                pass

            class Argument(Argumentation):
                """Clinical argument (pro or con)"""
                pass

            class Decision(Argumentation):
                """Treatment decision"""
                pass

            class Intent(Argumentation):
                """Treatment intent (curative/palliative)"""
                pass

            class argument_strength(DataProperty):
                """Strength of argument (Strong/Moderate/Weak)"""
                domain = [Argument]
                range = [str]

            class evidence_level(DataProperty):
                """Evidence level (Grade A/B/C)"""
                domain = [Argument]
                range = [str]

            # ========================================
            # OBJECT PROPERTIES
            # ========================================

            class has_clinical_finding(ObjectProperty):
                """Links patient to clinical findings"""
                domain = [Patient]
                range = [ClinicalFinding]

            class has_cancer_referral(ObjectProperty):
                """Links patient to referral info"""
                domain = [Patient]
                range = [PatientReferral]

            class has_treatment_plan(ObjectProperty):
                """Links patient to treatment plan"""
                domain = [Patient]
                range = [TreatmentPlan]

            class includes_treatment(ObjectProperty):
                """Links treatment plan to procedures"""
                domain = [TreatmentPlan]
                range = [Procedure]

            class has_procedure_site(ObjectProperty):
                """Links procedure to anatomical location"""
                domain = [Procedure]
                range = [BodyStructure]

            class has_histology(ObjectProperty):
                """Links clinical finding to histology"""
                domain = [ClinicalFinding]
                range = [Histology]

            class laterality(ObjectProperty):
                """Tumor laterality (left/right/bilateral)"""
                domain = [BodyStructure]
                range = [Side]

            class has_outcome(ObjectProperty):
                """Links treatment plan to outcome"""
                domain = [TreatmentPlan]
                range = [Outcome]

            class has_performance_status(ObjectProperty):
                """Links patient to performance status"""
                domain = [Patient]
                range = [PerformanceStatus]

            class has_comorbidity(ObjectProperty):
                """Links patient to comorbidities"""
                domain = [Patient]
                range = [Comorbidity]

            # Argumentation Object Properties
            class results_in_argument(ObjectProperty):
                """PatientScenario → Argument"""
                domain = [PatientScenario]
                range = [Argument]

            class supports_decision(ObjectProperty):
                """Argument → Decision (supporting)"""
                domain = [Argument]
                range = [Decision]

            class opposes_decision(ObjectProperty):
                """Argument → Decision (opposing)"""
                domain = [Argument]
                range = [Decision]

            class entails(ObjectProperty):
                """Decision → TreatmentPlan"""
                domain = [Decision]
                range = [TreatmentPlan]

            class has_intent(ObjectProperty):
                """TreatmentPlan → Intent"""
                domain = [TreatmentPlan]
                range = [Intent]

            # ========================================
            # REFERENCE INDIVIDUALS
            # ========================================

            # Side references
            Reference_Right = Side("Reference_Right")
            Reference_Left = Side("Reference_Left")
            Reference_Bilateral = Side("Reference_Bilateral")

            # Performance status references
            Reference_WHO_PS_0 = WHOPerfStatusGrade0("Reference_WHO_PS_0")
            Reference_WHO_PS_1 = WHOPerfStatusGrade1("Reference_WHO_PS_1")
            Reference_WHO_PS_2 = WHOPerfStatusGrade2("Reference_WHO_PS_2")
            Reference_WHO_PS_3 = WHOPerfStatusGrade3("Reference_WHO_PS_3")
            Reference_WHO_PS_4 = WHOPerfStatusGrade4("Reference_WHO_PS_4")

            # Intent references
            Reference_Curative_Intent = Intent("Reference_Curative_Intent")
            Reference_Palliative_Intent = Intent("Reference_Palliative_Intent")

        logger.info("✓ LUCADA ontology created successfully")
        logger.info(f"  Classes: {len(list(self.onto.classes()))}")
        logger.info(f"  Object Properties: {len(list(self.onto.object_properties()))}")
        logger.info(f"  Data Properties: {len(list(self.onto.data_properties()))}")

        return self.onto

    def create_patient_individual(
        self,
        patient_id: str,
        name: str,
        sex: str,
        age: int,
        diagnosis: str,
        tnm_stage: str,
        histology_type: str,
        laterality: str,
        performance_status: int,
        fev1_percent: Optional[float] = None
    ) -> Any:
        """
        Create a patient individual in the ontology.
        Follows the pattern from Figure 2 in the paper.

        Args:
            patient_id: Unique patient identifier
            name: Patient name
            sex: M/F/U
            age: Age at diagnosis
            diagnosis: Primary diagnosis
            tnm_stage: TNM staging (IA-IV)
            histology_type: Tumor histology
            laterality: Right/Left/Bilateral
            performance_status: WHO PS (0-4)
            fev1_percent: Optional FEV1 percentage

        Returns:
            Patient individual
        """
        with self.onto:
            # Create patient individual
            patient = self.onto.Patient(patient_id)
            patient.patient_name = [name]
            patient.sex = [sex]
            patient.age_at_diagnosis = [age]
            if fev1_percent:
                patient.fev1_absolute_amount = [fev1_percent]

            # Create cancer finding individual
            cancer = self.onto.MalignantNeoplasmOfLung(f"Cancer_{patient_id}")
            cancer.has_pre_tnm_staging = [tnm_stage]
            cancer.basis_of_diagnosis = ["Clinical"]

            # Create histology individual - map string to class
            histology_mapping = {
                "NonSmallCellCarcinoma": self.onto.NonSmallCellCarcinoma,
                "SmallCellCarcinoma": self.onto.SmallCellCarcinoma,
                "Carcinosarcoma": self.onto.Carcinosarcoma,
                "Adenocarcinoma": self.onto.Adenocarcinoma,
                "SquamousCellCarcinoma": self.onto.SquamousCellCarcinoma,
                "LargeCellCarcinoma": self.onto.LargeCellCarcinoma,
            }
            histology_class = histology_mapping.get(histology_type, self.onto.Histology)
            tumor = histology_class(f"Tumour_{patient_id}")

            # Create body structure for laterality
            neoplasm = self.onto.Neoplasm(f"Neoplasm_{patient_id}")

            # Link histology to cancer finding
            cancer.has_histology.append(tumor)

            # Link cancer finding to patient
            patient.has_clinical_finding.append(cancer)

            # Set laterality using reference individual
            laterality_mapping = {
                "Right": self.onto.Reference_Right,
                "Left": self.onto.Reference_Left,
                "Bilateral": self.onto.Reference_Bilateral,
            }
            lat_ref = laterality_mapping.get(laterality, self.onto.Reference_Right)
            if lat_ref:
                neoplasm.laterality = [lat_ref]

            # Set performance status
            ps_mapping = {
                0: self.onto.Reference_WHO_PS_0,
                1: self.onto.Reference_WHO_PS_1,
                2: self.onto.Reference_WHO_PS_2,
                3: self.onto.Reference_WHO_PS_3,
                4: self.onto.Reference_WHO_PS_4,
            }
            ps_ref = ps_mapping.get(performance_status)
            if ps_ref:
                patient.has_performance_status.append(ps_ref)

            logger.debug(f"Created patient individual: {patient_id}")
            return patient

    def save(self, filepath: str):
        """Save ontology to file"""
        if not self.onto:
            raise RuntimeError("Ontology not created. Call create() first.")

        self.onto.save(file=filepath)
        logger.info(f"✓ Ontology saved to: {filepath}")

    def load(self, filepath: str) -> Ontology:
        """Load existing ontology from file"""
        self.onto = get_ontology(f"file://{filepath}").load()
        logger.info(f"✓ Ontology loaded from: {filepath}")
        return self.onto


if __name__ == "__main__":
    # Create and save LUCADA ontology
    import os
    from pathlib import Path

    # Get output directory from environment or use default
    output_dir = os.getenv("LUCADA_ONTOLOGY_OUTPUT",
                          r"H:\akash\git\CoherencePLM\Version22\data\lca_ontologies\lucada")
    output_file = os.getenv("LUCADA_OWL_FILE", "lucada_ontology.owl")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Full path to output file
    output_path = Path(output_dir) / output_file

    lucada = LUCADAOntology()
    onto = lucada.create()
    lucada.save(str(output_path))
    print(f"\n✓ LUCADA ontology created and saved to: {output_path}")
