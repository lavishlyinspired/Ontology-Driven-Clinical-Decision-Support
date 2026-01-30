"""
SNOMED-CT OWL Ontology Loader
Loads and provides access to the complete SNOMED-CT ontology
"""

import os
import types
from pathlib import Path
from typing import Optional, List, Dict, Any
from owlready2 import *

# Centralized logging
from ..logging_config import get_logger, log_execution

logger = get_logger(__name__)

try:
    from ..config import LCAConfig
except (ImportError, ValueError):
    # Fallback if config module not available
    LCAConfig = None


class SNOMEDLoader:
    """
    Loader for SNOMED-CT OWL ontology.
    Provides utilities for accessing SNOMED concepts used in lung cancer DSS.
    """

    # Key SNOMED-CT concept codes for lung cancer
    LUNG_CANCER_CONCEPTS = {
        # Diagnoses
        "malignant_neoplasm_lung": "363358000",
        "nsclc": "254637007",
        "nsclc_stage_1": "424132000",
        "nsclc_stage_2": "425048006",
        "nsclc_stage_3": "422968005",
        "nsclc_stage_4": "423121009",
        "squamous_nsclc": "723301009",
        "nsclc_adenocarcinoma": "1255725002",
        "egfr_positive_nsclc": "426964009",
        "egfr_negative_nsclc": "427038005",
        "alk_fusion_positive_nsclc": "830151004",
        "sclc": "254632001",
        "adenocarcinoma": "35917007",
        "squamous_cell_carcinoma": "59367005",
        "large_cell_carcinoma": "67101007",
        "carcinosarcoma": "128885008",

        # Procedures/Treatments
        "surgical_procedure": "387713003",
        "lobectomy": "173171007",
        "pneumonectomy": "49795001",
        "chemotherapy": "367336001",
        "radiation_therapy": "108290001",
        "chemoradiotherapy": "703423002",
        "immunotherapy": "76334006",
        "palliative_care": "103735009",

        # Clinical Findings
        "clinical_finding": "404684003",
        "neoplastic_disease": "64572001",
        "body_structure": "123037004",

        # Performance Status
        "who_ps_grade_0": "373803006",
        "who_ps_grade_1": "373804000",
        "who_ps_grade_2": "373805004",
        "who_ps_grade_3": "373806003",
        "who_ps_grade_4": "373807007",
        
        # Treatment Outcomes
        "complete_response": "268910001",
        "partial_response": "268917003",
        "stable_disease": "359746009",
        "progressive_disease": "271299001",
        "disease_free_survival": "313239003",
        "overall_survival": "399307001",
        
        # Body Structures
        "right_lung": "39607008",
        "left_lung": "44029006",
        "bilateral_lungs": "51185008",
        "upper_lobe_lung": "45653009",
        "middle_lobe_lung": "72481006",
        "lower_lobe_lung": "4029006",
    }
    
    # Mapping dictionaries for common clinical terms to SNOMED codes
    HISTOLOGY_MAP = {
        "Adenocarcinoma": "35917007",
        "SquamousCellCarcinoma": "59367005",
        "SmallCellCarcinoma": "254632001",
        "LargeCellCarcinoma": "67101007",
        "NonSmallCellCarcinoma": "254637007",
        "Carcinosarcoma": "128885008",
        "NonSmallCellCarcinoma_NOS": "254637007",
    }
    
    STAGE_MAP = {
        "IA": "424132000",
        "IB": "424132000",
        "IIA": "425048006",
        "IIB": "425048006",
        "IIIA": "422968005",
        "IIIB": "422968005",
        "IV": "423121009",
    }
    
    TREATMENT_MAP = {
        "Surgery": "387713003",
        "Chemotherapy": "367336001",
        "Radiotherapy": "108290001",
        "Chemoradiotherapy": "703423002",
        "PalliativeCare": "103735009",
        "Immunotherapy": "76334006",
        "Lobectomy": "173171007",
        "Pneumonectomy": "49795001",
    }
    
    PERFORMANCE_STATUS_MAP = {
        0: "373803006",
        1: "373804000",
        2: "373805004",
        3: "373806003",
        4: "373807007",
    }
    
    OUTCOME_MAP = {
        "complete_response": "268910001",
        "partial_response": "268917003",
        "stable_disease": "359746009",
        "progressive_disease": "271299001",
        "disease_free": "313239003",
        "deceased": "419099009",
    }

    def __init__(self, owl_path: Optional[str] = None):
        """
        Initialize SNOMED loader.

        Args:
            owl_path: Path to SNOMED OWL file. If None, uses config or env variable.
        """
        # Priority: explicit parameter > config > env variable > default
        if owl_path:
            raw_path = owl_path
        elif LCAConfig:
            raw_path = LCAConfig.SNOMED_CT_PATH
        else:
            raw_path = os.getenv("SNOMED_CT_PATH", os.getenv("SNOMED_OWL_PATH", "ontology-2026-01-17_12-36-08.owl"))
        
        # Resolve to absolute path
        resolved_path = Path(raw_path)
        if not resolved_path.is_absolute():
            # Try relative to current working directory first
            if not resolved_path.exists():
                # Try relative to this module's directory (backend/src/ontology)
                module_dir = Path(__file__).parent
                # Go up to Version22 directory (../../../)
                project_root = module_dir.parent.parent.parent
                resolved_path = project_root / resolved_path.name
        
        self.owl_path = str(resolved_path.resolve()) if resolved_path.exists() else str(resolved_path)
        self.ontology: Optional[Ontology] = None
        self.loaded = False

        # Verify file exists (warn if missing but don't fail - can use mapping dictionaries)
        if not Path(self.owl_path).exists():
            logger.warning(f"SNOMED OWL file not found at {self.owl_path}")
            logger.info("Will use SNOMED code mapping dictionaries only (no full ontology loading)")
        else:
            logger.info(f"SNOMED OWL file found at {self.owl_path}")

    def load(self, load_full: bool = False) -> Ontology:
        """
        Load the SNOMED-CT ontology.

        Args:
            load_full: If True, loads entire ontology (memory intensive).
                      If False, loads minimal subset for lung cancer DSS.

        Returns:
            Loaded ontology object
        """
        logger.info(f"Loading SNOMED-CT ontology from: {self.owl_path}")

        # Check if file is in OWL Functional Syntax (starts with "Prefix(")
        with open(self.owl_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        is_functional_syntax = first_line.startswith("Prefix(")
        
        if is_functional_syntax:
            logger.info("Detected OWL Functional Syntax format")
            return self._load_functional_syntax(load_full)
        
        # Standard RDF/XML or N-Triples loading via owlready2
        try:
            if load_full:
                logger.warning("Loading full SNOMED-CT ontology - this may take several minutes and require significant RAM")
                self.ontology = get_ontology(f"file://{self.owl_path}").load()
            else:
                logger.info("Loading SNOMED-CT ontology structure (minimal mode)")
                self.ontology = get_ontology(f"file://{self.owl_path}").load()

            self.loaded = True
            logger.info(f"✓ SNOMED-CT ontology loaded successfully")
            logger.info(f"  Classes: {len(list(self.ontology.classes()))}")
            logger.info(f"  Properties: {len(list(self.ontology.properties()))}")

            return self.ontology

        except Exception as e:
            logger.error(f"Failed to load SNOMED-CT ontology: {e}")
            raise
    
    def _load_functional_syntax(self, load_full: bool = False) -> Ontology:
        """
        Load OWL Functional Syntax format by parsing class declarations.
        
        This creates a lightweight ontology with the SNOMED-CT class hierarchy.
        Full OWL Functional Syntax parsing is complex, so we extract key elements.
        """
        import re
        
        logger.info("Parsing OWL Functional Syntax ontology...")
        
        # Create a new ontology to hold the concepts
        self.ontology = get_ontology("http://snomed.info/sct/lca_module")
        
        class_count = 0
        subclass_count = 0
        annotation_count = 0
        obj_prop_count = 0
        
        # Define common patterns
        class_pattern = re.compile(r'Declaration\(Class\(:(\d+)\)\)')
        subclass_pattern = re.compile(r'SubClassOf\(:(\d+)\s+:(\d+)\)')
        annotation_pattern = re.compile(r'AnnotationAssertion\(rdfs:label\s+:(\d+)\s+"([^"]+)"')
        obj_prop_pattern = re.compile(r'Declaration\(ObjectProperty\(:(\d+)\)\)')
        
        with self.ontology:
            # Define sctid as a data property for storing SNOMED IDs
            class sctid(DataProperty):
                range = [str]
            
            # Dictionary to store class objects and their SCTIDs
            class_dict = {}
            sctid_map = {}  # Store SCTID separately
            subclass_relations = []
            labels = {}
            obj_properties = {}
            
            logger.info("Pass 1: Extracting declarations and annotations...")
            
            # Track whether we've seen all classes (to handle minimal mode properly)
            finished_classes = False
            
            with open(self.owl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Match class declarations
                    match = class_pattern.match(line)
                    if match:
                        concept_sctid = match.group(1)
                        # Create class dynamically
                        class_name = f"SCTID_{concept_sctid}"
                        cls = types.new_class(class_name, (Thing,))
                        class_dict[concept_sctid] = cls
                        sctid_map[concept_sctid] = concept_sctid
                        class_count += 1
                        
                        if class_count % 50000 == 0:
                            logger.info(f"  Loaded {class_count} classes...")
                        
                        # In minimal mode, limit classes but continue reading properties
                        if not load_full and class_count > 10000 and not finished_classes:
                            logger.info("Loaded 10,000 classes (minimal mode) - continuing for properties...")
                            finished_classes = True
                        continue
                    
                    # Match Object Property declarations
                    match = obj_prop_pattern.match(line)
                    if match:
                        prop_id = match.group(1)
                        prop_name = f"prop_{prop_id}"
                        try:
                            prop = types.new_class(prop_name, (ObjectProperty,))
                            obj_properties[prop_id] = prop
                            obj_prop_count += 1
                        except Exception:
                            pass  # Skip if property creation fails
                        continue
                    
                    # Match annotations (labels)
                    match = annotation_pattern.search(line)
                    if match:
                        concept_id = match.group(1)
                        label = match.group(2)
                        labels[concept_id] = label
                        annotation_count += 1
                        continue
                    
                    # Match subclass relationships
                    match = subclass_pattern.search(line)
                    if match:
                        child_id = match.group(1)
                        parent_id = match.group(2)
                        subclass_relations.append((child_id, parent_id))
                        subclass_count += 1
                        continue
            
            # Apply labels to classes
            logger.info(f"Applying {len(labels)} labels...")
            for concept_id, label in labels.items():
                if concept_id in class_dict:
                    class_dict[concept_id].label = [label]
                elif concept_id in obj_properties:
                    obj_properties[concept_id].label = [label]
            
            # Apply subclass relationships (limit for performance)
            if subclass_relations:
                logger.info(f"Applying subclass relationships ({len(subclass_relations)} total)...")
                applied = 0
                for child_id, parent_id in subclass_relations:
                    if child_id in class_dict and parent_id in class_dict:
                        # Note: owlready2 uses is_a for subclass relationships
                        child_cls = class_dict[child_id]
                        parent_cls = class_dict[parent_id]
                        if parent_cls not in child_cls.is_a:
                            child_cls.is_a.append(parent_cls)
                            applied += 1
                        
                        if applied >= 100000 and not load_full:
                            break
                
                logger.info(f"  Applied {applied} subclass relationships")
        
        self.loaded = True
        self._class_dict = class_dict  # Store for quick lookup
        self._obj_properties = obj_properties  # Store object properties
        
        logger.info(f"✓ SNOMED-CT ontology loaded successfully")
        logger.info(f"  Classes: {class_count}")
        logger.info(f"  Object Properties: {obj_prop_count}")
        logger.info(f"  Subclass Relations: {subclass_count}")
        logger.info(f"  Labels: {annotation_count}")
        
        # Store counts for external access
        self._class_count = class_count
        self._obj_prop_count = obj_prop_count
        self._annotation_count = annotation_count
        
        return self.ontology

    def get_concept_by_id(self, sctid: str) -> Optional[Thing]:
        """
        Retrieve a SNOMED concept by its SCTID.

        Args:
            sctid: SNOMED CT Identifier (e.g., "254637007" for NSCLC)

        Returns:
            Concept if found, None otherwise
        """
        if not self.loaded:
            raise RuntimeError("Ontology not loaded. Call load() first.")

        # First check our class dictionary (for OWL Functional Syntax parsed files)
        if hasattr(self, '_class_dict') and self._class_dict:
            return self._class_dict.get(sctid)
        
        # Fall back to standard IRI search for RDF/XML ontologies
        # SNOMED concepts use IRI format: http://snomed.info/id/{SCTID}
        concept_iri = f"http://snomed.info/id/{sctid}"

        # Search for the concept
        concept = self.ontology.search_one(iri=concept_iri)
        return concept

    def search_concepts(self, term: str, limit: int = 20) -> List[Any]:
        """
        Search for SNOMED concepts by term.

        Args:
            term: Search term
            limit: Maximum results to return

        Returns:
            List of matching concepts
        """
        if not self.loaded:
            raise RuntimeError("Ontology not loaded. Call load() first.")

        results = self.ontology.search(label=f"*{term}*")
        return list(results)[:limit]

    def get_lung_cancer_concepts(self) -> Dict[str, Any]:
        """
        Get all pre-defined lung cancer related concepts.

        Returns:
            Dictionary mapping concept names to ontology classes
        """
        concepts = {}
        for name, sctid in self.LUNG_CANCER_CONCEPTS.items():
            concept = self.get_concept_by_id(sctid)
            if concept:
                concepts[name] = concept
            else:
                logger.warning(f"Concept not found: {name} ({sctid})")

        return concepts

    def get_ancestors(self, concept: Thing) -> List[Thing]:
        """
        Get all ancestor concepts (parent hierarchy).

        Args:
            concept: SNOMED concept

        Returns:
            List of ancestor concepts
        """
        return concept.ancestors()

    def get_descendants(self, concept: Thing) -> List[Thing]:
        """
        Get all descendant concepts (child hierarchy).

        Args:
            concept: SNOMED concept

        Returns:
            List of descendant concepts
        """
        return concept.descendants()

    def is_a(self, child_sctid: str, parent_sctid: str) -> bool:
        """
        Check if one concept is a subclass of another.

        Args:
            child_sctid: Child concept SCTID
            parent_sctid: Parent concept SCTID

        Returns:
            True if child is a subclass of parent
        """
        child = self.get_concept_by_id(child_sctid)
        parent = self.get_concept_by_id(parent_sctid)

        if not child or not parent:
            return False

        try:
            return parent in child.ancestors()
        except Exception:
            # For parsed OWL Functional Syntax, check is_a relationship directly
            try:
                return parent in child.is_a or any(
                    self.is_a(str(getattr(p, 'sctid', '')), parent_sctid) 
                    for p in child.is_a if p != Thing and hasattr(p, 'sctid')
                )
            except Exception:
                return False

    def get_concept_info(self, sctid: str) -> Dict[str, Any]:
        """
        Get detailed information about a concept.

        Args:
            sctid: SNOMED CT Identifier

        Returns:
            Dictionary with concept details
        """
        concept = self.get_concept_by_id(sctid)
        if not concept:
            return {}

        # Handle both standard owlready2 classes and our custom parsed classes
        try:
            label = concept.label.first() if hasattr(concept, 'label') and concept.label else None
        except:
            label = getattr(concept, 'label', [None])[0] if getattr(concept, 'label', None) else None
        
        try:
            iri = concept.iri if hasattr(concept, 'iri') else f"http://snomed.info/id/{sctid}"
        except:
            iri = f"http://snomed.info/id/{sctid}"
            
        try:
            parents = [str(p) for p in concept.is_a if p != Thing]
        except:
            parents = []
            
        try:
            ancestors_count = len(list(concept.ancestors())) if hasattr(concept, 'ancestors') else 0
        except:
            ancestors_count = 0
            
        try:
            descendants_count = len(list(concept.descendants())) if hasattr(concept, 'descendants') else 0
        except:
            descendants_count = 0

        return {
            "sctid": sctid,
            "iri": iri,
            "label": label,
            "parents": parents,
            "ancestors_count": ancestors_count,
            "descendants_count": descendants_count,
        }

    def map_histology(self, histology_type: str) -> Optional[str]:
        """
        Map histology type to SNOMED code.

        Args:
            histology_type: Clinical histology type

        Returns:
            SNOMED code or None
        """
        return self.HISTOLOGY_MAP.get(histology_type)

    def map_stage(self, tnm_stage: str) -> Optional[str]:
        """
        Map TNM stage to SNOMED code.

        Args:
            tnm_stage: TNM staging

        Returns:
            SNOMED code or None
        """
        return self.STAGE_MAP.get(tnm_stage)

    def map_treatment(self, treatment: str) -> Optional[str]:
        """
        Map treatment to SNOMED code.

        Args:
            treatment: Treatment type

        Returns:
            SNOMED code or None
        """
        return self.TREATMENT_MAP.get(treatment)

    def map_performance_status(self, ps: int) -> Optional[str]:
        """
        Map WHO performance status to SNOMED code.

        Args:
            ps: Performance status (0-4)

        Returns:
            SNOMED code or None
        """
        return self.PERFORMANCE_STATUS_MAP.get(ps)

    def map_outcome(self, outcome: str) -> Optional[str]:
        """
        Map treatment outcome to SNOMED code.

        Args:
            outcome: Outcome description

        Returns:
            SNOMED code or None
        """
        return self.OUTCOME_MAP.get(outcome)

    def map_patient_to_snomed(self, patient_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Map all patient clinical data to SNOMED codes.

        Args:
            patient_data: Patient clinical information

        Returns:
            Dictionary of SNOMED codes for patient attributes
        """
        return {
            "diagnosis": self.map_histology(patient_data.get("histology_type", "")),
            "stage": self.map_stage(patient_data.get("tnm_stage", "")),
            "performance_status": self.map_performance_status(
                patient_data.get("performance_status", 0)
            ),
            "laterality": self._map_laterality(patient_data.get("laterality", "")),
        }

    def _map_laterality(self, laterality: str) -> Optional[str]:
        """Map laterality to SNOMED codes."""
        laterality_map = {
            "Right": "39607008",
            "Left": "44029006",
            "Bilateral": "51185008",
        }
        return laterality_map.get(laterality)

    # Public methods for semantic mapping (used by SemanticMappingAgent)
    @classmethod
    def map_histology(cls, histology: str) -> Optional[str]:
        """Map histology type to SNOMED-CT code."""
        return cls.HISTOLOGY_MAP.get(histology)

    @classmethod
    def map_stage(cls, stage: str) -> Optional[str]:
        """Map TNM stage to SNOMED-CT code."""
        return cls.STAGE_MAP.get(stage)

    @classmethod
    def map_performance_status(cls, ps: int) -> Optional[str]:
        """Map WHO performance status to SNOMED-CT code."""
        return cls.PERFORMANCE_STATUS_MAP.get(ps)

    @classmethod
    def map_laterality(cls, laterality: str) -> Optional[str]:
        """Map laterality to SNOMED-CT code."""
        laterality_map = {
            "Right": "39607008",
            "Left": "44029006",
            "Bilateral": "51185008",
        }
        return laterality_map.get(laterality)

    @classmethod
    def map_treatment(cls, treatment: str) -> Optional[str]:
        """Map treatment type to SNOMED-CT code."""
        return cls.TREATMENT_MAP.get(treatment)

    @classmethod
    def is_nsclc_subtype(cls, histology_code: str) -> bool:
        """Check if a SNOMED code represents an NSCLC subtype."""
        nsclc_subtypes = [
            "35917007",   # Adenocarcinoma
            "59367005",   # Squamous cell
            "67101007",   # Large cell
            "128885008",  # Carcinosarcoma
            "254637007",  # NSCLC NOS
        ]
        return histology_code in nsclc_subtypes

    @classmethod
    def is_sclc(cls, histology_code: str) -> bool:
        """Check if a SNOMED code represents SCLC."""
        return histology_code == "254632001"

    @classmethod
    def get_diagnosis_code(cls, diagnosis: str) -> Optional[str]:
        """Map diagnosis to SNOMED-CT code with fallback to generic lung cancer."""
        diagnosis_map = {
            "Malignant Neoplasm of Lung": "363358000",
            "NSCLC": "254637007",
            "SCLC": "254632001",
        }
        return diagnosis_map.get(diagnosis, "363358000")  # Default to malignant neoplasm of lung

    def generate_owl_expression(self, patient_data: Dict[str, Any]) -> str:
        """
        Generate OWL 2 class expression for patient classification.

        Args:
            patient_data: Patient clinical information

        Returns:
            OWL 2 Manchester syntax expression
        """
        codes = self.map_patient_to_snomed(patient_data)
        
        diagnosis_code = codes.get("diagnosis", "254637007")
        stage = patient_data.get("tnm_stage", "")
        ps_code = codes.get("performance_status", "373803006")
        
        expression = f"""
        (hasClinicalFinding some 
            (NeoplasticDisease and 
                (hasPreTNMStaging value "{stage}") and 
                (hasHistology some <http://snomed.info/id/{diagnosis_code}>)
            )
        ) 
        and 
        (hasPerformanceStatus some <http://snomed.info/id/{ps_code}>)
        """.strip()
        
        return expression

    def export_lung_cancer_module(self, output_path: str):
        """
        Export a minimal SNOMED module containing only lung cancer concepts.
        This creates a lightweight ontology for faster loading.

        Args:
            output_path: Path to save the module
        """
        if not self.loaded:
            raise RuntimeError("Ontology not loaded. Call load() first.")

        logger.info("Creating lung cancer SNOMED module...")

        # Create new ontology
        module = get_ontology("http://snomed.info/module/lung_cancer")

        with module:
            # Copy relevant classes
            lung_cancer_concepts = self.get_lung_cancer_concepts()
            for name, concept in lung_cancer_concepts.items():
                if concept:
                    # Create equivalent class in module
                    new_class = types.new_class(
                        concept.name,
                        (Thing,)
                    )
                    new_class.label = concept.label

        module.save(file=output_path)
        logger.info(f"✓ Lung cancer module saved to: {output_path}")


def test_snomed_loader():
    """Test the SNOMED loader functionality."""
    print("=" * 80)
    print("SNOMED-CT Loader Test")
    print("=" * 80)

    # Initialize loader
    loader = SNOMEDLoader()

    # Load ontology (minimal mode)
    try:
        onto = loader.load(load_full=False)
        print(f"\n✓ Ontology loaded successfully")
    except Exception as e:
        print(f"\n✗ Failed to load ontology: {e}")
        return

    # Test concept retrieval
    print("\n" + "-" * 80)
    print("Testing Concept Retrieval")
    print("-" * 80)

    test_concepts = [
        ("NSCLC", "254637007"),
        ("Adenocarcinoma", "35917007"),
        ("Chemotherapy", "367336001"),
    ]

    for name, sctid in test_concepts:
        info = loader.get_concept_info(sctid)
        print(f"\n{name} ({sctid}):")
        print(f"  IRI: {info.get('iri', 'Not found')}")
        print(f"  Label: {info.get('label', 'N/A')}")
        print(f"  Ancestors: {info.get('ancestors_count', 0)}")

    # Test subsumption
    print("\n" + "-" * 80)
    print("Testing Subsumption Reasoning")
    print("-" * 80)

    is_nsclc = loader.is_a("35917007", "254637007")  # Adenocarcinoma is-a NSCLC
    print(f"Adenocarcinoma is-a NSCLC: {is_nsclc}")

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)


if __name__ == "__main__":
    test_snomed_loader()
