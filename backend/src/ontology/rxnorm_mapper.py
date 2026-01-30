"""
RxNorm Medication Mapper

Maps medication names to RxNorm codes for standardization
and drug interaction checking.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Centralized logging
from ..logging_config import get_logger, log_execution

logger = get_logger(__name__)


@dataclass
class RxNormConcept:
    """RxNorm concept representation"""
    rxcui: str  # RxNorm Concept Unique Identifier
    name: str
    tty: str  # Term Type (IN=Ingredient, SCD=Semantic Clinical Drug, etc.)
    suppress: str = "N"  # Suppression flag


@dataclass
class MedicationMapping:
    """Mapped medication with RxNorm codes"""
    medication_name: str
    rxcui: str
    rxnorm_name: str
    term_type: str
    ingredient_rxcui: Optional[str] = None
    ingredient_name: Optional[str] = None
    dose_form: Optional[str] = None
    strength: Optional[str] = None


class RxNormMapper:
    """
    RxNorm medication mapping for standardization.

    Features:
    - Maps drug names to RxNorm codes
    - Identifies active ingredients
    - Supports dose forms and strengths
    - Drug class identification
    """

    def __init__(self):
        # Load common oncology drugs
        self.rxnorm_mappings = self._load_oncology_drugs()
        logger.info(f"✓ RxNorm Mapper initialized ({len(self.rxnorm_mappings)} drugs)")

    def _load_oncology_drugs(self) -> Dict[str, RxNormConcept]:
        """Load common oncology medication mappings"""

        drugs = {
            # EGFR TKIs
            "osimertinib": RxNormConcept(
                rxcui="1856076",
                name="osimertinib",
                tty="IN"
            ),
            "gefitinib": RxNormConcept(
                rxcui="282388",
                name="gefitinib",
                tty="IN"
            ),
            "erlotinib": RxNormConcept(
                rxcui="176326",
                name="erlotinib",
                tty="IN"
            ),
            "afatinib": RxNormConcept(
                rxcui="1430438",
                name="afatinib",
                tty="IN"
            ),

            # ALK Inhibitors
            "alectinib": RxNormConcept(
                rxcui="1791833",
                name="alectinib",
                tty="IN"
            ),
            "crizotinib": RxNormConcept(
                rxcui="1148619",
                name="crizotinib",
                tty="IN"
            ),
            "brigatinib": RxNormConcept(
                rxcui="1998316",
                name="brigatinib",
                tty="IN"
            ),
            "lorlatinib": RxNormConcept(
                rxcui="2058262",
                name="lorlatinib",
                tty="IN"
            ),

            # Immunotherapy
            "pembrolizumab": RxNormConcept(
                rxcui="1547545",
                name="pembrolizumab",
                tty="IN"
            ),
            "nivolumab": RxNormConcept(
                rxcui="1537471",
                name="nivolumab",
                tty="IN"
            ),
            "atezolizumab": RxNormConcept(
                rxcui="1792776",
                name="atezolizumab",
                tty="IN"
            ),
            "durvalumab": RxNormConcept(
                rxcui="1927851",
                name="durvalumab",
                tty="IN"
            ),

            # Chemotherapy - Platinum
            "cisplatin": RxNormConcept(
                rxcui="2555",
                name="cisplatin",
                tty="IN"
            ),
            "carboplatin": RxNormConcept(
                rxcui="2555",
                name="carboplatin",
                tty="IN"
            ),

            # Chemotherapy - Taxanes
            "paclitaxel": RxNormConcept(
                rxcui="56946",
                name="paclitaxel",
                tty="IN"
            ),
            "docetaxel": RxNormConcept(
                rxcui="72962",
                name="docetaxel",
                tty="IN"
            ),

            # Chemotherapy - Other
            "pemetrexed": RxNormConcept(
                rxcui="134736",
                name="pemetrexed",
                tty="IN"
            ),
            "gemcitabine": RxNormConcept(
                rxcui="2520",
                name="gemcitabine",
                tty="IN"
            ),
            "etoposide": RxNormConcept(
                rxcui="4492",
                name="etoposide",
                tty="IN"
            ),
            "vinorelbine": RxNormConcept(
                rxcui="75571",
                name="vinorelbine",
                tty="IN"
            ),

            # Targeted Therapy - Other
            "bevacizumab": RxNormConcept(
                rxcui="203992",
                name="bevacizumab",
                tty="IN"
            ),
            "ramucirumab": RxNormConcept(
                rxcui="1436676",
                name="ramucirumab",
                tty="IN"
            )
        }

        return drugs

    def map_medication(self, medication_name: str) -> Optional[MedicationMapping]:
        """
        Map medication name to RxNorm code.

        Args:
            medication_name: Drug name

        Returns:
            MedicationMapping or None
        """
        normalized = medication_name.lower().strip()

        # Check direct mappings
        if normalized in self.rxnorm_mappings:
            concept = self.rxnorm_mappings[normalized]
            return MedicationMapping(
                medication_name=medication_name,
                rxcui=concept.rxcui,
                rxnorm_name=concept.name,
                term_type=concept.tty
            )

        logger.warning(f"No RxNorm mapping found for: {medication_name}")
        return None

    def map_medication_list(
        self,
        medications: List[str]
    ) -> List[MedicationMapping]:
        """Map list of medications"""

        mappings = []
        for med in medications:
            mapping = self.map_medication(med)
            if mapping:
                mappings.append(mapping)

        return mappings

    def get_drug_class(self, medication_name: str) -> Optional[str]:
        """Get therapeutic class for medication"""

        drug_classes = {
            "osimertinib": "EGFR TKI (3rd generation)",
            "gefitinib": "EGFR TKI (1st generation)",
            "erlotinib": "EGFR TKI (1st generation)",
            "afatinib": "EGFR TKI (2nd generation)",
            "alectinib": "ALK Inhibitor (2nd generation)",
            "crizotinib": "ALK Inhibitor (1st generation)",
            "brigatinib": "ALK Inhibitor (2nd generation)",
            "lorlatinib": "ALK Inhibitor (3rd generation)",
            "pembrolizumab": "Anti-PD-1 Immunotherapy",
            "nivolumab": "Anti-PD-1 Immunotherapy",
            "atezolizumab": "Anti-PD-L1 Immunotherapy",
            "durvalumab": "Anti-PD-L1 Immunotherapy",
            "cisplatin": "Platinum-based Chemotherapy",
            "carboplatin": "Platinum-based Chemotherapy",
            "paclitaxel": "Taxane Chemotherapy",
            "docetaxel": "Taxane Chemotherapy",
            "pemetrexed": "Antifolate Chemotherapy",
            "gemcitabine": "Nucleoside Analog Chemotherapy",
            "bevacizumab": "Anti-VEGF Targeted Therapy",
            "ramucirumab": "Anti-VEGFR2 Targeted Therapy"
        }

        normalized = medication_name.lower().strip()
        return drug_classes.get(normalized)

    def check_drug_interactions(
        self,
        medication1: str,
        medication2: str
    ) -> List[str]:
        """
        Check for known drug-drug interactions.

        Args:
            medication1: First medication
            medication2: Second medication

        Returns:
            List of interaction warnings
        """
        interactions = []

        # Simplified interaction database
        interaction_db = {
            ("osimertinib", "rifampin"): "CYP3A4 induction - decreases osimertinib levels",
            ("osimertinib", "itraconazole"): "CYP3A4 inhibition - increases osimertinib levels",
            ("cisplatin", "gentamicin"): "Additive nephrotoxicity",
            ("pembrolizumab", "prednisone"): "Corticosteroids may decrease immunotherapy efficacy"
        }

        med1 = medication1.lower()
        med2 = medication2.lower()

        # Check both orders
        if (med1, med2) in interaction_db:
            interactions.append(interaction_db[(med1, med2)])
        elif (med2, med1) in interaction_db:
            interactions.append(interaction_db[(med2, med1)])

        return interactions

    def get_medication_info(self, medication_name: str) -> Dict[str, Any]:
        """Get comprehensive medication information"""

        mapping = self.map_medication(medication_name)
        if not mapping:
            return {"error": "Medication not found in database"}

        drug_class = self.get_drug_class(medication_name)

        return {
            "medication": medication_name,
            "rxcui": mapping.rxcui,
            "rxnorm_name": mapping.rxnorm_name,
            "drug_class": drug_class,
            "term_type": mapping.term_type
        }
