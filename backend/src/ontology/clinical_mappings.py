"""Centralized Clinical Mappings

Provides standardized clinical concept mappings used across the LCA system.
These are semantic/clinical mappings (not SNOMED codes).
For SNOMED-CT code mappings, use SNOMEDLoader.
"""

from typing import Dict, Optional

# Centralized logging
from ..logging_config import get_logger

logger = get_logger(__name__)


class ClinicalMappings:
    """Centralized clinical mappings for non-SNOMED concepts"""

    # Stage to numeric mapping for statistical analysis
    STAGE_TO_NUMERIC: Dict[str, int] = {
        'IA': 1, 'IB': 1,
        'IIA': 2, 'IIB': 2,
        'IIIA': 3, 'IIIB': 3, 'IIIC': 3,
        'IV': 4, 'IVA': 4, 'IVB': 4,
        # Roman numerals
        'I': 1, 'II': 2, 'III': 3, 'IV': 4
    }

    # Stage groupings for eligibility/matching
    STAGE_GROUPS = {
        'early': ['IA', 'IB', 'IIA', 'IIB'],
        'locally_advanced': ['IIIA', 'IIIB', 'IIIC'],
        'metastatic': ['IV', 'IVA', 'IVB'],
        'resectable': ['IA', 'IB', 'IIA', 'IIB', 'IIIA'],
        'unresectable': ['IIIB', 'IIIC', 'IV', 'IVA', 'IVB']
    }

    # Treatment response mapping
    RESPONSE_CATEGORIES = {
        'complete_response': 'CR',
        'partial_response': 'PR',
        'stable_disease': 'SD',
        'progressive_disease': 'PD'
    }

    # Progression status mapping
    PROGRESSION_STATUS = {
        'no_progression': 0,
        'local_progression': 1,
        'regional_progression': 2,
        'distant_metastasis': 3
    }

    @classmethod
    def stage_to_numeric(cls, stage: str) -> Optional[int]:
        """Convert TNM stage to numeric value for analysis"""
        return cls.STAGE_TO_NUMERIC.get(stage)

    @classmethod
    def is_early_stage(cls, stage: str) -> bool:
        """Check if stage is early (I-II)"""
        return stage in cls.STAGE_GROUPS['early']

    @classmethod
    def is_locally_advanced(cls, stage: str) -> bool:
        """Check if stage is locally advanced (III)"""
        return stage in cls.STAGE_GROUPS['locally_advanced']

    @classmethod
    def is_metastatic(cls, stage: str) -> bool:
        """Check if stage is metastatic (IV)"""
        return stage in cls.STAGE_GROUPS['metastatic']

    @classmethod
    def get_stage_keywords(cls, stage: str) -> list:
        """Get search keywords for stage (used in trial matching)"""
        stage_numeric = cls.stage_to_numeric(stage)
        keywords = [stage]

        if cls.is_early_stage(stage):
            keywords.extend(['early stage', 'stage I', 'stage II', 'resectable'])
        elif cls.is_locally_advanced(stage):
            keywords.extend(['locally advanced', 'stage III', 'unresectable'])
        elif cls.is_metastatic(stage):
            keywords.extend(['metastatic', 'stage IV', 'advanced'])

        return keywords
