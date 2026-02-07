"""
UMLS Crosswalk Service
======================

Parses UMLS MRCONSO.RRF to build a CUI-based crosswalk between
SNOMED-CT, LOINC, RxNorm, and NCI Thesaurus vocabularies.

MRCONSO columns (pipe-delimited):
  CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF

We filter to SAB in {SNOMEDCT_US, LNC, RXNORM, NCI} and LAT=ENG.
"""

import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..config import LCAConfig
from ..logging_config import get_logger

logger = get_logger(__name__)

# Vocabularies we care about
TARGET_SABS = {"SNOMEDCT_US", "LNC", "RXNORM", "NCI"}

# Column indices in MRCONSO.RRF
COL_CUI = 0
COL_LAT = 1
COL_SAB = 11
COL_CODE = 13
COL_STR = 14


class UMLSCrosswalkService:
    """
    In-memory crosswalk between SNOMED-CT, LOINC, RxNorm, and NCIt
    via UMLS CUI mappings.

    Usage:
        svc = UMLSCrosswalkService()
        svc.load()
        nci_code = svc.get_crosswalk("SNOMEDCT_US", "254637007", "NCI")
    """

    def __init__(self, mrconso_path: str = None):
        self.mrconso_path = mrconso_path or os.path.join(
            LCAConfig.UMLS_PATH, "MRCONSO.RRF"
        )
        # CUI → {sab: set(codes)}
        self._cui_map: Dict[str, Dict[str, set]] = {}
        # (sab, code) → CUI for reverse lookup
        self._code_to_cui: Dict[Tuple[str, str], str] = {}
        # name → list of CUIs for fuzzy lookup
        self._name_to_cuis: Dict[str, List[str]] = {}
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> Dict[str, int]:
        """
        Parse MRCONSO.RRF and build in-memory crosswalk.

        Returns:
            Dict with counts per vocabulary.
        """
        if not os.path.exists(self.mrconso_path):
            logger.warning(f"MRCONSO.RRF not found: {self.mrconso_path}")
            return {"success": False, "message": f"File not found: {self.mrconso_path}"}

        logger.info(f"Loading UMLS crosswalk from {self.mrconso_path}...")

        counts: Dict[str, int] = {sab: 0 for sab in TARGET_SABS}
        total_lines = 0

        with open(self.mrconso_path, "r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                parts = line.rstrip("\n").split("|")
                if len(parts) < 15:
                    continue

                lat = parts[COL_LAT]
                sab = parts[COL_SAB]

                if lat != "ENG" or sab not in TARGET_SABS:
                    continue

                cui = parts[COL_CUI]
                code = parts[COL_CODE]
                name = parts[COL_STR].lower()

                # Build CUI map
                if cui not in self._cui_map:
                    self._cui_map[cui] = {}
                if sab not in self._cui_map[cui]:
                    self._cui_map[cui][sab] = set()
                self._cui_map[cui][sab].add(code)

                # Build reverse lookup
                self._code_to_cui[(sab, code)] = cui

                # Build name lookup (first CUI wins for each name)
                if name not in self._name_to_cuis:
                    self._name_to_cuis[name] = []
                if cui not in self._name_to_cuis[name]:
                    self._name_to_cuis[name].append(cui)

                counts[sab] += 1

                if total_lines % 1_000_000 == 0:
                    logger.info(f"  Processed {total_lines:,} lines...")

        self._loaded = True
        logger.info(
            f"UMLS crosswalk loaded: {len(self._cui_map)} CUIs, "
            f"counts: {counts}"
        )
        return {"success": True, "cuis": len(self._cui_map), **counts}

    def get_crosswalk(
        self, source_vocab: str, source_code: str, target_vocab: str
    ) -> Optional[str]:
        """
        Translate a code from source vocabulary to target vocabulary.

        Args:
            source_vocab: Source SAB (e.g. "SNOMEDCT_US")
            source_code: Source concept code
            target_vocab: Target SAB (e.g. "NCI")

        Returns:
            First matching target code, or None.
        """
        cui = self._code_to_cui.get((source_vocab, source_code))
        if not cui:
            return None

        target_codes = self._cui_map.get(cui, {}).get(target_vocab, set())
        return next(iter(target_codes), None)

    def get_all_mappings(self, cui: str) -> Dict[str, List[str]]:
        """
        Get all vocabulary mappings for a CUI.

        Returns:
            Dict mapping SAB → list of codes.
        """
        entry = self._cui_map.get(cui, {})
        return {sab: sorted(codes) for sab, codes in entry.items()}

    def find_cui_by_name(self, name: str) -> Optional[str]:
        """
        Find a CUI by concept name (case-insensitive).

        Returns:
            First matching CUI, or None.
        """
        cuis = self._name_to_cuis.get(name.lower(), [])
        return cuis[0] if cuis else None

    def get_crosswalk_by_name(
        self, name: str, target_vocab: str
    ) -> Optional[str]:
        """
        Find a target code by concept name.
        """
        cui = self.find_cui_by_name(name)
        if not cui:
            return None
        codes = self._cui_map.get(cui, {}).get(target_vocab, set())
        return next(iter(codes), None)


# Singleton
_crosswalk_service: Optional[UMLSCrosswalkService] = None


def get_crosswalk_service() -> UMLSCrosswalkService:
    """Get or create the singleton crosswalk service."""
    global _crosswalk_service
    if _crosswalk_service is None:
        _crosswalk_service = UMLSCrosswalkService()
    return _crosswalk_service
