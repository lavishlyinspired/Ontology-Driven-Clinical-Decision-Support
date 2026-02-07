"""
LOINC Integration Service for Lab Result Interpretation

This service provides comprehensive LOINC (Logical Observation Identifiers Names and Codes)
integration for the Lung Cancer Assistant, enabling natural language interpretation of
laboratory results with clinical context.

Author: LCA Development Team
Version: 1.0.0
"""

import csv
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class LOINCCode:
    """Structured LOINC code data"""
    loinc_num: str
    component: str
    property: str
    time_aspect: str
    system: str
    scale_type: str
    method_type: str
    class_type: str
    short_name: str
    long_common_name: str
    status: str
    example_units: str
    order_obs: str


@dataclass
class LabInterpretation:
    """Interpretation result for a lab value"""
    loinc_code: str
    value: float
    unit: str
    interpretation: str  # "normal", "low", "high", "critical_low", "critical_high"
    reference_range: Dict[str, Any]
    clinical_significance: str
    recommendations: List[str]
    context: str


class LOINCService:
    """
    LOINC Integration Service for Lab Result Interpretation
    
    Provides:
    - LOINC code lookup and search
    - Lab result interpretation with clinical context
    - Reference range management
    - Lung cancer specific lab panels
    - Unit conversion utilities
    """
    
    # Lung Cancer Relevant LOINC Codes with reference ranges
    LUNG_CANCER_LOINC = {
        # Tumor Markers
        "17842-6": {
            "name": "Carcinoembryonic Ag [Mass/volume] in Serum or Plasma",
            "abbrev": "CEA",
            "category": "tumor_marker",
            "reference": {"low": 0, "high": 3.0, "unit": "ng/mL", "smoker_high": 5.0},
            "clinical_use": "Adenocarcinoma monitoring, treatment response"
        },
        "27775-1": {
            "name": "CYFRA 21-1 [Mass/volume] in Serum or Plasma",
            "abbrev": "CYFRA 21-1",
            "category": "tumor_marker",
            "reference": {"low": 0, "high": 3.3, "unit": "ng/mL"},
            "clinical_use": "NSCLC monitoring, especially squamous cell"
        },
        "2039-6": {
            "name": "Neuron specific enolase [Mass/volume] in Serum or Plasma",
            "abbrev": "NSE",
            "category": "tumor_marker",
            "reference": {"low": 0, "high": 16.3, "unit": "ng/mL"},
            "clinical_use": "SCLC monitoring and response assessment"
        },
        "33746-8": {
            "name": "Squamous cell carcinoma Ag [Mass/volume] in Serum or Plasma",
            "abbrev": "SCC-Ag",
            "category": "tumor_marker",
            "reference": {"low": 0, "high": 1.5, "unit": "ng/mL"},
            "clinical_use": "Squamous cell carcinoma monitoring"
        },
        "47527-7": {
            "name": "Pro-gastrin-releasing peptide [Mass/volume] in Serum or Plasma",
            "abbrev": "ProGRP",
            "category": "tumor_marker",
            "reference": {"low": 0, "high": 50, "unit": "pg/mL"},
            "clinical_use": "SCLC diagnosis and monitoring"
        },
        
        # Molecular Biomarkers
        "21667-1": {
            "name": "EGFR gene mutations [Identifier] in Blood or Tissue by Molecular genetics method",
            "abbrev": "EGFR",
            "category": "molecular",
            "reference": {"type": "categorical", "values": ["Positive", "Negative", "Indeterminate"]},
            "clinical_use": "EGFR TKI eligibility (osimertinib, erlotinib)"
        },
        "69725-2": {
            "name": "ALK gene rearrangements [Identifier] in Blood or Tissue by Molecular genetics method",
            "abbrev": "ALK",
            "category": "molecular",
            "reference": {"type": "categorical", "values": ["Positive", "Negative", "Indeterminate"]},
            "clinical_use": "ALK TKI eligibility (lorlatinib, alectinib)"
        },
        "81704-5": {
            "name": "ROS1 gene rearrangements [Identifier] in Blood or Tissue by Molecular genetics method",
            "abbrev": "ROS1",
            "category": "molecular",
            "reference": {"type": "categorical", "values": ["Positive", "Negative", "Indeterminate"]},
            "clinical_use": "ROS1 TKI eligibility (entrectinib, crizotinib)"
        },
        "69726-0": {
            "name": "KRAS gene mutations [Identifier] in Blood or Tissue by Molecular genetics method",
            "abbrev": "KRAS",
            "category": "molecular",
            "reference": {"type": "categorical", "values": ["G12C", "G12D", "G12V", "Other", "Negative"]},
            "clinical_use": "KRAS G12C inhibitor eligibility (sotorasib, adagrasib)"
        },
        "85336-2": {
            "name": "BRAF V600E mutation [Presence] in Blood or Tissue by Molecular genetics method",
            "abbrev": "BRAF V600E",
            "category": "molecular",
            "reference": {"type": "categorical", "values": ["Positive", "Negative"]},
            "clinical_use": "BRAF inhibitor eligibility (dabrafenib/trametinib)"
        },
        "40557-1": {
            "name": "PD-L1 by clone 22C3 [Score] in Tissue by Immune stain",
            "abbrev": "PD-L1",
            "category": "molecular",
            "reference": {"type": "percentage", "thresholds": {"high": 50, "positive": 1}},
            "clinical_use": "Immunotherapy selection (pembrolizumab eligibility)"
        },
        "85337-0": {
            "name": "MET gene exon 14 skipping mutation [Presence] in Blood or Tissue",
            "abbrev": "MET ex14",
            "category": "molecular",
            "reference": {"type": "categorical", "values": ["Positive", "Negative"]},
            "clinical_use": "MET inhibitor eligibility (capmatinib, tepotinib)"
        },
        "85338-8": {
            "name": "RET gene rearrangements [Identifier] in Blood or Tissue",
            "abbrev": "RET",
            "category": "molecular",
            "reference": {"type": "categorical", "values": ["Positive", "Negative"]},
            "clinical_use": "RET inhibitor eligibility (selpercatinib, pralsetinib)"
        },
        "85339-6": {
            "name": "NTRK gene fusions [Identifier] in Blood or Tissue",
            "abbrev": "NTRK",
            "category": "molecular",
            "reference": {"type": "categorical", "values": ["Positive", "Negative"]},
            "clinical_use": "NTRK inhibitor eligibility (larotrectinib, entrectinib)"
        },
        "85340-4": {
            "name": "HER2 gene mutations [Identifier] in Blood or Tissue",
            "abbrev": "HER2",
            "category": "molecular",
            "reference": {"type": "categorical", "values": ["Positive", "Negative"]},
            "clinical_use": "HER2 targeted therapy (trastuzumab deruxtecan)"
        },
        
        # Complete Blood Count
        "26464-8": {
            "name": "Leukocytes [#/volume] in Blood",
            "abbrev": "WBC",
            "category": "hematology",
            "reference": {"low": 4.5, "high": 11.0, "unit": "10^9/L", "critical_low": 2.0, "critical_high": 30.0},
            "clinical_use": "Infection monitoring, chemotherapy toxicity"
        },
        "718-7": {
            "name": "Hemoglobin [Mass/volume] in Blood",
            "abbrev": "Hgb",
            "category": "hematology",
            "reference": {"low_male": 13.5, "high_male": 17.5, "low_female": 12.0, "high_female": 16.0, "unit": "g/dL", "critical_low": 7.0},
            "clinical_use": "Anemia assessment, transfusion threshold"
        },
        "26515-7": {
            "name": "Platelets [#/volume] in Blood",
            "abbrev": "PLT",
            "category": "hematology",
            "reference": {"low": 150, "high": 400, "unit": "10^9/L", "critical_low": 50, "critical_high": 1000},
            "clinical_use": "Bleeding risk, chemotherapy hold threshold"
        },
        "26453-1": {
            "name": "Neutrophils [#/volume] in Blood",
            "abbrev": "ANC",
            "category": "hematology",
            "reference": {"low": 1.5, "high": 8.0, "unit": "10^9/L", "critical_low": 0.5, "chemo_hold": 1.5},
            "clinical_use": "Neutropenia grading, chemotherapy dosing"
        },
        "26449-9": {
            "name": "Lymphocytes [#/volume] in Blood",
            "abbrev": "Lymph",
            "category": "hematology",
            "reference": {"low": 1.0, "high": 4.8, "unit": "10^9/L"},
            "clinical_use": "Immune function, immunotherapy response"
        },
        
        # Chemistry Panel
        "2160-0": {
            "name": "Creatinine [Mass/volume] in Serum or Plasma",
            "abbrev": "Cr",
            "category": "chemistry",
            "reference": {"low_male": 0.7, "high_male": 1.3, "low_female": 0.5, "high_female": 1.1, "unit": "mg/dL"},
            "clinical_use": "Renal function, platinum dose calculation (CrCl)"
        },
        "3094-0": {
            "name": "Urea nitrogen [Mass/volume] in Serum or Plasma",
            "abbrev": "BUN",
            "category": "chemistry",
            "reference": {"low": 7, "high": 20, "unit": "mg/dL"},
            "clinical_use": "Renal function, dehydration assessment"
        },
        "1742-6": {
            "name": "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
            "abbrev": "ALT",
            "category": "chemistry",
            "reference": {"low": 0, "high": 35, "unit": "U/L", "hold_chemo": 105},
            "clinical_use": "Hepatotoxicity monitoring (TKIs, immunotherapy)"
        },
        "1920-8": {
            "name": "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
            "abbrev": "AST",
            "category": "chemistry",
            "reference": {"low": 0, "high": 35, "unit": "U/L", "hold_chemo": 105},
            "clinical_use": "Hepatotoxicity monitoring (TKIs, immunotherapy)"
        },
        "1975-2": {
            "name": "Bilirubin.total [Mass/volume] in Serum or Plasma",
            "abbrev": "T.Bili",
            "category": "chemistry",
            "reference": {"low": 0.1, "high": 1.2, "unit": "mg/dL", "hold_chemo": 1.5},
            "clinical_use": "Hepatic function, drug metabolism"
        },
        "6768-6": {
            "name": "Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma",
            "abbrev": "ALP",
            "category": "chemistry",
            "reference": {"low": 44, "high": 147, "unit": "U/L"},
            "clinical_use": "Bone/liver metastases screening"
        },
        "2345-7": {
            "name": "Glucose [Mass/volume] in Serum or Plasma",
            "abbrev": "Glucose",
            "category": "chemistry",
            "reference": {"low": 70, "high": 100, "unit": "mg/dL", "critical_high": 400},
            "clinical_use": "Steroid-induced hyperglycemia"
        },
        "2823-3": {
            "name": "Potassium [Moles/volume] in Serum or Plasma",
            "abbrev": "K",
            "category": "chemistry",
            "reference": {"low": 3.5, "high": 5.0, "unit": "mmol/L", "critical_low": 2.5, "critical_high": 6.5},
            "clinical_use": "Electrolyte monitoring, cisplatin nephrotoxicity"
        },
        "2951-2": {
            "name": "Sodium [Moles/volume] in Serum or Plasma",
            "abbrev": "Na",
            "category": "chemistry",
            "reference": {"low": 136, "high": 145, "unit": "mmol/L", "critical_low": 120, "critical_high": 160},
            "clinical_use": "SIADH monitoring (common in SCLC)"
        },
        "17861-6": {
            "name": "Calcium [Mass/volume] in Serum or Plasma",
            "abbrev": "Ca",
            "category": "chemistry",
            "reference": {"low": 8.5, "high": 10.5, "unit": "mg/dL", "critical_high": 14.0},
            "clinical_use": "Hypercalcemia of malignancy"
        },
        "2276-4": {
            "name": "Magnesium [Mass/volume] in Serum or Plasma",
            "abbrev": "Mg",
            "category": "chemistry",
            "reference": {"low": 1.7, "high": 2.2, "unit": "mg/dL"},
            "clinical_use": "Cisplatin-induced hypomagnesemia"
        },
        
        # Thyroid (immunotherapy monitoring)
        "3016-3": {
            "name": "Thyrotropin [Units/volume] in Serum or Plasma",
            "abbrev": "TSH",
            "category": "endocrine",
            "reference": {"low": 0.4, "high": 4.0, "unit": "mIU/L"},
            "clinical_use": "Immune-related thyroiditis (pembrolizumab, nivolumab)"
        },
        "3026-2": {
            "name": "Thyroxine (T4) free [Mass/volume] in Serum or Plasma",
            "abbrev": "Free T4",
            "category": "endocrine",
            "reference": {"low": 0.8, "high": 1.8, "unit": "ng/dL"},
            "clinical_use": "Thyroid dysfunction from immunotherapy"
        },
        
        # Pulmonary Function
        "19926-5": {
            "name": "FEV1 [Volume] Respiratory system",
            "abbrev": "FEV1",
            "category": "pulmonary",
            "reference": {"type": "percentage_predicted", "normal": 80},
            "clinical_use": "Surgical candidacy, pneumonectomy risk"
        },
        "19927-3": {
            "name": "FEV1/FVC [Ratio]",
            "abbrev": "FEV1/FVC",
            "category": "pulmonary",
            "reference": {"low": 0.70, "type": "ratio"},
            "clinical_use": "COPD assessment, surgical planning"
        },
        "19911-7": {
            "name": "Diffusing capacity of lung for carbon monoxide [Volume/Time/Pressure]",
            "abbrev": "DLCO",
            "category": "pulmonary",
            "reference": {"type": "percentage_predicted", "normal": 80, "high_risk": 40},
            "clinical_use": "Pneumonectomy risk, radiation tolerance"
        },
        
        # Coagulation
        "5902-2": {
            "name": "Prothrombin time (PT)",
            "abbrev": "PT",
            "category": "coagulation",
            "reference": {"low": 11, "high": 13.5, "unit": "seconds"},
            "clinical_use": "Bleeding risk, anticoagulation"
        },
        "6301-6": {
            "name": "INR in Platelet poor plasma by Coagulation assay",
            "abbrev": "INR",
            "category": "coagulation",
            "reference": {"low": 0.9, "high": 1.1, "therapeutic": {"low": 2.0, "high": 3.0}},
            "clinical_use": "Warfarin monitoring, VTE treatment"
        },
        "3173-2": {
            "name": "aPTT in Blood by Coagulation assay",
            "abbrev": "aPTT",
            "category": "coagulation",
            "reference": {"low": 25, "high": 35, "unit": "seconds"},
            "clinical_use": "Heparin monitoring"
        },
    }
    
    # Lung Cancer Lab Panels
    LAB_PANELS = {
        "baseline_staging": {
            "name": "Baseline Staging Panel",
            "description": "Initial workup for newly diagnosed lung cancer",
            "tests": ["17842-6", "26464-8", "718-7", "26515-7", "2160-0", "1742-6", "1920-8", "1975-2", "6768-6", "2823-3", "2951-2", "17861-6"]
        },
        "molecular_testing": {
            "name": "Molecular Biomarker Panel",
            "description": "Comprehensive biomarker testing for advanced NSCLC",
            "tests": ["21667-1", "69725-2", "81704-5", "69726-0", "85336-2", "40557-1", "85337-0", "85338-8", "85339-6", "85340-4"]
        },
        "chemotherapy_monitoring": {
            "name": "Chemotherapy Monitoring Panel",
            "description": "Pre-cycle labs for platinum-based chemotherapy",
            "tests": ["26464-8", "718-7", "26515-7", "26453-1", "2160-0", "1742-6", "1920-8", "2276-4", "2823-3"]
        },
        "immunotherapy_monitoring": {
            "name": "Immunotherapy Monitoring Panel",
            "description": "Monitoring for immune-related adverse events",
            "tests": ["26464-8", "1742-6", "1920-8", "1975-2", "3016-3", "3026-2", "2345-7", "2160-0"]
        },
        "tki_monitoring": {
            "name": "TKI Monitoring Panel",
            "description": "Monitoring for tyrosine kinase inhibitor toxicity",
            "tests": ["26464-8", "718-7", "26515-7", "1742-6", "1920-8", "1975-2", "2160-0"]
        },
        "sclc_panel": {
            "name": "SCLC Specific Panel",
            "description": "Tumor markers specific for small cell lung cancer",
            "tests": ["2039-6", "47527-7", "2951-2"]
        },
        "surgical_clearance": {
            "name": "Surgical Clearance Panel",
            "description": "Pre-operative assessment for lung resection",
            "tests": ["26464-8", "718-7", "26515-7", "5902-2", "3173-2", "2160-0", "2345-7", "19926-5", "19927-3", "19911-7"]
        }
    }
    
    def __init__(self, loinc_path: Optional[str] = None):
        """
        Initialize the LOINC service.
        
        Args:
            loinc_path: Path to the LOINC data directory
        """
        from ..config import LCAConfig
        self.loinc_path = loinc_path or LCAConfig.LOINC_PATH
        self._loinc_cache: Dict[str, LOINCCode] = {}
        self._loaded = False
    
    def _load_loinc_data(self):
        """Load LOINC data from CSV file"""
        if self._loaded:
            return
        
        csv_path = os.path.join(self.loinc_path, "LoincTable", "Loinc.csv")
        
        if not os.path.exists(csv_path):
            logger.warning(f"LOINC CSV not found at {csv_path}, using built-in codes only")
            self._loaded = True
            return
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    loinc_num = row.get("LOINC_NUM", "")
                    if loinc_num:
                        self._loinc_cache[loinc_num] = LOINCCode(
                            loinc_num=loinc_num,
                            component=row.get("COMPONENT", ""),
                            property=row.get("PROPERTY", ""),
                            time_aspect=row.get("TIME_ASPCT", ""),
                            system=row.get("SYSTEM", ""),
                            scale_type=row.get("SCALE_TYP", ""),
                            method_type=row.get("METHOD_TYP", ""),
                            class_type=row.get("CLASS", ""),
                            short_name=row.get("SHORTNAME", ""),
                            long_common_name=row.get("LONG_COMMON_NAME", ""),
                            status=row.get("STATUS", ""),
                            example_units=row.get("EXAMPLE_UNITS", ""),
                            order_obs=row.get("ORDER_OBS", "")
                        )
            self._loaded = True
            logger.info(f"Loaded {len(self._loinc_cache)} LOINC codes")
        except Exception as e:
            logger.error(f"Error loading LOINC data: {e}")
            self._loaded = True
    
    def search_loinc(
        self, 
        query: str, 
        category: Optional[str] = None,
        max_results: int = 20
    ) -> Dict[str, Any]:
        """
        Search LOINC codes by component name or keyword.
        
        Args:
            query: Search term
            category: Optional filter by category (tumor_marker, molecular, hematology, chemistry)
            max_results: Maximum results to return
            
        Returns:
            Dict with matching LOINC codes
        """
        self._load_loinc_data()
        
        results = []
        query_lower = query.lower()
        
        # First search lung cancer specific codes
        for code, data in self.LUNG_CANCER_LOINC.items():
            if category and data.get("category") != category:
                continue
            
            name_lower = data["name"].lower()
            abbrev_lower = data.get("abbrev", "").lower()
            
            if query_lower in name_lower or query_lower in abbrev_lower or query_lower == code:
                results.append({
                    "loinc_code": code,
                    "name": data["name"],
                    "abbrev": data.get("abbrev"),
                    "category": data.get("category"),
                    "clinical_use": data.get("clinical_use"),
                    "reference": data.get("reference"),
                    "source": "lung_cancer_curated"
                })
        
        # Then search full LOINC database if available
        if len(results) < max_results:
            for code, loinc in self._loinc_cache.items():
                if code in [r["loinc_code"] for r in results]:
                    continue
                
                if (query_lower in loinc.component.lower() or 
                    query_lower in loinc.long_common_name.lower() or
                    query_lower in loinc.short_name.lower()):
                    results.append({
                        "loinc_code": code,
                        "name": loinc.long_common_name,
                        "abbrev": loinc.short_name,
                        "component": loinc.component,
                        "system": loinc.system,
                        "status": loinc.status,
                        "source": "loinc_database"
                    })
                
                if len(results) >= max_results:
                    break
        
        return {
            "status": "success",
            "query": query,
            "category_filter": category,
            "results": results[:max_results],
            "total_found": len(results)
        }
    
    def get_loinc_details(self, loinc_code: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific LOINC code.
        
        Args:
            loinc_code: The LOINC code to look up
            
        Returns:
            Detailed LOINC information
        """
        self._load_loinc_data()
        
        # Check lung cancer curated codes first
        if loinc_code in self.LUNG_CANCER_LOINC:
            data = self.LUNG_CANCER_LOINC[loinc_code]
            return {
                "status": "success",
                "loinc_code": loinc_code,
                "name": data["name"],
                "abbrev": data.get("abbrev"),
                "category": data.get("category"),
                "clinical_use": data.get("clinical_use"),
                "reference_range": data.get("reference"),
                "lung_cancer_relevance": True,
                "source": "lung_cancer_curated"
            }
        
        # Check LOINC database
        if loinc_code in self._loinc_cache:
            loinc = self._loinc_cache[loinc_code]
            return {
                "status": "success",
                "loinc_code": loinc_code,
                "name": loinc.long_common_name,
                "abbrev": loinc.short_name,
                "component": loinc.component,
                "property": loinc.property,
                "system": loinc.system,
                "scale_type": loinc.scale_type,
                "method_type": loinc.method_type,
                "class_type": loinc.class_type,
                "status": loinc.status,
                "example_units": loinc.example_units,
                "lung_cancer_relevance": False,
                "source": "loinc_database"
            }
        
        return {
            "status": "not_found",
            "loinc_code": loinc_code,
            "message": f"LOINC code {loinc_code} not found"
        }
    
    def interpret_lab_result(
        self,
        loinc_code: str,
        value: float,
        unit: Optional[str] = None,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Interpret a laboratory result with clinical context.
        
        Args:
            loinc_code: The LOINC code for the test
            value: The numeric value
            unit: The unit of measurement
            patient_context: Optional patient context (age, sex, diagnosis, treatment)
            
        Returns:
            Clinical interpretation with recommendations
        """
        context = patient_context or {}
        sex = context.get("sex", "unknown")
        is_smoker = context.get("smoking_status", "").lower() in ["current", "former"]
        current_treatment = context.get("current_treatment", "")
        diagnosis = context.get("diagnosis", {})
        
        if loinc_code not in self.LUNG_CANCER_LOINC:
            return {
                "status": "error",
                "loinc_code": loinc_code,
                "message": f"No interpretation rules for LOINC code {loinc_code}",
                "value": value,
                "unit": unit
            }
        
        test_data = self.LUNG_CANCER_LOINC[loinc_code]
        ref = test_data.get("reference", {})
        
        interpretation = "normal"
        clinical_significance = ""
        recommendations = []
        
        # Get appropriate reference range based on context
        if ref.get("type") == "categorical":
            # For molecular biomarkers
            return self._interpret_categorical(loinc_code, str(value), test_data, context)
        
        elif ref.get("type") == "percentage":
            # For PD-L1
            return self._interpret_percentage(loinc_code, value, test_data, context)
        
        elif ref.get("type") == "percentage_predicted":
            # For pulmonary function tests
            return self._interpret_pft(loinc_code, value, test_data, context)
        
        else:
            # Numeric interpretation
            low = ref.get("low_male" if sex == "male" else "low_female", ref.get("low", 0))
            high = ref.get("high_male" if sex == "male" else "high_female", ref.get("high", float('inf')))
            
            # Adjust for smoker status if applicable
            if is_smoker and ref.get("smoker_high"):
                high = ref["smoker_high"]
            
            critical_low = ref.get("critical_low")
            critical_high = ref.get("critical_high")
            
            # Determine interpretation
            if critical_low and value < critical_low:
                interpretation = "critical_low"
                clinical_significance = "CRITICAL: Immediate attention required"
                recommendations.append("Urgent clinical review needed")
            elif critical_high and value > critical_high:
                interpretation = "critical_high"
                clinical_significance = "CRITICAL: Immediate attention required"
                recommendations.append("Urgent clinical review needed")
            elif value < low:
                interpretation = "low"
                clinical_significance = f"Below normal range ({low}-{high} {ref.get('unit', '')})"
            elif value > high:
                interpretation = "high"
                clinical_significance = f"Above normal range ({low}-{high} {ref.get('unit', '')})"
            else:
                interpretation = "normal"
                clinical_significance = f"Within normal range ({low}-{high} {ref.get('unit', '')})"
        
        # Add specific recommendations based on test and interpretation
        recommendations.extend(self._get_recommendations(loinc_code, interpretation, value, context))
        
        return {
            "status": "success",
            "loinc_code": loinc_code,
            "test_name": test_data["name"],
            "abbrev": test_data.get("abbrev"),
            "value": value,
            "unit": unit or ref.get("unit", ""),
            "interpretation": interpretation,
            "reference_range": ref,
            "clinical_significance": clinical_significance,
            "recommendations": recommendations,
            "clinical_use": test_data.get("clinical_use"),
            "patient_context": context
        }
    
    def _interpret_categorical(
        self, 
        loinc_code: str, 
        value: str, 
        test_data: Dict, 
        context: Dict
    ) -> Dict[str, Any]:
        """Interpret categorical biomarker results"""
        biomarker = test_data.get("abbrev", "")
        clinical_use = test_data.get("clinical_use", "")
        
        recommendations = []
        
        if "positive" in value.lower() or value.upper() in ["G12C", "L858R", "T790M"]:
            interpretation = "positive"
            clinical_significance = f"{biomarker} positive - {clinical_use}"
            
            if "EGFR" in biomarker:
                recommendations.append("Consider osimertinib as first-line therapy")
                recommendations.append("Refer for genetic counseling if germline mutation suspected")
            elif "ALK" in biomarker:
                recommendations.append("Consider lorlatinib or alectinib as first-line therapy")
            elif "KRAS" in value.upper():
                recommendations.append("Consider sotorasib or adagrasib for KRAS G12C")
            elif "PD-L1" in biomarker:
                recommendations.append("Eligible for pembrolizumab monotherapy if TPS ≥50%")
        else:
            interpretation = "negative"
            clinical_significance = f"{biomarker} negative"
            recommendations.append("Continue comprehensive molecular testing")
        
        return {
            "status": "success",
            "loinc_code": loinc_code,
            "test_name": test_data["name"],
            "abbrev": biomarker,
            "value": value,
            "interpretation": interpretation,
            "clinical_significance": clinical_significance,
            "recommendations": recommendations,
            "clinical_use": clinical_use
        }
    
    def _interpret_percentage(
        self, 
        loinc_code: str, 
        value: float, 
        test_data: Dict, 
        context: Dict
    ) -> Dict[str, Any]:
        """Interpret percentage-based results (PD-L1)"""
        ref = test_data.get("reference", {})
        thresholds = ref.get("thresholds", {})
        
        recommendations = []
        
        if value >= thresholds.get("high", 50):
            interpretation = "high_expression"
            clinical_significance = f"PD-L1 TPS ≥50% - High expression"
            recommendations.append("Eligible for pembrolizumab monotherapy (KEYNOTE-024)")
            recommendations.append("Consider pembrolizumab + chemotherapy for faster response")
        elif value >= thresholds.get("positive", 1):
            interpretation = "positive"
            clinical_significance = f"PD-L1 TPS 1-49% - Positive expression"
            recommendations.append("Consider pembrolizumab + platinum-doublet chemotherapy")
            recommendations.append("Not eligible for pembrolizumab monotherapy")
        else:
            interpretation = "negative"
            clinical_significance = f"PD-L1 TPS <1% - Negative"
            recommendations.append("Consider chemotherapy with or without immunotherapy")
            recommendations.append("Check for other actionable biomarkers")
        
        return {
            "status": "success",
            "loinc_code": loinc_code,
            "test_name": test_data["name"],
            "abbrev": test_data.get("abbrev"),
            "value": value,
            "unit": "%",
            "interpretation": interpretation,
            "clinical_significance": clinical_significance,
            "recommendations": recommendations,
            "thresholds": thresholds
        }
    
    def _interpret_pft(
        self, 
        loinc_code: str, 
        value: float, 
        test_data: Dict, 
        context: Dict
    ) -> Dict[str, Any]:
        """Interpret pulmonary function tests"""
        ref = test_data.get("reference", {})
        normal_threshold = ref.get("normal", 80)
        high_risk_threshold = ref.get("high_risk", 40)
        
        recommendations = []
        
        if value >= normal_threshold:
            interpretation = "normal"
            clinical_significance = f"≥{normal_threshold}% predicted - Normal pulmonary function"
            recommendations.append("Likely tolerate surgical resection")
        elif value >= high_risk_threshold:
            interpretation = "reduced"
            clinical_significance = f"{high_risk_threshold}-{normal_threshold}% predicted - Reduced function"
            recommendations.append("May tolerate lobectomy; pneumonectomy higher risk")
            recommendations.append("Consider additional functional assessment")
        else:
            interpretation = "severely_reduced"
            clinical_significance = f"<{high_risk_threshold}% predicted - Severely reduced"
            recommendations.append("High surgical risk - consider non-surgical options")
            recommendations.append("SBRT may be preferred over surgery")
        
        return {
            "status": "success",
            "loinc_code": loinc_code,
            "test_name": test_data["name"],
            "abbrev": test_data.get("abbrev"),
            "value": value,
            "unit": "% predicted",
            "interpretation": interpretation,
            "clinical_significance": clinical_significance,
            "recommendations": recommendations
        }
    
    def _get_recommendations(
        self, 
        loinc_code: str, 
        interpretation: str, 
        value: float, 
        context: Dict
    ) -> List[str]:
        """Get specific recommendations based on test and interpretation"""
        recommendations = []
        test_data = self.LUNG_CANCER_LOINC.get(loinc_code, {})
        category = test_data.get("category", "")
        current_treatment = context.get("current_treatment", "").lower()
        
        if category == "tumor_marker":
            if interpretation == "high":
                recommendations.append("Consider imaging to assess disease status")
                recommendations.append("Repeat in 4-6 weeks to assess trend")
        
        elif category == "hematology":
            if "chemotherapy" in current_treatment or "chemo" in current_treatment:
                if loinc_code == "26453-1":  # ANC
                    if interpretation in ["low", "critical_low"]:
                        if value < 0.5:
                            recommendations.append("HOLD chemotherapy - febrile neutropenia risk")
                            recommendations.append("Consider G-CSF (filgrastim) support")
                        elif value < 1.5:
                            recommendations.append("Consider dose reduction or G-CSF prophylaxis")
                elif loinc_code == "26515-7":  # Platelets
                    if interpretation == "low" and value < 100:
                        recommendations.append("Hold chemotherapy if platelets < 100K")
                        recommendations.append("Consider platelet transfusion if < 10K or bleeding")
                elif loinc_code == "718-7":  # Hemoglobin
                    if interpretation == "low":
                        if value < 8.0:
                            recommendations.append("Consider RBC transfusion")
                        else:
                            recommendations.append("Monitor for symptoms, consider EPO if persistent")
        
        elif category == "chemistry":
            if loinc_code == "2160-0":  # Creatinine
                if interpretation == "high":
                    recommendations.append("Recalculate CrCl for platinum dosing")
                    if "cisplatin" in current_treatment:
                        recommendations.append("Consider switching to carboplatin")
            elif loinc_code in ["1742-6", "1920-8"]:  # ALT, AST
                if interpretation == "high":
                    ref = test_data.get("reference", {})
                    if value > ref.get("hold_chemo", 105):
                        recommendations.append("Consider holding treatment until LFTs improve")
                        if "tki" in current_treatment or "osimertinib" in current_treatment:
                            recommendations.append("May need TKI dose reduction")
        
        elif category == "endocrine":
            if "immunotherapy" in current_treatment or "pembrolizumab" in current_treatment:
                if loinc_code == "3016-3":  # TSH
                    if interpretation == "high":
                        recommendations.append("Suspect immune-related hypothyroidism")
                        recommendations.append("Check Free T4, consider levothyroxine replacement")
                    elif interpretation == "low":
                        recommendations.append("Suspect immune-related thyroiditis or hyperthyroidism")
                        recommendations.append("Check Free T4 and T3, monitor closely")
        
        return recommendations
    
    def get_lung_cancer_panel(self, panel_name: str) -> Dict[str, Any]:
        """
        Get a recommended lab panel for a specific clinical scenario.
        
        Args:
            panel_name: Name of the panel (baseline_staging, molecular_testing, etc.)
            
        Returns:
            Panel details with included tests
        """
        if panel_name not in self.LAB_PANELS:
            return {
                "status": "error",
                "message": f"Unknown panel: {panel_name}",
                "available_panels": list(self.LAB_PANELS.keys())
            }
        
        panel = self.LAB_PANELS[panel_name]
        tests = []
        
        for code in panel["tests"]:
            if code in self.LUNG_CANCER_LOINC:
                test_data = self.LUNG_CANCER_LOINC[code]
                tests.append({
                    "loinc_code": code,
                    "name": test_data["name"],
                    "abbrev": test_data.get("abbrev"),
                    "category": test_data.get("category"),
                    "reference": test_data.get("reference"),
                    "clinical_use": test_data.get("clinical_use")
                })
        
        return {
            "status": "success",
            "panel_name": panel_name,
            "name": panel["name"],
            "description": panel["description"],
            "tests": tests,
            "test_count": len(tests)
        }
    
    def list_panels(self) -> Dict[str, Any]:
        """List all available lab panels"""
        panels = []
        for key, panel in self.LAB_PANELS.items():
            panels.append({
                "panel_key": key,
                "name": panel["name"],
                "description": panel["description"],
                "test_count": len(panel["tests"])
            })
        
        return {
            "status": "success",
            "panels": panels,
            "total": len(panels)
        }
    
    def interpret_panel(
        self,
        panel_name: str,
        results: Dict[str, float],
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Interpret all results from a lab panel.
        
        Args:
            panel_name: The panel name
            results: Dict mapping LOINC codes to values
            patient_context: Optional patient context
            
        Returns:
            Comprehensive panel interpretation
        """
        panel_info = self.get_lung_cancer_panel(panel_name)
        if panel_info.get("status") == "error":
            return panel_info
        
        interpretations = []
        abnormal_count = 0
        critical_count = 0
        
        for code, value in results.items():
            result = self.interpret_lab_result(code, value, patient_context=patient_context)
            if result.get("status") == "success":
                interpretations.append(result)
                if result["interpretation"] not in ["normal", "negative"]:
                    abnormal_count += 1
                if "critical" in result["interpretation"]:
                    critical_count += 1
        
        # Generate summary
        summary = []
        if critical_count > 0:
            summary.append(f"⚠️ {critical_count} CRITICAL value(s) requiring immediate attention")
        if abnormal_count > 0:
            summary.append(f"⚡ {abnormal_count} abnormal result(s) identified")
        if abnormal_count == 0 and critical_count == 0:
            summary.append("✅ All results within normal limits")
        
        return {
            "status": "success",
            "panel_name": panel_name,
            "panel_description": panel_info.get("description"),
            "summary": summary,
            "critical_count": critical_count,
            "abnormal_count": abnormal_count,
            "interpretations": interpretations,
            "total_tests": len(interpretations)
        }


# Singleton instance
_service_instance: Optional[LOINCService] = None


def get_loinc_service(loinc_path: Optional[str] = None) -> LOINCService:
    """Get or create the LOINCService singleton"""
    global _service_instance
    if _service_instance is None:
        _service_instance = LOINCService(loinc_path)
    return _service_instance
