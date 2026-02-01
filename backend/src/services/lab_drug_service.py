"""
Lab-Drug Integration Service

This service provides cross-integration between LOINC (lab tests) and RXNORM (drugs)
for comprehensive monitoring protocols, drug-induced lab changes, and dose adjustment
recommendations in lung cancer treatment.

Author: LCA Development Team
Version: 1.0.0
"""

import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

from .loinc_service import get_loinc_service, LOINCService
from .rxnorm_service import get_rxnorm_service, RXNORMService

logger = logging.getLogger(__name__)


@dataclass
class MonitoringParameter:
    """A lab parameter to monitor for a drug"""
    loinc_code: str
    test_name: str
    frequency: str
    rationale: str
    critical_threshold: Optional[str] = None
    action_required: Optional[str] = None


@dataclass
class DrugLabEffect:
    """Expected lab effect from a drug"""
    drug: str
    loinc_code: str
    test_name: str
    direction: str  # increase, decrease, variable
    mechanism: str
    clinical_significance: str
    expected_magnitude: Optional[str] = None
    time_course: Optional[str] = None


class LabDrugService:
    """
    Lab-Drug Integration Service
    
    Provides:
    - Drug effect on lab parameters
    - Monitoring protocols for regimens
    - Dose adjustments based on lab values
    - Lab-based toxicity assessment
    """
    
    # Drug effects on lab values (lung cancer drugs)
    DRUG_LAB_EFFECTS = {
        # EGFR TKIs
        "osimertinib": [
            {
                "loinc": "1742-6",
                "test": "ALT",
                "direction": "increase",
                "mechanism": "Hepatocellular injury",
                "significance": "May require dose modification or discontinuation",
                "magnitude": "Grade 1-2 in ~10% of patients",
                "time_course": "Usually within first 3 months"
            },
            {
                "loinc": "2160-0",
                "test": "Creatinine",
                "direction": "increase",
                "mechanism": "Creatinine transporter inhibition (not true nephrotoxicity)",
                "significance": "Typically reversible, rarely clinically significant",
                "magnitude": "~0.2-0.3 mg/dL increase common"
            },
            {
                "loinc": "6768-6",
                "test": "ALP",
                "direction": "increase",
                "mechanism": "Hepatic enzyme induction",
                "significance": "Usually mild, isolated elevation"
            }
        ],
        "erlotinib": [
            {
                "loinc": "1742-6",
                "test": "ALT",
                "direction": "increase",
                "mechanism": "Hepatotoxicity",
                "significance": "Dose reduction or discontinuation may be needed"
            },
            {
                "loinc": "1920-8",
                "test": "AST",
                "direction": "increase",
                "mechanism": "Hepatotoxicity",
                "significance": "Monitor concurrently with ALT"
            }
        ],
        
        # ALK TKIs
        "alectinib": [
            {
                "loinc": "1742-6",
                "test": "ALT",
                "direction": "increase",
                "mechanism": "Hepatotoxicity",
                "significance": "Dose modification per label"
            },
            {
                "loinc": "2157-6",
                "test": "CPK",
                "direction": "increase",
                "mechanism": "Myopathy",
                "significance": "Can be significant; monitor for myalgia",
                "magnitude": "Grade 3-4 in ~5% of patients"
            },
            {
                "loinc": "718-7",
                "test": "Hemoglobin",
                "direction": "decrease",
                "mechanism": "Anemia (mechanism unclear)",
                "significance": "Usually mild"
            }
        ],
        "lorlatinib": [
            {
                "loinc": "2093-3",
                "test": "Total Cholesterol",
                "direction": "increase",
                "mechanism": "Lipid metabolism alteration",
                "significance": "Common; may require lipid-lowering therapy",
                "magnitude": "Grade 3-4 in ~16%"
            },
            {
                "loinc": "2571-8",
                "test": "Triglycerides",
                "direction": "increase",
                "mechanism": "Lipid metabolism alteration",
                "significance": "Common; may require lipid-lowering therapy",
                "magnitude": "Grade 3-4 in ~16%"
            },
            {
                "loinc": "1742-6",
                "test": "ALT",
                "direction": "increase",
                "mechanism": "Hepatotoxicity",
                "significance": "Monitor per label"
            }
        ],
        "crizotinib": [
            {
                "loinc": "1742-6",
                "test": "ALT",
                "direction": "increase",
                "mechanism": "Hepatotoxicity",
                "significance": "Common; may require dose modification",
                "magnitude": "Grade 3-4 in ~14%"
            },
            {
                "loinc": "751-8",
                "test": "ANC",
                "direction": "decrease",
                "mechanism": "Myelosuppression",
                "significance": "Usually mild to moderate"
            }
        ],
        
        # Immunotherapy
        "pembrolizumab": [
            {
                "loinc": "3016-3",
                "test": "TSH",
                "direction": "variable",
                "mechanism": "Immune-related thyroiditis",
                "significance": "Hypo- or hyperthyroidism; may need hormone replacement",
                "magnitude": "Hypothyroidism in ~10-15%",
                "time_course": "Usually within 6 months"
            },
            {
                "loinc": "1742-6",
                "test": "ALT",
                "direction": "increase",
                "mechanism": "Immune-related hepatitis",
                "significance": "May require steroids or discontinuation"
            },
            {
                "loinc": "2160-0",
                "test": "Creatinine",
                "direction": "increase",
                "mechanism": "Immune-related nephritis",
                "significance": "Rare but can be severe"
            },
            {
                "loinc": "2345-7",
                "test": "Glucose",
                "direction": "increase",
                "mechanism": "Immune-related diabetes (insulitis)",
                "significance": "Rare but can present as DKA",
                "time_course": "Variable, can be rapid onset"
            },
            {
                "loinc": "2132-9",
                "test": "Cortisol",
                "direction": "decrease",
                "mechanism": "Immune-related hypophysitis/adrenalitis",
                "significance": "May need lifelong replacement"
            },
            {
                "loinc": "6690-2",
                "test": "WBC",
                "direction": "increase",
                "mechanism": "Inflammation (if irAE present)",
                "significance": "Can indicate active immune response"
            }
        ],
        "nivolumab": [
            {
                "loinc": "3016-3",
                "test": "TSH",
                "direction": "variable",
                "mechanism": "Immune-related thyroiditis",
                "significance": "Similar to pembrolizumab"
            },
            {
                "loinc": "1742-6",
                "test": "ALT",
                "direction": "increase",
                "mechanism": "Immune-related hepatitis",
                "significance": "May require steroids"
            },
            {
                "loinc": "1751-7",
                "test": "Albumin",
                "direction": "decrease",
                "mechanism": "Inflammation, colitis",
                "significance": "Can indicate severe colitis if with diarrhea"
            }
        ],
        "ipilimumab": [
            {
                "loinc": "3016-3",
                "test": "TSH",
                "direction": "variable",
                "mechanism": "Immune-related thyroiditis/hypophysitis",
                "significance": "Higher rate than PD-1 inhibitors alone"
            },
            {
                "loinc": "1742-6",
                "test": "ALT",
                "direction": "increase",
                "mechanism": "Immune-related hepatitis",
                "significance": "Higher rate with combination therapy"
            },
            {
                "loinc": "2951-2",
                "test": "Sodium",
                "direction": "decrease",
                "mechanism": "Hypophysitis with SIADH or adrenal insufficiency",
                "significance": "Check cortisol and pituitary function"
            }
        ],
        
        # Chemotherapy
        "cisplatin": [
            {
                "loinc": "2160-0",
                "test": "Creatinine",
                "direction": "increase",
                "mechanism": "Direct tubular toxicity",
                "significance": "Dose-limiting; requires hydration",
                "magnitude": "Significant nephrotoxicity in ~28-36%"
            },
            {
                "loinc": "2823-3",
                "test": "Potassium",
                "direction": "decrease",
                "mechanism": "Renal wasting",
                "significance": "Requires monitoring and replacement"
            },
            {
                "loinc": "19123-9",
                "test": "Magnesium",
                "direction": "decrease",
                "mechanism": "Renal wasting",
                "significance": "Common; requires monitoring and replacement"
            },
            {
                "loinc": "718-7",
                "test": "Hemoglobin",
                "direction": "decrease",
                "mechanism": "Myelosuppression",
                "significance": "Cumulative; may need ESA or transfusion"
            }
        ],
        "carboplatin": [
            {
                "loinc": "777-3",
                "test": "Platelets",
                "direction": "decrease",
                "mechanism": "Myelosuppression (dose-limiting)",
                "significance": "Thrombocytopenia is dose-limiting",
                "magnitude": "Nadir ~day 21",
                "time_course": "Recovery by day 28-35"
            },
            {
                "loinc": "751-8",
                "test": "ANC",
                "direction": "decrease",
                "mechanism": "Myelosuppression",
                "significance": "Less severe than cisplatin equivalent"
            },
            {
                "loinc": "2160-0",
                "test": "Creatinine",
                "direction": "increase",
                "mechanism": "Nephrotoxicity (less than cisplatin)",
                "significance": "Monitor for dose adjustments"
            }
        ],
        "pemetrexed": [
            {
                "loinc": "751-8",
                "test": "ANC",
                "direction": "decrease",
                "mechanism": "Myelosuppression",
                "significance": "Neutropenia common"
            },
            {
                "loinc": "777-3",
                "test": "Platelets",
                "direction": "decrease",
                "mechanism": "Myelosuppression",
                "significance": "Can be dose-limiting"
            },
            {
                "loinc": "2160-0",
                "test": "Creatinine",
                "direction": "increase",
                "mechanism": "Renal elimination affects clearance",
                "significance": "Adjust dose for renal function"
            }
        ],
        "paclitaxel": [
            {
                "loinc": "751-8",
                "test": "ANC",
                "direction": "decrease",
                "mechanism": "Myelosuppression",
                "significance": "Dose-limiting neutropenia",
                "magnitude": "Nadir day 8-11",
                "time_course": "Recovery by day 15-21"
            },
            {
                "loinc": "1742-6",
                "test": "ALT",
                "direction": "increase",
                "mechanism": "Hepatic metabolism",
                "significance": "May require dose adjustment"
            }
        ],
        "docetaxel": [
            {
                "loinc": "751-8",
                "test": "ANC",
                "direction": "decrease",
                "mechanism": "Myelosuppression",
                "significance": "Severe neutropenia common",
                "magnitude": "Grade 4 in ~75%",
                "time_course": "Nadir day 7, recovery day 21"
            },
            {
                "loinc": "1751-7",
                "test": "Albumin",
                "direction": "decrease",
                "mechanism": "Fluid retention syndrome",
                "significance": "Can cause pleural effusions, edema"
            }
        ],
        "etoposide": [
            {
                "loinc": "751-8",
                "test": "ANC",
                "direction": "decrease",
                "mechanism": "Myelosuppression",
                "significance": "Dose-limiting",
                "magnitude": "Nadir day 10-14"
            },
            {
                "loinc": "777-3",
                "test": "Platelets",
                "direction": "decrease",
                "mechanism": "Myelosuppression",
                "significance": "Less common than neutropenia"
            }
        ],
        
        # KRAS inhibitors
        "sotorasib": [
            {
                "loinc": "1742-6",
                "test": "ALT",
                "direction": "increase",
                "mechanism": "Hepatotoxicity",
                "significance": "Common; may require dose modification",
                "magnitude": "Grade 3-4 in ~11%"
            },
            {
                "loinc": "1920-8",
                "test": "AST",
                "direction": "increase",
                "mechanism": "Hepatotoxicity",
                "significance": "Monitor with ALT"
            }
        ],
        "adagrasib": [
            {
                "loinc": "1742-6",
                "test": "ALT",
                "direction": "increase",
                "mechanism": "Hepatotoxicity",
                "significance": "Monitor LFTs"
            },
            {
                "loinc": "2160-0",
                "test": "Creatinine",
                "direction": "increase",
                "mechanism": "Creatinine transporter inhibition",
                "significance": "May not reflect true GFR change"
            }
        ],
        
        # Targeted therapies
        "selpercatinib": [
            {
                "loinc": "1742-6",
                "test": "ALT",
                "direction": "increase",
                "mechanism": "Hepatotoxicity",
                "significance": "Common; may require dose modification"
            }
        ],
        "dabrafenib": [
            {
                "loinc": "2345-7",
                "test": "Glucose",
                "direction": "increase",
                "mechanism": "Hyperglycemia",
                "significance": "Monitor blood glucose"
            },
            {
                "loinc": "6768-6",
                "test": "ALP",
                "direction": "increase",
                "mechanism": "Hepatic effects",
                "significance": "Usually mild"
            }
        ],
        "trametinib": [
            {
                "loinc": "1742-6",
                "test": "ALT",
                "direction": "increase",
                "mechanism": "Hepatotoxicity",
                "significance": "Monitor with dabrafenib combination"
            },
            {
                "loinc": "2157-6",
                "test": "CPK",
                "direction": "increase",
                "mechanism": "Rhabdomyolysis risk",
                "significance": "Monitor for myalgias"
            }
        ]
    }
    
    # Monitoring protocols by regimen
    MONITORING_PROTOCOLS = {
        "osimertinib_monotherapy": {
            "regimen": "Osimertinib 80mg daily",
            "indication": "EGFR+ NSCLC",
            "baseline_labs": ["CBC", "CMP", "LFTs", "ECG"],
            "monitoring": [
                {"test": "LFTs", "frequency": "Monthly x3, then Q3 months", "rationale": "Hepatotoxicity surveillance"},
                {"test": "ECG", "frequency": "At baseline, then if symptoms", "rationale": "QTc monitoring"},
                {"test": "Creatinine", "frequency": "Q3 months", "rationale": "Renal function (note: may be elevated due to transporter inhibition)"}
            ],
            "alert_parameters": [
                {"test": "ALT", "threshold": ">3x ULN", "action": "Withhold until ≤2.5x ULN, resume at same dose. If >5x ULN, discontinue"},
                {"test": "QTc", "threshold": ">500ms", "action": "Withhold until ≤470ms, resume with dose reduction"}
            ]
        },
        "pembrolizumab_monotherapy": {
            "regimen": "Pembrolizumab 200mg Q3W",
            "indication": "PD-L1 TPS ≥50% NSCLC",
            "baseline_labs": ["CBC", "CMP", "LFTs", "TSH", "fT4"],
            "monitoring": [
                {"test": "TSH, fT4", "frequency": "Every cycle x4, then Q6 weeks", "rationale": "Thyroid dysfunction"},
                {"test": "LFTs", "frequency": "Every cycle", "rationale": "Immune hepatitis"},
                {"test": "Creatinine", "frequency": "Every cycle", "rationale": "Immune nephritis"},
                {"test": "Glucose", "frequency": "Every cycle", "rationale": "Immune-related diabetes"}
            ],
            "alert_parameters": [
                {"test": "TSH", "threshold": "<0.1 or >10 mIU/L", "action": "Endocrine consult, consider levothyroxine or beta blockers"},
                {"test": "ALT", "threshold": ">3x ULN", "action": "Hold pembrolizumab, start prednisone 1-2mg/kg"},
                {"test": "Creatinine", "threshold": ">1.5x baseline", "action": "Evaluate for nephritis, consider biopsy and steroids"}
            ]
        },
        "carboplatin_pemetrexed_pembrolizumab": {
            "regimen": "Carboplatin AUC5 + Pemetrexed 500mg/m² + Pembrolizumab 200mg Q3W x4, then Pem + Pembro maintenance",
            "indication": "Non-squamous NSCLC",
            "baseline_labs": ["CBC", "CMP", "LFTs", "TSH", "fT4", "B12", "Folate"],
            "monitoring": [
                {"test": "CBC", "frequency": "Day 1 of each cycle, Day 8-10 if symptomatic", "rationale": "Myelosuppression"},
                {"test": "Creatinine", "frequency": "Before each cycle", "rationale": "Carboplatin dose calculation, pemetrexed clearance"},
                {"test": "LFTs", "frequency": "Every cycle", "rationale": "Hepatotoxicity"},
                {"test": "TSH, fT4", "frequency": "Q6 weeks", "rationale": "Thyroid dysfunction from pembrolizumab"},
                {"test": "Platelets", "frequency": "Weekly during induction", "rationale": "Carboplatin thrombocytopenia"}
            ],
            "alert_parameters": [
                {"test": "ANC", "threshold": "<1.5 x10^9/L", "action": "Delay cycle until recovery. Consider G-CSF"},
                {"test": "Platelets", "threshold": "<100 x10^9/L", "action": "Delay cycle. If <75, reduce carboplatin to AUC 4"},
                {"test": "CrCl", "threshold": "<45 mL/min", "action": "Hold pemetrexed (contraindicated)"}
            ]
        },
        "alectinib_monotherapy": {
            "regimen": "Alectinib 600mg BID",
            "indication": "ALK+ NSCLC",
            "baseline_labs": ["CBC", "CMP", "LFTs", "CPK"],
            "monitoring": [
                {"test": "LFTs", "frequency": "Q2 weeks x2 months, then monthly", "rationale": "Hepatotoxicity"},
                {"test": "CPK", "frequency": "Q2 weeks x1 month, then monthly if elevated", "rationale": "Myopathy"},
                {"test": "CBC", "frequency": "Monthly", "rationale": "Anemia, cytopenias"}
            ],
            "alert_parameters": [
                {"test": "ALT", "threshold": ">5x ULN", "action": "Withhold until ≤3x ULN, resume at reduced dose (450mg BID)"},
                {"test": "CPK", "threshold": ">5x ULN with symptoms", "action": "Withhold until ≤2.5x ULN or baseline, resume at 450mg BID"}
            ]
        },
        "cisplatin_etoposide": {
            "regimen": "Cisplatin 75mg/m² D1 + Etoposide 100mg/m² D1-3 Q3W",
            "indication": "SCLC",
            "baseline_labs": ["CBC", "CMP", "LFTs", "Mg", "Audiogram (if prior ototoxic exposure)"],
            "monitoring": [
                {"test": "CBC", "frequency": "Weekly x3 per cycle", "rationale": "Severe myelosuppression expected"},
                {"test": "Creatinine, electrolytes", "frequency": "Pre-each cycle, Day 2", "rationale": "Nephrotoxicity, electrolyte wasting"},
                {"test": "Mg, K", "frequency": "Each cycle, Day 2, weekly PRN", "rationale": "Renal wasting common with cisplatin"},
                {"test": "CrCl", "frequency": "Pre-each cycle", "rationale": "For cisplatin dosing"}
            ],
            "alert_parameters": [
                {"test": "CrCl", "threshold": "<60 mL/min", "action": "Switch to carboplatin"},
                {"test": "ANC", "threshold": "<0.5 x10^9/L on Day 1", "action": "Delay 1 week. Consider dose reduction"},
                {"test": "Mg", "threshold": "<1.5 mg/dL", "action": "IV magnesium replacement before cisplatin"}
            ]
        },
        "lorlatinib_monotherapy": {
            "regimen": "Lorlatinib 100mg daily",
            "indication": "ALK+ NSCLC (post-prior ALK TKI)",
            "baseline_labs": ["CBC", "CMP", "LFTs", "Lipid panel"],
            "monitoring": [
                {"test": "Lipid panel", "frequency": "Q2 weeks x2 months, then monthly", "rationale": "Hyperlipidemia very common"},
                {"test": "LFTs", "frequency": "Monthly", "rationale": "Hepatotoxicity"},
                {"test": "CNS assessment", "frequency": "Each visit", "rationale": "Cognitive/mood effects"}
            ],
            "alert_parameters": [
                {"test": "LDL", "threshold": ">190 mg/dL", "action": "Start statin therapy"},
                {"test": "Triglycerides", "threshold": ">500 mg/dL", "action": "Consider fibrate, lifestyle modification"},
                {"test": "ALT", "threshold": ">3x ULN", "action": "Withhold, resume at 75mg when ≤2.5x ULN"}
            ]
        }
    }
    
    # Dose adjustment guidelines based on lab values
    DOSE_ADJUSTMENTS = {
        "osimertinib": {
            "hepatic": [
                {"condition": "ALT or AST >3x ULN but ≤5x ULN", "recommendation": "Withhold until ≤2.5x ULN, resume at same dose (80mg)"},
                {"condition": "ALT or AST >5x ULN", "recommendation": "Permanently discontinue"},
                {"condition": "Bilirubin >2x ULN", "recommendation": "Withhold, resume at 40mg when ≤1.5x ULN"}
            ],
            "renal": [
                {"condition": "CrCl 15-29 mL/min", "recommendation": "No adjustment needed (limited data)"},
                {"condition": "CrCl <15 mL/min", "recommendation": "Not studied, use caution"}
            ]
        },
        "carboplatin": {
            "renal": [
                {"condition": "CrCl >60 mL/min", "recommendation": "Standard AUC dosing (Calvert formula)"},
                {"condition": "CrCl 45-59 mL/min", "recommendation": "Reduce target AUC by 1"},
                {"condition": "CrCl 30-44 mL/min", "recommendation": "Consider reduced AUC, close monitoring"},
                {"condition": "CrCl <30 mL/min", "recommendation": "Not recommended; consider alternative"}
            ],
            "hematologic": [
                {"condition": "ANC <1.5 x10^9/L on Day 1", "recommendation": "Delay until recovery"},
                {"condition": "Platelets 75-99 x10^9/L on Day 1", "recommendation": "Reduce AUC by 1"},
                {"condition": "Platelets <75 x10^9/L on Day 1", "recommendation": "Delay until ≥100; reduce AUC"},
                {"condition": "Prior Grade 4 thrombocytopenia", "recommendation": "Reduce AUC by 25%"}
            ]
        },
        "pemetrexed": {
            "renal": [
                {"condition": "CrCl ≥45 mL/min", "recommendation": "Full dose 500mg/m²"},
                {"condition": "CrCl <45 mL/min", "recommendation": "Do not use (contraindicated)"}
            ],
            "hematologic": [
                {"condition": "ANC nadir <0.5 x10^9/L and platelet nadir ≥50 x10^9/L", "recommendation": "Reduce to 75% dose"},
                {"condition": "Platelet nadir <50 x10^9/L regardless of ANC nadir", "recommendation": "Reduce to 75% dose"},
                {"condition": "Grade 3-4 mucositis", "recommendation": "Reduce to 50% dose"}
            ]
        },
        "pembrolizumab": {
            "hepatic": [
                {"condition": "ALT 3-5x ULN", "recommendation": "Hold until ≤1.5x ULN; may rechallenge"},
                {"condition": "ALT >5x ULN or with elevated bili", "recommendation": "Permanently discontinue"},
                {"condition": "Any grade hepatitis with concurrent transaminase rise", "recommendation": "Start prednisone 1-2mg/kg, hold pembrolizumab"}
            ],
            "endocrine": [
                {"condition": "TSH >10 mIU/L (hypothyroidism)", "recommendation": "Start levothyroxine, continue pembrolizumab"},
                {"condition": "TSH <0.1 with high fT4 (hyperthyroidism)", "recommendation": "Consider beta blocker, follow to hypothyroid phase"}
            ]
        },
        "alectinib": {
            "hepatic": [
                {"condition": "ALT or AST >5x ULN with bilirubin ≤2x ULN", "recommendation": "Withhold until ≤3x ULN, resume at 450mg BID"},
                {"condition": "ALT or AST >3x ULN with bilirubin >2x ULN (no cholestasis)", "recommendation": "Permanently discontinue"}
            ],
            "CPK_elevation": [
                {"condition": "CPK >5x ULN", "recommendation": "Withhold until ≤2.5x ULN, resume at 450mg BID"},
                {"condition": "CPK >10x ULN or rhabdomyolysis", "recommendation": "Permanently discontinue"}
            ]
        },
        "lorlatinib": {
            "hepatic": [
                {"condition": "ALT or AST >3x ULN", "recommendation": "Withhold until ≤1.5x ULN, resume at 75mg"},
                {"condition": "Recurrent ALT or AST >3x ULN", "recommendation": "Permanently discontinue"}
            ],
            "lipid": [
                {"condition": "Grade 3-4 hypercholesterolemia/hypertriglyceridemia", "recommendation": "Initiate or increase lipid-lowering therapy; no dose adjustment needed"}
            ]
        }
    }
    
    def __init__(self, loinc_service: Optional[LOINCService] = None, 
                 rxnorm_service: Optional[RXNORMService] = None):
        """Initialize the lab-drug integration service"""
        self.loinc_service = loinc_service or get_loinc_service()
        self.rxnorm_service = rxnorm_service or get_rxnorm_service()
    
    def check_drug_lab_effects(
        self,
        drug: str,
        current_labs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check expected lab effects for a drug.
        
        Args:
            drug: Drug name
            current_labs: Optional dict of current lab values for context
            
        Returns:
            Expected lab effects and monitoring recommendations
        """
        drug_lower = drug.lower()
        
        effects = self.DRUG_LAB_EFFECTS.get(drug_lower, [])
        
        if not effects:
            return {
                "status": "not_found",
                "drug": drug,
                "message": f"No lab effect data available for '{drug}'"
            }
        
        effect_list = []
        for effect in effects:
            effect_entry = {
                "loinc_code": effect["loinc"],
                "test_name": effect["test"],
                "direction": effect["direction"],
                "mechanism": effect["mechanism"],
                "clinical_significance": effect["significance"],
                "expected_magnitude": effect.get("magnitude"),
                "time_course": effect.get("time_course")
            }
            
            # If current labs provided, check current values
            if current_labs and effect["test"] in current_labs:
                current_value = current_labs[effect["test"]]
                effect_entry["current_value"] = current_value
                
                # Use LOINC service to interpret
                interpretation = self.loinc_service.interpret_lab_result(
                    loinc_code=effect["loinc"],
                    value=current_value
                )
                effect_entry["current_interpretation"] = interpretation.get("status")
            
            effect_list.append(effect_entry)
        
        # Get monitoring recommendations
        monitoring_tests = list(set(e["test"] for e in effects))
        
        return {
            "status": "success",
            "drug": drug,
            "total_effects": len(effect_list),
            "effects": effect_list,
            "recommended_monitoring": monitoring_tests,
            "clinical_note": f"{drug.title()} can affect these lab parameters. Monitor accordingly."
        }
    
    def get_monitoring_protocol(
        self,
        regimen: Optional[str] = None,
        drugs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get monitoring protocol for a regimen or drug combination.
        
        Args:
            regimen: Named regimen (e.g., "carboplatin_pemetrexed_pembrolizumab")
            drugs: List of drugs if not using named regimen
            
        Returns:
            Comprehensive monitoring protocol
        """
        # Try to find matching protocol
        if regimen:
            regimen_lower = regimen.lower().replace(" ", "_").replace("+", "_")
            protocol = self.MONITORING_PROTOCOLS.get(regimen_lower)
            
            if protocol:
                return {
                    "status": "success",
                    "regimen": protocol["regimen"],
                    "indication": protocol["indication"],
                    "baseline_labs": protocol["baseline_labs"],
                    "monitoring_schedule": protocol["monitoring"],
                    "alert_parameters": protocol["alert_parameters"],
                    "source": "predefined_protocol"
                }
        
        # Build protocol from individual drugs
        if drugs:
            combined_monitoring = []
            all_baseline = set(["CBC", "CMP"])
            all_alerts = []
            
            for drug in drugs:
                drug_lower = drug.lower()
                
                # Add drug-specific labs
                effects = self.DRUG_LAB_EFFECTS.get(drug_lower, [])
                for effect in effects:
                    test = effect["test"]
                    if test not in [m.get("test") for m in combined_monitoring]:
                        combined_monitoring.append({
                            "test": test,
                            "frequency": "Each cycle or monthly",
                            "rationale": f"{drug.title()}: {effect['mechanism']}"
                        })
                    all_baseline.add(test)
                
                # Add from dose adjustment guidelines
                adjustments = self.DOSE_ADJUSTMENTS.get(drug_lower, {})
                for category, rules in adjustments.items():
                    for rule in rules:
                        all_alerts.append({
                            "drug": drug,
                            "category": category,
                            "condition": rule["condition"],
                            "action": rule["recommendation"]
                        })
            
            return {
                "status": "success",
                "drugs": drugs,
                "baseline_labs": list(all_baseline),
                "monitoring_schedule": combined_monitoring,
                "alert_parameters": all_alerts,
                "source": "generated_from_drugs",
                "note": "Protocol generated from individual drug profiles. Review with clinical team."
            }
        
        # List available protocols
        return {
            "status": "info",
            "message": "No regimen or drugs specified",
            "available_protocols": list(self.MONITORING_PROTOCOLS.keys()),
            "usage": "Provide 'regimen' name or 'drugs' list"
        }
    
    def assess_dose_for_labs(
        self,
        drug: str,
        lab_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess dose adjustments needed based on lab values.
        
        Args:
            drug: Drug name
            lab_results: Dict of lab test names/codes to values
            
        Returns:
            Dose adjustment recommendations
        """
        drug_lower = drug.lower()
        adjustments = self.DOSE_ADJUSTMENTS.get(drug_lower)
        
        if not adjustments:
            return {
                "status": "not_found",
                "drug": drug,
                "message": f"No dose adjustment guidelines available for '{drug}'"
            }
        
        recommendations = []
        warnings = []
        
        # Normalize lab result keys
        lab_lower = {k.lower(): v for k, v in lab_results.items()}
        
        # Check hepatic
        if "hepatic" in adjustments:
            alt = lab_lower.get("alt") or lab_lower.get("sgpt")
            ast = lab_lower.get("ast") or lab_lower.get("sgot")
            bili = lab_lower.get("bilirubin") or lab_lower.get("total bilirubin")
            
            # Simple threshold checks (assuming ULN: ALT/AST = 40, Bili = 1.2)
            if alt and alt > 200:  # >5x ULN
                for rule in adjustments["hepatic"]:
                    if ">5x ULN" in rule["condition"]:
                        recommendations.append({
                            "category": "Hepatic",
                            "trigger": f"ALT = {alt} (>5x ULN)",
                            "recommendation": rule["recommendation"]
                        })
                        warnings.append("⚠️ Significant hepatotoxicity - review immediately")
                        break
            elif alt and alt > 120:  # >3x ULN
                for rule in adjustments["hepatic"]:
                    if ">3x ULN" in rule["condition"] and "≤5x" in rule["condition"]:
                        recommendations.append({
                            "category": "Hepatic",
                            "trigger": f"ALT = {alt} (>3x ULN)",
                            "recommendation": rule["recommendation"]
                        })
                        break
        
        # Check renal
        if "renal" in adjustments:
            crcl = lab_lower.get("crcl") or lab_lower.get("creatinine clearance") or lab_lower.get("gfr")
            cr = lab_lower.get("creatinine") or lab_lower.get("cr")
            
            if crcl:
                for rule in adjustments["renal"]:
                    condition = rule["condition"].lower()
                    if "<45" in condition and crcl < 45:
                        recommendations.append({
                            "category": "Renal",
                            "trigger": f"CrCl = {crcl} mL/min",
                            "recommendation": rule["recommendation"]
                        })
                        warnings.append("⚠️ Renal impairment - dose adjustment required")
                        break
                    elif "30-44" in condition and 30 <= crcl < 45:
                        recommendations.append({
                            "category": "Renal",
                            "trigger": f"CrCl = {crcl} mL/min",
                            "recommendation": rule["recommendation"]
                        })
                        break
        
        # Check hematologic
        if "hematologic" in adjustments:
            anc = lab_lower.get("anc") or lab_lower.get("absolute neutrophil count")
            plt = lab_lower.get("platelets") or lab_lower.get("plt") or lab_lower.get("platelet count")
            
            if anc is not None and anc < 1.5:
                for rule in adjustments["hematologic"]:
                    if "<1.5" in rule["condition"]:
                        recommendations.append({
                            "category": "Hematologic",
                            "trigger": f"ANC = {anc} x10^9/L",
                            "recommendation": rule["recommendation"]
                        })
                        if anc < 0.5:
                            warnings.append("⚠️ Severe neutropenia - hold treatment")
                        break
            
            if plt is not None and plt < 100:
                for rule in adjustments["hematologic"]:
                    if "<75" in rule["condition"] and plt < 75:
                        recommendations.append({
                            "category": "Hematologic",
                            "trigger": f"Platelets = {plt} x10^9/L",
                            "recommendation": rule["recommendation"]
                        })
                        warnings.append("⚠️ Thrombocytopenia - delay treatment")
                        break
                    elif "75-99" in rule["condition"] and 75 <= plt < 100:
                        recommendations.append({
                            "category": "Hematologic",
                            "trigger": f"Platelets = {plt} x10^9/L",
                            "recommendation": rule["recommendation"]
                        })
                        break
        
        # Check endocrine (for immunotherapy)
        if "endocrine" in adjustments:
            tsh = lab_lower.get("tsh")
            if tsh is not None:
                for rule in adjustments["endocrine"]:
                    if ">10" in rule["condition"] and tsh > 10:
                        recommendations.append({
                            "category": "Endocrine",
                            "trigger": f"TSH = {tsh} mIU/L (hypothyroidism)",
                            "recommendation": rule["recommendation"]
                        })
                        break
                    elif "<0.1" in rule["condition"] and tsh < 0.1:
                        recommendations.append({
                            "category": "Endocrine",
                            "trigger": f"TSH = {tsh} mIU/L (hyperthyroidism)",
                            "recommendation": rule["recommendation"]
                        })
                        break
        
        # Check lipid (for lorlatinib)
        if "lipid" in adjustments:
            chol = lab_lower.get("cholesterol") or lab_lower.get("total cholesterol")
            tg = lab_lower.get("triglycerides")
            
            if chol and chol > 300:  # Grade 3
                for rule in adjustments["lipid"]:
                    recommendations.append({
                        "category": "Lipid",
                        "trigger": f"Cholesterol = {chol} mg/dL",
                        "recommendation": rule["recommendation"]
                    })
                    break
        
        # Check CPK (for alectinib)
        if "CPK_elevation" in adjustments:
            cpk = lab_lower.get("cpk") or lab_lower.get("ck") or lab_lower.get("creatine kinase")
            if cpk and cpk > 1000:  # ~5x ULN
                for rule in adjustments["CPK_elevation"]:
                    if ">5x" in rule["condition"] and cpk < 2000:
                        recommendations.append({
                            "category": "CPK",
                            "trigger": f"CPK = {cpk} U/L",
                            "recommendation": rule["recommendation"]
                        })
                        break
                    elif ">10x" in rule["condition"] and cpk > 2000:
                        recommendations.append({
                            "category": "CPK",
                            "trigger": f"CPK = {cpk} U/L",
                            "recommendation": rule["recommendation"]
                        })
                        warnings.append("⚠️ Significant CPK elevation - evaluate for rhabdomyolysis")
                        break
        
        if not recommendations:
            return {
                "status": "success",
                "drug": drug,
                "lab_results": lab_results,
                "recommendations": [],
                "warnings": [],
                "summary": "✅ No dose adjustments indicated based on provided lab values"
            }
        
        return {
            "status": "success",
            "drug": drug,
            "lab_results": lab_results,
            "recommendations": recommendations,
            "warnings": warnings,
            "summary": f"⚠️ {len(recommendations)} dose modification(s) recommended"
        }
    
    def list_available_protocols(self) -> Dict[str, Any]:
        """List all available monitoring protocols"""
        protocols = []
        for key, data in self.MONITORING_PROTOCOLS.items():
            protocols.append({
                "id": key,
                "regimen": data["regimen"],
                "indication": data["indication"],
                "num_monitoring_items": len(data["monitoring"]),
                "num_alerts": len(data["alert_parameters"])
            })
        
        return {
            "status": "success",
            "protocols": protocols,
            "total": len(protocols)
        }


# Singleton instance
_service_instance: Optional[LabDrugService] = None


def get_lab_drug_service() -> LabDrugService:
    """Get or create the LabDrugService singleton"""
    global _service_instance
    if _service_instance is None:
        _service_instance = LabDrugService()
    return _service_instance
