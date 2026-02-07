"""
RXNORM Integration Service for Medication Management

This service provides comprehensive RXNORM integration for the Lung Cancer Assistant,
enabling drug lookup, interaction checking, and medication management with clinical
context specific to lung cancer treatment.

Author: LCA Development Team
Version: 1.0.0
"""

import os
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class Drug:
    """Structured drug data"""
    rxcui: str
    name: str
    tty: str  # Term type (IN=Ingredient, BN=Brand Name, etc.)
    synonyms: List[str] = field(default_factory=list)
    therapeutic_class: str = ""
    brand_names: List[str] = field(default_factory=list)
    routes: List[str] = field(default_factory=list)
    forms: List[str] = field(default_factory=list)


@dataclass
class DrugInteraction:
    """Drug-drug interaction data"""
    drug1: str
    drug2: str
    severity: str  # mild, moderate, severe, contraindicated
    mechanism: str
    clinical_effect: str
    management: str
    evidence_level: str


class RXNORMService:
    """
    RXNORM Integration Service for Medication Management
    
    Provides:
    - Drug name normalization and lookup
    - Drug-drug interaction checking
    - Therapeutic class lookup
    - Route and form information
    - Lung cancer drug formulary
    """
    
    # Lung Cancer Drug Formulary with detailed information
    LUNG_CANCER_DRUGS = {
        # EGFR Inhibitors
        "osimertinib": {
            "rxcui": "1601380",
            "class": "EGFR TKI",
            "generation": "3rd",
            "brand_names": ["Tagrisso"],
            "indication": "EGFR+ NSCLC (exon 19 del, L858R, T790M)",
            "route": "oral",
            "forms": ["40mg tablet", "80mg tablet"],
            "standard_dose": "80mg once daily",
            "metabolism": "CYP3A4",
            "key_toxicities": ["Diarrhea", "Rash", "Paronychia", "ILD", "QTc prolongation"],
            "monitoring": ["ECG", "LFTs", "Symptoms of ILD"]
        },
        "erlotinib": {
            "rxcui": "340257",
            "class": "EGFR TKI",
            "generation": "1st",
            "brand_names": ["Tarceva"],
            "indication": "EGFR+ NSCLC",
            "route": "oral",
            "forms": ["25mg tablet", "100mg tablet", "150mg tablet"],
            "standard_dose": "150mg once daily",
            "metabolism": "CYP3A4, CYP1A2",
            "key_toxicities": ["Rash", "Diarrhea", "Anorexia", "ILD"],
            "monitoring": ["LFTs", "Symptoms of ILD"]
        },
        "gefitinib": {
            "rxcui": "285018",
            "class": "EGFR TKI",
            "generation": "1st",
            "brand_names": ["Iressa"],
            "indication": "EGFR+ NSCLC",
            "route": "oral",
            "forms": ["250mg tablet"],
            "standard_dose": "250mg once daily",
            "metabolism": "CYP3A4",
            "key_toxicities": ["Diarrhea", "Rash", "ILD", "Hepatotoxicity"],
            "monitoring": ["LFTs"]
        },
        "afatinib": {
            "rxcui": "1430449",
            "class": "EGFR TKI",
            "generation": "2nd",
            "brand_names": ["Gilotrif"],
            "indication": "EGFR+ NSCLC, including exon 20 insertions",
            "route": "oral",
            "forms": ["20mg tablet", "30mg tablet", "40mg tablet"],
            "standard_dose": "40mg once daily",
            "metabolism": "P-glycoprotein",
            "key_toxicities": ["Diarrhea", "Rash", "Stomatitis", "Paronychia"],
            "monitoring": ["LFTs", "Renal function"]
        },
        "amivantamab": {
            "rxcui": "2468322",
            "class": "EGFR-MET bispecific antibody",
            "brand_names": ["Rybrevant"],
            "indication": "EGFR exon 20 insertion NSCLC",
            "route": "IV",
            "forms": ["350mg/7mL vial"],
            "standard_dose": "1050mg (1400mg if ≥80kg) weekly x4, then Q2W",
            "metabolism": "Proteolytic degradation",
            "key_toxicities": ["Infusion reactions", "Rash", "Paronychia", "Hypoalbuminemia"],
            "monitoring": ["Infusion reactions", "Dermatologic toxicity"]
        },
        
        # ALK Inhibitors
        "alectinib": {
            "rxcui": "1732254",
            "class": "ALK TKI",
            "generation": "2nd",
            "brand_names": ["Alecensa"],
            "indication": "ALK+ NSCLC",
            "route": "oral",
            "forms": ["150mg capsule"],
            "standard_dose": "600mg twice daily with food",
            "metabolism": "CYP3A4",
            "key_toxicities": ["Myalgia", "Edema", "Hepatotoxicity", "Bradycardia"],
            "monitoring": ["LFTs", "CPK", "Heart rate"]
        },
        "lorlatinib": {
            "rxcui": "2102230",
            "class": "ALK TKI",
            "generation": "3rd",
            "brand_names": ["Lorbrena"],
            "indication": "ALK+ NSCLC (including CNS disease)",
            "route": "oral",
            "forms": ["25mg tablet", "100mg tablet"],
            "standard_dose": "100mg once daily",
            "metabolism": "CYP3A4, UGT1A4",
            "key_toxicities": ["Hyperlipidemia", "CNS effects", "Edema", "Peripheral neuropathy"],
            "monitoring": ["Lipid panel", "CNS symptoms", "LFTs"]
        },
        "brigatinib": {
            "rxcui": "1920149",
            "class": "ALK TKI",
            "generation": "2nd",
            "brand_names": ["Alunbrig"],
            "indication": "ALK+ NSCLC",
            "route": "oral",
            "forms": ["30mg tablet", "90mg tablet", "180mg tablet"],
            "standard_dose": "90mg daily x7d, then 180mg daily",
            "metabolism": "CYP3A4, CYP2C8",
            "key_toxicities": ["Pulmonary symptoms", "Hypertension", "Bradycardia", "CPK elevation"],
            "monitoring": ["Pulmonary symptoms", "BP", "Heart rate", "LFTs"]
        },
        "crizotinib": {
            "rxcui": "1148495",
            "class": "ALK/ROS1/MET TKI",
            "generation": "1st",
            "brand_names": ["Xalkori"],
            "indication": "ALK+ or ROS1+ NSCLC",
            "route": "oral",
            "forms": ["200mg capsule", "250mg capsule"],
            "standard_dose": "250mg twice daily",
            "metabolism": "CYP3A4",
            "key_toxicities": ["Visual disturbances", "GI toxicity", "Bradycardia", "QTc prolongation"],
            "monitoring": ["ECG", "LFTs", "Vision"]
        },
        
        # PD-1/PD-L1 Inhibitors
        "pembrolizumab": {
            "rxcui": "1657003",
            "class": "PD-1 inhibitor",
            "brand_names": ["Keytruda"],
            "indication": "NSCLC (PD-L1+ or with chemotherapy), SCLC",
            "route": "IV",
            "forms": ["100mg/4mL vial"],
            "standard_dose": "200mg Q3W or 400mg Q6W",
            "metabolism": "Proteolytic degradation",
            "key_toxicities": ["Immune-related AEs (pneumonitis, colitis, hepatitis, thyroiditis)"],
            "monitoring": ["TSH", "LFTs", "Signs of irAEs"]
        },
        "nivolumab": {
            "rxcui": "1545995",
            "class": "PD-1 inhibitor",
            "brand_names": ["Opdivo"],
            "indication": "NSCLC, SCLC (maintenance)",
            "route": "IV",
            "forms": ["40mg/4mL vial", "100mg/10mL vial", "240mg/24mL vial"],
            "standard_dose": "240mg Q2W or 480mg Q4W",
            "metabolism": "Proteolytic degradation",
            "key_toxicities": ["Immune-related AEs"],
            "monitoring": ["TSH", "LFTs", "Signs of irAEs"]
        },
        "atezolizumab": {
            "rxcui": "1792776",
            "class": "PD-L1 inhibitor",
            "brand_names": ["Tecentriq"],
            "indication": "NSCLC, ES-SCLC",
            "route": "IV",
            "forms": ["840mg/14mL vial", "1200mg/20mL vial"],
            "standard_dose": "1200mg Q3W",
            "metabolism": "Proteolytic degradation",
            "key_toxicities": ["Immune-related AEs"],
            "monitoring": ["TSH", "LFTs", "Signs of irAEs"]
        },
        "durvalumab": {
            "rxcui": "1927879",
            "class": "PD-L1 inhibitor",
            "brand_names": ["Imfinzi"],
            "indication": "Unresectable Stage III NSCLC (consolidation), ES-SCLC",
            "route": "IV",
            "forms": ["120mg/2.4mL vial", "500mg/10mL vial"],
            "standard_dose": "10mg/kg Q2W or 1500mg Q4W",
            "metabolism": "Proteolytic degradation",
            "key_toxicities": ["Immune-related AEs", "Pneumonitis"],
            "monitoring": ["TSH", "LFTs", "Pulmonary symptoms"]
        },
        "ipilimumab": {
            "rxcui": "1094833",
            "class": "CTLA-4 inhibitor",
            "brand_names": ["Yervoy"],
            "indication": "NSCLC (with nivolumab)",
            "route": "IV",
            "forms": ["50mg/10mL vial", "200mg/40mL vial"],
            "standard_dose": "1mg/kg Q6W (with nivolumab)",
            "metabolism": "Proteolytic degradation",
            "key_toxicities": ["Immune-related AEs (higher rate than PD-1 alone)", "Colitis"],
            "monitoring": ["LFTs", "TSH", "Signs of colitis"]
        },
        
        # KRAS G12C Inhibitors
        "sotorasib": {
            "rxcui": "2468306",
            "class": "KRAS G12C inhibitor",
            "brand_names": ["Lumakras"],
            "indication": "KRAS G12C+ NSCLC",
            "route": "oral",
            "forms": ["120mg tablet"],
            "standard_dose": "960mg once daily",
            "metabolism": "CYP3A4",
            "key_toxicities": ["Diarrhea", "Musculoskeletal pain", "Hepatotoxicity", "Nausea"],
            "monitoring": ["LFTs"]
        },
        "adagrasib": {
            "rxcui": "2549428",
            "class": "KRAS G12C inhibitor",
            "brand_names": ["Krazati"],
            "indication": "KRAS G12C+ NSCLC",
            "route": "oral",
            "forms": ["200mg tablet"],
            "standard_dose": "600mg twice daily",
            "metabolism": "CYP3A4",
            "key_toxicities": ["Nausea", "Diarrhea", "Fatigue", "QTc prolongation"],
            "monitoring": ["ECG", "LFTs"]
        },
        
        # Other Targeted Therapies
        "entrectinib": {
            "rxcui": "2182575",
            "class": "ROS1/NTRK TKI",
            "brand_names": ["Rozlytrek"],
            "indication": "ROS1+ NSCLC, NTRK+ solid tumors",
            "route": "oral",
            "forms": ["100mg capsule", "200mg capsule"],
            "standard_dose": "600mg once daily",
            "metabolism": "CYP3A4",
            "key_toxicities": ["Cognitive effects", "Weight gain", "Edema", "Dizziness"],
            "monitoring": ["CNS symptoms", "Weight", "LFTs"]
        },
        "selpercatinib": {
            "rxcui": "2371720",
            "class": "RET inhibitor",
            "brand_names": ["Retevmo"],
            "indication": "RET fusion+ NSCLC",
            "route": "oral",
            "forms": ["40mg capsule", "80mg capsule"],
            "standard_dose": "160mg twice daily",
            "metabolism": "CYP3A4",
            "key_toxicities": ["Hepatotoxicity", "Hypertension", "QTc prolongation"],
            "monitoring": ["LFTs", "BP", "ECG"]
        },
        "capmatinib": {
            "rxcui": "2371764",
            "class": "MET inhibitor",
            "brand_names": ["Tabrecta"],
            "indication": "MET exon 14 skipping NSCLC",
            "route": "oral",
            "forms": ["150mg tablet", "200mg tablet"],
            "standard_dose": "400mg twice daily",
            "metabolism": "CYP3A4",
            "key_toxicities": ["Peripheral edema", "Nausea", "Creatinine elevation"],
            "monitoring": ["LFTs", "Renal function"]
        },
        "tepotinib": {
            "rxcui": "2468278",
            "class": "MET inhibitor",
            "brand_names": ["Tepmetko"],
            "indication": "MET exon 14 skipping NSCLC",
            "route": "oral",
            "forms": ["225mg tablet"],
            "standard_dose": "450mg once daily",
            "metabolism": "CYP3A4",
            "key_toxicities": ["Peripheral edema", "Nausea", "Hypoalbuminemia"],
            "monitoring": ["LFTs", "Albumin"]
        },
        "dabrafenib": {
            "rxcui": "1424911",
            "class": "BRAF inhibitor",
            "brand_names": ["Tafinlar"],
            "indication": "BRAF V600E+ NSCLC (with trametinib)",
            "route": "oral",
            "forms": ["50mg capsule", "75mg capsule"],
            "standard_dose": "150mg twice daily (with trametinib)",
            "metabolism": "CYP2C8, CYP3A4",
            "key_toxicities": ["Pyrexia", "Fatigue", "Arthralgias"],
            "monitoring": ["Temperature", "Skin exams", "LFTs"]
        },
        "trametinib": {
            "rxcui": "1424923",
            "class": "MEK inhibitor",
            "brand_names": ["Mekinist"],
            "indication": "BRAF V600E+ NSCLC (with dabrafenib)",
            "route": "oral",
            "forms": ["0.5mg tablet", "2mg tablet"],
            "standard_dose": "2mg once daily (with dabrafenib)",
            "metabolism": "Deacetylation, oxidation",
            "key_toxicities": ["Rash", "Diarrhea", "Cardiomyopathy", "Retinopathy"],
            "monitoring": ["Echo/MUGA", "Ophthalmologic exams", "Skin"]
        },
        
        # Chemotherapy
        "carboplatin": {
            "rxcui": "40048",
            "class": "Platinum",
            "brand_names": ["Paraplatin"],
            "indication": "NSCLC, SCLC",
            "route": "IV",
            "forms": ["50mg vial", "150mg vial", "450mg vial", "600mg vial"],
            "standard_dose": "AUC 5-6 Q3W",
            "metabolism": "Renal elimination",
            "key_toxicities": ["Myelosuppression", "Nausea/vomiting", "Nephrotoxicity (less than cisplatin)"],
            "monitoring": ["CBC", "Renal function", "Electrolytes"]
        },
        "cisplatin": {
            "rxcui": "2555",
            "class": "Platinum",
            "brand_names": ["Platinol"],
            "indication": "NSCLC, SCLC",
            "route": "IV",
            "forms": ["50mg vial", "100mg vial"],
            "standard_dose": "75mg/m² Q3W",
            "metabolism": "Renal elimination",
            "key_toxicities": ["Nephrotoxicity", "Ototoxicity", "Nausea/vomiting", "Peripheral neuropathy"],
            "monitoring": ["CBC", "Renal function", "Hearing", "Electrolytes (Mg, K)"]
        },
        "pemetrexed": {
            "rxcui": "282437",
            "class": "Antifolate",
            "brand_names": ["Alimta"],
            "indication": "Non-squamous NSCLC",
            "route": "IV",
            "forms": ["100mg vial", "500mg vial"],
            "standard_dose": "500mg/m² Q3W",
            "metabolism": "Renal elimination",
            "key_toxicities": ["Myelosuppression", "Fatigue", "Rash"],
            "monitoring": ["CBC", "Renal function"],
            "special_requirements": ["Folic acid supplementation", "Vitamin B12 injection"]
        },
        "docetaxel": {
            "rxcui": "72962",
            "class": "Taxane",
            "brand_names": ["Taxotere"],
            "indication": "NSCLC (2L+)",
            "route": "IV",
            "forms": ["20mg/mL vial", "80mg/4mL vial"],
            "standard_dose": "75mg/m² Q3W",
            "metabolism": "CYP3A4",
            "key_toxicities": ["Myelosuppression", "Peripheral neuropathy", "Fluid retention", "Hypersensitivity"],
            "monitoring": ["CBC", "Premedication required"]
        },
        "paclitaxel": {
            "rxcui": "56946",
            "class": "Taxane",
            "brand_names": ["Taxol"],
            "indication": "NSCLC",
            "route": "IV",
            "forms": ["30mg/5mL vial", "100mg/16.7mL vial"],
            "standard_dose": "200mg/m² Q3W or 80mg/m² weekly",
            "metabolism": "CYP2C8, CYP3A4",
            "key_toxicities": ["Myelosuppression", "Peripheral neuropathy", "Hypersensitivity", "Alopecia"],
            "monitoring": ["CBC", "Premedication required"]
        },
        "etoposide": {
            "rxcui": "4179",
            "class": "Topoisomerase II inhibitor",
            "brand_names": ["VePesid", "Toposar"],
            "indication": "SCLC, NSCLC",
            "route": "IV, oral",
            "forms": ["100mg/5mL vial", "50mg capsule"],
            "standard_dose": "100mg/m² days 1-3 Q3W (IV)",
            "metabolism": "CYP3A4",
            "key_toxicities": ["Myelosuppression", "Alopecia", "Mucositis", "Secondary malignancies"],
            "monitoring": ["CBC"]
        },
        "gemcitabine": {
            "rxcui": "12574",
            "class": "Antimetabolite",
            "brand_names": ["Gemzar"],
            "indication": "NSCLC",
            "route": "IV",
            "forms": ["200mg vial", "1g vial", "2g vial"],
            "standard_dose": "1250mg/m² days 1, 8 Q3W",
            "metabolism": "Intracellular deamination",
            "key_toxicities": ["Myelosuppression", "Hepatotoxicity", "Pulmonary toxicity"],
            "monitoring": ["CBC", "LFTs"]
        },
        "vinorelbine": {
            "rxcui": "11198",
            "class": "Vinca alkaloid",
            "brand_names": ["Navelbine"],
            "indication": "NSCLC",
            "route": "IV, oral",
            "forms": ["10mg/mL vial", "20mg capsule", "30mg capsule"],
            "standard_dose": "30mg/m² weekly",
            "metabolism": "CYP3A4",
            "key_toxicities": ["Myelosuppression", "Peripheral neuropathy", "Constipation"],
            "monitoring": ["CBC"]
        },
        
        # Supportive Care
        "ondansetron": {
            "rxcui": "26225",
            "class": "5-HT3 antagonist antiemetic",
            "brand_names": ["Zofran"],
            "indication": "Chemotherapy-induced nausea/vomiting",
            "route": "IV, oral",
            "forms": ["4mg tablet", "8mg tablet", "4mg/2mL vial"],
            "standard_dose": "8mg pre-chemo, then Q8H PRN",
            "metabolism": "CYP3A4, CYP1A2, CYP2D6",
            "key_toxicities": ["QTc prolongation", "Constipation", "Headache"],
            "monitoring": ["ECG in high-risk patients"]
        },
        "granisetron": {
            "rxcui": "14878",
            "class": "5-HT3 antagonist antiemetic",
            "brand_names": ["Kytril", "Sancuso"],
            "indication": "Chemotherapy-induced nausea/vomiting",
            "route": "IV, oral, transdermal",
            "forms": ["1mg tablet", "1mg/mL vial", "3.1mg/24hr patch"],
            "standard_dose": "1mg pre-chemo or 2mg oral",
            "metabolism": "CYP3A4",
            "key_toxicities": ["Constipation", "Headache", "QTc prolongation"],
            "monitoring": []
        },
        "filgrastim": {
            "rxcui": "24330",
            "class": "G-CSF",
            "brand_names": ["Neupogen", "Zarxio"],
            "indication": "Chemotherapy-induced neutropenia",
            "route": "SC, IV",
            "forms": ["300mcg/0.5mL syringe", "480mcg/0.8mL syringe"],
            "standard_dose": "5mcg/kg daily until ANC recovery",
            "metabolism": "Proteolytic degradation",
            "key_toxicities": ["Bone pain", "Splenic rupture (rare)"],
            "monitoring": ["CBC"]
        },
        "pegfilgrastim": {
            "rxcui": "352052",
            "class": "Pegylated G-CSF",
            "brand_names": ["Neulasta"],
            "indication": "Chemotherapy-induced neutropenia prophylaxis",
            "route": "SC",
            "forms": ["6mg/0.6mL syringe"],
            "standard_dose": "6mg once per chemo cycle (24-72h post)",
            "metabolism": "Proteolytic degradation",
            "key_toxicities": ["Bone pain", "Splenic rupture (rare)"],
            "monitoring": ["CBC"]
        },
        "dexamethasone": {
            "rxcui": "3264",
            "class": "Corticosteroid",
            "brand_names": ["Decadron"],
            "indication": "Antiemetic, brain metastases, irAE management",
            "route": "IV, oral",
            "forms": ["0.5mg tablet", "4mg tablet", "4mg/mL vial"],
            "standard_dose": "8-20mg pre-chemo; variable for other indications",
            "metabolism": "CYP3A4",
            "key_toxicities": ["Hyperglycemia", "Insomnia", "Immunosuppression", "Osteoporosis"],
            "monitoring": ["Blood glucose", "Bone density (long-term)"]
        },
    }
    
    # Drug-Drug Interactions (lung cancer relevant)
    INTERACTIONS = {
        # EGFR TKI interactions
        ("osimertinib", "rifampin"): {
            "severity": "severe",
            "mechanism": "CYP3A4 induction",
            "clinical_effect": "Significantly decreased osimertinib exposure (78%)",
            "management": "Avoid concomitant use. If rifampin necessary, consider alternative to osimertinib",
            "evidence": "FDA label, PKPD studies"
        },
        ("osimertinib", "carbamazepine"): {
            "severity": "severe",
            "mechanism": "CYP3A4 induction",
            "clinical_effect": "Decreased osimertinib exposure",
            "management": "Avoid concomitant use",
            "evidence": "FDA label"
        },
        ("osimertinib", "phenytoin"): {
            "severity": "severe",
            "mechanism": "CYP3A4 induction",
            "clinical_effect": "Decreased osimertinib exposure",
            "management": "Avoid concomitant use",
            "evidence": "FDA label"
        },
        ("osimertinib", "warfarin"): {
            "severity": "moderate",
            "mechanism": "Unknown, possible CYP interaction",
            "clinical_effect": "Altered warfarin effect",
            "management": "Monitor INR more frequently",
            "evidence": "Case reports"
        },
        ("erlotinib", "omeprazole"): {
            "severity": "moderate",
            "mechanism": "Reduced gastric pH decreases erlotinib absorption",
            "clinical_effect": "Decreased erlotinib exposure (46%)",
            "management": "Avoid PPIs if possible. If needed, take erlotinib 10h after PPI or 2h before next PPI dose",
            "evidence": "PKPD studies"
        },
        ("erlotinib", "ciprofloxacin"): {
            "severity": "moderate",
            "mechanism": "CYP1A2 inhibition",
            "clinical_effect": "Increased erlotinib exposure (39%)",
            "management": "Monitor for erlotinib toxicity",
            "evidence": "PKPD studies"
        },
        
        # ALK TKI interactions
        ("crizotinib", "ketoconazole"): {
            "severity": "severe",
            "mechanism": "Strong CYP3A4 inhibition",
            "clinical_effect": "3.2-fold increase in crizotinib exposure",
            "management": "Avoid concomitant use or reduce crizotinib dose to 250mg once daily",
            "evidence": "FDA label"
        },
        ("lorlatinib", "rifampin"): {
            "severity": "severe",
            "mechanism": "Strong CYP3A4 induction",
            "clinical_effect": "85% decrease in lorlatinib exposure",
            "management": "Avoid concomitant use",
            "evidence": "FDA label"
        },
        
        # Immunotherapy interactions
        ("pembrolizumab", "prednisone"): {
            "severity": "moderate",
            "mechanism": "Immunosuppression may reduce efficacy",
            "clinical_effect": "Potentially decreased immunotherapy efficacy with >10mg prednisone equivalent",
            "management": "Use lowest effective dose. OK for short courses (irAE management). Avoid chronic use >10mg/day prior to treatment",
            "evidence": "Retrospective analyses"
        },
        ("nivolumab", "prednisone"): {
            "severity": "moderate",
            "mechanism": "Immunosuppression may reduce efficacy",
            "clinical_effect": "Potentially decreased immunotherapy efficacy with >10mg prednisone equivalent",
            "management": "Use lowest effective dose. OK for irAE management",
            "evidence": "Retrospective analyses"
        },
        
        # Chemotherapy interactions
        ("cisplatin", "aminoglycosides"): {
            "severity": "severe",
            "mechanism": "Additive ototoxicity and nephrotoxicity",
            "clinical_effect": "Increased risk of hearing loss and kidney damage",
            "management": "Avoid combination if possible. If necessary, monitor renal function and hearing closely",
            "evidence": "Multiple studies"
        },
        ("cisplatin", "vancomycin"): {
            "severity": "moderate",
            "mechanism": "Additive nephrotoxicity",
            "clinical_effect": "Increased risk of acute kidney injury",
            "management": "Monitor renal function closely, ensure adequate hydration",
            "evidence": "Clinical experience"
        },
        ("paclitaxel", "ketoconazole"): {
            "severity": "moderate",
            "mechanism": "CYP3A4 and CYP2C8 inhibition",
            "clinical_effect": "Increased paclitaxel exposure",
            "management": "Monitor for increased toxicity",
            "evidence": "PKPD studies"
        },
        ("docetaxel", "ketoconazole"): {
            "severity": "severe",
            "mechanism": "CYP3A4 inhibition",
            "clinical_effect": "50% increase in docetaxel exposure",
            "management": "Avoid concomitant use or reduce docetaxel dose by 50%",
            "evidence": "FDA label"
        },
        
        # QTc interactions
        ("osimertinib", "ondansetron"): {
            "severity": "moderate",
            "mechanism": "Additive QTc prolongation",
            "clinical_effect": "Increased risk of arrhythmia",
            "management": "Use alternative antiemetic (granisetron) if possible. If used together, monitor ECG",
            "evidence": "Pharmacologic effect"
        },
        ("sotorasib", "ondansetron"): {
            "severity": "moderate",
            "mechanism": "Both can prolong QTc",
            "clinical_effect": "Increased risk of arrhythmia",
            "management": "Monitor ECG, use alternative antiemetic if baseline QTc prolonged",
            "evidence": "Pharmacologic effect"
        },
        ("adagrasib", "ondansetron"): {
            "severity": "moderate",
            "mechanism": "Additive QTc prolongation",
            "clinical_effect": "Increased risk of arrhythmia",
            "management": "Monitor ECG, consider alternative antiemetic",
            "evidence": "FDA label"
        },
        
        # Anticoagulant interactions
        ("carboplatin", "warfarin"): {
            "severity": "moderate",
            "mechanism": "Unknown",
            "clinical_effect": "Variable effect on INR",
            "management": "Monitor INR closely during and after chemotherapy",
            "evidence": "Clinical experience"
        },
        ("pemetrexed", "nsaid"): {
            "severity": "severe",
            "mechanism": "Reduced renal clearance of pemetrexed",
            "clinical_effect": "Increased pemetrexed toxicity",
            "management": "Hold short-acting NSAIDs for 2 days before, day of, and 2 days after pemetrexed. Hold long-acting NSAIDs for 5 days before through 2 days after",
            "evidence": "FDA label"
        },
    }
    
    # Therapeutic Classes
    THERAPEUTIC_CLASSES = {
        "EGFR TKI": ["osimertinib", "erlotinib", "gefitinib", "afatinib"],
        "ALK TKI": ["lorlatinib", "alectinib", "brigatinib", "crizotinib"],
        "PD-1 inhibitor": ["pembrolizumab", "nivolumab"],
        "PD-L1 inhibitor": ["atezolizumab", "durvalumab"],
        "CTLA-4 inhibitor": ["ipilimumab"],
        "KRAS G12C inhibitor": ["sotorasib", "adagrasib"],
        "ROS1 TKI": ["entrectinib", "crizotinib"],
        "RET inhibitor": ["selpercatinib"],
        "MET inhibitor": ["capmatinib", "tepotinib"],
        "BRAF inhibitor": ["dabrafenib"],
        "MEK inhibitor": ["trametinib"],
        "Platinum": ["carboplatin", "cisplatin"],
        "Taxane": ["paclitaxel", "docetaxel"],
        "Antifolate": ["pemetrexed"],
        "Antimetabolite": ["gemcitabine"],
        "Vinca alkaloid": ["vinorelbine"],
        "Topoisomerase II inhibitor": ["etoposide"],
        "G-CSF": ["filgrastim", "pegfilgrastim"],
        "5-HT3 antagonist": ["ondansetron", "granisetron"],
        "Corticosteroid": ["dexamethasone"]
    }
    
    def __init__(self, rxnorm_path: Optional[str] = None):
        """
        Initialize the RXNORM service.
        
        Args:
            rxnorm_path: Path to the RXNORM data directory
        """
        from ..config import LCAConfig
        self.rxnorm_path = rxnorm_path or LCAConfig.RXNORM_PATH
        self._rxnorm_cache: Dict[str, Drug] = {}
        self._name_to_rxcui: Dict[str, str] = {}
        self._loaded = False
    
    def _load_rxnorm_data(self):
        """Load RXNORM data from RRF files"""
        if self._loaded:
            return
        
        # First, populate from our curated lung cancer drugs
        for name, data in self.LUNG_CANCER_DRUGS.items():
            rxcui = data["rxcui"]
            self._rxnorm_cache[rxcui] = Drug(
                rxcui=rxcui,
                name=name,
                tty="IN",
                synonyms=data.get("brand_names", []),
                therapeutic_class=data.get("class", ""),
                brand_names=data.get("brand_names", []),
                routes=[data.get("route", "")],
                forms=data.get("forms", [])
            )
            self._name_to_rxcui[name.lower()] = rxcui
            for brand in data.get("brand_names", []):
                self._name_to_rxcui[brand.lower()] = rxcui
        
        # Load from RXNCONSO.RRF if available
        rrf_path = os.path.join(self.rxnorm_path, "rrf", "RXNCONSO.RRF")
        if os.path.exists(rrf_path):
            try:
                with open(rrf_path, 'r', encoding='utf-8') as f:
                    count = 0
                    for line in f:
                        if count > 500000:  # Limit initial load
                            break
                        parts = line.strip().split('|')
                        if len(parts) >= 15:
                            rxcui = parts[0]
                            name = parts[14]
                            tty = parts[12]
                            sab = parts[11]
                            
                            # Only load RXNORM source entries
                            if sab == "RXNORM" and tty in ["IN", "BN", "SCD", "SBD"]:
                                if rxcui not in self._rxnorm_cache:
                                    self._rxnorm_cache[rxcui] = Drug(
                                        rxcui=rxcui,
                                        name=name,
                                        tty=tty
                                    )
                                self._name_to_rxcui[name.lower()] = rxcui
                        count += 1
                
                logger.info(f"Loaded {len(self._rxnorm_cache)} RXNORM concepts")
            except Exception as e:
                logger.error(f"Error loading RXNORM data: {e}")
        
        self._loaded = True
    
    def search_drug(
        self,
        query: str,
        therapeutic_class: Optional[str] = None,
        max_results: int = 20
    ) -> Dict[str, Any]:
        """
        Search for drugs by name or therapeutic class.
        
        Args:
            query: Search term (drug name, brand name, or keyword)
            therapeutic_class: Optional filter by class
            max_results: Maximum results to return
            
        Returns:
            Dict with matching drugs
        """
        self._load_rxnorm_data()
        
        results = []
        query_lower = query.lower()
        
        # First search lung cancer curated drugs
        for name, data in self.LUNG_CANCER_DRUGS.items():
            if therapeutic_class and data.get("class", "").lower() != therapeutic_class.lower():
                continue
            
            if (query_lower in name.lower() or
                any(query_lower in brand.lower() for brand in data.get("brand_names", [])) or
                query_lower in data.get("class", "").lower() or
                query_lower in data.get("indication", "").lower()):
                
                results.append({
                    "rxcui": data["rxcui"],
                    "name": name,
                    "brand_names": data.get("brand_names", []),
                    "class": data.get("class"),
                    "indication": data.get("indication"),
                    "route": data.get("route"),
                    "standard_dose": data.get("standard_dose"),
                    "source": "lung_cancer_formulary"
                })
        
        # Then search RXNORM database
        if len(results) < max_results:
            for rxcui, drug in self._rxnorm_cache.items():
                if rxcui in [r["rxcui"] for r in results]:
                    continue
                
                if query_lower in drug.name.lower():
                    results.append({
                        "rxcui": rxcui,
                        "name": drug.name,
                        "tty": drug.tty,
                        "class": drug.therapeutic_class,
                        "source": "rxnorm_database"
                    })
                
                if len(results) >= max_results:
                    break
        
        return {
            "status": "success",
            "query": query,
            "therapeutic_class_filter": therapeutic_class,
            "results": results[:max_results],
            "total_found": len(results)
        }
    
    def get_drug_details(self, identifier: str) -> Dict[str, Any]:
        """
        Get detailed information for a drug.
        
        Args:
            identifier: RXCUI or drug name
            
        Returns:
            Detailed drug information
        """
        self._load_rxnorm_data()
        
        # Try to find by name first
        drug_name = identifier.lower()
        if drug_name in self._name_to_rxcui:
            rxcui = self._name_to_rxcui[drug_name]
        else:
            rxcui = identifier
        
        # Check lung cancer formulary first
        for name, data in self.LUNG_CANCER_DRUGS.items():
            if data["rxcui"] == rxcui or name.lower() == drug_name:
                return {
                    "status": "success",
                    "rxcui": data["rxcui"],
                    "name": name,
                    "brand_names": data.get("brand_names", []),
                    "therapeutic_class": data.get("class"),
                    "generation": data.get("generation"),
                    "indication": data.get("indication"),
                    "route": data.get("route"),
                    "forms": data.get("forms", []),
                    "standard_dose": data.get("standard_dose"),
                    "metabolism": data.get("metabolism"),
                    "key_toxicities": data.get("key_toxicities", []),
                    "monitoring": data.get("monitoring", []),
                    "special_requirements": data.get("special_requirements", []),
                    "lung_cancer_drug": True,
                    "source": "lung_cancer_formulary"
                }
        
        # Check RXNORM cache
        if rxcui in self._rxnorm_cache:
            drug = self._rxnorm_cache[rxcui]
            return {
                "status": "success",
                "rxcui": rxcui,
                "name": drug.name,
                "tty": drug.tty,
                "therapeutic_class": drug.therapeutic_class,
                "brand_names": drug.brand_names,
                "routes": drug.routes,
                "forms": drug.forms,
                "lung_cancer_drug": False,
                "source": "rxnorm_database"
            }
        
        return {
            "status": "not_found",
            "identifier": identifier,
            "message": f"Drug '{identifier}' not found"
        }
    
    def check_drug_interactions(
        self,
        drugs: List[str],
        include_all_severities: bool = True
    ) -> Dict[str, Any]:
        """
        Check for drug-drug interactions.
        
        Args:
            drugs: List of drug names to check
            include_all_severities: Include mild/moderate interactions
            
        Returns:
            Interaction analysis
        """
        self._load_rxnorm_data()
        
        interactions_found = []
        drugs_normalized = [d.lower() for d in drugs]
        
        # Check all pairs
        for i, drug1 in enumerate(drugs_normalized):
            for drug2 in drugs_normalized[i+1:]:
                # Check both orderings
                key1 = (drug1, drug2)
                key2 = (drug2, drug1)
                
                interaction = None
                if key1 in self.INTERACTIONS:
                    interaction = self.INTERACTIONS[key1]
                elif key2 in self.INTERACTIONS:
                    interaction = self.INTERACTIONS[key2]
                
                if interaction:
                    if include_all_severities or interaction["severity"] in ["severe", "contraindicated"]:
                        interactions_found.append({
                            "drug1": drug1,
                            "drug2": drug2,
                            "severity": interaction["severity"],
                            "mechanism": interaction["mechanism"],
                            "clinical_effect": interaction["clinical_effect"],
                            "management": interaction["management"],
                            "evidence": interaction.get("evidence", "")
                        })
        
        # Sort by severity
        severity_order = {"contraindicated": 0, "severe": 1, "moderate": 2, "mild": 3}
        interactions_found.sort(key=lambda x: severity_order.get(x["severity"], 4))
        
        # Summary
        severe_count = sum(1 for i in interactions_found if i["severity"] in ["severe", "contraindicated"])
        moderate_count = sum(1 for i in interactions_found if i["severity"] == "moderate")
        
        summary = []
        if severe_count > 0:
            summary.append(f"⚠️ {severe_count} SEVERE interaction(s) found - review required")
        if moderate_count > 0:
            summary.append(f"⚡ {moderate_count} MODERATE interaction(s) - monitor recommended")
        if not interactions_found:
            summary.append("✅ No significant interactions identified")
        
        return {
            "status": "success",
            "drugs_checked": drugs,
            "summary": summary,
            "severe_count": severe_count,
            "moderate_count": moderate_count,
            "interactions": interactions_found,
            "total_interactions": len(interactions_found)
        }
    
    def get_therapeutic_alternatives(self, drug: str) -> Dict[str, Any]:
        """
        Get therapeutic alternatives for a drug.
        
        Args:
            drug: Drug name to find alternatives for
            
        Returns:
            List of therapeutic alternatives
        """
        self._load_rxnorm_data()
        
        drug_lower = drug.lower()
        
        # Find the drug's class
        drug_class = None
        for name, data in self.LUNG_CANCER_DRUGS.items():
            if name.lower() == drug_lower:
                drug_class = data.get("class")
                break
        
        if not drug_class:
            return {
                "status": "not_found",
                "drug": drug,
                "message": f"Drug '{drug}' not found in lung cancer formulary"
            }
        
        # Find alternatives in same class
        alternatives = []
        for name, data in self.LUNG_CANCER_DRUGS.items():
            if data.get("class") == drug_class and name.lower() != drug_lower:
                alternatives.append({
                    "name": name,
                    "rxcui": data["rxcui"],
                    "brand_names": data.get("brand_names", []),
                    "indication": data.get("indication"),
                    "generation": data.get("generation"),
                    "route": data.get("route"),
                    "standard_dose": data.get("standard_dose")
                })
        
        return {
            "status": "success",
            "drug": drug,
            "therapeutic_class": drug_class,
            "alternatives": alternatives,
            "total_alternatives": len(alternatives)
        }
    
    def get_lung_cancer_formulary(
        self,
        therapeutic_class: Optional[str] = None,
        route: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get the complete lung cancer drug formulary.
        
        Args:
            therapeutic_class: Optional filter by class
            route: Optional filter by route (oral, IV)
            
        Returns:
            Filtered formulary
        """
        drugs = []
        
        for name, data in self.LUNG_CANCER_DRUGS.items():
            if therapeutic_class and data.get("class", "").lower() != therapeutic_class.lower():
                continue
            if route and data.get("route", "").lower() != route.lower():
                continue
            
            drugs.append({
                "name": name,
                "rxcui": data["rxcui"],
                "brand_names": data.get("brand_names", []),
                "class": data.get("class"),
                "indication": data.get("indication"),
                "route": data.get("route"),
                "standard_dose": data.get("standard_dose")
            })
        
        # Group by class
        by_class = {}
        for drug in drugs:
            cls = drug.get("class", "Other")
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(drug)
        
        return {
            "status": "success",
            "filters": {"therapeutic_class": therapeutic_class, "route": route},
            "drugs": drugs,
            "by_class": by_class,
            "total_drugs": len(drugs),
            "classes": list(by_class.keys())
        }
    
    def list_therapeutic_classes(self) -> Dict[str, Any]:
        """List all therapeutic classes in the formulary"""
        classes = []
        for cls, drugs in self.THERAPEUTIC_CLASSES.items():
            classes.append({
                "class": cls,
                "drugs": drugs,
                "count": len(drugs)
            })
        
        return {
            "status": "success",
            "classes": classes,
            "total_classes": len(classes)
        }


# Singleton instance
_service_instance: Optional[RXNORMService] = None


def get_rxnorm_service(rxnorm_path: Optional[str] = None) -> RXNORMService:
    """Get or create the RXNORMService singleton"""
    global _service_instance
    if _service_instance is None:
        _service_instance = RXNORMService(rxnorm_path)
    return _service_instance
