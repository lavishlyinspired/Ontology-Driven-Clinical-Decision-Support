"""
Advanced Analytics Suite for Lung Cancer Assistant

Provides:
- Uncertainty Quantification (epistemic + aleatoric)
- Survival Analysis (Kaplan-Meier, Cox PH models)
- Counterfactual Reasoning (what-if scenarios)
- Clinical Trial Matching (ClinicalTrials.gov integration)
"""

# Lazy imports to avoid loading heavy dependencies unless needed
UncertaintyQuantifier = None
UncertaintyMetrics = None
SurvivalAnalyzer = None
CounterfactualEngine = None
CounterfactualAnalysis = None
CounterfactualScenario = None
ClinicalTrialMatcher = None
ClinicalTrial = None
TrialMatch = None

# Lazy loaders
def get_uncertainty_quantifier():
    global UncertaintyQuantifier, UncertaintyMetrics
    if UncertaintyQuantifier is None:
        from .uncertainty_quantifier import UncertaintyQuantifier, UncertaintyMetrics
    return UncertaintyQuantifier, UncertaintyMetrics

def get_survival_analyzer():
    global SurvivalAnalyzer
    if SurvivalAnalyzer is None:
        from .survival_analyzer import SurvivalAnalyzer
    return SurvivalAnalyzer

def get_counterfactual_engine():
    global CounterfactualEngine, CounterfactualAnalysis, CounterfactualScenario
    if CounterfactualEngine is None:
        from .counterfactual_engine import CounterfactualEngine, CounterfactualAnalysis, CounterfactualScenario
    return CounterfactualEngine, CounterfactualAnalysis, CounterfactualScenario

def get_clinical_trial_matcher():
    global ClinicalTrialMatcher, ClinicalTrial, TrialMatch
    if ClinicalTrialMatcher is None:
        from .clinical_trial_matcher import ClinicalTrialMatcher, ClinicalTrial, TrialMatch
    return ClinicalTrialMatcher, ClinicalTrial, TrialMatch

__all__ = [
    # Uncertainty Quantification
    'UncertaintyQuantifier',
    'UncertaintyMetrics',
    
    # Survival Analysis
    'SurvivalAnalyzer',
    
    # Counterfactual Reasoning
    'CounterfactualEngine',
    'CounterfactualAnalysis',
    'CounterfactualScenario',
    
    # Clinical Trial Matching
    'ClinicalTrialMatcher',
    'ClinicalTrial',
    'TrialMatch',
]
