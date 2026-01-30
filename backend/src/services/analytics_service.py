"""
Enhanced Analytics Service

Integrates advanced analytics (survival analysis, uncertainty, clinical trials)
with beautiful visualizations and actionable insights.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from pydantic import BaseModel

# Centralized logging
from ..logging_config import get_logger, log_execution

logger = get_logger(__name__)


class SurvivalPrediction(BaseModel):
    """Survival prediction with confidence intervals."""
    time_points: List[int]  # months
    survival_probabilities: List[float]
    lower_ci: List[float]  # 95% CI
    upper_ci: List[float]
    median_survival_months: float
    one_year_survival: float
    two_year_survival: float
    five_year_survival: float


class UncertaintyAnalysis(BaseModel):
    """Uncertainty quantification for predictions."""
    overall_confidence: float
    agent_uncertainties: Dict[str, float]
    data_quality_score: float
    evidence_completeness: float
    
    # Confidence intervals for key predictions
    treatment_confidence: float
    survival_confidence: float
    biomarker_confidence: float
    
    # Sources of uncertainty
    uncertainty_sources: List[Dict[str, Any]]
    recommendations: List[str]


class ClinicalTrialMatch(BaseModel):
    """Clinical trial matching result."""
    trial_id: str
    nct_number: str
    title: str
    phase: str
    status: str
    
    match_score: float
    eligibility_criteria: Dict[str, bool]
    match_reasons: List[str]
    
    location: str
    contact: Optional[str] = None
    url: str


class TreatmentComparison(BaseModel):
    """Comparison of treatment options."""
    treatments: List[str]
    metrics: Dict[str, List[float]]  # metric -> values for each treatment
    
    # Survival comparison
    median_survival: List[float]
    one_year_survival: List[float]
    
    # Quality of life metrics
    toxicity_scores: List[float]
    qol_scores: List[float]
    
    # Recommendations
    recommended_treatment: str
    rationale: str


class AnalyticsService:
    """Service for generating analytics and visualizations."""
    
    def __init__(self):
        """Initialize analytics service."""
        # Integration with specialized analyzers
        from ..analytics.survival_analyzer import SurvivalAnalyzer
        from ..analytics.uncertainty_quantifier import UncertaintyQuantifier
        from ..analytics.clinical_trial_matcher import ClinicalTrialMatcher
        
        self.survival_analyzer = SurvivalAnalyzer()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.trial_matcher = ClinicalTrialMatcher()
    
    def generate_survival_prediction(
        self,
        patient_data: Dict[str, Any],
        treatment_plan: str
    ) -> SurvivalPrediction:
        """
        Generate survival prediction with confidence intervals.
        
        Returns KM curves and survival probabilities at key timepoints.
        """
        # Use survival analyzer
        prediction = self.survival_analyzer.predict_survival(
            age=patient_data.get('age', 65),
            stage=patient_data.get('stage', 'IIIb'),
            histology=patient_data.get('histology', 'adenocarcinoma'),
            performance_status=patient_data.get('ps', 1),
            biomarkers=patient_data.get('biomarkers', {}),
            comorbidities=patient_data.get('comorbidities', []),
            treatment=treatment_plan
        )
        
        return SurvivalPrediction(
            time_points=prediction['time_points'],
            survival_probabilities=prediction['survival_curve'],
            lower_ci=prediction['lower_ci'],
            upper_ci=prediction['upper_ci'],
            median_survival_months=prediction['median_survival'],
            one_year_survival=prediction['survival_1y'],
            two_year_survival=prediction['survival_2y'],
            five_year_survival=prediction['survival_5y']
        )
    
    def analyze_uncertainty(
        self,
        patient_data: Dict[str, Any],
        agent_results: Dict[str, Any]
    ) -> UncertaintyAnalysis:
        """
        Quantify uncertainty in the analysis.
        
        Identifies sources of uncertainty and provides recommendations.
        """
        uncertainty = self.uncertainty_quantifier.quantify_uncertainty(
            patient_data=patient_data,
            predictions=agent_results
        )
        
        return UncertaintyAnalysis(
            overall_confidence=uncertainty['overall_confidence'],
            agent_uncertainties=uncertainty['agent_uncertainties'],
            data_quality_score=uncertainty['data_quality'],
            evidence_completeness=uncertainty['evidence_completeness'],
            treatment_confidence=uncertainty['treatment_confidence'],
            survival_confidence=uncertainty['survival_confidence'],
            biomarker_confidence=uncertainty['biomarker_confidence'],
            uncertainty_sources=uncertainty['sources'],
            recommendations=uncertainty['recommendations']
        )
    
    def find_clinical_trials(
        self,
        patient_data: Dict[str, Any],
        treatment_context: Dict[str, Any]
    ) -> List[ClinicalTrialMatch]:
        """
        Find matching clinical trials for the patient.
        
        Returns ranked list of relevant trials.
        """
        matches = self.trial_matcher.find_matches(
            age=patient_data.get('age', 65),
            stage=patient_data.get('stage', 'IIIb'),
            histology=patient_data.get('histology', 'adenocarcinoma'),
            biomarkers=patient_data.get('biomarkers', {}),
            prior_treatments=patient_data.get('prior_treatments', []),
            location=patient_data.get('location', 'US')
        )
        
        return [
            ClinicalTrialMatch(
                trial_id=match['trial_id'],
                nct_number=match['nct_number'],
                title=match['title'],
                phase=match['phase'],
                status=match['status'],
                match_score=match['match_score'],
                eligibility_criteria=match['eligibility'],
                match_reasons=match['reasons'],
                location=match['location'],
                contact=match.get('contact'),
                url=f"https://clinicaltrials.gov/study/{match['nct_number']}"
            )
            for match in matches
        ]
    
    def compare_treatments(
        self,
        patient_data: Dict[str, Any],
        treatment_options: List[str]
    ) -> TreatmentComparison:
        """
        Compare multiple treatment options.
        
        Provides side-by-side comparison of outcomes, toxicity, QoL.
        """
        # Predict outcomes for each treatment
        comparisons = []
        for treatment in treatment_options:
            survival = self.generate_survival_prediction(patient_data, treatment)
            
            # Estimate toxicity and QoL (simplified)
            toxicity = self._estimate_toxicity(treatment)
            qol = self._estimate_qol(treatment, toxicity)
            
            comparisons.append({
                'treatment': treatment,
                'median_survival': survival.median_survival_months,
                'one_year_survival': survival.one_year_survival,
                'toxicity': toxicity,
                'qol': qol
            })
        
        # Determine best option
        best_idx = max(range(len(comparisons)), 
                       key=lambda i: comparisons[i]['median_survival'] * comparisons[i]['qol'])
        
        return TreatmentComparison(
            treatments=treatment_options,
            metrics={
                'median_survival': [c['median_survival'] for c in comparisons],
                'one_year_survival': [c['one_year_survival'] for c in comparisons],
                'toxicity': [c['toxicity'] for c in comparisons],
                'qol': [c['qol'] for c in comparisons]
            },
            median_survival=[c['median_survival'] for c in comparisons],
            one_year_survival=[c['one_year_survival'] for c in comparisons],
            toxicity_scores=[c['toxicity'] for c in comparisons],
            qol_scores=[c['qol'] for c in comparisons],
            recommended_treatment=treatment_options[best_idx],
            rationale=f"Best balance of survival benefit ({comparisons[best_idx]['median_survival']:.1f} months) and quality of life ({comparisons[best_idx]['qol']:.0%})"
        )
    
    def generate_kaplan_meier_chart(
        self,
        survival: SurvivalPrediction,
        title: str = "Survival Prediction"
    ) -> Dict[str, Any]:
        """
        Generate Kaplan-Meier chart data for frontend visualization.
        
        Returns chart configuration for Chart.js or similar.
        """
        return {
            'type': 'line',
            'title': title,
            'data': {
                'labels': [f"{t} months" for t in survival.time_points],
                'datasets': [
                    {
                        'label': 'Survival Probability',
                        'data': survival.survival_probabilities,
                        'borderColor': 'rgb(59, 130, 246)',
                        'backgroundColor': 'rgba(59, 130, 246, 0.1)',
                        'stepped': True,
                        'fill': False
                    },
                    {
                        'label': '95% CI Upper',
                        'data': survival.upper_ci,
                        'borderColor': 'rgba(59, 130, 246, 0.3)',
                        'borderDash': [5, 5],
                        'fill': '+1',
                        'stepped': True
                    },
                    {
                        'label': '95% CI Lower',
                        'data': survival.lower_ci,
                        'borderColor': 'rgba(59, 130, 246, 0.3)',
                        'borderDash': [5, 5],
                        'fill': False,
                        'stepped': True
                    }
                ]
            },
            'options': {
                'responsive': True,
                'scales': {
                    'y': {
                        'title': {'display': True, 'text': 'Survival Probability'},
                        'min': 0,
                        'max': 1,
                        'ticks': {'format': {'style': 'percent'}}
                    },
                    'x': {
                        'title': {'display': True, 'text': 'Time (months)'}
                    }
                },
                'plugins': {
                    'legend': {'display': True, 'position': 'top'},
                    'annotation': {
                        'annotations': {
                            'median': {
                                'type': 'line',
                                'yMin': 0.5,
                                'yMax': 0.5,
                                'borderColor': 'rgba(255, 99, 132, 0.5)',
                                'borderDash': [10, 5],
                                'label': {
                                    'content': f"Median: {survival.median_survival_months:.1f} months",
                                    'enabled': True
                                }
                            }
                        }
                    }
                }
            }
        }
    
    def generate_uncertainty_chart(
        self,
        uncertainty: UncertaintyAnalysis
    ) -> Dict[str, Any]:
        """Generate uncertainty visualization (radar chart)."""
        return {
            'type': 'radar',
            'title': 'Uncertainty Analysis',
            'data': {
                'labels': [
                    'Overall Confidence',
                    'Data Quality',
                    'Evidence Completeness',
                    'Treatment Confidence',
                    'Survival Confidence',
                    'Biomarker Confidence'
                ],
                'datasets': [{
                    'label': 'Confidence Scores',
                    'data': [
                        uncertainty.overall_confidence,
                        uncertainty.data_quality_score,
                        uncertainty.evidence_completeness,
                        uncertainty.treatment_confidence,
                        uncertainty.survival_confidence,
                        uncertainty.biomarker_confidence
                    ],
                    'backgroundColor': 'rgba(34, 197, 94, 0.2)',
                    'borderColor': 'rgb(34, 197, 94)',
                    'pointBackgroundColor': 'rgb(34, 197, 94)'
                }]
            },
            'options': {
                'responsive': True,
                'scales': {
                    'r': {
                        'min': 0,
                        'max': 1,
                        'ticks': {'format': {'style': 'percent'}}
                    }
                }
            }
        }
    
    def generate_treatment_comparison_chart(
        self,
        comparison: TreatmentComparison
    ) -> Dict[str, Any]:
        """Generate treatment comparison chart (grouped bar)."""
        return {
            'type': 'bar',
            'title': 'Treatment Comparison',
            'data': {
                'labels': comparison.treatments,
                'datasets': [
                    {
                        'label': 'Median Survival (months)',
                        'data': comparison.median_survival,
                        'backgroundColor': 'rgba(59, 130, 246, 0.7)',
                        'yAxisID': 'y'
                    },
                    {
                        'label': 'Quality of Life Score',
                        'data': comparison.qol_scores,
                        'backgroundColor': 'rgba(34, 197, 94, 0.7)',
                        'yAxisID': 'y1'
                    }
                ]
            },
            'options': {
                'responsive': True,
                'scales': {
                    'y': {
                        'type': 'linear',
                        'display': True,
                        'position': 'left',
                        'title': {'display': True, 'text': 'Survival (months)'}
                    },
                    'y1': {
                        'type': 'linear',
                        'display': True,
                        'position': 'right',
                        'title': {'display': True, 'text': 'QoL Score'},
                        'min': 0,
                        'max': 1,
                        'grid': {'drawOnChartArea': False}
                    }
                }
            }
        }
    
    def _estimate_toxicity(self, treatment: str) -> float:
        """Estimate treatment toxicity (0-1 scale)."""
        # Simplified toxicity estimates
        toxicity_map = {
            'pembrolizumab': 0.3,
            'carboplatin_pemetrexed': 0.6,
            'carboplatin_paclitaxel': 0.7,
            'cisplatin_etoposide': 0.8,
            'radiation': 0.5,
            'surgery': 0.4
        }
        
        # Default medium toxicity
        return toxicity_map.get(treatment.lower(), 0.5)
    
    def _estimate_qol(self, treatment: str, toxicity: float) -> float:
        """Estimate quality of life impact (0-1 scale)."""
        # QoL inversely related to toxicity, but not linearly
        base_qol = 0.8
        toxicity_impact = toxicity * 0.4
        return max(0.3, base_qol - toxicity_impact)
    
    def generate_comprehensive_analytics(
        self,
        patient_data: Dict[str, Any],
        agent_results: Dict[str, Any],
        treatment_options: List[str]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analytics dashboard.
        
        Combines survival, uncertainty, trials, and treatment comparison.
        """
        # Get recommended treatment
        recommended_treatment = agent_results.get('recommended_treatment', treatment_options[0])
        
        # Generate all analytics
        survival = self.generate_survival_prediction(patient_data, recommended_treatment)
        uncertainty = self.analyze_uncertainty(patient_data, agent_results)
        trials = self.find_clinical_trials(patient_data, {'treatment': recommended_treatment})
        comparison = self.compare_treatments(patient_data, treatment_options)
        
        # Generate charts
        km_chart = self.generate_kaplan_meier_chart(survival)
        uncertainty_chart = self.generate_uncertainty_chart(uncertainty)
        comparison_chart = self.generate_treatment_comparison_chart(comparison)
        
        # Key insights
        insights = self._generate_insights(
            survival, uncertainty, trials, comparison
        )
        
        return {
            'survival': survival.dict(),
            'uncertainty': uncertainty.dict(),
            'clinical_trials': [t.dict() for t in trials],
            'treatment_comparison': comparison.dict(),
            'charts': {
                'kaplan_meier': km_chart,
                'uncertainty': uncertainty_chart,
                'treatment_comparison': comparison_chart
            },
            'insights': insights,
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_insights(
        self,
        survival: SurvivalPrediction,
        uncertainty: UncertaintyAnalysis,
        trials: List[ClinicalTrialMatch],
        comparison: TreatmentComparison
    ) -> List[Dict[str, str]]:
        """Generate actionable insights from analytics."""
        insights = []
        
        # Survival insights
        if survival.median_survival_months > 18:
            insights.append({
                'type': 'positive',
                'category': 'survival',
                'message': f'Favorable prognosis with median survival of {survival.median_survival_months:.1f} months'
            })
        elif survival.median_survival_months < 12:
            insights.append({
                'type': 'warning',
                'category': 'survival',
                'message': f'Aggressive disease with median survival of {survival.median_survival_months:.1f} months. Consider clinical trial enrollment.'
            })
        
        # Uncertainty insights
        if uncertainty.overall_confidence < 0.7:
            insights.append({
                'type': 'warning',
                'category': 'uncertainty',
                'message': f'Moderate confidence ({uncertainty.overall_confidence:.0%}). Consider additional testing or expert consultation.'
            })
        
        # Clinical trial insights
        high_match_trials = [t for t in trials if t.match_score > 0.8]
        if high_match_trials:
            insights.append({
                'type': 'info',
                'category': 'trials',
                'message': f'{len(high_match_trials)} highly relevant clinical trial(s) available'
            })
        
        # Treatment comparison insights
        survival_range = max(comparison.median_survival) - min(comparison.median_survival)
        if survival_range > 6:
            insights.append({
                'type': 'important',
                'category': 'treatment',
                'message': f'Significant survival difference between treatments (up to {survival_range:.1f} months). Treatment choice is critical.'
            })
        
        return insights


# Global analytics service instance
analytics_service = AnalyticsService()
