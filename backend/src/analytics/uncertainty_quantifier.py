"""
Uncertainty Quantification for Clinical Decision Support

Quantifies epistemic and aleatoric uncertainty in treatment recommendations
to provide confidence intervals and reliability metrics.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

from ..db.models import TreatmentRecommendation, PatientFact
from ..db.neo4j_tools import Neo4jReadTools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UncertaintyMetrics:
    """Uncertainty metrics for a recommendation"""
    recommendation: str
    confidence_score: float  # 0.0 to 1.0
    epistemic_uncertainty: float  # Uncertainty due to lack of knowledge/data
    aleatoric_uncertainty: float  # Inherent variability in outcomes
    total_uncertainty: float  # Combined uncertainty
    confidence_level: str  # High, Moderate, Low
    sample_size: int  # Number of similar cases
    explanation: str
    confidence_interval: Optional[Tuple[float, float]] = None


class UncertaintyQuantifier:
    """
    Quantifies uncertainty in clinical decision support recommendations.

    Types of uncertainty:
    1. Epistemic (knowledge uncertainty): Due to insufficient data or evidence
    2. Aleatoric (inherent uncertainty): Natural variability in patient outcomes
    """

    def __init__(self, neo4j_tools: Optional[Neo4jReadTools] = None):
        """
        Initialize uncertainty quantifier.

        Args:
            neo4j_tools: Neo4j read tools for historical data lookup
        """
        self.neo4j_tools = neo4j_tools

    # ========================================
    # MAIN QUANTIFICATION INTERFACE
    # ========================================

    def quantify_recommendation_uncertainty(
        self,
        recommendation: TreatmentRecommendation,
        patient: PatientFact,
        similar_patients: Optional[List[Dict[str, Any]]] = None
    ) -> UncertaintyMetrics:
        """
        Quantify uncertainty for a treatment recommendation.

        Args:
            recommendation: Treatment recommendation
            patient: Current patient
            similar_patients: List of similar historical patients (optional)

        Returns:
            UncertaintyMetrics with detailed uncertainty assessment
        """
        # Get similar patients if not provided
        if similar_patients is None and self.neo4j_tools:
            similar_patients = self._get_similar_patients(patient, recommendation.primary_treatment)

        if not similar_patients:
            # No historical data - high epistemic uncertainty
            return UncertaintyMetrics(
                recommendation=recommendation.primary_treatment,
                confidence_score=0.3,
                epistemic_uncertainty=0.9,
                aleatoric_uncertainty=0.5,
                total_uncertainty=0.95,
                confidence_level="Low",
                sample_size=0,
                explanation="No historical data available for similar patients with this treatment"
            )

        # Calculate epistemic uncertainty (sample size based)
        epistemic = self._calculate_epistemic_uncertainty(similar_patients)

        # Calculate aleatoric uncertainty (outcome variability)
        aleatoric = self._calculate_aleatoric_uncertainty(similar_patients, recommendation.primary_treatment)

        # Combine uncertainties
        total_uncertainty = np.sqrt(epistemic**2 + aleatoric**2)

        # Convert to confidence score (inverse of uncertainty)
        confidence_score = 1.0 - total_uncertainty

        # Classify confidence level
        if confidence_score > 0.8:
            confidence_level = "High"
        elif confidence_score > 0.6:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"

        # Generate explanation
        explanation = self._generate_explanation(
            confidence_score,
            epistemic,
            aleatoric,
            len(similar_patients)
        )

        # Calculate confidence interval for success rate
        confidence_interval = self._calculate_confidence_interval(similar_patients, recommendation.primary_treatment)

        return UncertaintyMetrics(
            recommendation=recommendation.primary_treatment,
            confidence_score=confidence_score,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total_uncertainty,
            confidence_level=confidence_level,
            sample_size=len(similar_patients),
            explanation=explanation,
            confidence_interval=confidence_interval
        )

    # ========================================
    # EPISTEMIC UNCERTAINTY
    # ========================================

    def _calculate_epistemic_uncertainty(
        self,
        similar_patients: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate epistemic uncertainty based on sample size.

        More data = less epistemic uncertainty.

        Args:
            similar_patients: List of similar historical patients

        Returns:
            Epistemic uncertainty (0.0 to 1.0)
        """
        n = len(similar_patients)

        if n == 0:
            return 1.0  # Maximum uncertainty

        # Use inverse relationship with sample size
        # Formula: uncertainty = exp(-n/threshold)
        # threshold = 30 (rule of thumb for clinical studies)
        threshold = 30
        epistemic = np.exp(-n / threshold)

        return float(epistemic)

    # ========================================
    # ALEATORIC UNCERTAINTY
    # ========================================

    def _calculate_aleatoric_uncertainty(
        self,
        similar_patients: List[Dict[str, Any]],
        treatment: str
    ) -> float:
        """
        Calculate aleatoric uncertainty based on outcome variability.

        More variable outcomes = higher aleatoric uncertainty.

        Args:
            similar_patients: List of similar patients
            treatment: Treatment name

        Returns:
            Aleatoric uncertainty (0.0 to 1.0)
        """
        # Extract outcomes for this treatment
        outcomes = []
        for patient in similar_patients:
            if patient.get("treatment_received") == treatment:
                outcome = patient.get("outcome")
                if outcome:
                    # Convert outcome to binary (success/failure)
                    success = 1 if outcome in ["Complete Response", "Partial Response"] else 0
                    outcomes.append(success)

        if not outcomes:
            return 0.5  # Default medium uncertainty

        # Calculate variance in outcomes (binomial)
        success_rate = np.mean(outcomes)
        variance = success_rate * (1 - success_rate)

        # Normalize variance to 0-1 scale
        # Maximum variance is 0.25 (at p=0.5)
        aleatoric = variance / 0.25

        return float(aleatoric)

    # ========================================
    # CONFIDENCE INTERVALS
    # ========================================

    def _calculate_confidence_interval(
        self,
        similar_patients: List[Dict[str, Any]],
        treatment: str,
        confidence_level: float = 0.95
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate confidence interval for treatment success rate.

        Args:
            similar_patients: List of similar patients
            treatment: Treatment name
            confidence_level: Confidence level (default 0.95 for 95% CI)

        Returns:
            (lower_bound, upper_bound) or None
        """
        outcomes = []
        for patient in similar_patients:
            if patient.get("treatment_received") == treatment:
                outcome = patient.get("outcome")
                if outcome:
                    success = 1 if outcome in ["Complete Response", "Partial Response"] else 0
                    outcomes.append(success)

        if len(outcomes) < 5:
            return None  # Too few data points for reliable CI

        n = len(outcomes)
        successes = sum(outcomes)
        success_rate = successes / n

        # Wilson score interval (better for small samples than normal approximation)
        z = 1.96 if confidence_level == 0.95 else 2.576  # z-score for 95% or 99% CI

        denominator = 1 + z**2 / n
        center = (success_rate + z**2 / (2*n)) / denominator
        margin = (z / denominator) * np.sqrt(success_rate * (1-success_rate) / n + z**2 / (4*n**2))

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

        return (lower, upper)

    # ========================================
    # EXPLANATION GENERATION
    # ========================================

    def _generate_explanation(
        self,
        confidence_score: float,
        epistemic: float,
        aleatoric: float,
        sample_size: int
    ) -> str:
        """Generate human-readable explanation of uncertainty"""

        explanations = []

        # Confidence level
        if confidence_score > 0.8:
            explanations.append(f"High confidence (score: {confidence_score:.2f})")
        elif confidence_score > 0.6:
            explanations.append(f"Moderate confidence (score: {confidence_score:.2f})")
        else:
            explanations.append(f"Low confidence (score: {confidence_score:.2f})")

        # Sample size
        if sample_size > 50:
            explanations.append(f"based on {sample_size} similar cases (strong evidence)")
        elif sample_size > 20:
            explanations.append(f"based on {sample_size} similar cases (moderate evidence)")
        elif sample_size > 5:
            explanations.append(f"based on {sample_size} similar cases (limited evidence)")
        else:
            explanations.append(f"based on only {sample_size} similar cases (very limited evidence)")

        # Dominant uncertainty type
        if epistemic > aleatoric:
            explanations.append("Limited by insufficient data (epistemic uncertainty)")
        else:
            explanations.append("Limited by inherent outcome variability (aleatoric uncertainty)")

        return ". ".join(explanations) + "."

    # ========================================
    # DATA RETRIEVAL
    # ========================================

    def _get_similar_patients(
        self,
        patient: PatientFact,
        treatment: str
    ) -> List[Dict[str, Any]]:
        """Get similar patients from Neo4j"""

        if not self.neo4j_tools or not self.neo4j_tools.is_available:
            return []

        # Use Neo4j tools to find similar patients
        similar = self.neo4j_tools.find_similar_patients(patient, k=50)

        # Filter for those who received the specific treatment
        filtered = []
        for sim in similar:
            if sim.treatment_received == treatment:
                filtered.append({
                    "patient_id": sim.patient_id,
                    "treatment_received": sim.treatment_received,
                    "outcome": sim.outcome,
                    "survival_days": sim.survival_days,
                    "similarity_score": sim.similarity_score
                })

        return filtered

    # ========================================
    # BATCH PROCESSING
    # ========================================

    def quantify_multiple_recommendations(
        self,
        recommendations: List[TreatmentRecommendation],
        patient: PatientFact
    ) -> List[UncertaintyMetrics]:
        """
        Quantify uncertainty for multiple recommendations.

        Args:
            recommendations: List of recommendations
            patient: Patient data

        Returns:
            List of uncertainty metrics for each recommendation
        """
        metrics = []

        for rec in recommendations:
            metric = self.quantify_recommendation_uncertainty(rec, patient)
            metrics.append(metric)

        # Sort by confidence score (highest first)
        metrics.sort(key=lambda x: x.confidence_score, reverse=True)

        return metrics

    # ========================================
    # REPORTING
    # ========================================

    def generate_uncertainty_report(
        self,
        metrics: UncertaintyMetrics
    ) -> Dict[str, Any]:
        """
        Generate detailed uncertainty report.

        Args:
            metrics: Uncertainty metrics

        Returns:
            Formatted report dictionary
        """
        report = {
            "recommendation": metrics.recommendation,
            "confidence_assessment": {
                "overall_confidence": f"{metrics.confidence_score:.1%}",
                "confidence_level": metrics.confidence_level,
                "sample_size": metrics.sample_size
            },
            "uncertainty_breakdown": {
                "epistemic_uncertainty": {
                    "value": f"{metrics.epistemic_uncertainty:.1%}",
                    "description": "Uncertainty due to limited data/knowledge"
                },
                "aleatoric_uncertainty": {
                    "value": f"{metrics.aleatoric_uncertainty:.1%}",
                    "description": "Inherent variability in patient outcomes"
                },
                "total_uncertainty": f"{metrics.total_uncertainty:.1%}"
            },
            "explanation": metrics.explanation
        }

        # Add confidence interval if available
        if metrics.confidence_interval:
            lower, upper = metrics.confidence_interval
            report["confidence_interval"] = {
                "lower": f"{lower:.1%}",
                "upper": f"{upper:.1%}",
                "interpretation": f"95% CI: success rate between {lower:.1%} and {upper:.1%}"
            }

        # Add recommendations for improving confidence
        report["recommendations_for_improvement"] = self._suggest_improvements(metrics)

        return report

    def _suggest_improvements(
        self,
        metrics: UncertaintyMetrics
    ) -> List[str]:
        """Suggest ways to improve confidence"""

        suggestions = []

        if metrics.sample_size < 10:
            suggestions.append(
                "Consider enrolling patient in clinical trial to contribute to evidence base"
            )

        if metrics.epistemic_uncertainty > 0.5:
            suggestions.append(
                "Collect more comprehensive patient data for better matching with historical cases"
            )

        if metrics.aleatoric_uncertainty > 0.5:
            suggestions.append(
                "High outcome variability - consider personalized factors (biomarkers, comorbidities)"
            )

        if metrics.confidence_score < 0.6:
            suggestions.append(
                "Low confidence - consider multidisciplinary team review before treatment decision"
            )

        return suggestions

    # ========================================
    # VISUALIZATION DATA
    # ========================================

    def get_uncertainty_visualization_data(
        self,
        metrics_list: List[UncertaintyMetrics]
    ) -> Dict[str, Any]:
        """
        Prepare data for uncertainty visualization.

        Args:
            metrics_list: List of uncertainty metrics

        Returns:
            Data formatted for visualization
        """
        viz_data = {
            "recommendations": [],
            "confidence_scores": [],
            "epistemic_uncertainty": [],
            "aleatoric_uncertainty": [],
            "sample_sizes": []
        }

        for m in metrics_list:
            viz_data["recommendations"].append(m.recommendation)
            viz_data["confidence_scores"].append(m.confidence_score)
            viz_data["epistemic_uncertainty"].append(m.epistemic_uncertainty)
            viz_data["aleatoric_uncertainty"].append(m.aleatoric_uncertainty)
            viz_data["sample_sizes"].append(m.sample_size)

        return viz_data
