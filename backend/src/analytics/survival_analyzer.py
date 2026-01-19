"""
Survival Analysis Module

Implements Kaplan-Meier survival curves and Cox Proportional Hazards models
for outcome prediction and treatment comparison.

Requires: lifelines library
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
import pandas as pd

# Optional: lifelines for survival analysis
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("lifelines not installed - survival analysis features limited")

from ..db.neo4j_tools import Neo4jReadTools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SurvivalAnalyzer:
    """
    Survival analysis for clinical outcomes.

    Features:
    - Kaplan-Meier survival curves
    - Cox Proportional Hazards regression
    - Log-rank test for group comparisons
    - Hazard ratio calculations
    - Risk stratification
    """

    def __init__(self, neo4j_tools: Optional[Neo4jReadTools] = None):
        """
        Initialize survival analyzer.

        Args:
            neo4j_tools: Neo4j read tools for data retrieval
        """
        self.neo4j_tools = neo4j_tools
        self.lifelines_available = LIFELINES_AVAILABLE

        if not LIFELINES_AVAILABLE:
            logger.warning(
                "Install lifelines for full survival analysis: pip install lifelines"
            )

    # ========================================
    # KAPLAN-MEIER ANALYSIS
    # ========================================

    def kaplan_meier_analysis(
        self,
        treatment: str,
        stage: Optional[str] = None,
        histology: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform Kaplan-Meier survival analysis for a treatment.

        Args:
            treatment: Treatment name
            stage: Optional stage filter
            histology: Optional histology filter

        Returns:
            Survival statistics and curve data
        """
        if not self.lifelines_available:
            return {
                "error": "lifelines not installed",
                "message": "Install with: pip install lifelines"
            }

        # Fetch historical data
        survival_data = self._fetch_survival_data(treatment, stage, histology)

        if not survival_data or len(survival_data) < 5:
            return {
                "error": "Insufficient data",
                "message": f"Need at least 5 patients, found {len(survival_data) if survival_data else 0}"
            }

        # Convert to DataFrame
        df = pd.DataFrame(survival_data)

        # Fit Kaplan-Meier
        kmf = KaplanMeierFitter()

        try:
            kmf.fit(
                durations=df['survival_days'],
                event_observed=df['event'],  # 1=death, 0=censored
                label=f"{treatment}"
            )

            # Extract key statistics
            median_survival = kmf.median_survival_time_

            # Survival probabilities at key timepoints
            timepoints = {
                "6_months": 180,
                "1_year": 365,
                "2_years": 730,
                "5_years": 1825
            }

            survival_probs = {}
            for label, days in timepoints.items():
                if days <= df['survival_days'].max():
                    prob = kmf.survival_function_at_times(days).values[0]
                    survival_probs[label] = float(prob)

            # Get confidence intervals
            ci = kmf.confidence_interval_survival_function_

            result = {
                "treatment": treatment,
                "stage": stage,
                "histology": histology,
                "sample_size": len(df),
                "events": int(df['event'].sum()),
                "censored": int((1 - df['event']).sum()),
                "median_survival_days": float(median_survival) if not pd.isna(median_survival) else None,
                "survival_probabilities": survival_probs,
                "survival_curve": {
                    "timeline": kmf.survival_function_.index.tolist(),
                    "survival": kmf.survival_function_.values.flatten().tolist(),
                    "ci_lower": ci.iloc[:, 0].tolist() if not ci.empty else [],
                    "ci_upper": ci.iloc[:, 1].tolist() if not ci.empty else []
                }
            }

            logger.info(
                f"KM analysis complete: {treatment}, median={median_survival:.0f} days, n={len(df)}"
            )

            return result

        except Exception as e:
            logger.error(f"Kaplan-Meier analysis failed: {e}")
            return {"error": str(e)}

    def compare_survival_curves(
        self,
        treatment1: str,
        treatment2: str,
        stage: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare survival curves between two treatments using log-rank test.

        Args:
            treatment1: First treatment
            treatment2: Second treatment
            stage: Optional stage filter

        Returns:
            Comparison statistics
        """
        if not self.lifelines_available:
            return {"error": "lifelines not installed"}

        # Get data for both treatments
        data1 = self._fetch_survival_data(treatment1, stage=stage)
        data2 = self._fetch_survival_data(treatment2, stage=stage)

        if not data1 or not data2:
            return {"error": "Insufficient data for comparison"}

        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)

        try:
            # Log-rank test
            results = logrank_test(
                df1['survival_days'],
                df2['survival_days'],
                df1['event'],
                df2['event']
            )

            # Fit KM curves for both
            kmf1 = KaplanMeierFitter()
            kmf1.fit(df1['survival_days'], df1['event'], label=treatment1)

            kmf2 = KaplanMeierFitter()
            kmf2.fit(df2['survival_days'], df2['event'], label=treatment2)

            comparison = {
                "treatment1": {
                    "name": treatment1,
                    "n": len(df1),
                    "median_survival": float(kmf1.median_survival_time_)
                },
                "treatment2": {
                    "name": treatment2,
                    "n": len(df2),
                    "median_survival": float(kmf2.median_survival_time_)
                },
                "log_rank_test": {
                    "p_value": float(results.p_value),
                    "test_statistic": float(results.test_statistic),
                    "significant": results.p_value < 0.05,
                    "interpretation": (
                        f"Survival curves are {'significantly' if results.p_value < 0.05 else 'not significantly'} "
                        f"different (p={results.p_value:.4f})"
                    )
                }
            }

            return comparison

        except Exception as e:
            logger.error(f"Survival comparison failed: {e}")
            return {"error": str(e)}

    # ========================================
    # COX PROPORTIONAL HAZARDS
    # ========================================

    def cox_proportional_hazards(
        self,
        patient_data: Dict[str, Any],
        covariates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fit Cox Proportional Hazards model and predict survival.

        Args:
            patient_data: Patient characteristics for prediction
            covariates: List of covariates to include (default: standard set)

        Returns:
            Hazard ratios and survival prediction
        """
        if not self.lifelines_available:
            return {"error": "lifelines not installed"}

        # Default covariates
        if covariates is None:
            covariates = [
                'age_at_diagnosis',
                'stage_numeric',  # I=1, II=2, III=3, IV=4
                'ps',  # Performance status
                'histology_nsclc'  # 1=NSCLC, 0=SCLC
            ]

        # Fetch training data
        training_data = self._fetch_cox_training_data()

        if not training_data or len(training_data) < 20:
            return {
                "error": "Insufficient training data",
                "message": f"Need at least 20 patients, found {len(training_data) if training_data else 0}"
            }

        df_train = pd.DataFrame(training_data)

        try:
            # Fit Cox model
            cph = CoxPHFitter()
            cph.fit(
                df_train,
                duration_col='survival_days',
                event_col='event',
                formula=' + '.join(covariates)
            )

            # Get hazard ratios
            hazard_ratios = cph.hazard_ratios_.to_dict()
            p_values = cph.summary['p'].to_dict()

            # Prepare patient data for prediction
            patient_df = self._prepare_patient_for_cox(patient_data, covariates)

            # Predict survival function
            survival_func = cph.predict_survival_function(patient_df)

            # Extract survival probabilities
            timepoints = [180, 365, 730, 1825]  # 6mo, 1yr, 2yr, 5yr
            survival_probs = {}

            for days in timepoints:
                if days <= survival_func.index.max():
                    prob = survival_func.loc[days].values[0]
                    survival_probs[f"{days}_days"] = float(prob)

            # Median survival
            median_idx = (survival_func.values <= 0.5).argmax()
            median_survival = survival_func.index[median_idx] if median_idx > 0 else None

            result = {
                "model": "Cox Proportional Hazards",
                "covariates": covariates,
                "training_sample_size": len(df_train),
                "hazard_ratios": hazard_ratios,
                "p_values": p_values,
                "significant_predictors": [
                    var for var, p in p_values.items() if p < 0.05
                ],
                "patient_prediction": {
                    "median_survival_days": float(median_survival) if median_survival else None,
                    "survival_probabilities": survival_probs
                },
                "model_performance": {
                    "concordance_index": float(cph.concordance_index_)
                }
            }

            return result

        except Exception as e:
            logger.error(f"Cox regression failed: {e}")
            return {"error": str(e)}

    # ========================================
    # RISK STRATIFICATION
    # ========================================

    def stratify_risk(
        self,
        patient: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Stratify patient into risk groups based on prognostic factors.

        Args:
            patient: Patient data

        Returns:
            Risk stratification
        """
        risk_score = 0
        risk_factors = []

        # Age
        age = patient.get('age_at_diagnosis', 65)
        if age > 70:
            risk_score += 2
            risk_factors.append(f"Age >70 ({age})")
        elif age > 60:
            risk_score += 1
            risk_factors.append(f"Age 60-70 ({age})")

        # Stage (using ClinicalMappings)
        stage = patient.get('tnm_stage', 'IV')
        if ClinicalMappings.is_metastatic(stage):
            risk_score += 3
            risk_factors.append(f"Stage IV disease")
        elif stage in ['IIIB', 'IIIC']:
            risk_score += 2
            risk_factors.append(f"Stage IIIB/IIIC disease")
        elif stage == 'IIIA':
            risk_score += 1
            risk_factors.append(f"Stage IIIA disease")

        # Performance Status
        ps = patient.get('performance_status', 1)
        if ps >= 3:
            risk_score += 3
            risk_factors.append(f"Poor PS ({ps})")
        elif ps == 2:
            risk_score += 2
            risk_factors.append(f"Moderate PS ({ps})")
        elif ps == 1:
            risk_score += 1
            risk_factors.append(f"PS 1")

        # Weight loss
        weight_loss = patient.get('weight_loss_percent', 0)
        if weight_loss > 10:
            risk_score += 2
            risk_factors.append(f"Weight loss >{weight_loss}%")

        # Histology
        histology = patient.get('histology_type', 'Adenocarcinoma')
        if 'SmallCell' in histology:
            risk_score += 1
            risk_factors.append("SCLC histology")

        # Risk group classification
        if risk_score >= 6:
            risk_group = "High"
            prognosis = "Poor"
            median_survival_estimate = "6-9 months"
        elif risk_score >= 3:
            risk_group = "Intermediate"
            prognosis = "Moderate"
            median_survival_estimate = "12-18 months"
        else:
            risk_group = "Low"
            prognosis = "Favorable"
            median_survival_estimate = ">24 months"

        return {
            "risk_group": risk_group,
            "risk_score": risk_score,
            "prognosis": prognosis,
            "estimated_median_survival": median_survival_estimate,
            "risk_factors": risk_factors,
            "recommendation": self._get_risk_recommendation(risk_group)
        }

    def _get_risk_recommendation(self, risk_group: str) -> str:
        """Get treatment recommendation based on risk group"""

        recommendations = {
            "Low": "Consider aggressive treatment with curative intent",
            "Intermediate": "Standard therapy with close monitoring",
            "High": "Consider palliative approach, symptom management priority"
        }

        return recommendations.get(risk_group, "Individualized approach")

    # ========================================
    # DATA FETCHING
    # ========================================

    def _fetch_survival_data(
        self,
        treatment: str,
        stage: Optional[str] = None,
        histology: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch survival data from Neo4j"""

        if not self.neo4j_tools or not self.neo4j_tools.is_available:
            # Return synthetic data for demonstration
            return self._generate_synthetic_survival_data(treatment, 30)

        # Build query with filters
        where_clauses = ["t.type = $treatment"]
        params = {"treatment": treatment}

        if stage:
            where_clauses.append("p.tnm_stage = $stage")
            params["stage"] = stage

        if histology:
            where_clauses.append("p.histology_type = $histology")
            params["histology"] = histology

        query = f"""
        MATCH (p:Patient)-[:RECEIVED_TREATMENT]->(t:TreatmentPlan)
        MATCH (t)-[:HAS_OUTCOME]->(o:Outcome)
        WHERE {' AND '.join(where_clauses)}

        RETURN p.patient_id as patient_id,
               o.survival_days as survival_days,
               CASE WHEN o.status = 'Death' THEN 1 ELSE 0 END as event,
               p.age_at_diagnosis as age,
               p.tnm_stage as stage,
               p.performance_status as ps
        """

        try:
            with self.neo4j_tools.driver.session(database=self.neo4j_tools.database) as session:
                result = session.run(query, **params)

                data = []
                for record in result:
                    data.append({
                        'patient_id': record['patient_id'],
                        'survival_days': record['survival_days'],
                        'event': record['event'],
                        'age': record['age'],
                        'stage': record['stage'],
                        'ps': record['ps']
                    })

                return data

        except Exception as e:
            logger.error(f"Failed to fetch survival data: {e}")
            return []

    def _fetch_cox_training_data(self) -> List[Dict[str, Any]]:
        """Fetch comprehensive training data for Cox model"""

        # Simplified - would fetch from Neo4j in production
        return self._generate_synthetic_survival_data("mixed", 50)

    def _generate_synthetic_survival_data(
        self,
        treatment: str,
        n: int
    ) -> List[Dict[str, Any]]:
        """Generate synthetic survival data for demonstration"""

        np.random.seed(42)

        data = []
        for i in range(n):
            # Synthetic survival times (exponential distribution)
            survival_days = int(np.random.exponential(scale=400))
            event = 1 if np.random.random() < 0.6 else 0  # 60% death rate

            data.append({
                'patient_id': f"PT-{i:04d}",
                'survival_days': survival_days,
                'event': event,
                'age_at_diagnosis': int(np.random.normal(65, 10)),
                'stage_numeric': np.random.choice([1, 2, 3, 4], p=[0.1, 0.2, 0.3, 0.4]),
                'ps': np.random.choice([0, 1, 2, 3], p=[0.2, 0.4, 0.3, 0.1]),
                'histology_nsclc': 1 if np.random.random() < 0.85 else 0
            })

        return data

    def _prepare_patient_for_cox(
        self,
        patient_data: Dict[str, Any],
        covariates: List[str]
    ) -> pd.DataFrame:
        """Prepare patient data for Cox model prediction"""

        # Map patient data to covariates
        patient_row = {}

        if 'age_at_diagnosis' in covariates:
            patient_row['age_at_diagnosis'] = patient_data.get('age_at_diagnosis', 65)

        if 'stage_numeric' in covariates:
            stage = patient_data.get('tnm_stage', 'IV')
            stage_map = {'I': 1, 'IA': 1, 'IB': 1, 'II': 2, 'IIA': 2, 'IIB': 2,
                        'III': 3, 'IIIA': 3, 'IIIB': 3, 'IIIC': 3, 'IV': 4, 'IVA': 4, 'IVB': 4}
            patient_row['stage_numeric'] = stage_map.get(stage, 4)

        if 'ps' in covariates:
            patient_row['ps'] = patient_data.get('performance_status', 1)

        if 'histology_nsclc' in covariates:
            histology = patient_data.get('histology_type', 'Adenocarcinoma')
            patient_row['histology_nsclc'] = 1 if 'NonSmallCell' in histology or 'Adenocarcinoma' in histology else 0

        return pd.DataFrame([patient_row])
