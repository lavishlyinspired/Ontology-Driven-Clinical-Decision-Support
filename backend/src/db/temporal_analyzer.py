"""
Temporal Pattern Analysis for Clinical Decision Support

Analyzes disease progression, treatment response, and temporal patterns
in patient data over time.
"""

from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Analyzes temporal patterns in clinical data:
    - Disease progression trajectories
    - Treatment response timelines
    - Outcome prediction based on temporal features
    - Critical intervention windows
    """

    def __init__(self, uri: str = None, user: str = None, password: str = None, database: str = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            self._available = True
            logger.info("✓ Temporal Analyzer initialized")
        except Exception as e:
            logger.warning(f"Neo4j not available: {e}")
            self._available = False
            self.driver = None

    @property
    def is_available(self) -> bool:
        return self._available

    # ========================================
    # DISEASE PROGRESSION ANALYSIS
    # ========================================

    def analyze_disease_progression(self, patient_id: str) -> Dict[str, Any]:
        """
        Track how disease stage and characteristics change over time.

        Args:
            patient_id: Patient identifier

        Returns:
            Timeline of disease progression with key events
        """
        if not self._available:
            return {"patient_id": patient_id, "timeline": []}

        query = """
        MATCH (p:Patient {patient_id: $patient_id})-[:HAS_INFERENCE]->(i:Inference)
        WHERE i.status = 'completed'

        WITH i
        ORDER BY i.created_at

        RETURN i.inference_id as inference_id,
               i.created_at as timestamp,
               i.classification_result as classification,
               i.workflow_duration_ms as processing_time
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, patient_id=patient_id)

                timeline = []
                prev_scenario = None

                for record in result:
                    timestamp = record["timestamp"]
                    classification = record["classification"]

                    # Extract scenario from classification result (stored as string)
                    # You may need to parse this based on your actual storage format
                    current_scenario = self._extract_scenario(classification)

                    event = {
                        "inference_id": record["inference_id"],
                        "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                        "scenario": current_scenario,
                        "processing_time_ms": record["processing_time"]
                    }

                    # Detect progression events
                    if prev_scenario and current_scenario != prev_scenario:
                        event["progression_detected"] = True
                        event["progression_type"] = self._classify_progression(prev_scenario, current_scenario)
                    else:
                        event["progression_detected"] = False

                    timeline.append(event)
                    prev_scenario = current_scenario

                # Calculate progression metrics
                metrics = self._calculate_progression_metrics(timeline)

                return {
                    "patient_id": patient_id,
                    "timeline": timeline,
                    "total_inferences": len(timeline),
                    "progression_events": metrics["progression_count"],
                    "avg_time_between_assessments": metrics["avg_interval"],
                    "current_scenario": timeline[-1]["scenario"] if timeline else None
                }

        except Exception as e:
            logger.error(f"Failed to analyze disease progression: {e}")
            return {"patient_id": patient_id, "error": str(e)}

    def _extract_scenario(self, classification_str: str) -> str:
        """Extract scenario from classification result"""
        # Simple extraction - adjust based on your data format
        if isinstance(classification_str, dict):
            return classification_str.get("scenario", "Unknown")
        return "Unknown"

    def _classify_progression(self, old_scenario: str, new_scenario: str) -> str:
        """Classify type of disease progression"""
        progression_map = {
            ("Early Stage", "Locally Advanced"): "Disease Progression",
            ("Early Stage", "Metastatic"): "Rapid Progression",
            ("Locally Advanced", "Metastatic"): "Disease Progression",
            ("Metastatic", "Early Stage"): "Treatment Response",
            ("Metastatic", "Locally Advanced"): "Treatment Response"
        }

        return progression_map.get((old_scenario, new_scenario), "Status Change")

    def _calculate_progression_metrics(self, timeline: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics about disease progression"""
        progression_count = sum(1 for event in timeline if event.get("progression_detected", False))

        # Calculate average time between assessments
        if len(timeline) > 1:
            timestamps = [datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")) for e in timeline]
            intervals = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]
            avg_interval = np.mean(intervals) if intervals else 0
        else:
            avg_interval = 0

        return {
            "progression_count": progression_count,
            "avg_interval": avg_interval
        }

    # ========================================
    # TREATMENT RESPONSE TIMELINE
    # ========================================

    def analyze_treatment_response(self, patient_id: str) -> Dict[str, Any]:
        """
        Analyze treatment response over time.

        Args:
            patient_id: Patient identifier

        Returns:
            Treatment timeline with response indicators
        """
        if not self._available:
            return {"patient_id": patient_id, "treatments": []}

        query = """
        MATCH (p:Patient {patient_id: $patient_id})-[:RECEIVED_TREATMENT]->(t:TreatmentPlan)
        OPTIONAL MATCH (t)-[:HAS_OUTCOME]->(o:Outcome)

        RETURN t.type as treatment,
               t.start_date as start_date,
               t.end_date as end_date,
               o.status as outcome_status,
               o.response_date as response_date,
               o.survival_days as survival_days
        ORDER BY t.start_date
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, patient_id=patient_id)

                treatments = []
                for record in result:
                    treatment = {
                        "treatment": record["treatment"],
                        "start_date": record["start_date"],
                        "end_date": record["end_date"],
                        "outcome_status": record["outcome_status"],
                        "response_date": record["response_date"],
                        "survival_days": record["survival_days"]
                    }

                    # Calculate treatment duration
                    if record["start_date"] and record["end_date"]:
                        duration = (record["end_date"] - record["start_date"]).days
                        treatment["duration_days"] = duration

                    # Classify response
                    treatment["response_classification"] = self._classify_response(
                        record["outcome_status"]
                    )

                    treatments.append(treatment)

                # Identify treatment patterns
                patterns = self._identify_treatment_patterns(treatments)

                return {
                    "patient_id": patient_id,
                    "treatments": treatments,
                    "treatment_count": len(treatments),
                    "patterns": patterns
                }

        except Exception as e:
            logger.error(f"Failed to analyze treatment response: {e}")
            return {"patient_id": patient_id, "error": str(e)}

    def _classify_response(self, outcome_status: str) -> str:
        """Classify treatment response"""
        if not outcome_status:
            return "Unknown"

        response_map = {
            "Complete Response": "Excellent",
            "Partial Response": "Good",
            "Stable Disease": "Moderate",
            "Progressive Disease": "Poor",
            "Death": "Poor"
        }

        return response_map.get(outcome_status, "Unknown")

    def _identify_treatment_patterns(self, treatments: List[Dict]) -> Dict[str, Any]:
        """Identify patterns in treatment sequences"""
        if not treatments:
            return {}

        patterns = {
            "line_of_therapy": len(treatments),
            "treatments_used": [t["treatment"] for t in treatments],
            "responses": [t.get("response_classification") for t in treatments]
        }

        # Check for treatment escalation
        if len(treatments) > 1:
            patterns["escalation_detected"] = True

        # Check for multiple successful treatments
        success_count = sum(1 for t in treatments if t.get("response_classification") in ["Excellent", "Good"])
        patterns["successful_treatments"] = success_count

        return patterns

    # ========================================
    # TEMPORAL PATTERN MINING
    # ========================================

    def mine_temporal_patterns(
        self,
        cohort_criteria: Dict[str, Any],
        min_support: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Mine frequent temporal patterns in treatment sequences.

        Args:
            cohort_criteria: Criteria for patient cohort
            min_support: Minimum support threshold

        Returns:
            List of frequent temporal patterns
        """
        if not self._available:
            return []

        query = """
        MATCH (p:Patient)-[:RECEIVED_TREATMENT]->(t:TreatmentPlan)
        WHERE p.tnm_stage = $stage AND p.histology_type = $histology

        WITH p, collect(t.type ORDER BY t.start_date) as treatment_sequence

        RETURN treatment_sequence,
               count(*) as frequency
        ORDER BY frequency DESC
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    stage=cohort_criteria.get("tnm_stage"),
                    histology=cohort_criteria.get("histology_type")
                )

                patterns = []
                total_patients = 0

                for record in result:
                    total_patients += record["frequency"]

                # Calculate support for each pattern
                result = session.run(
                    query,
                    stage=cohort_criteria.get("tnm_stage"),
                    histology=cohort_criteria.get("histology_type")
                )

                for record in result:
                    frequency = record["frequency"]
                    support = frequency / total_patients if total_patients > 0 else 0

                    if support >= min_support:
                        patterns.append({
                            "sequence": record["treatment_sequence"],
                            "frequency": frequency,
                            "support": support
                        })

                logger.info(f"Found {len(patterns)} frequent temporal patterns")
                return patterns

        except Exception as e:
            logger.error(f"Failed to mine temporal patterns: {e}")
            return []

    # ========================================
    # CRITICAL INTERVENTION WINDOWS
    # ========================================

    def identify_intervention_windows(
        self,
        patient_id: str,
        lookahead_days: int = 90
    ) -> Dict[str, Any]:
        """
        Identify critical time windows for intervention based on historical patterns.

        Args:
            patient_id: Patient identifier
            lookahead_days: Days to look ahead for predictions

        Returns:
            Intervention window recommendations
        """
        if not self._available:
            return {}

        # Get patient's progression timeline
        progression = self.analyze_disease_progression(patient_id)

        if not progression.get("timeline"):
            return {"recommendation": "Insufficient data for temporal analysis"}

        timeline = progression["timeline"]
        last_assessment = timeline[-1]

        # Calculate time since last assessment
        last_timestamp = datetime.fromisoformat(last_assessment["timestamp"].replace("Z", "+00:00"))
        days_since = (datetime.now() - last_timestamp).days

        # Get similar patients' progression patterns
        similar_patterns = self._get_similar_progression_patterns(patient_id, timeline)

        # Predict next assessment window
        recommendations = {
            "patient_id": patient_id,
            "last_assessment": last_timestamp.isoformat(),
            "days_since_last_assessment": days_since,
            "current_scenario": last_assessment["scenario"],
            "intervention_windows": []
        }

        # Recommend early re-assessment if progression detected
        if any(e.get("progression_detected") for e in timeline[-3:]):
            recommendations["intervention_windows"].append({
                "type": "Early Re-assessment",
                "urgency": "High",
                "recommended_timeframe": "Within 14 days",
                "rationale": "Recent disease progression detected"
            })

        # Recommend routine follow-up
        if days_since > 90:
            recommendations["intervention_windows"].append({
                "type": "Routine Follow-up",
                "urgency": "Medium",
                "recommended_timeframe": "Immediate",
                "rationale": "Overdue for routine assessment (>90 days)"
            })

        # Based on similar patients
        if similar_patterns:
            avg_progression_time = similar_patterns.get("avg_time_to_progression")
            if avg_progression_time and days_since > (avg_progression_time * 0.8):
                recommendations["intervention_windows"].append({
                    "type": "Proactive Monitoring",
                    "urgency": "Medium",
                    "recommended_timeframe": f"Within {int(avg_progression_time - days_since)} days",
                    "rationale": f"Similar patients progressed after {avg_progression_time} days on average"
                })

        return recommendations

    def _get_similar_progression_patterns(
        self,
        patient_id: str,
        timeline: List[Dict]
    ) -> Dict[str, Any]:
        """Get progression patterns from similar patients"""

        if not timeline:
            return {}

        current_scenario = timeline[-1]["scenario"]

        query = """
        MATCH (similar:Patient)-[:HAS_INFERENCE]->(i:Inference)
        WHERE similar.patient_id <> $patient_id
          AND i.classification_result CONTAINS $scenario

        WITH similar, i
        ORDER BY i.created_at

        WITH similar, collect(i) as inferences
        WHERE size(inferences) > 1

        UNWIND range(0, size(inferences)-2) as idx
        WITH similar,
             inferences[idx] as current_inf,
             inferences[idx+1] as next_inf
        WHERE current_inf.classification_result CONTAINS $scenario
          AND next_inf.classification_result <> current_inf.classification_result

        WITH duration.between(current_inf.created_at, next_inf.created_at).days as time_to_progression

        RETURN avg(time_to_progression) as avg_time_to_progression,
               count(*) as sample_size
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, patient_id=patient_id, scenario=current_scenario)
                record = result.single()

                if record:
                    return {
                        "avg_time_to_progression": record["avg_time_to_progression"],
                        "sample_size": record["sample_size"]
                    }

        except Exception as e:
            logger.error(f"Failed to get similar patterns: {e}")

        return {}

    # ========================================
    # OUTCOME PREDICTION
    # ========================================

    def predict_outcome_timeline(
        self,
        patient_id: str,
        treatment: str
    ) -> Dict[str, Any]:
        """
        Predict outcome timeline based on similar patients' temporal patterns.

        Args:
            patient_id: Patient identifier
            treatment: Proposed treatment

        Returns:
            Predicted timeline with confidence intervals
        """
        if not self._available:
            return {}

        query = """
        MATCH (p:Patient {patient_id: $patient_id})
        MATCH (similar:Patient)-[:RECEIVED_TREATMENT]->(t:TreatmentPlan {type: $treatment})
        WHERE similar.tnm_stage = p.tnm_stage
          AND similar.histology_type = p.histology_type
          AND abs(similar.performance_status - p.performance_status) <= 1

        OPTIONAL MATCH (t)-[:HAS_OUTCOME]->(o:Outcome)

        RETURN similar.patient_id as similar_id,
               t.start_date as treatment_start,
               o.response_date as response_date,
               o.survival_days as survival_days,
               duration.between(t.start_date, o.response_date).days as time_to_response
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, patient_id=patient_id, treatment=treatment)

                response_times = []
                survival_times = []

                for record in result:
                    if record["time_to_response"]:
                        response_times.append(record["time_to_response"])
                    if record["survival_days"]:
                        survival_times.append(record["survival_days"])

                if not response_times:
                    return {"message": "Insufficient historical data for prediction"}

                # Calculate statistics
                prediction = {
                    "treatment": treatment,
                    "expected_response_timeline": {
                        "median_days": int(np.median(response_times)),
                        "range": [int(np.percentile(response_times, 25)), int(np.percentile(response_times, 75))],
                        "sample_size": len(response_times)
                    }
                }

                if survival_times:
                    prediction["expected_survival"] = {
                        "median_days": int(np.median(survival_times)),
                        "range": [int(np.percentile(survival_times, 25)), int(np.percentile(survival_times, 75))],
                        "sample_size": len(survival_times)
                    }

                return prediction

        except Exception as e:
            logger.error(f"Failed to predict outcome timeline: {e}")
            return {"error": str(e)}

    # ========================================
    # CLEANUP
    # ========================================

    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
