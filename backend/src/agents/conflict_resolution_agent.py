"""
Conflict Resolution Agent (Agent 4 of 6)
Handles conflicting recommendations and applies evidence hierarchy.

Responsibilities:
- Detect conflicts between recommendations
- Apply evidence hierarchy (Grade A > B > C)
- Consider patient preferences and comorbidities
- Return ranked, deduplicated recommendations

Tools: compare_evidence_levels(), resolve_conflict(), get_conflict_rules()
Data Sources: Evidence hierarchy rules, patient context
NEVER: Direct Neo4j writes
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Centralized logging
from ..logging_config import get_logger, log_agent_action

logger = get_logger(__name__)

from ..db.models import (
    ClassificationResult,
    TreatmentRecommendation,
    EvidenceLevel,
    TreatmentIntent
)


@dataclass
class ConflictReport:
    """Report of detected conflicts and resolutions."""
    conflict_type: str
    original_recommendations: List[str]
    resolution: str
    rationale: str


class ConflictResolutionAgent:
    """
    Agent 4: Conflict Resolution Agent
    Handles conflicting recommendations and applies evidence hierarchy.
    READ-ONLY: Never writes to Neo4j.
    """

    # Evidence level priority (higher number = stronger evidence)
    EVIDENCE_PRIORITY = {
        EvidenceLevel.GRADE_A: 4,
        EvidenceLevel.GRADE_B: 3,
        EvidenceLevel.GRADE_C: 2,
        EvidenceLevel.EXPERT_OPINION: 1,
    }

    # Treatment intent priority (higher = stronger preference)
    INTENT_PRIORITY = {
        TreatmentIntent.CURATIVE: 4,
        TreatmentIntent.PALLIATIVE: 3,
        TreatmentIntent.SUPPORTIVE: 2,
        TreatmentIntent.UNKNOWN: 1,
    }

    # Conflict resolution rules
    CONFLICT_RULES = {
        "surgery_vs_radiotherapy": {
            "condition": "Both surgery and radiotherapy recommended for early stage",
            "resolution": "Surgery preferred if operable; radiotherapy if inoperable",
            "rule": "Check operability status"
        },
        "concurrent_vs_sequential": {
            "condition": "Both concurrent and sequential chemoRT recommended",
            "resolution": "Concurrent preferred if tolerable; sequential if not",
            "rule": "Check performance status and comorbidities"
        },
        "immunotherapy_vs_chemotherapy": {
            "condition": "Multiple first-line systemic options",
            "resolution": "Check PD-L1 and driver mutations first",
            "rule": "Biomarker-guided therapy selection"
        }
    }

    def __init__(self):
        self.name = "ConflictResolutionAgent"
        self.version = "1.0.0"

    def execute(self, classification: ClassificationResult) -> Tuple[ClassificationResult, List[ConflictReport]]:
        """
        Execute conflict resolution: detect and resolve conflicting recommendations.
        
        Args:
            classification: Classification result with recommendations
            
        Returns:
            Tuple of (updated ClassificationResult, list of ConflictReports)
        """
        logger.info(f"[{self.name}] Resolving conflicts for patient {classification.patient_id}...")

        recommendations = classification.recommendations
        conflict_reports = []

        # Step 1: Detect conflicts
        conflicts = self._detect_conflicts(recommendations)

        # Step 2: Resolve each conflict
        for conflict in conflicts:
            resolution = self._resolve_conflict(conflict, recommendations)
            conflict_reports.append(resolution)

        # Step 3: Apply evidence hierarchy ranking
        ranked_recommendations = self._rank_by_evidence(recommendations)

        # Step 4: Remove duplicates and contradictions
        final_recommendations = self._deduplicate(ranked_recommendations)

        # Update classification with resolved recommendations
        updated_classification = ClassificationResult(
            patient_id=classification.patient_id,
            scenario=classification.scenario,
            scenario_confidence=classification.scenario_confidence,
            recommendations=final_recommendations,
            reasoning_chain=classification.reasoning_chain + [
                f"Conflict resolution: {len(conflict_reports)} conflicts detected and resolved"
            ],
            ontology_concepts_matched=classification.ontology_concepts_matched,
            guideline_refs=classification.guideline_refs
        )

        logger.info(f"[{self.name}] ✓ Resolved {len(conflict_reports)} conflicts for {classification.patient_id}")
        return updated_classification, conflict_reports

    def _detect_conflicts(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect conflicts between recommendations."""
        conflicts = []
        
        treatments = [r.get("treatment", "").lower() for r in recommendations]
        
        # Check surgery vs radiotherapy conflict
        has_surgery = any("surgery" in t or "resection" in t for t in treatments)
        has_radiotherapy = any("radiotherapy" in t or "sabr" in t for t in treatments)
        
        if has_surgery and has_radiotherapy:
            conflicts.append({
                "type": "surgery_vs_radiotherapy",
                "recommendations": [
                    r for r in recommendations 
                    if "surgery" in r.get("treatment", "").lower() or 
                       "resection" in r.get("treatment", "").lower() or
                       "radiotherapy" in r.get("treatment", "").lower() or
                       "sabr" in r.get("treatment", "").lower()
                ]
            })

        # Check concurrent vs sequential chemo conflict
        has_concurrent = any("concurrent" in t for t in treatments)
        has_sequential = any("sequential" in t for t in treatments)
        
        if has_concurrent and has_sequential:
            conflicts.append({
                "type": "concurrent_vs_sequential",
                "recommendations": [
                    r for r in recommendations 
                    if "concurrent" in r.get("treatment", "").lower() or 
                       "sequential" in r.get("treatment", "").lower()
                ]
            })

        return conflicts

    def _resolve_conflict(
        self, 
        conflict: Dict[str, Any], 
        all_recommendations: List[Dict[str, Any]]
    ) -> ConflictReport:
        """Resolve a specific conflict."""
        conflict_type = conflict["type"]
        conflict_recs = conflict["recommendations"]
        
        rule = self.CONFLICT_RULES.get(conflict_type, {})
        
        # Sort by evidence level to determine resolution
        sorted_recs = sorted(
            conflict_recs,
            key=lambda r: self.EVIDENCE_PRIORITY.get(EvidenceLevel(r.get("evidence_level", "Grade C")), 0),
            reverse=True
        )
        
        preferred = sorted_recs[0] if sorted_recs else None
        
        return ConflictReport(
            conflict_type=conflict_type,
            original_recommendations=[r.get("treatment", "") for r in conflict_recs],
            resolution=f"Prefer '{preferred.get('treatment', '')}' based on evidence level" if preferred else "No resolution",
            rationale=rule.get("resolution", "Applied evidence hierarchy")
        )

    def _rank_by_evidence(
        self, 
        recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rank recommendations by evidence level."""
        # Sort by evidence level (highest first), then by intent
        sorted_recs = sorted(
            recommendations,
            key=lambda r: (
                self.EVIDENCE_PRIORITY.get(EvidenceLevel(r.get("evidence_level", "Grade C")), 0),
                self.INTENT_PRIORITY.get(TreatmentIntent(r.get("intent", "Unknown")), 0)
            ),
            reverse=True
        )
        
        # Update ranks
        ranked = []
        for i, rec in enumerate(sorted_recs):
            rec_copy = rec.copy()
            rec_copy["rank"] = i + 1
            ranked.append(rec_copy)
        
        return ranked

    def _deduplicate(
        self, 
        recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate or very similar recommendations."""
        seen_treatments = set()
        deduplicated = []
        
        for rec in recommendations:
            # Normalize treatment name for comparison
            normalized = rec.get("treatment", "").lower().strip()
            
            # Check for near-duplicates
            is_duplicate = False
            for seen in seen_treatments:
                if self._is_similar(normalized, seen):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_treatments.add(normalized)
                deduplicated.append(rec)
        
        return deduplicated

    def _is_similar(self, treatment1: str, treatment2: str) -> bool:
        """Check if two treatments are similar enough to be duplicates."""
        # Simple word overlap check
        words1 = set(treatment1.split())
        words2 = set(treatment2.split())
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return (overlap / total) > 0.7 if total > 0 else False

    def compare_evidence_levels(
        self, 
        level1: EvidenceLevel, 
        level2: EvidenceLevel
    ) -> int:
        """
        Compare two evidence levels.
        Returns: positive if level1 > level2, negative if level1 < level2, 0 if equal
        """
        priority1 = self.EVIDENCE_PRIORITY.get(level1, 0)
        priority2 = self.EVIDENCE_PRIORITY.get(level2, 0)
        return priority1 - priority2

    def get_conflict_rules(self) -> Dict[str, Any]:
        """Get all conflict resolution rules."""
        return self.CONFLICT_RULES
