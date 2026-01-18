"""
Agent Negotiation Protocol for Multi-Agent Decision Support

Implements negotiation strategies for resolving conflicts between
multiple specialized agents. Based on 2025 research showing 43%
reduction in system deadlocks with robust negotiation protocols.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

from ..db.models import ClassificationResult, TreatmentRecommendation, EvidenceLevel, TreatmentIntent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NegotiationStrategy(Enum):
    """Strategies for agent negotiation"""
    EVIDENCE_HIERARCHY = "evidence_hierarchy"      # Prioritize by evidence grade
    CONSENSUS_VOTING = "consensus_voting"           # Majority vote
    WEIGHTED_EXPERTISE = "weighted_expertise"       # Weight by agent expertise
    PATIENT_PREFERENCE = "patient_preference"       # Patient-centered
    SAFETY_FIRST = "safety_first"                   # Prioritize safety over efficacy
    HYBRID = "hybrid"                               # Combine multiple strategies


@dataclass
class AgentProposal:
    """Proposal from a specialized agent"""
    agent_id: str
    agent_type: str
    treatment: str
    confidence: float  # 0.0 to 1.0
    evidence_level: str  # Grade A, B, C
    treatment_intent: str  # Curative, Palliative
    rationale: str
    guideline_reference: str
    contraindications: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0  # 0.0 (low risk) to 1.0 (high risk)
    expected_benefit: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class NegotiationResult:
    """Result of negotiation between agents"""
    selected_treatment: str
    selected_agent: str
    negotiation_strategy: str
    confidence: float
    consensus_score: float  # 0.0 (no consensus) to 1.0 (full consensus)
    all_proposals: List[AgentProposal]
    negotiation_rounds: int
    reasoning: str
    alternative_treatments: List[Dict[str, Any]] = field(default_factory=list)
    conflicts_resolved: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class NegotiationProtocol:
    """
    Negotiation protocol for multi-agent decision making.

    Handles conflicts between specialized agents and produces
    consensus recommendations with full auditability.
    """

    def __init__(self, strategy: NegotiationStrategy = NegotiationStrategy.HYBRID):
        self.strategy = strategy
        self.negotiation_history: List[NegotiationResult] = []

    # ========================================
    # MAIN NEGOTIATION INTERFACE
    # ========================================

    def negotiate(
        self,
        proposals: List[AgentProposal],
        patient_context: Optional[Dict[str, Any]] = None
    ) -> NegotiationResult:
        """
        Negotiate between competing agent proposals.

        Args:
            proposals: List of proposals from different agents
            patient_context: Additional patient information for decision making

        Returns:
            NegotiationResult with selected treatment and reasoning
        """
        if not proposals:
            raise ValueError("No proposals to negotiate")

        if len(proposals) == 1:
            # Single proposal - no negotiation needed
            return self._create_single_proposal_result(proposals[0])

        logger.info(f"Negotiating between {len(proposals)} proposals using {self.strategy.value}")

        # Check for conflicts
        conflicts = self._detect_conflicts(proposals)
        logger.info(f"Detected {len(conflicts)} conflicts")

        # Select strategy
        if self.strategy == NegotiationStrategy.EVIDENCE_HIERARCHY:
            result = self._evidence_based_negotiation(proposals, patient_context)
        elif self.strategy == NegotiationStrategy.CONSENSUS_VOTING:
            result = self._consensus_voting(proposals, patient_context)
        elif self.strategy == NegotiationStrategy.WEIGHTED_EXPERTISE:
            result = self._weighted_expertise(proposals, patient_context)
        elif self.strategy == NegotiationStrategy.SAFETY_FIRST:
            result = self._safety_first_negotiation(proposals, patient_context)
        elif self.strategy == NegotiationStrategy.HYBRID:
            result = self._hybrid_negotiation(proposals, patient_context)
        else:
            result = self._evidence_based_negotiation(proposals, patient_context)

        result.conflicts_resolved = len(conflicts)

        # Store in history
        self.negotiation_history.append(result)

        logger.info(f"Negotiation complete: Selected {result.selected_treatment} "
                   f"(consensus: {result.consensus_score:.2f})")

        return result

    # ========================================
    # NEGOTIATION STRATEGIES
    # ========================================

    def _evidence_based_negotiation(
        self,
        proposals: List[AgentProposal],
        patient_context: Optional[Dict[str, Any]]
    ) -> NegotiationResult:
        """
        Prioritize by evidence grade, then confidence, then intent.

        Evidence hierarchy: Grade A > Grade B > Grade C
        Intent hierarchy: Curative > Palliative (for similar stages)
        """
        # Score evidence levels
        evidence_scores = {"Grade A": 3, "Grade B": 2, "Grade C": 1, "Unknown": 0}

        # Score proposals
        scored_proposals = []
        for p in proposals:
            evidence_score = evidence_scores.get(p.evidence_level, 0)

            # Composite score: evidence (60%) + confidence (30%) + intent (10%)
            intent_bonus = 0.1 if p.treatment_intent == "Curative" else 0.0
            score = (evidence_score / 3.0) * 0.6 + p.confidence * 0.3 + intent_bonus

            scored_proposals.append((score, p))

        # Sort by score
        scored_proposals.sort(key=lambda x: x[0], reverse=True)

        top_score, top_proposal = scored_proposals[0]

        # Calculate consensus (how many agents agree with top choice)
        agreement_count = sum(1 for p in proposals if p.treatment == top_proposal.treatment)
        consensus = agreement_count / len(proposals)

        # Build alternatives
        alternatives = [
            {
                "treatment": p.treatment,
                "agent": p.agent_id,
                "score": score,
                "evidence": p.evidence_level
            }
            for score, p in scored_proposals[1:]
        ]

        reasoning = (
            f"Selected based on evidence hierarchy: {top_proposal.evidence_level} evidence, "
            f"{top_proposal.confidence:.2%} confidence. "
            f"Guideline: {top_proposal.guideline_reference}"
        )

        return NegotiationResult(
            selected_treatment=top_proposal.treatment,
            selected_agent=top_proposal.agent_id,
            negotiation_strategy="evidence_hierarchy",
            confidence=top_proposal.confidence,
            consensus_score=consensus,
            all_proposals=proposals,
            negotiation_rounds=1,
            reasoning=reasoning,
            alternative_treatments=alternatives
        )

    def _consensus_voting(
        self,
        proposals: List[AgentProposal],
        patient_context: Optional[Dict[str, Any]]
    ) -> NegotiationResult:
        """
        Majority vote weighted by agent confidence.
        """
        # Count votes for each treatment (weighted by confidence)
        vote_counts: Dict[str, float] = {}
        treatment_proposals: Dict[str, List[AgentProposal]] = {}

        for p in proposals:
            vote_weight = p.confidence
            vote_counts[p.treatment] = vote_counts.get(p.treatment, 0) + vote_weight

            if p.treatment not in treatment_proposals:
                treatment_proposals[p.treatment] = []
            treatment_proposals[p.treatment].append(p)

        # Find winner
        winning_treatment = max(vote_counts, key=vote_counts.get)
        winning_proposals = treatment_proposals[winning_treatment]

        # Select highest confidence proposal for winner
        selected_proposal = max(winning_proposals, key=lambda p: p.confidence)

        # Calculate consensus
        total_votes = sum(vote_counts.values())
        consensus = vote_counts[winning_treatment] / total_votes if total_votes > 0 else 0

        reasoning = (
            f"Selected by consensus voting: {len(winning_proposals)} agents recommended {winning_treatment}, "
            f"total vote weight: {vote_counts[winning_treatment]:.2f}/{total_votes:.2f}"
        )

        alternatives = [
            {"treatment": t, "votes": v, "agents": len(treatment_proposals[t])}
            for t, v in vote_counts.items() if t != winning_treatment
        ]

        return NegotiationResult(
            selected_treatment=winning_treatment,
            selected_agent=selected_proposal.agent_id,
            negotiation_strategy="consensus_voting",
            confidence=selected_proposal.confidence,
            consensus_score=consensus,
            all_proposals=proposals,
            negotiation_rounds=1,
            reasoning=reasoning,
            alternative_treatments=alternatives
        )

    def _weighted_expertise(
        self,
        proposals: List[AgentProposal],
        patient_context: Optional[Dict[str, Any]]
    ) -> NegotiationResult:
        """
        Weight proposals by agent expertise in specific domains.
        """
        # Define agent expertise weights
        expertise_weights = {
            "BiomarkerAgent": 1.5,  # Highest weight for biomarker-driven decisions
            "ClassificationAgent": 1.2,
            "ComorbidityAgent": 1.3,  # High weight for safety considerations
            "NSCLCAgent": 1.2,
            "SCLCAgent": 1.2,
            "GenericAgent": 1.0
        }

        # Score proposals with expertise weighting
        scored_proposals = []
        for p in proposals:
            weight = expertise_weights.get(p.agent_type, 1.0)
            score = p.confidence * weight
            scored_proposals.append((score, p))

        # Sort by weighted score
        scored_proposals.sort(key=lambda x: x[0], reverse=True)

        top_score, top_proposal = scored_proposals[0]

        # Calculate consensus
        agreement_count = sum(1 for p in proposals if p.treatment == top_proposal.treatment)
        consensus = agreement_count / len(proposals)

        reasoning = (
            f"Selected based on expert agent weighting: {top_proposal.agent_type} "
            f"(weight: {expertise_weights.get(top_proposal.agent_type, 1.0)}) "
            f"with {top_proposal.confidence:.2%} confidence"
        )

        return NegotiationResult(
            selected_treatment=top_proposal.treatment,
            selected_agent=top_proposal.agent_id,
            negotiation_strategy="weighted_expertise",
            confidence=top_proposal.confidence,
            consensus_score=consensus,
            all_proposals=proposals,
            negotiation_rounds=1,
            reasoning=reasoning
        )

    def _safety_first_negotiation(
        self,
        proposals: List[AgentProposal],
        patient_context: Optional[Dict[str, Any]]
    ) -> NegotiationResult:
        """
        Prioritize safety - select treatment with lowest risk score
        and fewest contraindications.
        """
        # Filter out proposals with contraindications
        safe_proposals = [p for p in proposals if not p.contraindications]

        if not safe_proposals:
            # All have contraindications - select least risky
            safe_proposals = sorted(proposals, key=lambda p: (len(p.contraindications), p.risk_score))

        # Select safest option (lowest risk score)
        selected = min(safe_proposals, key=lambda p: p.risk_score)

        # Calculate consensus for safety
        agreement = sum(1 for p in proposals if p.treatment == selected.treatment)
        consensus = agreement / len(proposals)

        reasoning = (
            f"Selected for safety profile: {len(selected.contraindications)} contraindications, "
            f"risk score: {selected.risk_score:.2f}. "
            f"{selected.rationale}"
        )

        return NegotiationResult(
            selected_treatment=selected.treatment,
            selected_agent=selected.agent_id,
            negotiation_strategy="safety_first",
            confidence=selected.confidence,
            consensus_score=consensus,
            all_proposals=proposals,
            negotiation_rounds=1,
            reasoning=reasoning
        )

    def _hybrid_negotiation(
        self,
        proposals: List[AgentProposal],
        patient_context: Optional[Dict[str, Any]]
    ) -> NegotiationResult:
        """
        Hybrid strategy combining evidence, consensus, and safety.

        Multi-criteria decision making:
        - Evidence level (30%)
        - Agent consensus (25%)
        - Safety profile (25%)
        - Confidence (20%)
        """
        evidence_scores = {"Grade A": 1.0, "Grade B": 0.67, "Grade C": 0.33, "Unknown": 0.0}

        # Calculate multi-criteria scores
        scored_proposals = []

        for p in proposals:
            # Evidence score (30%)
            evidence_score = evidence_scores.get(p.evidence_level, 0.0) * 0.30

            # Consensus score (25%) - how many agents agree
            agreement = sum(1 for other in proposals if other.treatment == p.treatment)
            consensus_score = (agreement / len(proposals)) * 0.25

            # Safety score (25%) - inverse of risk
            safety_score = (1.0 - p.risk_score) * 0.25

            # Confidence score (20%)
            confidence_score = p.confidence * 0.20

            # Total score
            total_score = evidence_score + consensus_score + safety_score + confidence_score

            scored_proposals.append((total_score, p))

        # Sort by total score
        scored_proposals.sort(key=lambda x: x[0], reverse=True)

        top_score, top_proposal = scored_proposals[0]

        # Calculate overall consensus
        agreement_count = sum(1 for p in proposals if p.treatment == top_proposal.treatment)
        consensus = agreement_count / len(proposals)

        reasoning = (
            f"Selected using hybrid multi-criteria analysis: "
            f"Score {top_score:.2f} (evidence: {top_proposal.evidence_level}, "
            f"safety: {(1-top_proposal.risk_score):.2f}, "
            f"consensus: {consensus:.2%}, "
            f"confidence: {top_proposal.confidence:.2%})"
        )

        alternatives = [
            {
                "treatment": p.treatment,
                "agent": p.agent_id,
                "score": score,
                "evidence": p.evidence_level,
                "risk": p.risk_score
            }
            for score, p in scored_proposals[1:]
        ]

        return NegotiationResult(
            selected_treatment=top_proposal.treatment,
            selected_agent=top_proposal.agent_id,
            negotiation_strategy="hybrid",
            confidence=top_proposal.confidence,
            consensus_score=consensus,
            all_proposals=proposals,
            negotiation_rounds=1,
            reasoning=reasoning,
            alternative_treatments=alternatives
        )

    # ========================================
    # CONFLICT DETECTION
    # ========================================

    def _detect_conflicts(self, proposals: List[AgentProposal]) -> List[Dict[str, Any]]:
        """
        Detect conflicts between agent proposals.

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Check for treatment disagreements
        treatments = set(p.treatment for p in proposals)
        if len(treatments) > 1:
            conflicts.append({
                "type": "treatment_disagreement",
                "treatments": list(treatments),
                "severity": "high" if len(treatments) > 2 else "medium"
            })

        # Check for intent conflicts
        intents = set(p.treatment_intent for p in proposals)
        if len(intents) > 1:
            conflicts.append({
                "type": "intent_conflict",
                "intents": list(intents),
                "severity": "high"
            })

        # Check for evidence quality disagreements
        evidence_levels = set(p.evidence_level for p in proposals)
        if "Grade C" in evidence_levels and "Grade A" in evidence_levels:
            conflicts.append({
                "type": "evidence_quality_gap",
                "evidence_range": f"Grade A to Grade C",
                "severity": "medium"
            })

        # Check for contraindication warnings
        for p in proposals:
            if p.contraindications:
                conflicts.append({
                    "type": "contraindication_warning",
                    "agent": p.agent_id,
                    "treatment": p.treatment,
                    "contraindications": p.contraindications,
                    "severity": "high"
                })

        return conflicts

    # ========================================
    # UTILITIES
    # ========================================

    def _create_single_proposal_result(self, proposal: AgentProposal) -> NegotiationResult:
        """Create result for single proposal (no negotiation needed)"""
        return NegotiationResult(
            selected_treatment=proposal.treatment,
            selected_agent=proposal.agent_id,
            negotiation_strategy="single_proposal",
            confidence=proposal.confidence,
            consensus_score=1.0,
            all_proposals=[proposal],
            negotiation_rounds=0,
            reasoning="Single proposal - no negotiation required"
        )

    def get_negotiation_history(self) -> List[NegotiationResult]:
        """Get history of all negotiations"""
        return self.negotiation_history

    def clear_history(self):
        """Clear negotiation history"""
        self.negotiation_history = []

    def generate_negotiation_report(self) -> Dict[str, Any]:
        """Generate summary report of negotiation patterns"""
        if not self.negotiation_history:
            return {"message": "No negotiation history"}

        total_negotiations = len(self.negotiation_history)
        avg_consensus = sum(n.consensus_score for n in self.negotiation_history) / total_negotiations
        avg_conflicts = sum(n.conflicts_resolved for n in self.negotiation_history) / total_negotiations

        # Strategy distribution
        strategy_counts = {}
        for n in self.negotiation_history:
            strategy_counts[n.negotiation_strategy] = strategy_counts.get(n.negotiation_strategy, 0) + 1

        return {
            "total_negotiations": total_negotiations,
            "avg_consensus_score": avg_consensus,
            "avg_conflicts_resolved": avg_conflicts,
            "strategy_distribution": strategy_counts,
            "high_consensus_rate": sum(1 for n in self.negotiation_history if n.consensus_score > 0.8) / total_negotiations
        }
