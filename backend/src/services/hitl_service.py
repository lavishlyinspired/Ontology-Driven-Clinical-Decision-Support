"""
Human-in-the-Loop (HITL) Service

Enables clinician review and override of low-confidence recommendations.
Implements a review queue system with feedback loop for continuous improvement.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel
import asyncio

# Centralized logging
from ..logging_config import get_logger, log_execution

logger = get_logger(__name__)


class ReviewStatus(str, Enum):
    """Status of review cases."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    OVERRIDDEN = "overridden"
    ESCALATED = "escalated"


class ReviewPriority(str, Enum):
    """Priority levels for review."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ReviewReason(str, Enum):
    """Reasons for human review."""
    LOW_CONFIDENCE = "low_confidence"
    CONFLICTING_RECOMMENDATIONS = "conflicting_recommendations"
    COMPLEX_CASE = "complex_case"
    RARE_BIOMARKER = "rare_biomarker"
    GUIDELINE_AMBIGUITY = "guideline_ambiguity"
    MANUAL_REQUEST = "manual_request"


class ReviewCase(BaseModel):
    """A case requiring human review."""
    case_id: str
    patient_id: str
    analysis_id: str
    
    # Review metadata
    status: ReviewStatus = ReviewStatus.PENDING
    priority: ReviewPriority
    reason: ReviewReason
    
    # Original analysis
    original_recommendation: Dict[str, Any]
    overall_confidence: float
    agent_confidences: Dict[str, float]
    conflicts: List[Dict[str, Any]] = []
    
    # Patient context
    patient_data: Dict[str, Any]
    
    # Review timeline
    created_at: datetime
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    
    # Review outcome
    reviewer_decision: Optional[str] = None
    override_recommendation: Optional[Dict[str, Any]] = None
    rationale: Optional[str] = None
    feedback: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = {}


class ReviewDecision(BaseModel):
    """Clinician's review decision."""
    case_id: str
    reviewer_id: str
    reviewer_name: str
    
    decision: ReviewStatus  # APPROVED, REJECTED, or OVERRIDDEN
    override_recommendation: Optional[Dict[str, Any]] = None
    rationale: str
    confidence_adjustment: Optional[float] = None
    
    # Learning feedback
    feedback_to_agents: Optional[Dict[str, str]] = None
    suggestions: Optional[List[str]] = None


class HumanInTheLoopService:
    """Service for managing human-in-the-loop review process."""
    
    def __init__(self, confidence_threshold: float = 0.65):
        """
        Initialize HITL service.
        
        Args:
            confidence_threshold: Below this threshold, cases are flagged for review
        """
        self.confidence_threshold = confidence_threshold
        self.review_queue: List[ReviewCase] = []
        self._case_counter = 0
        
        # Track reviewer performance
        self.reviewer_stats: Dict[str, Dict[str, Any]] = {}
    
    def should_review(
        self,
        overall_confidence: float,
        agent_confidences: Dict[str, float],
        conflicts: List[Dict[str, Any]],
        patient_data: Dict[str, Any]
    ) -> tuple[bool, ReviewReason, ReviewPriority]:
        """
        Determine if a case needs human review.
        
        Returns:
            (needs_review, reason, priority)
        """
        # Low overall confidence
        if overall_confidence < self.confidence_threshold:
            priority = self._calculate_priority(overall_confidence, "low_confidence")
            return True, ReviewReason.LOW_CONFIDENCE, priority
        
        # Conflicting recommendations
        if len(conflicts) > 0:
            priority = ReviewPriority.HIGH if len(conflicts) > 2 else ReviewPriority.MEDIUM
            return True, ReviewReason.CONFLICTING_RECOMMENDATIONS, priority
        
        # Any agent with very low confidence
        if any(conf < 0.5 for conf in agent_confidences.values()):
            return True, ReviewReason.LOW_CONFIDENCE, ReviewPriority.MEDIUM
        
        # Complex case indicators
        biomarkers = patient_data.get('biomarkers', {})
        comorbidities = patient_data.get('comorbidities', [])
        
        # Rare biomarker combinations
        if len(biomarkers) > 3:
            return True, ReviewReason.COMPLEX_CASE, ReviewPriority.MEDIUM
        
        # Multiple comorbidities
        if len(comorbidities) > 3:
            return True, ReviewReason.COMPLEX_CASE, ReviewPriority.MEDIUM
        
        # Rare biomarkers (example: multiple mutations)
        rare_biomarkers = ['braf', 'ret', 'ntrk', 'met']
        if any(b in biomarkers for b in rare_biomarkers):
            return True, ReviewReason.RARE_BIOMARKER, ReviewPriority.HIGH
        
        return False, None, None
    
    def _calculate_priority(self, confidence: float, reason: str) -> ReviewPriority:
        """Calculate review priority based on confidence and reason."""
        if confidence < 0.4:
            return ReviewPriority.URGENT
        elif confidence < 0.5:
            return ReviewPriority.HIGH
        elif confidence < 0.6:
            return ReviewPriority.MEDIUM
        else:
            return ReviewPriority.LOW
    
    def create_review_case(
        self,
        patient_id: str,
        analysis_id: str,
        original_recommendation: Dict[str, Any],
        overall_confidence: float,
        agent_confidences: Dict[str, float],
        patient_data: Dict[str, Any],
        reason: ReviewReason,
        priority: ReviewPriority,
        conflicts: Optional[List[Dict[str, Any]]] = None
    ) -> ReviewCase:
        """Create a new review case and add to queue."""
        self._case_counter += 1
        
        review_case = ReviewCase(
            case_id=f"review_{self._case_counter:08d}",
            patient_id=patient_id,
            analysis_id=analysis_id,
            status=ReviewStatus.PENDING,
            priority=priority,
            reason=reason,
            original_recommendation=original_recommendation,
            overall_confidence=overall_confidence,
            agent_confidences=agent_confidences,
            conflicts=conflicts or [],
            patient_data=patient_data,
            created_at=datetime.now()
        )
        
        self.review_queue.append(review_case)
        
        # Sort queue by priority
        self._sort_queue()
        
        print(f"📋 Review case created: {review_case.case_id} ({priority.value} priority)")
        print(f"   Reason: {reason.value}")
        print(f"   Confidence: {overall_confidence:.1%}")
        
        return review_case
    
    def _sort_queue(self):
        """Sort review queue by priority and age."""
        priority_order = {
            ReviewPriority.URGENT: 0,
            ReviewPriority.HIGH: 1,
            ReviewPriority.MEDIUM: 2,
            ReviewPriority.LOW: 3
        }
        
        self.review_queue.sort(
            key=lambda x: (
                priority_order[x.priority],
                x.created_at
            )
        )
    
    def get_review_queue(
        self,
        status: Optional[ReviewStatus] = None,
        priority: Optional[ReviewPriority] = None,
        assigned_to: Optional[str] = None,
        limit: int = 50
    ) -> List[ReviewCase]:
        """Get review queue with filters."""
        filtered = self.review_queue
        
        if status:
            filtered = [c for c in filtered if c.status == status]
        
        if priority:
            filtered = [c for c in filtered if c.priority == priority]
        
        if assigned_to:
            filtered = [c for c in filtered if c.assigned_to == assigned_to]
        
        return filtered[:limit]
    
    def assign_case(self, case_id: str, reviewer_id: str) -> bool:
        """Assign a case to a reviewer."""
        case = self._find_case(case_id)
        if not case:
            return False
        
        case.assigned_to = reviewer_id
        case.assigned_at = datetime.now()
        case.status = ReviewStatus.IN_REVIEW
        
        print(f"👤 Case {case_id} assigned to {reviewer_id}")
        
        return True
    
    def submit_review(self, decision: ReviewDecision) -> bool:
        """Submit a review decision."""
        case = self._find_case(decision.case_id)
        if not case:
            return False
        
        # Update case
        case.status = decision.decision
        case.reviewed_at = datetime.now()
        case.reviewer_decision = decision.decision.value
        case.override_recommendation = decision.override_recommendation
        case.rationale = decision.rationale
        case.feedback = json.dumps(decision.feedback_to_agents) if decision.feedback_to_agents else None
        
        # Track reviewer stats
        self._update_reviewer_stats(decision.reviewer_id, decision)
        
        # Log the decision
        from .audit_service import audit_logger, AuditAction
        
        if decision.decision == ReviewStatus.OVERRIDDEN:
            audit_logger.log_override(
                user_id=decision.reviewer_id,
                username=decision.reviewer_name,
                user_role="clinician",
                patient_id=case.patient_id,
                original_recommendation=str(case.original_recommendation),
                override_recommendation=str(decision.override_recommendation),
                rationale=decision.rationale
            )
        
        print(f"✅ Review completed for {decision.case_id}: {decision.decision.value}")
        print(f"   Rationale: {decision.rationale}")
        
        return True
    
    def escalate_case(self, case_id: str, escalation_reason: str) -> bool:
        """Escalate a case to senior clinician."""
        case = self._find_case(case_id)
        if not case:
            return False
        
        case.status = ReviewStatus.ESCALATED
        case.priority = ReviewPriority.URGENT
        case.metadata['escalation_reason'] = escalation_reason
        case.metadata['escalated_at'] = datetime.now().isoformat()
        
        # Re-sort queue
        self._sort_queue()
        
        print(f"⬆️ Case {case_id} escalated: {escalation_reason}")
        
        return True
    
    def get_case_details(self, case_id: str) -> Optional[ReviewCase]:
        """Get detailed information about a review case."""
        return self._find_case(case_id)
    
    def get_reviewer_stats(self, reviewer_id: str) -> Dict[str, Any]:
        """Get statistics for a reviewer."""
        return self.reviewer_stats.get(reviewer_id, {
            'total_reviews': 0,
            'approved': 0,
            'rejected': 0,
            'overridden': 0,
            'average_time_minutes': 0
        })
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get overall queue statistics."""
        total = len(self.review_queue)
        
        by_status = {}
        for status in ReviewStatus:
            count = len([c for c in self.review_queue if c.status == status])
            by_status[status.value] = count
        
        by_priority = {}
        for priority in ReviewPriority:
            count = len([c for c in self.review_queue if c.priority == priority])
            by_priority[priority.value] = count
        
        # Calculate average wait time for pending cases
        pending_cases = [c for c in self.review_queue if c.status == ReviewStatus.PENDING]
        if pending_cases:
            avg_wait = sum(
                (datetime.now() - c.created_at).total_seconds()
                for c in pending_cases
            ) / len(pending_cases) / 60  # in minutes
        else:
            avg_wait = 0
        
        return {
            'total_cases': total,
            'by_status': by_status,
            'by_priority': by_priority,
            'average_wait_minutes': avg_wait,
            'oldest_pending': pending_cases[0].case_id if pending_cases else None
        }
    
    def _find_case(self, case_id: str) -> Optional[ReviewCase]:
        """Find a case by ID."""
        for case in self.review_queue:
            if case.case_id == case_id:
                return case
        return None
    
    def _update_reviewer_stats(self, reviewer_id: str, decision: ReviewDecision):
        """Update reviewer performance statistics."""
        if reviewer_id not in self.reviewer_stats:
            self.reviewer_stats[reviewer_id] = {
                'total_reviews': 0,
                'approved': 0,
                'rejected': 0,
                'overridden': 0,
                'total_time_seconds': 0
            }
        
        stats = self.reviewer_stats[reviewer_id]
        stats['total_reviews'] += 1
        
        if decision.decision == ReviewStatus.APPROVED:
            stats['approved'] += 1
        elif decision.decision == ReviewStatus.REJECTED:
            stats['rejected'] += 1
        elif decision.decision == ReviewStatus.OVERRIDDEN:
            stats['overridden'] += 1
        
        # Calculate review time
        case = self._find_case(decision.case_id)
        if case and case.assigned_at:
            review_time = (datetime.now() - case.assigned_at).total_seconds()
            stats['total_time_seconds'] += review_time
            stats['average_time_minutes'] = stats['total_time_seconds'] / stats['total_reviews'] / 60
    
    async def auto_review_simple_cases(self):
        """Automatically approve very simple, high-confidence cases."""
        auto_approved = 0
        
        for case in self.review_queue:
            if case.status != ReviewStatus.PENDING:
                continue
            
            # Auto-approve if confidence is just below threshold but close
            if (case.overall_confidence >= self.confidence_threshold - 0.05 and
                case.priority == ReviewPriority.LOW and
                len(case.conflicts) == 0):
                
                case.status = ReviewStatus.APPROVED
                case.reviewed_at = datetime.now()
                case.reviewer_decision = "auto_approved"
                case.rationale = "Automatically approved (confidence close to threshold, no conflicts)"
                
                auto_approved += 1
        
        if auto_approved > 0:
            print(f"🤖 Auto-approved {auto_approved} simple cases")
        
        return auto_approved


# Global HITL service instance
hitl_service = HumanInTheLoopService()


# Import json for serialization
import json
