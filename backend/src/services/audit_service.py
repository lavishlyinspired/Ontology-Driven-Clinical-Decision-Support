"""
Audit Logging Service

HIPAA/GDPR compliant audit trail for all system actions.
Tracks who did what, when, and why.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel
import json

# Centralized logging
from ..logging_config import get_logger, log_execution

logger = get_logger(__name__)


class AuditAction(str, Enum):
    """Types of auditable actions."""
    # Authentication
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    LOGIN_FAILED = "auth.login_failed"
    TOKEN_REFRESH = "auth.token_refresh"
    
    # Patient operations
    PATIENT_CREATE = "patient.create"
    PATIENT_READ = "patient.read"
    PATIENT_UPDATE = "patient.update"
    PATIENT_DELETE = "patient.delete"
    
    # Analysis operations
    ANALYSIS_START = "analysis.start"
    ANALYSIS_COMPLETE = "analysis.complete"
    ANALYSIS_FAIL = "analysis.fail"
    ANALYSIS_EXPORT = "analysis.export"
    
    # File operations
    FILE_UPLOAD = "file.upload"
    FILE_DOWNLOAD = "file.download"
    
    # Administrative
    USER_CREATE = "admin.user_create"
    USER_UPDATE = "admin.user_update"
    USER_DEACTIVATE = "admin.user_deactivate"
    ROLE_CHANGE = "admin.role_change"
    SETTINGS_CHANGE = "admin.settings_change"
    
    # Clinical operations
    RECOMMENDATION_OVERRIDE = "clinical.override_recommendation"
    CASE_REVIEW = "clinical.case_review"
    CASE_APPROVE = "clinical.case_approve"
    
    # System events
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditLog(BaseModel):
    """Audit log entry."""
    log_id: str
    timestamp: datetime
    action: AuditAction
    severity: AuditSeverity
    
    # User context
    user_id: Optional[str] = None
    username: Optional[str] = None
    user_role: Optional[str] = None
    
    # Resource context
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    
    # Request context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    
    # Details
    description: str
    metadata: Dict[str, Any] = {}
    
    # Outcome
    success: bool
    error_message: Optional[str] = None


class AuditLogger:
    """Service for logging audit events."""
    
    def __init__(self):
        # In-memory storage (replace with database in production)
        self.logs: List[AuditLog] = []
        self._log_counter = 0
    
    def log(
        self,
        action: AuditAction,
        description: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        user_role: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.INFO
    ) -> AuditLog:
        """Log an audit event."""
        self._log_counter += 1
        
        audit_log = AuditLog(
            log_id=f"audit_{self._log_counter:08d}",
            timestamp=datetime.now(),
            action=action,
            severity=severity,
            user_id=user_id,
            username=username,
            user_role=user_role,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            description=description,
            metadata=metadata or {},
            success=success,
            error_message=error_message
        )
        
        self.logs.append(audit_log)
        
        # Print to console (replace with proper logging in production)
        self._print_log(audit_log)
        
        return audit_log
    
    def _print_log(self, log: AuditLog):
        """Print log entry to console."""
        severity_icon = {
            AuditSeverity.INFO: "ℹ️",
            AuditSeverity.WARNING: "⚠️",
            AuditSeverity.ERROR: "❌",
            AuditSeverity.CRITICAL: "🚨"
        }
        
        icon = severity_icon.get(log.severity, "📝")
        status = "✅" if log.success else "❌"
        
        print(f"{icon} [{log.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
              f"{status} {log.action.value} - {log.username or 'system'} - {log.description}")
    
    def log_authentication(
        self,
        username: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """Log authentication attempt."""
        action = AuditAction.LOGIN if success else AuditAction.LOGIN_FAILED
        severity = AuditSeverity.INFO if success else AuditSeverity.WARNING
        
        description = f"User '{username}' {'logged in successfully' if success else 'failed to log in'}"
        
        self.log(
            action=action,
            description=description,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
            severity=severity
        )
    
    def log_patient_access(
        self,
        user_id: str,
        username: str,
        patient_id: str,
        action: AuditAction,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log patient data access."""
        self.log(
            action=action,
            description=description,
            user_id=user_id,
            username=username,
            resource_type="patient",
            resource_id=patient_id,
            metadata=metadata,
            severity=AuditSeverity.INFO
        )
    
    def log_analysis(
        self,
        user_id: str,
        username: str,
        patient_id: str,
        success: bool,
        duration_ms: Optional[int] = None,
        error_message: Optional[str] = None
    ):
        """Log analysis execution."""
        action = AuditAction.ANALYSIS_COMPLETE if success else AuditAction.ANALYSIS_FAIL
        severity = AuditSeverity.INFO if success else AuditSeverity.ERROR
        
        description = f"Analysis {'completed' if success else 'failed'} for patient {patient_id}"
        
        metadata = {}
        if duration_ms:
            metadata['duration_ms'] = duration_ms
        
        self.log(
            action=action,
            description=description,
            user_id=user_id,
            username=username,
            resource_type="analysis",
            resource_id=patient_id,
            metadata=metadata,
            success=success,
            error_message=error_message,
            severity=severity
        )
    
    def log_override(
        self,
        user_id: str,
        username: str,
        user_role: str,
        patient_id: str,
        original_recommendation: str,
        override_recommendation: str,
        rationale: str
    ):
        """Log clinical override."""
        self.log(
            action=AuditAction.RECOMMENDATION_OVERRIDE,
            description=f"Clinician override for patient {patient_id}",
            user_id=user_id,
            username=username,
            user_role=user_role,
            resource_type="recommendation",
            resource_id=patient_id,
            metadata={
                'original': original_recommendation,
                'override': override_recommendation,
                'rationale': rationale
            },
            severity=AuditSeverity.WARNING  # Overrides are important
        )
    
    def log_export(
        self,
        user_id: str,
        username: str,
        patient_id: str,
        export_format: str,
        ip_address: Optional[str] = None
    ):
        """Log data export."""
        self.log(
            action=AuditAction.ANALYSIS_EXPORT,
            description=f"Exported {export_format} for patient {patient_id}",
            user_id=user_id,
            username=username,
            resource_type="export",
            resource_id=patient_id,
            ip_address=ip_address,
            metadata={'format': export_format},
            severity=AuditSeverity.INFO
        )
    
    def query_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """Query audit logs with filters."""
        filtered_logs = self.logs
        
        if user_id:
            filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
        
        if action:
            filtered_logs = [log for log in filtered_logs if log.action == action]
        
        if resource_id:
            filtered_logs = [log for log in filtered_logs if log.resource_id == resource_id]
        
        if start_date:
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_date]
        
        if end_date:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_date]
        
        # Sort by timestamp descending (most recent first)
        filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered_logs[:limit]
    
    def get_user_activity(self, user_id: str, days: int = 7) -> List[AuditLog]:
        """Get recent activity for a user."""
        start_date = datetime.now() - timedelta(days=days)
        return self.query_logs(user_id=user_id, start_date=start_date)
    
    def get_patient_access_history(self, patient_id: str) -> List[AuditLog]:
        """Get all access history for a patient."""
        return self.query_logs(resource_id=patient_id)
    
    def get_failed_logins(self, hours: int = 24) -> List[AuditLog]:
        """Get recent failed login attempts."""
        start_date = datetime.now() - timedelta(hours=hours)
        return self.query_logs(
            action=AuditAction.LOGIN_FAILED,
            start_date=start_date
        )
    
    def export_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        format: str = "json"
    ) -> str:
        """Export audit logs for compliance reporting."""
        logs = self.query_logs(start_date=start_date, end_date=end_date, limit=10000)
        
        if format == "json":
            return json.dumps([log.dict() for log in logs], indent=2, default=str)
        elif format == "csv":
            # Simple CSV format
            lines = ["timestamp,action,username,resource_type,resource_id,success,description"]
            for log in logs:
                lines.append(
                    f"{log.timestamp},{log.action.value},{log.username or ''},"
                    f"{log.resource_type or ''},{log.resource_id or ''},{log.success},{log.description}"
                )
            return "\n".join(lines)
        
        return json.dumps([log.dict() for log in logs], indent=2, default=str)


# Global audit logger instance
audit_logger = AuditLogger()


# Import for timedelta
from datetime import timedelta
