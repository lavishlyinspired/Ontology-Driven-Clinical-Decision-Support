"""
Guideline Version Management Service

Manages versions of clinical guidelines (NCCN, LUCADA, etc.).
Supports version control, migration, A/B testing, and rollback.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel
import hashlib
import json


class GuidelineStatus(str, Enum):
    """Status of a guideline version."""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class GuidelineType(str, Enum):
    """Type of clinical guideline."""
    NCCN_NSCLC = "nccn_nsclc"
    NCCN_SCLC = "nccn_sclc"
    LUCADA = "lucada"
    ASCO = "asco"
    ESMO = "esmo"
    CUSTOM = "custom"


class GuidelineVersion(BaseModel):
    """A specific version of a clinical guideline."""
    guideline_id: str
    version: str
    guideline_type: GuidelineType
    
    # Metadata
    title: str
    description: Optional[str] = None
    status: GuidelineStatus = GuidelineStatus.DRAFT
    
    # Content
    content: Dict[str, Any]  # The actual guideline rules/logic
    content_hash: str  # SHA256 hash of content for integrity
    
    # Dates
    effective_date: date
    created_at: datetime
    updated_at: datetime
    deprecated_at: Optional[datetime] = None
    
    # Source tracking
    source_url: Optional[str] = None
    source_document: Optional[str] = None
    
    # Version control
    previous_version: Optional[str] = None
    changelog: List[str] = []
    
    # Usage tracking
    usage_count: int = 0
    last_used: Optional[datetime] = None


class MigrationScript(BaseModel):
    """Script to migrate from one guideline version to another."""
    from_version: str
    to_version: str
    description: str
    
    # Migration steps
    transformations: List[Dict[str, Any]]
    
    # Validation
    validation_rules: List[str]
    
    created_at: datetime


class ABTest(BaseModel):
    """A/B test configuration for guideline versions."""
    test_id: str
    test_name: str
    
    # Versions being tested
    version_a: str
    version_b: str
    
    # Traffic split (0-1)
    traffic_split: float = 0.5
    
    # Test configuration
    start_date: datetime
    end_date: Optional[datetime] = None
    status: str = "active"  # active, paused, completed
    
    # Results tracking
    results_a: Dict[str, Any] = {}
    results_b: Dict[str, Any] = {}
    
    # Metrics
    metrics_to_track: List[str] = []


class GuidelineVersionManager:
    """Manages guideline versions and migrations."""
    
    def __init__(self):
        """Initialize version manager."""
        # Store all guideline versions
        # guideline_id -> version -> GuidelineVersion
        self.versions: Dict[str, Dict[str, GuidelineVersion]] = {}
        
        # Active version for each guideline
        # guideline_id -> version
        self.active_versions: Dict[str, str] = {}
        
        # Migration scripts
        # (from_version, to_version) -> MigrationScript
        self.migrations: Dict[Tuple[str, str], MigrationScript] = {}
        
        # Active A/B tests
        self.ab_tests: Dict[str, ABTest] = {}
    
    def create_version(
        self,
        guideline_id: str,
        version: str,
        guideline_type: GuidelineType,
        title: str,
        content: Dict[str, Any],
        effective_date: date,
        description: Optional[str] = None,
        source_url: Optional[str] = None,
        previous_version: Optional[str] = None,
        changelog: Optional[List[str]] = None
    ) -> GuidelineVersion:
        """
        Create a new guideline version.
        
        Args:
            guideline_id: Unique identifier for the guideline
            version: Version string (e.g., "2024.1", "v2.0")
            guideline_type: Type of guideline
            title: Human-readable title
            content: The actual guideline content
            effective_date: When this version becomes effective
            description: Optional description
            source_url: URL to source document
            previous_version: Previous version for changelog
            changelog: List of changes from previous version
        
        Returns:
            Created GuidelineVersion
        """
        # Calculate content hash
        content_hash = self._calculate_hash(content)
        
        # Create version
        guideline_version = GuidelineVersion(
            guideline_id=guideline_id,
            version=version,
            guideline_type=guideline_type,
            title=title,
            description=description,
            status=GuidelineStatus.DRAFT,
            content=content,
            content_hash=content_hash,
            effective_date=effective_date,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_url=source_url,
            previous_version=previous_version,
            changelog=changelog or []
        )
        
        # Store version
        if guideline_id not in self.versions:
            self.versions[guideline_id] = {}
        
        self.versions[guideline_id][version] = guideline_version
        
        print(f"📝 Created guideline version: {guideline_id} v{version}")
        
        return guideline_version
    
    def activate_version(self, guideline_id: str, version: str) -> bool:
        """Activate a specific version of a guideline."""
        if guideline_id not in self.versions:
            print(f"❌ Guideline {guideline_id} not found")
            return False
        
        if version not in self.versions[guideline_id]:
            print(f"❌ Version {version} not found for {guideline_id}")
            return False
        
        # Deactivate previous active version
        if guideline_id in self.active_versions:
            old_version = self.active_versions[guideline_id]
            old_guideline = self.versions[guideline_id][old_version]
            old_guideline.status = GuidelineStatus.DEPRECATED
            old_guideline.deprecated_at = datetime.now()
        
        # Activate new version
        guideline_version = self.versions[guideline_id][version]
        guideline_version.status = GuidelineStatus.ACTIVE
        self.active_versions[guideline_id] = version
        
        print(f"✅ Activated {guideline_id} v{version}")
        
        return True
    
    def get_active_version(self, guideline_id: str) -> Optional[GuidelineVersion]:
        """Get the currently active version of a guideline."""
        if guideline_id not in self.active_versions:
            return None
        
        version = self.active_versions[guideline_id]
        return self.versions[guideline_id][version]
    
    def get_version(self, guideline_id: str, version: str) -> Optional[GuidelineVersion]:
        """Get a specific version of a guideline."""
        if guideline_id not in self.versions:
            return None
        
        return self.versions[guideline_id].get(version)
    
    def list_versions(
        self,
        guideline_id: str,
        status: Optional[GuidelineStatus] = None
    ) -> List[GuidelineVersion]:
        """List all versions of a guideline."""
        if guideline_id not in self.versions:
            return []
        
        versions = list(self.versions[guideline_id].values())
        
        if status:
            versions = [v for v in versions if v.status == status]
        
        # Sort by effective date (newest first)
        versions.sort(key=lambda v: v.effective_date, reverse=True)
        
        return versions
    
    def create_migration(
        self,
        guideline_id: str,
        from_version: str,
        to_version: str,
        description: str,
        transformations: List[Dict[str, Any]],
        validation_rules: Optional[List[str]] = None
    ) -> MigrationScript:
        """
        Create a migration script between versions.
        
        Transformations define how to convert patient data / analysis
        from one guideline version to another.
        """
        migration = MigrationScript(
            from_version=from_version,
            to_version=to_version,
            description=description,
            transformations=transformations,
            validation_rules=validation_rules or [],
            created_at=datetime.now()
        )
        
        key = (from_version, to_version)
        self.migrations[key] = migration
        
        print(f"🔄 Created migration: {from_version} → {to_version}")
        
        return migration
    
    def migrate(
        self,
        guideline_id: str,
        from_version: str,
        to_version: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Migrate data from one guideline version to another.
        
        Applies transformation rules defined in migration script.
        """
        key = (from_version, to_version)
        
        if key not in self.migrations:
            print(f"⚠️ No migration script found for {from_version} → {to_version}")
            return data
        
        migration = self.migrations[key]
        migrated_data = data.copy()
        
        # Apply transformations
        for transformation in migration.transformations:
            transform_type = transformation.get('type')
            
            if transform_type == 'rename_field':
                old_name = transformation['old_name']
                new_name = transformation['new_name']
                if old_name in migrated_data:
                    migrated_data[new_name] = migrated_data.pop(old_name)
            
            elif transform_type == 'map_values':
                field = transformation['field']
                mapping = transformation['mapping']
                if field in migrated_data:
                    migrated_data[field] = mapping.get(migrated_data[field], migrated_data[field])
            
            elif transform_type == 'add_field':
                field = transformation['field']
                default_value = transformation.get('default_value')
                if field not in migrated_data:
                    migrated_data[field] = default_value
            
            elif transform_type == 'remove_field':
                field = transformation['field']
                migrated_data.pop(field, None)
        
        print(f"🔄 Migrated data: {from_version} → {to_version}")
        
        return migrated_data
    
    def rollback(self, guideline_id: str) -> bool:
        """
        Rollback to the previous version of a guideline.
        
        Useful if issues are found with current version.
        """
        current_version = self.active_versions.get(guideline_id)
        if not current_version:
            print(f"❌ No active version for {guideline_id}")
            return False
        
        current = self.versions[guideline_id][current_version]
        
        if not current.previous_version:
            print(f"❌ No previous version to rollback to")
            return False
        
        # Rollback to previous version
        previous_version = current.previous_version
        
        print(f"⏪ Rolling back {guideline_id}: {current_version} → {previous_version}")
        
        return self.activate_version(guideline_id, previous_version)
    
    def create_ab_test(
        self,
        test_id: str,
        test_name: str,
        guideline_id: str,
        version_a: str,
        version_b: str,
        traffic_split: float = 0.5,
        metrics_to_track: Optional[List[str]] = None
    ) -> ABTest:
        """
        Create an A/B test between two guideline versions.
        
        Allows comparing outcomes between versions before full rollout.
        """
        ab_test = ABTest(
            test_id=test_id,
            test_name=test_name,
            version_a=version_a,
            version_b=version_b,
            traffic_split=traffic_split,
            start_date=datetime.now(),
            status="active",
            metrics_to_track=metrics_to_track or [
                'confidence',
                'treatment_efficacy',
                'clinician_override_rate'
            ]
        )
        
        self.ab_tests[test_id] = ab_test
        
        print(f"🧪 Started A/B test: {test_name}")
        print(f"   Version A ({version_a}): {traffic_split*100:.0f}%")
        print(f"   Version B ({version_b}): {(1-traffic_split)*100:.0f}%")
        
        return ab_test
    
    def get_version_for_test(self, test_id: str, patient_id: str) -> str:
        """
        Determine which version to use for a patient in an A/B test.
        
        Uses deterministic hash for consistent assignment.
        """
        if test_id not in self.ab_tests:
            return None
        
        ab_test = self.ab_tests[test_id]
        
        # Hash patient ID to get consistent assignment
        hash_value = int(hashlib.md5(patient_id.encode()).hexdigest(), 16)
        normalized = (hash_value % 10000) / 10000  # 0-1
        
        if normalized < ab_test.traffic_split:
            return ab_test.version_a
        else:
            return ab_test.version_b
    
    def record_test_result(
        self,
        test_id: str,
        version: str,
        metric: str,
        value: float
    ):
        """Record a result for an A/B test."""
        if test_id not in self.ab_tests:
            return
        
        ab_test = self.ab_tests[test_id]
        
        # Determine which version
        if version == ab_test.version_a:
            results = ab_test.results_a
        elif version == ab_test.version_b:
            results = ab_test.results_b
        else:
            return
        
        # Initialize metric if needed
        if metric not in results:
            results[metric] = []
        
        # Record value
        results[metric].append(value)
    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get results summary for an A/B test."""
        if test_id not in self.ab_tests:
            return {}
        
        ab_test = self.ab_tests[test_id]
        
        # Calculate statistics for each metric
        summary = {
            'test_id': test_id,
            'test_name': ab_test.test_name,
            'status': ab_test.status,
            'version_a': ab_test.version_a,
            'version_b': ab_test.version_b,
            'results': {}
        }
        
        for metric in ab_test.metrics_to_track:
            values_a = ab_test.results_a.get(metric, [])
            values_b = ab_test.results_b.get(metric, [])
            
            summary['results'][metric] = {
                'version_a': {
                    'count': len(values_a),
                    'mean': sum(values_a) / len(values_a) if values_a else 0,
                    'min': min(values_a) if values_a else 0,
                    'max': max(values_a) if values_a else 0
                },
                'version_b': {
                    'count': len(values_b),
                    'mean': sum(values_b) / len(values_b) if values_b else 0,
                    'min': min(values_b) if values_b else 0,
                    'max': max(values_b) if values_b else 0
                }
            }
        
        return summary
    
    def _calculate_hash(self, content: Dict[str, Any]) -> str:
        """Calculate SHA256 hash of content."""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def verify_integrity(self, guideline_id: str, version: str) -> bool:
        """Verify content integrity using hash."""
        guideline = self.get_version(guideline_id, version)
        if not guideline:
            return False
        
        current_hash = self._calculate_hash(guideline.content)
        return current_hash == guideline.content_hash
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about guideline versions."""
        total_guidelines = len(self.versions)
        total_versions = sum(len(versions) for versions in self.versions.values())
        active_versions = len(self.active_versions)
        active_tests = len([t for t in self.ab_tests.values() if t.status == 'active'])
        
        return {
            'total_guidelines': total_guidelines,
            'total_versions': total_versions,
            'active_versions': active_versions,
            'active_ab_tests': active_tests,
            'migrations': len(self.migrations)
        }


# Global version manager instance
version_manager = GuidelineVersionManager()
version_service = version_manager  # Alias for consistency with other services


# Initialize default NCCN versions
def initialize_default_versions():
    """Initialize default NCCN guideline versions."""
    # NCCN NSCLC 2024 Version 1
    version_manager.create_version(
        guideline_id="nccn_nsclc",
        version="2024.1",
        guideline_type=GuidelineType.NCCN_NSCLC,
        title="NCCN Guidelines for Non-Small Cell Lung Cancer",
        content={
            'staging_rules': {},
            'treatment_algorithms': {},
            'biomarker_guidelines': {}
        },
        effective_date=date(2024, 1, 15),
        description="NCCN Clinical Practice Guidelines in Oncology: NSCLC Version 1.2024",
        source_url="https://www.nccn.org/professionals/physician_gls/pdf/nscl.pdf"
    )
    
    # NCCN SCLC 2024 Version 1
    version_manager.create_version(
        guideline_id="nccn_sclc",
        version="2024.1",
        guideline_type=GuidelineType.NCCN_SCLC,
        title="NCCN Guidelines for Small Cell Lung Cancer",
        content={
            'staging_rules': {},
            'treatment_algorithms': {}
        },
        effective_date=date(2024, 1, 15),
        description="NCCN Clinical Practice Guidelines in Oncology: SCLC Version 1.2024",
        source_url="https://www.nccn.org/professionals/physician_gls/pdf/sclc.pdf"
    )
    
    # Activate both versions
    version_manager.activate_version("nccn_nsclc", "2024.1")
    version_manager.activate_version("nccn_sclc", "2024.1")
    
    print("✅ Initialized default NCCN guideline versions")
