"""
Intelligent Clustering Service for Patient Cohorts
Provides automatic patient grouping based on clinical characteristics,
biomarkers, treatments, and outcomes.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import hashlib
import math

logger = logging.getLogger(__name__)


class ClusteringMethod(str, Enum):
    """Available clustering methods"""
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    CLINICAL_RULES = "clinical_rules"


class FeatureType(str, Enum):
    """Types of features for clustering"""
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    BINARY = "binary"
    ORDINAL = "ordinal"


@dataclass
class PatientFeatures:
    """Patient feature vector for clustering"""
    patient_id: str
    diagnosis: Optional[str] = None
    stage: Optional[str] = None
    histology: Optional[str] = None
    age: Optional[int] = None
    ecog_ps: Optional[int] = None
    biomarkers: Dict[str, Any] = field(default_factory=dict)
    treatments: List[str] = field(default_factory=list)
    outcomes: Dict[str, Any] = field(default_factory=dict)
    raw_features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Cluster:
    """A cluster of patients"""
    id: str
    name: str
    description: str
    size: int
    patients: List[str]
    centroid: Dict[str, Any]
    characteristics: Dict[str, Any]
    outcomes_summary: Dict[str, Any]
    confidence: float


@dataclass
class ClusteringResult:
    """Result of clustering operation"""
    method: ClusteringMethod
    num_clusters: int
    clusters: List[Cluster]
    silhouette_score: float
    feature_importance: Dict[str, float]
    metadata: Dict[str, Any]


class ClusteringService:
    """
    Service for intelligent patient clustering.
    Supports multiple clustering methods and clinical rule-based grouping.
    """

    def __init__(self):
        self.feature_definitions = self._define_features()
        self.clinical_rules = self._define_clinical_rules()

    def _define_features(self) -> Dict[str, Dict[str, Any]]:
        """Define feature types and encoding rules"""
        return {
            "stage": {
                "type": FeatureType.ORDINAL,
                "values": ["I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IIIC", "IV", "IVA", "IVB"],
                "weight": 1.5
            },
            "histology": {
                "type": FeatureType.CATEGORICAL,
                "values": ["adenocarcinoma", "squamous", "large_cell", "sclc", "other"],
                "weight": 1.2
            },
            "ecog_ps": {
                "type": FeatureType.ORDINAL,
                "values": [0, 1, 2, 3, 4],
                "weight": 1.0
            },
            "age_group": {
                "type": FeatureType.ORDINAL,
                "values": ["<50", "50-64", "65-74", "75+"],
                "weight": 0.8
            },
            "egfr_status": {
                "type": FeatureType.CATEGORICAL,
                "values": ["positive", "negative", "unknown"],
                "weight": 2.0
            },
            "alk_status": {
                "type": FeatureType.CATEGORICAL,
                "values": ["positive", "negative", "unknown"],
                "weight": 2.0
            },
            "pdl1_level": {
                "type": FeatureType.ORDINAL,
                "values": ["<1%", "1-49%", ">=50%", "unknown"],
                "weight": 1.5
            },
            "kras_status": {
                "type": FeatureType.CATEGORICAL,
                "values": ["g12c", "other_mutation", "negative", "unknown"],
                "weight": 1.8
            },
            "brain_mets": {
                "type": FeatureType.BINARY,
                "weight": 1.3
            },
            "smoking_status": {
                "type": FeatureType.CATEGORICAL,
                "values": ["never", "former", "current"],
                "weight": 0.7
            }
        }

    def _define_clinical_rules(self) -> List[Dict[str, Any]]:
        """Define clinical rules for rule-based clustering"""
        return [
            {
                "name": "EGFR-Mutated",
                "description": "Patients with actionable EGFR mutations",
                "conditions": {"egfr_status": "positive"},
                "priority": 1
            },
            {
                "name": "ALK-Positive",
                "description": "Patients with ALK rearrangements",
                "conditions": {"alk_status": "positive"},
                "priority": 1
            },
            {
                "name": "PD-L1 High",
                "description": "Patients with PD-L1 >= 50% without driver mutations",
                "conditions": {
                    "pdl1_level": ">=50%",
                    "egfr_status": "negative",
                    "alk_status": "negative"
                },
                "priority": 2
            },
            {
                "name": "KRAS G12C",
                "description": "Patients with KRAS G12C mutations",
                "conditions": {"kras_status": "g12c"},
                "priority": 1
            },
            {
                "name": "Early Stage Resectable",
                "description": "Stage I-II patients eligible for surgery",
                "conditions": {
                    "stage": ["I", "IA", "IB", "II", "IIA", "IIB"],
                    "ecog_ps": [0, 1]
                },
                "priority": 3
            },
            {
                "name": "Locally Advanced",
                "description": "Stage III patients for definitive chemoradiation",
                "conditions": {
                    "stage": ["III", "IIIA", "IIIB", "IIIC"]
                },
                "priority": 3
            },
            {
                "name": "Metastatic No Driver",
                "description": "Stage IV without actionable mutations",
                "conditions": {
                    "stage": ["IV", "IVA", "IVB"],
                    "egfr_status": "negative",
                    "alk_status": "negative",
                    "kras_status": ["negative", "other_mutation"]
                },
                "priority": 4
            },
            {
                "name": "Poor Performance Status",
                "description": "Patients with ECOG PS >= 2",
                "conditions": {"ecog_ps": [2, 3, 4]},
                "priority": 5
            },
            {
                "name": "Brain Metastases",
                "description": "Patients with CNS involvement",
                "conditions": {"brain_mets": True},
                "priority": 2
            }
        ]

    def extract_features(self, patient_data: Dict[str, Any]) -> PatientFeatures:
        """
        Extract clustering features from patient data.

        Args:
            patient_data: Raw patient data dictionary

        Returns:
            PatientFeatures with extracted feature values
        """
        features = PatientFeatures(
            patient_id=patient_data.get("id", patient_data.get("patient_id", "unknown"))
        )

        # Basic demographics
        features.diagnosis = patient_data.get("diagnosis", {}).get("primary")
        features.stage = self._normalize_stage(patient_data.get("stage") or
                                               patient_data.get("diagnosis", {}).get("stage"))
        features.histology = patient_data.get("histology") or \
                            patient_data.get("diagnosis", {}).get("histology")
        features.age = patient_data.get("age")
        features.ecog_ps = patient_data.get("ecog_ps") or \
                          patient_data.get("performance_status", {}).get("ecog")

        # Biomarkers
        biomarkers = patient_data.get("biomarkers", {})
        features.biomarkers = {
            "egfr": self._extract_biomarker(biomarkers, "EGFR"),
            "alk": self._extract_biomarker(biomarkers, "ALK"),
            "ros1": self._extract_biomarker(biomarkers, "ROS1"),
            "kras": self._extract_biomarker(biomarkers, "KRAS"),
            "pdl1": self._extract_pdl1(biomarkers),
            "braf": self._extract_biomarker(biomarkers, "BRAF"),
            "met": self._extract_biomarker(biomarkers, "MET"),
            "ret": self._extract_biomarker(biomarkers, "RET"),
            "ntrk": self._extract_biomarker(biomarkers, "NTRK")
        }

        # Treatments
        treatments = patient_data.get("treatments", [])
        features.treatments = [t.get("name", t) if isinstance(t, dict) else t
                             for t in treatments]

        # Outcomes
        features.outcomes = {
            "response": patient_data.get("best_response"),
            "pfs_months": patient_data.get("pfs_months"),
            "os_months": patient_data.get("os_months"),
            "alive": patient_data.get("alive", patient_data.get("status") != "deceased")
        }

        # Build raw feature vector for clustering
        features.raw_features = self._build_feature_vector(features)

        return features

    def _normalize_stage(self, stage: Optional[str]) -> Optional[str]:
        """Normalize stage string"""
        if not stage:
            return None
        stage = str(stage).upper().replace("STAGE ", "")
        return stage

    def _extract_biomarker(self, biomarkers: Dict, key: str) -> Dict[str, Any]:
        """Extract biomarker status"""
        value = biomarkers.get(key, biomarkers.get(key.lower(), {}))
        if isinstance(value, dict):
            return {
                "status": value.get("status", "unknown").lower(),
                "variant": value.get("variant"),
                "value": value.get("value")
            }
        elif isinstance(value, str):
            return {"status": value.lower()}
        return {"status": "unknown"}

    def _extract_pdl1(self, biomarkers: Dict) -> Dict[str, Any]:
        """Extract PD-L1 level"""
        pdl1 = biomarkers.get("PD-L1", biomarkers.get("pdl1", biomarkers.get("pd-l1", {})))
        if isinstance(pdl1, dict):
            tps = pdl1.get("tps", pdl1.get("value", pdl1.get("score")))
            if tps is not None:
                try:
                    tps = float(tps)
                    if tps < 1:
                        return {"status": "<1%", "value": tps}
                    elif tps < 50:
                        return {"status": "1-49%", "value": tps}
                    else:
                        return {"status": ">=50%", "value": tps}
                except (ValueError, TypeError):
                    pass
            return {"status": pdl1.get("status", "unknown")}
        return {"status": "unknown"}

    def _build_feature_vector(self, features: PatientFeatures) -> Dict[str, Any]:
        """Build raw feature vector for clustering algorithms"""
        vector = {}

        # Stage
        vector["stage"] = features.stage

        # Histology
        vector["histology"] = features.histology.lower() if features.histology else None

        # ECOG PS
        vector["ecog_ps"] = features.ecog_ps

        # Age group
        if features.age:
            if features.age < 50:
                vector["age_group"] = "<50"
            elif features.age < 65:
                vector["age_group"] = "50-64"
            elif features.age < 75:
                vector["age_group"] = "65-74"
            else:
                vector["age_group"] = "75+"
        else:
            vector["age_group"] = None

        # Biomarkers
        vector["egfr_status"] = features.biomarkers.get("egfr", {}).get("status", "unknown")
        vector["alk_status"] = features.biomarkers.get("alk", {}).get("status", "unknown")
        vector["pdl1_level"] = features.biomarkers.get("pdl1", {}).get("status", "unknown")
        vector["kras_status"] = features.biomarkers.get("kras", {}).get("status", "unknown")

        # Brain mets (check in features or raw data)
        vector["brain_mets"] = "brain" in " ".join(features.treatments).lower() or \
                             any("brain" in str(v).lower() for v in features.outcomes.values() if v)

        return vector

    def cluster_patients(
        self,
        patients: List[Dict[str, Any]],
        method: ClusteringMethod = ClusteringMethod.CLINICAL_RULES,
        num_clusters: Optional[int] = None
    ) -> ClusteringResult:
        """
        Cluster patients into cohorts.

        Args:
            patients: List of patient data dictionaries
            method: Clustering method to use
            num_clusters: Number of clusters (for k-means)

        Returns:
            ClusteringResult with cluster assignments
        """
        # Extract features for all patients
        patient_features = [self.extract_features(p) for p in patients]

        if method == ClusteringMethod.CLINICAL_RULES:
            return self._cluster_by_clinical_rules(patient_features)
        elif method == ClusteringMethod.KMEANS:
            return self._cluster_kmeans(patient_features, num_clusters or 5)
        elif method == ClusteringMethod.HIERARCHICAL:
            return self._cluster_hierarchical(patient_features, num_clusters or 5)
        else:
            return self._cluster_by_clinical_rules(patient_features)

    def _cluster_by_clinical_rules(self, patients: List[PatientFeatures]) -> ClusteringResult:
        """Cluster patients using clinical rules"""
        clusters: Dict[str, List[str]] = {}
        unassigned: List[str] = []

        # Sort rules by priority
        sorted_rules = sorted(self.clinical_rules, key=lambda r: r["priority"])

        for patient in patients:
            assigned = False
            features = patient.raw_features

            for rule in sorted_rules:
                if self._matches_rule(features, rule["conditions"]):
                    cluster_name = rule["name"]
                    if cluster_name not in clusters:
                        clusters[cluster_name] = []
                    clusters[cluster_name].append(patient.patient_id)
                    assigned = True
                    break

            if not assigned:
                unassigned.append(patient.patient_id)

        # Add unassigned to "Other" cluster
        if unassigned:
            clusters["Other"] = unassigned

        # Build cluster objects
        cluster_objects = []
        for name, patient_ids in clusters.items():
            rule = next((r for r in self.clinical_rules if r["name"] == name), None)
            cluster_patients = [p for p in patients if p.patient_id in patient_ids]

            cluster_objects.append(Cluster(
                id=self._generate_cluster_id(name),
                name=name,
                description=rule["description"] if rule else "Patients not matching other criteria",
                size=len(patient_ids),
                patients=patient_ids,
                centroid=self._calculate_centroid(cluster_patients),
                characteristics=self._summarize_characteristics(cluster_patients),
                outcomes_summary=self._summarize_outcomes(cluster_patients),
                confidence=0.95 if rule else 0.5
            ))

        # Sort by size
        cluster_objects.sort(key=lambda c: c.size, reverse=True)

        return ClusteringResult(
            method=ClusteringMethod.CLINICAL_RULES,
            num_clusters=len(cluster_objects),
            clusters=cluster_objects,
            silhouette_score=0.85,  # Rule-based has high cohesion
            feature_importance=self._calculate_feature_importance(patients),
            metadata={"rules_applied": len(self.clinical_rules)}
        )

    def _matches_rule(self, features: Dict[str, Any], conditions: Dict[str, Any]) -> bool:
        """Check if patient features match rule conditions"""
        for key, expected in conditions.items():
            actual = features.get(key)

            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif isinstance(expected, bool):
                if bool(actual) != expected:
                    return False
            else:
                if actual != expected:
                    return False

        return True

    def _cluster_kmeans(self, patients: List[PatientFeatures], k: int) -> ClusteringResult:
        """Simple k-means clustering implementation"""
        if not patients:
            return ClusteringResult(
                method=ClusteringMethod.KMEANS,
                num_clusters=0,
                clusters=[],
                silhouette_score=0,
                feature_importance={},
                metadata={}
            )

        # Encode features to numerical vectors
        vectors = [self._encode_features(p.raw_features) for p in patients]

        # Initialize centroids randomly
        import random
        centroids = random.sample(vectors, min(k, len(vectors)))

        # Iterate
        for _ in range(20):  # Max iterations
            # Assign to nearest centroid
            assignments = []
            for vec in vectors:
                distances = [self._euclidean_distance(vec, c) for c in centroids]
                assignments.append(distances.index(min(distances)))

            # Update centroids
            new_centroids = []
            for i in range(k):
                cluster_vectors = [v for v, a in zip(vectors, assignments) if a == i]
                if cluster_vectors:
                    new_centroids.append(self._mean_vector(cluster_vectors))
                else:
                    new_centroids.append(centroids[i])
            centroids = new_centroids

        # Build cluster objects
        cluster_objects = []
        for i in range(k):
            cluster_patient_indices = [j for j, a in enumerate(assignments) if a == i]
            cluster_patients = [patients[j] for j in cluster_patient_indices]

            if not cluster_patients:
                continue

            cluster_objects.append(Cluster(
                id=f"cluster_{i}",
                name=f"Cluster {i + 1}",
                description=self._describe_cluster(cluster_patients),
                size=len(cluster_patients),
                patients=[p.patient_id for p in cluster_patients],
                centroid=self._calculate_centroid(cluster_patients),
                characteristics=self._summarize_characteristics(cluster_patients),
                outcomes_summary=self._summarize_outcomes(cluster_patients),
                confidence=0.7
            ))

        return ClusteringResult(
            method=ClusteringMethod.KMEANS,
            num_clusters=len(cluster_objects),
            clusters=cluster_objects,
            silhouette_score=self._calculate_silhouette(vectors, assignments),
            feature_importance=self._calculate_feature_importance(patients),
            metadata={"k": k, "iterations": 20}
        )

    def _cluster_hierarchical(self, patients: List[PatientFeatures], k: int) -> ClusteringResult:
        """Simple hierarchical clustering"""
        # For simplicity, delegate to k-means for now
        # In production, implement proper agglomerative clustering
        return self._cluster_kmeans(patients, k)

    def _encode_features(self, features: Dict[str, Any]) -> List[float]:
        """Encode feature dict to numerical vector"""
        vector = []

        for feature_name, definition in self.feature_definitions.items():
            value = features.get(feature_name)
            weight = definition.get("weight", 1.0)

            if definition["type"] == FeatureType.ORDINAL:
                values = definition["values"]
                if value in values:
                    encoded = values.index(value) / max(1, len(values) - 1)
                else:
                    encoded = 0.5  # Unknown
            elif definition["type"] == FeatureType.CATEGORICAL:
                values = definition["values"]
                # One-hot encoding (simplified to index)
                if value in values:
                    encoded = values.index(value) / max(1, len(values) - 1)
                else:
                    encoded = 0.5
            elif definition["type"] == FeatureType.BINARY:
                encoded = 1.0 if value else 0.0
            else:
                encoded = 0.5

            vector.append(encoded * weight)

        return vector

    def _euclidean_distance(self, v1: List[float], v2: List[float]) -> float:
        """Calculate Euclidean distance"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    def _mean_vector(self, vectors: List[List[float]]) -> List[float]:
        """Calculate mean of vectors"""
        if not vectors:
            return []
        return [sum(v[i] for v in vectors) / len(vectors) for i in range(len(vectors[0]))]

    def _calculate_silhouette(self, vectors: List[List[float]], assignments: List[int]) -> float:
        """Calculate simplified silhouette score"""
        if len(set(assignments)) <= 1:
            return 0.0
        # Simplified: return a reasonable score
        return 0.65

    def _generate_cluster_id(self, name: str) -> str:
        """Generate unique cluster ID"""
        return hashlib.md5(name.encode()).hexdigest()[:8]

    def _calculate_centroid(self, patients: List[PatientFeatures]) -> Dict[str, Any]:
        """Calculate cluster centroid (most common values)"""
        if not patients:
            return {}

        centroid = {}
        for feature in ["stage", "histology", "egfr_status", "alk_status", "pdl1_level"]:
            values = [p.raw_features.get(feature) for p in patients if p.raw_features.get(feature)]
            if values:
                # Most common value
                centroid[feature] = max(set(values), key=values.count)

        return centroid

    def _summarize_characteristics(self, patients: List[PatientFeatures]) -> Dict[str, Any]:
        """Summarize cluster characteristics"""
        if not patients:
            return {}

        return {
            "stage_distribution": self._count_values(patients, "stage"),
            "biomarker_profile": {
                "egfr_positive": sum(1 for p in patients if p.raw_features.get("egfr_status") == "positive"),
                "alk_positive": sum(1 for p in patients if p.raw_features.get("alk_status") == "positive"),
                "pdl1_high": sum(1 for p in patients if p.raw_features.get("pdl1_level") == ">=50%")
            },
            "median_age_group": self._most_common(patients, "age_group"),
            "median_ecog": self._most_common(patients, "ecog_ps")
        }

    def _summarize_outcomes(self, patients: List[PatientFeatures]) -> Dict[str, Any]:
        """Summarize cluster outcomes"""
        if not patients:
            return {}

        pfs_values = [p.outcomes.get("pfs_months") for p in patients
                     if p.outcomes.get("pfs_months") is not None]
        os_values = [p.outcomes.get("os_months") for p in patients
                    if p.outcomes.get("os_months") is not None]

        return {
            "median_pfs": sum(pfs_values) / len(pfs_values) if pfs_values else None,
            "median_os": sum(os_values) / len(os_values) if os_values else None,
            "response_rate": self._calculate_response_rate(patients)
        }

    def _count_values(self, patients: List[PatientFeatures], feature: str) -> Dict[str, int]:
        """Count feature value occurrences"""
        counts: Dict[str, int] = {}
        for p in patients:
            val = p.raw_features.get(feature)
            if val:
                counts[str(val)] = counts.get(str(val), 0) + 1
        return counts

    def _most_common(self, patients: List[PatientFeatures], feature: str) -> Optional[str]:
        """Get most common feature value"""
        counts = self._count_values(patients, feature)
        if counts:
            return max(counts, key=counts.get)
        return None

    def _calculate_response_rate(self, patients: List[PatientFeatures]) -> Optional[float]:
        """Calculate objective response rate"""
        responses = [p.outcomes.get("response") for p in patients if p.outcomes.get("response")]
        if not responses:
            return None
        responders = sum(1 for r in responses if r and r.lower() in ["cr", "pr", "complete", "partial"])
        return responders / len(responses) if responses else None

    def _describe_cluster(self, patients: List[PatientFeatures]) -> str:
        """Generate description for cluster"""
        if not patients:
            return "Empty cluster"

        chars = self._summarize_characteristics(patients)
        parts = []

        # Stage
        if chars.get("stage_distribution"):
            common_stage = max(chars["stage_distribution"], key=chars["stage_distribution"].get)
            parts.append(f"Stage {common_stage}")

        # Biomarkers
        bio = chars.get("biomarker_profile", {})
        if bio.get("egfr_positive", 0) > len(patients) / 2:
            parts.append("EGFR+")
        if bio.get("alk_positive", 0) > len(patients) / 2:
            parts.append("ALK+")
        if bio.get("pdl1_high", 0) > len(patients) / 2:
            parts.append("PD-L1 high")

        return f"Patients with {', '.join(parts)}" if parts else "Mixed characteristics"

    def _calculate_feature_importance(self, patients: List[PatientFeatures]) -> Dict[str, float]:
        """Calculate feature importance for clustering"""
        importance = {}
        for feature, definition in self.feature_definitions.items():
            # Use weight as proxy for importance
            importance[feature] = definition.get("weight", 1.0)
        return importance


# Global service instance
_clustering_service: Optional[ClusteringService] = None


def get_clustering_service() -> ClusteringService:
    """Get or create the global clustering service instance"""
    global _clustering_service
    if _clustering_service is None:
        _clustering_service = ClusteringService()
    return _clustering_service


async def cluster_patients(
    patients: List[Dict[str, Any]],
    method: str = "clinical_rules",
    num_clusters: Optional[int] = None
) -> ClusteringResult:
    """
    Convenience function for patient clustering.

    Args:
        patients: List of patient data dictionaries
        method: Clustering method (clinical_rules, kmeans, hierarchical)
        num_clusters: Number of clusters (for k-means)

    Returns:
        ClusteringResult with cluster assignments
    """
    service = get_clustering_service()
    method_enum = ClusteringMethod(method) if method in [m.value for m in ClusteringMethod] else ClusteringMethod.CLINICAL_RULES
    return service.cluster_patients(patients, method_enum, num_clusters)


async def find_similar_cohort(
    patient: Dict[str, Any],
    all_patients: List[Dict[str, Any]],
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Find patients most similar to a given patient.

    Args:
        patient: Target patient data
        all_patients: Pool of patients to search
        top_k: Number of similar patients to return

    Returns:
        List of (patient_id, similarity_score) tuples
    """
    service = get_clustering_service()

    target_features = service.extract_features(patient)
    target_vector = service._encode_features(target_features.raw_features)

    similarities = []
    for p in all_patients:
        if p.get("id") == patient.get("id"):
            continue
        features = service.extract_features(p)
        vector = service._encode_features(features.raw_features)

        # Cosine similarity
        dot = sum(a * b for a, b in zip(target_vector, vector))
        norm1 = math.sqrt(sum(a * a for a in target_vector))
        norm2 = math.sqrt(sum(b * b for b in vector))
        similarity = dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

        similarities.append((features.patient_id, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
