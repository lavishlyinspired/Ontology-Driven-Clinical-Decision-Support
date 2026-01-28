"""
Clinical Trial Matcher

Matches patients to relevant clinical trials based on eligibility criteria.
Uses ClinicalTrials.gov API for real-time trial discovery.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import requests
from datetime import datetime

# Import ClinicalMappings for stage/histology mappings
try:
    from ..ontology.clinical_mappings import ClinicalMappings
except ImportError:
    # Fallback - define minimal ClinicalMappings inline
    class ClinicalMappings:
        @staticmethod
        def is_metastatic(stage: str) -> bool:
            return 'IV' in stage.upper() or 'METASTATIC' in stage.upper()
        
        @staticmethod
        def is_locally_advanced(stage: str) -> bool:
            return 'III' in stage.upper()
        
        @staticmethod
        def get_stage_keywords(stage: str) -> List[str]:
            if 'IV' in stage.upper():
                return ['metastatic', 'stage IV', 'advanced']
            elif 'III' in stage.upper():
                return ['locally advanced', 'stage III']
            elif 'II' in stage.upper():
                return ['stage II', 'early stage']
            else:
                return ['stage I', 'early stage']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClinicalTrial:
    """Clinical trial representation"""
    nct_id: str
    title: str
    status: str
    phase: str
    conditions: List[str]
    interventions: List[str]
    eligibility_criteria: str
    locations: List[str]
    sponsor: str
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    enrollment: Optional[int] = None
    start_date: Optional[str] = None
    completion_date: Optional[str] = None


@dataclass
class TrialMatch:
    """Trial match with eligibility score"""
    trial: ClinicalTrial
    match_score: float  # 0.0 to 1.0
    matched_criteria: List[str]
    potential_barriers: List[str]
    recommendation: str


class ClinicalTrialMatcher:
    """
    Match patients to relevant clinical trials.

    Features:
    - ClinicalTrials.gov API integration
    - Eligibility scoring
    - Geographic filtering
    - Phase and status filtering
    """

    def __init__(self, use_online_api: bool = True):
        """
        Initialize clinical trial matcher.

        Args:
            use_online_api: Use ClinicalTrials.gov API (requires internet)
        """
        self.use_online_api = use_online_api
        self.api_base = "https://clinicaltrials.gov/api/v2"

        # Cache for API responses
        self._cache = {}

    def find_eligible_trials(
        self,
        patient: Dict[str, Any],
        max_results: int = 10,
        phase_filter: Optional[List[str]] = None,
        location: Optional[str] = None
    ) -> List[TrialMatch]:
        """
        Find clinical trials matching patient eligibility.

        Args:
            patient: Patient data dictionary
            max_results: Maximum number of trials to return
            phase_filter: Filter by phase (e.g., ["Phase 2", "Phase 3"])
            location: Geographic location filter (e.g., "United States", "California")

        Returns:
            List of matched trials with scores
        """
        logger.info(f"Searching for clinical trials for patient {patient.get('patient_id')}")

        # Build search query
        search_criteria = self._build_search_criteria(patient, phase_filter, location)

        # Search for trials
        trials = self._search_trials(search_criteria, max_results)

        if not trials:
            logger.warning("No trials found matching criteria")
            return []

        # Score each trial for eligibility
        matches = []
        for trial in trials:
            match = self._score_trial_eligibility(patient, trial)
            matches.append(match)

        # Sort by match score
        matches.sort(key=lambda x: x.match_score, reverse=True)

        logger.info(f"Found {len(matches)} potential trial matches")

        return matches[:max_results]

    # ========================================
    # ELIGIBILITY MATCHING
    # ========================================

    def _score_trial_eligibility(
        self,
        patient: Dict[str, Any],
        trial: ClinicalTrial
    ) -> TrialMatch:
        """
        Score patient eligibility for a trial.

        Args:
            patient: Patient data
            trial: Clinical trial

        Returns:
            TrialMatch with score and details
        """
        matched_criteria = []
        potential_barriers = []
        score = 0.0

        # Basic criteria matching

        # 1. Histology match (30% weight)
        histology = patient.get('histology_type', '')
        if any(h.lower() in trial.title.lower() + trial.eligibility_criteria.lower()
               for h in ['nsclc', 'non-small cell', histology.lower()]):
            matched_criteria.append(f"Histology: {histology}")
            score += 0.3

        # 2. Stage match (25% weight)
        stage = patient.get('tnm_stage', '')
        stage_keywords = ClinicalMappings.get_stage_keywords(stage)
        if any(kw.lower() in trial.eligibility_criteria.lower() for kw in stage_keywords):
            matched_criteria.append(f"Stage: {stage}")
            score += 0.25

        # 3. Biomarker match (25% weight)
        biomarker_matches = []
        if patient.get('egfr_mutation') == "Positive":
            if 'egfr' in trial.eligibility_criteria.lower():
                biomarker_matches.append("EGFR mutation")
                score += 0.25
        if patient.get('alk_rearrangement') == "Positive":
            if 'alk' in trial.eligibility_criteria.lower():
                biomarker_matches.append("ALK rearrangement")
                score += 0.25
        if patient.get('pdl1_tps', 0) >= 50:
            if 'pd-l1' in trial.eligibility_criteria.lower():
                biomarker_matches.append(f"PD-L1 ≥50%")
                score += 0.15

        if biomarker_matches:
            matched_criteria.append(f"Biomarkers: {', '.join(biomarker_matches)}")

        # 4. Performance Status (10% weight)
        ps = patient.get('performance_status', 1)
        if ps <= 1:
            matched_criteria.append(f"Performance Status: {ps}")
            score += 0.1
        elif ps >= 3:
            potential_barriers.append(f"Poor Performance Status ({ps}) may exclude from most trials")

        # 5. Age eligibility (10% weight)
        age = patient.get('age_at_diagnosis', 65)
        if 18 <= age <= 75:  # Most trials have this range
            matched_criteria.append(f"Age: {age}")
            score += 0.1
        elif age < 18:
            potential_barriers.append("Age <18 - pediatric protocols may differ")
        elif age > 80:
            potential_barriers.append("Age >80 - may be excluded from some trials")

        # Additional barriers
        comorbidities = patient.get('comorbidities', [])
        if "autoimmune_disease" in comorbidities and "immunotherapy" in trial.interventions[0].lower():
            potential_barriers.append("Autoimmune disease may exclude from immunotherapy trials")

        # Cap score at 1.0
        score = min(score, 1.0)

        # Generate recommendation
        if score >= 0.7:
            recommendation = "Highly Recommended - Strong eligibility match"
        elif score >= 0.5:
            recommendation = "Recommended - Good eligibility match, review criteria"
        elif score >= 0.3:
            recommendation = "Consider - Partial match, discuss with investigator"
        else:
            recommendation = "Not Recommended - Poor eligibility match"

        return TrialMatch(
            trial=trial,
            match_score=score,
            matched_criteria=matched_criteria,
            potential_barriers=potential_barriers,
            recommendation=recommendation
        )

    # ========================================
    # CLINICALTRIALS.GOV API
    # ========================================

    def _search_trials(
        self,
        search_criteria: Dict[str, Any],
        max_results: int
    ) -> List[ClinicalTrial]:
        """Search ClinicalTrials.gov API"""

        if not self.use_online_api:
            return self._get_example_trials()

        # Check cache
        cache_key = str(search_criteria)
        if cache_key in self._cache:
            logger.info("Using cached trial results")
            return self._cache[cache_key]

        try:
            # Build API query for ClinicalTrials.gov v2 API
            # Note: v2 API uses different parameter format
            query_parts = ["Lung Cancer"]
            terms = search_criteria.get("terms", "")
            if terms:
                query_parts.append(terms)

            params = {
                "query.cond": " AND ".join(query_parts) if len(query_parts) > 1 else query_parts[0],
                "filter.overallStatus": "RECRUITING|NOT_YET_RECRUITING|ACTIVE_NOT_RECRUITING",
                "pageSize": min(max_results, 100),  # API max is 100
                "countTotal": "true"
            }

            # Add phase filter (v2 API format)
            phases = search_criteria.get("phases", [])
            if phases:
                phase_map = {
                    "Phase 1": "PHASE1",
                    "Phase 2": "PHASE2",
                    "Phase 3": "PHASE3",
                    "Phase 4": "PHASE4"
                }
                phase_codes = [phase_map.get(p, p) for p in phases]
                params["filter.advanced"] = f"SEARCH[Location](AREA[LocationCountry]United States) AND AREA[Phase]({' OR '.join(phase_codes)})"

            # Make API request
            response = requests.get(
                f"{self.api_base}/studies",
                params=params,
                timeout=15,
                headers={"Accept": "application/json"}
            )

            if response.status_code != 200:
                logger.error(f"ClinicalTrials.gov API error: {response.status_code}")
                return self._get_example_trials()

            data = response.json()

            # Parse response - handle different API response formats
            trials = []
            studies_data = data.get('studies', [])

            # Log response structure for debugging
            logger.debug(f"API response keys: {data.keys()}")
            if studies_data:
                logger.debug(f"First study type: {type(studies_data[0])}")

            for study in studies_data:
                # Handle case where study is a string (NCT ID only)
                if isinstance(study, str):
                    logger.debug(f"Skipping string study: {study}")
                    continue

                # Handle nested study structure
                if isinstance(study, dict):
                    trial = self._parse_trial_data(study)
                    if trial:
                        trials.append(trial)

            # If no trials parsed, return example trials
            if not trials:
                logger.info("No trials parsed from API, using example trials")
                return self._get_example_trials()

            # Cache results
            self._cache[cache_key] = trials

            return trials

        except Exception as e:
            logger.error(f"Failed to search trials: {e}")
            return self._get_example_trials()

    def _parse_trial_data(self, study) -> Optional[ClinicalTrial]:
        """Parse trial data from API response"""

        try:
            # Handle case where study is a string (NCT ID only)
            if isinstance(study, str):
                return None

            # Handle case where study is not a dict
            if not isinstance(study, dict):
                return None

            # Check if data is nested in 'protocolSection' (v2 API) or flat (v1 API)
            if 'protocolSection' in study:
                protocol = study.get('protocolSection', {})
            else:
                protocol = study

            identification = protocol.get('identificationModule', {}) or protocol
            status = protocol.get('statusModule', {}) or protocol
            conditions = protocol.get('conditionsModule', {}) or {}
            interventions = protocol.get('armsInterventionsModule', {}) or {}
            eligibility = protocol.get('eligibilityModule', {}) or {}
            contacts = protocol.get('contactsLocationsModule', {}) or {}

            nct_id = identification.get('nctId') or study.get('nctId') or study.get('NCTId', '')
            if not nct_id:
                return None

            return ClinicalTrial(
                nct_id=nct_id,
                title=identification.get('briefTitle') or identification.get('officialTitle', ''),
                status=status.get('overallStatus', ''),
                phase=status.get('phase', 'N/A'),
                conditions=conditions.get('conditions', []),
                interventions=[i.get('name', '') for i in interventions.get('interventions', []) if isinstance(i, dict)],
                eligibility_criteria=eligibility.get('eligibilityCriteria', ''),
                locations=[loc.get('facility', {}).get('name', '') if isinstance(loc, dict) else ''
                          for loc in contacts.get('locations', [])],
                sponsor=identification.get('organization', {}).get('fullName', '') if isinstance(identification.get('organization'), dict) else '',
                enrollment=status.get('enrollmentInfo', {}).get('count') if isinstance(status.get('enrollmentInfo'), dict) else None,
                start_date=status.get('startDateStruct', {}).get('date') if isinstance(status.get('startDateStruct'), dict) else None,
                completion_date=status.get('completionDateStruct', {}).get('date') if isinstance(status.get('completionDateStruct'), dict) else None
            )

        except Exception as e:
            # Silently fail - example trials will be used as fallback
            return None

    def _build_search_criteria(
        self,
        patient: Dict[str, Any],
        phase_filter: Optional[List[str]],
        location: Optional[str]
    ) -> Dict[str, Any]:
        """Build search criteria from patient data"""

        search_terms = []

        # Histology
        histology = patient.get('histology_type', '')
        if 'NonSmallCell' in histology or 'Adenocarcinoma' in histology:
            search_terms.append("NSCLC")
        elif 'SmallCell' in histology:
            search_terms.append("SCLC")

        # Stage (using ClinicalMappings)
        stage = patient.get('tnm_stage', '')
        if ClinicalMappings.is_metastatic(stage):
            search_terms.append("metastatic")
        elif ClinicalMappings.is_locally_advanced(stage):
            search_terms.append("locally advanced")

        # Biomarkers
        if patient.get('egfr_mutation') == "Positive":
            search_terms.append("EGFR")
        if patient.get('alk_rearrangement') == "Positive":
            search_terms.append("ALK")

        return {
            "terms": " ".join(search_terms),
            "phases": phase_filter or ["Phase 2", "Phase 3"],
            "location": location or "United States"
        }

    def _get_stage_keywords(self, stage: str) -> List[str]:
        """Get keywords associated with stage - delegates to ClinicalMappings"""
        return ClinicalMappings.get_stage_keywords(stage)

    # ========================================
    # EXAMPLE DATA
    # ========================================

    def _get_example_trials(self) -> List[ClinicalTrial]:
        """Return example trials for demonstration"""

        return [
            ClinicalTrial(
                nct_id="NCT12345678",
                title="Phase 3 Study of Osimertinib in EGFR-Mutant NSCLC",
                status="RECRUITING",
                phase="Phase 3",
                conditions=["Non-Small Cell Lung Cancer", "EGFR Mutation"],
                interventions=["Osimertinib 80mg daily"],
                eligibility_criteria=(
                    "Inclusion: Stage IV NSCLC with EGFR Ex19del or L858R mutation, "
                    "PS 0-1, age 18+. Exclusion: Prior EGFR TKI, ILD, QTc >470ms"
                ),
                locations=["Memorial Sloan Kettering Cancer Center, New York"],
                sponsor="AstraZeneca",
                contact_email="trials@astrazeneca.com",
                enrollment=500,
                start_date="2025-01-15"
            ),
            ClinicalTrial(
                nct_id="NCT87654321",
                title="Pembrolizumab + Chemotherapy vs Chemotherapy in PD-L1+ NSCLC",
                status="RECRUITING",
                phase="Phase 3",
                conditions=["Non-Small Cell Lung Cancer"],
                interventions=["Pembrolizumab", "Carboplatin", "Pemetrexed"],
                eligibility_criteria=(
                    "Inclusion: Metastatic NSCLC, PD-L1 TPS 1-49%, PS 0-1, "
                    "no prior systemic therapy. Exclusion: Active autoimmune disease"
                ),
                locations=["MD Anderson Cancer Center, Houston"],
                sponsor="Merck Sharp & Dohme",
                enrollment=600,
                start_date="2024-11-01"
            )
        ]

    # ========================================
    # REPORTING
    # ========================================

    def generate_trial_summary(self, match: TrialMatch) -> str:
        """Generate human-readable trial summary"""

        trial = match.trial

        lines = [
            f"Clinical Trial: {trial.nct_id}",
            "=" * 70,
            f"Title: {trial.title}",
            f"Phase: {trial.phase}",
            f"Status: {trial.status}",
            f"Sponsor: {trial.sponsor}",
            "",
            f"Interventions:",
        ]

        for intervention in trial.interventions:
            lines.append(f"  • {intervention}")

        lines.extend([
            "",
            f"Match Score: {match.match_score:.1%}",
            f"Recommendation: {match.recommendation}",
            "",
            "Matched Criteria:"
        ])

        for criterion in match.matched_criteria:
            lines.append(f"  ✓ {criterion}")

        if match.potential_barriers:
            lines.extend([
                "",
                "Potential Barriers:"
            ])
            for barrier in match.potential_barriers:
                lines.append(f"  ⚠ {barrier}")

        lines.extend([
            "",
            "Eligibility Summary:",
            trial.eligibility_criteria[:300] + "..." if len(trial.eligibility_criteria) > 300 else trial.eligibility_criteria,
            "",
            "Contact:",
            f"  Email: {trial.contact_email or 'See ClinicalTrials.gov'}",
            f"  Study Page: https://clinicaltrials.gov/study/{trial.nct_id}"
        ])

        return "\n".join(lines)
