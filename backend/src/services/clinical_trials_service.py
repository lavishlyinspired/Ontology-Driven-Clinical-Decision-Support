"""
ClinicalTrials.gov Integration Service

This service provides real integration with the ClinicalTrials.gov API v2
for fetching clinical trials and mapping them through the pipeline:
ClinicalTrials.gov → SNOMED → LUCADA → Neo4j → FHIR

Author: LCA Development Team
Version: 1.0.0
"""

import aiohttp
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class ClinicalTrial:
    """Structured clinical trial data"""
    nct_id: str
    title: str
    brief_summary: str
    status: str
    phase: str
    conditions: List[str]
    interventions: List[Dict[str, str]]
    eligibility_criteria: str
    primary_outcomes: List[Dict[str, str]]
    secondary_outcomes: List[Dict[str, str]]
    enrollment: int
    start_date: Optional[str]
    completion_date: Optional[str]
    sponsor: str
    locations: List[Dict[str, str]]
    snomed_mappings: List[Dict[str, Any]] = field(default_factory=list)
    lucada_concepts: List[Dict[str, Any]] = field(default_factory=list)
    fhir_resources: List[Dict[str, Any]] = field(default_factory=list)


class ClinicalTrialsService:
    """
    Service for fetching and processing clinical trials from ClinicalTrials.gov API v2.
    
    API Documentation: https://clinicaltrials.gov/api/v2/
    """
    
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
    
    # SNOMED CT codes for lung cancer concepts
    SNOMED_MAPPINGS = {
        # Conditions
        "lung cancer": "254637007",
        "non-small cell lung cancer": "254637007",
        "nsclc": "254637007",
        "small cell lung cancer": "254632001",
        "sclc": "254632001",
        "adenocarcinoma": "35917007",
        "squamous cell carcinoma": "402815007",
        
        # Biomarkers
        "egfr": "448264001",
        "alk": "448263007",
        "ros1": "415116010",
        "pd-l1": "415116008",
        "kras": "415116009",
        "braf": "702838006",
        "met": "715816008",
        "ret": "715817004",
        "ntrk": "715818009",
        "her2": "715819001",
        
        # Treatment modalities
        "immunotherapy": "76334006",
        "chemotherapy": "367336001",
        "radiation therapy": "108290001",
        "targeted therapy": "416608005",
        "surgery": "387713003",
        
        # Drugs (common lung cancer drugs)
        "pembrolizumab": "716299006",
        "nivolumab": "716066001",
        "atezolizumab": "716067005",
        "osimertinib": "716068000",
        "erlotinib": "414382003",
        "gefitinib": "407007000",
        "alectinib": "716069008",
        "lorlatinib": "716070009",
        "crizotinib": "716071008",
        "sotorasib": "716072007",
        "carboplatin": "386904004",
        "cisplatin": "387318005",
        "pemetrexed": "386910003",
        "docetaxel": "386917000"
    }
    
    def __init__(self, snomed_loader=None, neo4j_driver=None, fhir_client=None):
        """
        Initialize the ClinicalTrials service.
        
        Args:
            snomed_loader: Optional SNOMEDLoader instance for concept mapping
            neo4j_driver: Optional Neo4j driver for graph storage
            fhir_client: Optional FHIR client for resource creation
        """
        self.snomed_loader = snomed_loader
        self.neo4j_driver = neo4j_driver
        self.fhir_client = fhir_client
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session
    
    async def close(self):
        """Close the HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def search_trials(
        self,
        condition: str = "lung cancer",
        intervention: Optional[str] = None,
        status: List[str] = None,
        phase: List[str] = None,
        page_size: int = 20,
        page_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for clinical trials from ClinicalTrials.gov API v2.
        
        Args:
            condition: Disease condition to search for
            intervention: Treatment intervention to filter by
            status: List of recruitment statuses (RECRUITING, ACTIVE_NOT_RECRUITING, etc.)
            phase: List of trial phases (PHASE1, PHASE2, PHASE3, PHASE4)
            page_size: Number of results per page (max 1000)
            page_token: Token for pagination
            
        Returns:
            Dict with trials list and pagination info
        """
        session = await self._get_session()
        
        # Build query parameters
        params = {
            "format": "json",
            "pageSize": min(page_size, 1000),
            "fields": ",".join([
                "NCTId", "BriefTitle", "BriefSummary", "OverallStatus",
                "Phase", "Condition", "InterventionName", "InterventionType",
                "EligibilityCriteria", "PrimaryOutcomeMeasure", "PrimaryOutcomeTimeFrame",
                "SecondaryOutcomeMeasure", "SecondaryOutcomeTimeFrame",
                "EnrollmentCount", "StartDate", "CompletionDate",
                "LeadSponsorName", "LocationFacility", "LocationCity", "LocationCountry"
            ])
        }
        
        # Add condition filter
        query_parts = []
        if condition:
            query_parts.append(f"AREA[Condition]{condition}")
        
        if intervention:
            query_parts.append(f"AREA[InterventionName]{intervention}")
        
        if query_parts:
            params["query.cond"] = condition
        
        if status:
            params["filter.overallStatus"] = ",".join(status)
        
        if phase:
            params["filter.phase"] = ",".join(phase)
        
        if page_token:
            params["pageToken"] = page_token
        
        try:
            async with session.get(self.BASE_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._process_api_response(data)
                else:
                    error_text = await response.text()
                    logger.error(f"ClinicalTrials.gov API error: {response.status} - {error_text}")
                    return {
                        "status": "error",
                        "error": f"API returned status {response.status}",
                        "details": error_text
                    }
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error: {e}")
            return {"status": "error", "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error fetching trials: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _process_api_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the API response and extract trial data"""
        studies = data.get("studies", [])
        trials = []
        
        for study in studies:
            protocol = study.get("protocolSection", {})
            id_module = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            desc_module = protocol.get("descriptionModule", {})
            design_module = protocol.get("designModule", {})
            eligibility_module = protocol.get("eligibilityModule", {})
            outcomes_module = protocol.get("outcomesModule", {})
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            contacts_module = protocol.get("contactsLocationsModule", {})
            arms_module = protocol.get("armsInterventionsModule", {})
            conditions_module = protocol.get("conditionsModule", {})
            
            # Extract interventions
            interventions = []
            for intervention in arms_module.get("interventions", []):
                interventions.append({
                    "type": intervention.get("type", ""),
                    "name": intervention.get("name", ""),
                    "description": intervention.get("description", "")
                })
            
            # Extract outcomes
            primary_outcomes = []
            for outcome in outcomes_module.get("primaryOutcomes", []):
                primary_outcomes.append({
                    "measure": outcome.get("measure", ""),
                    "time_frame": outcome.get("timeFrame", ""),
                    "description": outcome.get("description", "")
                })
            
            secondary_outcomes = []
            for outcome in outcomes_module.get("secondaryOutcomes", []):
                secondary_outcomes.append({
                    "measure": outcome.get("measure", ""),
                    "time_frame": outcome.get("timeFrame", ""),
                    "description": outcome.get("description", "")
                })
            
            # Extract locations
            locations = []
            for location in contacts_module.get("locations", []):
                locations.append({
                    "facility": location.get("facility", ""),
                    "city": location.get("city", ""),
                    "state": location.get("state", ""),
                    "country": location.get("country", "")
                })
            
            trial = ClinicalTrial(
                nct_id=id_module.get("nctId", ""),
                title=id_module.get("briefTitle", ""),
                brief_summary=desc_module.get("briefSummary", ""),
                status=status_module.get("overallStatus", ""),
                phase=",".join(design_module.get("phases", ["N/A"])),
                conditions=conditions_module.get("conditions", []),
                interventions=interventions,
                eligibility_criteria=eligibility_module.get("eligibilityCriteria", ""),
                primary_outcomes=primary_outcomes,
                secondary_outcomes=secondary_outcomes,
                enrollment=design_module.get("enrollmentInfo", {}).get("count", 0),
                start_date=status_module.get("startDateStruct", {}).get("date"),
                completion_date=status_module.get("completionDateStruct", {}).get("date"),
                sponsor=sponsor_module.get("leadSponsor", {}).get("name", ""),
                locations=locations[:5]  # Limit to 5 locations
            )
            
            # Map to SNOMED
            trial.snomed_mappings = self._map_to_snomed(trial)
            
            trials.append(trial)
        
        return {
            "status": "success",
            "total_count": data.get("totalCount", len(trials)),
            "trials": [self._trial_to_dict(t) for t in trials],
            "next_page_token": data.get("nextPageToken")
        }
    
    def _map_to_snomed(self, trial: ClinicalTrial) -> List[Dict[str, Any]]:
        """Map trial concepts to SNOMED-CT codes"""
        mappings = []
        
        # Map conditions
        for condition in trial.conditions:
            condition_lower = condition.lower()
            for term, code in self.SNOMED_MAPPINGS.items():
                if term in condition_lower:
                    mappings.append({
                        "source_term": condition,
                        "snomed_code": code,
                        "concept_type": "condition",
                        "confidence": 0.9
                    })
                    break
        
        # Map interventions
        for intervention in trial.interventions:
            intervention_name = intervention.get("name", "").lower()
            intervention_type = intervention.get("type", "").lower()
            
            for term, code in self.SNOMED_MAPPINGS.items():
                if term in intervention_name or term in intervention_type:
                    mappings.append({
                        "source_term": intervention.get("name"),
                        "snomed_code": code,
                        "concept_type": "intervention",
                        "confidence": 0.85
                    })
                    break
        
        return mappings
    
    def _trial_to_dict(self, trial: ClinicalTrial) -> Dict[str, Any]:
        """Convert ClinicalTrial dataclass to dictionary"""
        return {
            "nct_id": trial.nct_id,
            "title": trial.title,
            "brief_summary": trial.brief_summary[:500] + "..." if len(trial.brief_summary) > 500 else trial.brief_summary,
            "status": trial.status,
            "phase": trial.phase,
            "conditions": trial.conditions,
            "interventions": trial.interventions,
            "eligibility_criteria": trial.eligibility_criteria[:1000] + "..." if len(trial.eligibility_criteria) > 1000 else trial.eligibility_criteria,
            "primary_outcomes": trial.primary_outcomes,
            "secondary_outcomes": trial.secondary_outcomes[:3],  # Limit
            "enrollment": trial.enrollment,
            "start_date": trial.start_date,
            "completion_date": trial.completion_date,
            "sponsor": trial.sponsor,
            "locations": trial.locations,
            "snomed_mappings": trial.snomed_mappings
        }
    
    async def get_trial_by_nct_id(self, nct_id: str) -> Dict[str, Any]:
        """
        Fetch a specific trial by NCT ID.
        
        Args:
            nct_id: The NCT identifier (e.g., "NCT04321096")
            
        Returns:
            Trial details or error
        """
        session = await self._get_session()
        url = f"{self.BASE_URL}/{nct_id}"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._process_api_response({"studies": [data]})
                elif response.status == 404:
                    return {"status": "error", "error": f"Trial {nct_id} not found"}
                else:
                    return {"status": "error", "error": f"API returned status {response.status}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def match_patient_to_trials(
        self,
        patient_data: Dict[str, Any],
        max_trials: int = 10
    ) -> Dict[str, Any]:
        """
        Match a patient to eligible clinical trials.
        
        Args:
            patient_data: Patient information including diagnosis, biomarkers, etc.
            max_trials: Maximum number of trials to return
            
        Returns:
            List of matching trials with eligibility scores
        """
        diagnosis = patient_data.get("diagnosis", {})
        histology = diagnosis.get("histology", "lung cancer")
        stage = diagnosis.get("stage", "")
        biomarkers = patient_data.get("biomarkers", {})
        age = patient_data.get("age", 65)
        ecog_ps = patient_data.get("ecog_ps", 1)
        
        # Build search query based on patient characteristics
        condition = histology if histology else "non-small cell lung cancer"
        
        # Add biomarker context
        intervention = None
        for biomarker, value in biomarkers.items():
            if biomarker.upper() == "EGFR" and "positive" in str(value).lower() or "mutation" in str(value).lower():
                intervention = "osimertinib OR erlotinib"
                break
            elif biomarker.upper() == "ALK" and "positive" in str(value).lower():
                intervention = "alectinib OR lorlatinib"
                break
            elif biomarker.upper() == "PD-L1":
                intervention = "pembrolizumab OR nivolumab"
                break
        
        # Search for trials
        result = await self.search_trials(
            condition=condition,
            intervention=intervention,
            status=["RECRUITING", "ACTIVE_NOT_RECRUITING"],
            page_size=max_trials * 2  # Fetch more to filter
        )
        
        if result.get("status") != "success":
            return result
        
        # Score and filter trials based on patient eligibility
        matched_trials = []
        for trial in result.get("trials", []):
            score = self._calculate_eligibility_score(trial, patient_data)
            if score > 0:
                matched_trials.append({
                    **trial,
                    "eligibility_score": score,
                    "eligibility_notes": self._get_eligibility_notes(trial, patient_data)
                })
        
        # Sort by eligibility score
        matched_trials.sort(key=lambda x: x["eligibility_score"], reverse=True)
        
        return {
            "status": "success",
            "patient_summary": {
                "histology": histology,
                "stage": stage,
                "biomarkers": biomarkers,
                "age": age,
                "ecog_ps": ecog_ps
            },
            "matched_trials": matched_trials[:max_trials],
            "total_matches": len(matched_trials)
        }
    
    def _calculate_eligibility_score(self, trial: Dict[str, Any], patient: Dict[str, Any]) -> float:
        """Calculate patient-trial eligibility score (0-100)"""
        score = 50.0  # Base score
        criteria = trial.get("eligibility_criteria", "").lower()
        patient_diagnosis = patient.get("diagnosis", {})
        patient_biomarkers = patient.get("biomarkers", {})
        patient_age = patient.get("age", 65)
        patient_ecog = patient.get("ecog_ps", 1)
        
        # Check age eligibility
        if "18 years" in criteria or "≥18" in criteria:
            if patient_age >= 18:
                score += 10
        
        # Check ECOG eligibility
        if "ecog" in criteria:
            if "ecog 0-1" in criteria or "ecog 0 or 1" in criteria:
                if patient_ecog <= 1:
                    score += 15
                else:
                    score -= 30  # Likely ineligible
            elif "ecog 0-2" in criteria:
                if patient_ecog <= 2:
                    score += 10
        
        # Check stage
        stage = patient_diagnosis.get("stage", "").lower()
        if stage in criteria:
            score += 15
        elif "advanced" in criteria and stage in ["iii", "iiia", "iiib", "iiic", "iv", "iva", "ivb"]:
            score += 10
        
        # Check histology match
        histology = patient_diagnosis.get("histology", "").lower()
        if histology in criteria:
            score += 10
        
        # Check biomarker requirements
        for biomarker, value in patient_biomarkers.items():
            biomarker_lower = biomarker.lower()
            if biomarker_lower in criteria:
                if "positive" in str(value).lower() or "mutation" in str(value).lower():
                    if f"{biomarker_lower} positive" in criteria or f"{biomarker_lower} mutation" in criteria:
                        score += 20
                elif "negative" in str(value).lower():
                    if f"{biomarker_lower} negative" in criteria:
                        score += 10
        
        # Exclusion criteria checks
        exclusion_patterns = ["brain metastases", "prior treatment", "autoimmune"]
        for pattern in exclusion_patterns:
            if pattern in criteria:
                score -= 5  # Potential exclusion
        
        return min(100, max(0, score))
    
    def _get_eligibility_notes(self, trial: Dict[str, Any], patient: Dict[str, Any]) -> List[str]:
        """Generate eligibility assessment notes"""
        notes = []
        criteria = trial.get("eligibility_criteria", "").lower()
        patient_ecog = patient.get("ecog_ps", 1)
        patient_age = patient.get("age", 65)
        
        # Age note
        notes.append(f"Patient age {patient_age} - verify age requirements")
        
        # ECOG note
        if patient_ecog <= 1:
            notes.append(f"ECOG PS {patient_ecog} - likely meets performance status requirement")
        else:
            notes.append(f"ECOG PS {patient_ecog} - verify performance status eligibility")
        
        # Prior treatment check
        if "prior" in criteria and "treatment" in criteria:
            notes.append("Review prior treatment requirements/exclusions")
        
        # Biomarker notes
        for biomarker in patient.get("biomarkers", {}).keys():
            if biomarker.lower() in criteria:
                notes.append(f"{biomarker} status may affect eligibility")
        
        return notes
    
    async def store_trial_in_neo4j(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a clinical trial in Neo4j graph database.
        
        Args:
            trial: Trial data dictionary
            
        Returns:
            Result of the storage operation
        """
        if not self.neo4j_driver:
            return {"status": "error", "error": "Neo4j driver not configured"}
        
        try:
            async with self.neo4j_driver.session() as session:
                # Create trial node
                query = """
                MERGE (t:ClinicalTrial {nct_id: $nct_id})
                SET t.title = $title,
                    t.status = $status,
                    t.phase = $phase,
                    t.sponsor = $sponsor,
                    t.enrollment = $enrollment,
                    t.start_date = $start_date,
                    t.brief_summary = $brief_summary,
                    t.updated_at = datetime()
                
                WITH t
                
                // Create condition relationships
                UNWIND $conditions as condition
                MERGE (c:Condition {name: condition})
                MERGE (t)-[:STUDIES]->(c)
                
                WITH t
                
                // Create intervention relationships
                UNWIND $interventions as intervention
                MERGE (i:Intervention {name: intervention.name, type: intervention.type})
                MERGE (t)-[:TESTS]->(i)
                
                WITH t
                
                // Create SNOMED mappings
                UNWIND $snomed_mappings as mapping
                MERGE (s:SNOMEDConcept {code: mapping.snomed_code})
                SET s.source_term = mapping.source_term
                MERGE (t)-[:MAPS_TO {confidence: mapping.confidence}]->(s)
                
                RETURN t.nct_id as nct_id
                """
                
                result = await session.run(query, **trial)
                record = await result.single()
                
                return {
                    "status": "success",
                    "nct_id": record["nct_id"],
                    "message": f"Trial {trial['nct_id']} stored in Neo4j"
                }
                
        except Exception as e:
            logger.error(f"Error storing trial in Neo4j: {e}")
            return {"status": "error", "error": str(e)}
    
    async def convert_trial_to_fhir(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a clinical trial to FHIR ResearchStudy resource.
        
        Args:
            trial: Trial data dictionary
            
        Returns:
            FHIR ResearchStudy resource
        """
        # Map trial status to FHIR status
        status_mapping = {
            "RECRUITING": "active",
            "ACTIVE_NOT_RECRUITING": "active",
            "COMPLETED": "completed",
            "TERMINATED": "stopped",
            "WITHDRAWN": "withdrawn",
            "SUSPENDED": "temporarily-closed-to-accrual",
            "NOT_YET_RECRUITING": "in-review"
        }
        
        fhir_status = status_mapping.get(trial.get("status", ""), "unknown")
        
        # Map phase to FHIR phase
        phase_mapping = {
            "PHASE1": "phase-1",
            "PHASE2": "phase-2",
            "PHASE3": "phase-3",
            "PHASE4": "phase-4",
            "EARLY_PHASE1": "early-phase-1",
            "NA": "n-a"
        }
        
        phase = trial.get("phase", "N/A").upper().replace(" ", "")
        fhir_phase = phase_mapping.get(phase, "n-a")
        
        # Build FHIR ResearchStudy resource
        research_study = {
            "resourceType": "ResearchStudy",
            "id": trial.get("nct_id", "").lower(),
            "identifier": [
                {
                    "use": "official",
                    "system": "https://clinicaltrials.gov",
                    "value": trial.get("nct_id")
                }
            ],
            "title": trial.get("title"),
            "status": fhir_status,
            "phase": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/research-study-phase",
                        "code": fhir_phase
                    }
                ]
            },
            "condition": [
                {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": mapping.get("snomed_code"),
                            "display": mapping.get("source_term")
                        }
                    ]
                }
                for mapping in trial.get("snomed_mappings", [])
                if mapping.get("concept_type") == "condition"
            ],
            "description": trial.get("brief_summary"),
            "enrollment": [
                {
                    "reference": f"Group/{trial.get('nct_id')}-enrollment"
                }
            ],
            "period": {
                "start": trial.get("start_date"),
                "end": trial.get("completion_date")
            },
            "sponsor": {
                "display": trial.get("sponsor")
            },
            "arm": [
                {
                    "name": intervention.get("name"),
                    "type": {
                        "text": intervention.get("type")
                    },
                    "description": intervention.get("description")
                }
                for intervention in trial.get("interventions", [])
            ],
            "outcome": [
                {
                    "name": outcome.get("measure"),
                    "type": {
                        "text": "primary"
                    }
                }
                for outcome in trial.get("primary_outcomes", [])
            ]
        }
        
        return {
            "status": "success",
            "fhir_resource": research_study,
            "resource_type": "ResearchStudy",
            "nct_id": trial.get("nct_id")
        }
    
    async def full_integration_pipeline(
        self,
        condition: str = "non-small cell lung cancer",
        max_trials: int = 10
    ) -> Dict[str, Any]:
        """
        Execute the full integration pipeline:
        ClinicalTrials.gov → SNOMED → LUCADA → Neo4j → FHIR
        
        Args:
            condition: Disease condition to search
            max_trials: Maximum number of trials to process
            
        Returns:
            Complete pipeline results
        """
        pipeline_results = {
            "status": "success",
            "pipeline_steps": [],
            "processed_trials": []
        }
        
        # Step 1: Fetch from ClinicalTrials.gov
        step1_start = datetime.now()
        search_result = await self.search_trials(
            condition=condition,
            status=["RECRUITING", "ACTIVE_NOT_RECRUITING"],
            page_size=max_trials
        )
        
        pipeline_results["pipeline_steps"].append({
            "step": 1,
            "name": "ClinicalTrials.gov API",
            "status": search_result.get("status"),
            "trials_fetched": len(search_result.get("trials", [])),
            "duration_ms": (datetime.now() - step1_start).total_seconds() * 1000
        })
        
        if search_result.get("status") != "success":
            pipeline_results["status"] = "partial"
            return pipeline_results
        
        # Process each trial through the pipeline
        for trial in search_result.get("trials", []):
            trial_result = {
                "nct_id": trial.get("nct_id"),
                "title": trial.get("title")
            }
            
            # Step 2: SNOMED mapping (already done in search)
            trial_result["snomed_mappings"] = trial.get("snomed_mappings", [])
            
            # Step 3: LUCADA concept mapping
            trial_result["lucada_concepts"] = self._map_to_lucada(trial)
            
            # Step 4: Store in Neo4j (if driver available)
            if self.neo4j_driver:
                neo4j_result = await self.store_trial_in_neo4j(trial)
                trial_result["neo4j_stored"] = neo4j_result.get("status") == "success"
            else:
                trial_result["neo4j_stored"] = False
                trial_result["neo4j_note"] = "Driver not configured"
            
            # Step 5: Convert to FHIR
            fhir_result = await self.convert_trial_to_fhir(trial)
            trial_result["fhir_resource"] = fhir_result.get("fhir_resource")
            trial_result["fhir_status"] = fhir_result.get("status")
            
            pipeline_results["processed_trials"].append(trial_result)
        
        pipeline_results["pipeline_steps"].append({
            "step": 2,
            "name": "SNOMED Mapping",
            "status": "success",
            "concepts_mapped": sum(len(t.get("snomed_mappings", [])) for t in pipeline_results["processed_trials"])
        })
        
        pipeline_results["pipeline_steps"].append({
            "step": 3,
            "name": "LUCADA Enrichment",
            "status": "success",
            "concepts_mapped": sum(len(t.get("lucada_concepts", [])) for t in pipeline_results["processed_trials"])
        })
        
        pipeline_results["pipeline_steps"].append({
            "step": 4,
            "name": "Neo4j Storage",
            "status": "success" if any(t.get("neo4j_stored") for t in pipeline_results["processed_trials"]) else "skipped"
        })
        
        pipeline_results["pipeline_steps"].append({
            "step": 5,
            "name": "FHIR Conversion",
            "status": "success",
            "resources_created": len([t for t in pipeline_results["processed_trials"] if t.get("fhir_status") == "success"])
        })
        
        return pipeline_results
    
    def _map_to_lucada(self, trial: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Map trial concepts to LUCADA ontology"""
        lucada_concepts = []
        
        # LUCADA lung cancer specific mappings
        lucada_mappings = {
            # Histology types (LUCADA codes)
            "adenocarcinoma": "LUCADA:LC001",
            "squamous cell": "LUCADA:LC002",
            "large cell": "LUCADA:LC003",
            "small cell": "LUCADA:LC004",
            "non-small cell": "LUCADA:LC005",
            
            # Treatment modalities
            "surgery": "LUCADA:TX001",
            "chemotherapy": "LUCADA:TX002",
            "radiation": "LUCADA:TX003",
            "immunotherapy": "LUCADA:TX004",
            "targeted therapy": "LUCADA:TX005",
            
            # Biomarkers
            "egfr": "LUCADA:BM001",
            "alk": "LUCADA:BM002",
            "ros1": "LUCADA:BM003",
            "pd-l1": "LUCADA:BM004",
            "kras": "LUCADA:BM005"
        }
        
        # Map conditions
        for condition in trial.get("conditions", []):
            condition_lower = condition.lower()
            for term, code in lucada_mappings.items():
                if term in condition_lower:
                    lucada_concepts.append({
                        "source_term": condition,
                        "lucada_code": code,
                        "concept_type": "histology" if "LC" in code else "condition"
                    })
        
        # Map interventions
        for intervention in trial.get("interventions", []):
            intervention_name = intervention.get("name", "").lower()
            for term, code in lucada_mappings.items():
                if term in intervention_name:
                    lucada_concepts.append({
                        "source_term": intervention.get("name"),
                        "lucada_code": code,
                        "concept_type": "treatment" if "TX" in code else "intervention"
                    })
        
        return lucada_concepts


# Singleton instance
_service_instance: Optional[ClinicalTrialsService] = None


def get_clinical_trials_service(
    snomed_loader=None,
    neo4j_driver=None,
    fhir_client=None
) -> ClinicalTrialsService:
    """Get or create the ClinicalTrialsService singleton"""
    global _service_instance
    if _service_instance is None:
        _service_instance = ClinicalTrialsService(
            snomed_loader=snomed_loader,
            neo4j_driver=neo4j_driver,
            fhir_client=fhir_client
        )
    return _service_instance


# Example usage
async def main():
    """Example usage of the ClinicalTrialsService"""
    service = get_clinical_trials_service()
    
    try:
        # Search for lung cancer trials
        print("Searching for NSCLC trials...")
        result = await service.search_trials(
            condition="non-small cell lung cancer",
            status=["RECRUITING"],
            page_size=5
        )
        
        if result.get("status") == "success":
            print(f"Found {result.get('total_count')} trials")
            for trial in result.get("trials", [])[:3]:
                print(f"\n{trial['nct_id']}: {trial['title']}")
                print(f"  Status: {trial['status']}, Phase: {trial['phase']}")
                print(f"  SNOMED mappings: {len(trial.get('snomed_mappings', []))}")
        
        # Match a patient to trials
        print("\n\nMatching patient to trials...")
        patient = {
            "age": 62,
            "diagnosis": {
                "histology": "Adenocarcinoma",
                "stage": "IV"
            },
            "biomarkers": {
                "EGFR": "L858R mutation positive",
                "PD-L1": "80%"
            },
            "ecog_ps": 1
        }
        
        matches = await service.match_patient_to_trials(patient, max_trials=3)
        if matches.get("status") == "success":
            print(f"Found {matches.get('total_matches')} matching trials")
            for trial in matches.get("matched_trials", []):
                print(f"\n{trial['nct_id']}: Score {trial['eligibility_score']}")
                print(f"  Notes: {trial['eligibility_notes'][:2]}")
        
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
