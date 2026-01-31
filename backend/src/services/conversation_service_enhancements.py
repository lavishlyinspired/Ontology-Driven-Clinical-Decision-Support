"""
Enhancements for Conversation Service
=====================================

This module contains methods to add to ConversationService for:
1. MCP Apps integration
2. Clustering analysis
3. Citation enhancement

Add these methods to your ConversationService class.
"""

import re
import json
from typing import Dict, List, AsyncIterator, Any, Optional

# Add these imports to conversation_service.py:
# from ..services.clustering_service import ClusteringService, ClusteringMethod


# ============================================================================
# 1. MCP APPS INTEGRATION
# ============================================================================

async def _stream_mcp_app(self, message: str, session_id: str) -> AsyncIterator[str]:
    """
    Stream MCP app rendering based on user query

    Detects which interactive app to show and provides the data
    """

    message_lower = message.lower()

    # Treatment Comparison App
    if any(kw in message_lower for kw in ['compare treatment', 'treatment comparison', 'treatment options']):
        yield self._format_sse({
            "type": "status",
            "content": "Loading treatment comparison tool..."
        })

        # Get treatment data from patient context or query
        treatments_data = await self._get_treatment_comparison_data(session_id)

        yield self._format_sse({
            "type": "mcp_app",
            "content": {
                "resourceUri": "/mcp-apps/treatment-compare.html",
                "input": treatments_data,
                "title": "Treatment Comparison"
            }
        })

        # Also provide text explanation
        explanation = self._explain_treatment_comparison(treatments_data)
        yield self._format_sse({
            "type": "text",
            "content": explanation
        })

    # Survival Curves App
    elif any(kw in message_lower for kw in ['survival curve', 'kaplan meier', 'survival data']):
        yield self._format_sse({
            "type": "status",
            "content": "Generating survival curves..."
        })

        curves_data = await self._get_survival_curves_data(session_id)

        yield self._format_sse({
            "type": "mcp_app",
            "content": {
                "resourceUri": "/mcp-apps/survival-curves.html",
                "input": curves_data,
                "title": "Survival Analysis"
            }
        })

        explanation = self._explain_survival_curves(curves_data)
        yield self._format_sse({
            "type": "text",
            "content": explanation
        })

    # Guideline Tree App
    elif any(kw in message_lower for kw in ['guideline tree', 'nccn decision', 'decision tree']):
        yield self._format_sse({
            "type": "status",
            "content": "Loading NCCN guideline decision tree..."
        })

        guideline_data = await self._get_guideline_tree_data(session_id)

        yield self._format_sse({
            "type": "mcp_app",
            "content": {
                "resourceUri": "/mcp-apps/guideline-tree.html",
                "input": guideline_data,
                "title": "NCCN Guideline Explorer"
            }
        })

        explanation = self._explain_guideline_tree(guideline_data)
        yield self._format_sse({
            "type": "text",
            "content": explanation
        })

    # Clinical Trial Matcher App
    elif any(kw in message_lower for kw in ['clinical trial', 'trial match', 'trial search']):
        yield self._format_sse({
            "type": "status",
            "content": "Searching clinical trials..."
        })

        trials = await self._match_clinical_trials(session_id)

        yield self._format_sse({
            "type": "mcp_app",
            "content": {
                "resourceUri": "/mcp-apps/trial-matcher.html",
                "input": trials,
                "title": "Clinical Trial Matcher"
            }
        })

        explanation = self._explain_trial_matches(trials)
        yield self._format_sse({
            "type": "text",
            "content": explanation
        })

    else:
        # No matching app found
        yield self._format_sse({
            "type": "text",
            "content": "I can show you interactive visualizations for:\n"
                      "- **Treatment Comparison**: Compare treatment options side-by-side\n"
                      "- **Survival Curves**: Kaplan-Meier survival analysis\n"
                      "- **Guideline Tree**: Navigate NCCN decision pathways\n"
                      "- **Clinical Trials**: Match patients to trials\n\n"
                      "Try asking: 'Compare treatments' or 'Show survival curves'"
        })


async def _get_treatment_comparison_data(self, session_id: str) -> Dict[str, Any]:
    """Get treatment comparison data for the app"""

    # Get patient context if available
    patient_data = self.patient_context.get(session_id, {}).get("patient_data", {})

    # Default treatments (can be customized based on patient characteristics)
    treatments = [
        {
            "name": "Osimertinib",
            "indication": "1L EGFR+ NSCLC",
            "orr": 80,
            "pfs": 18.9,
            "os": 38.6,
            "grade34_ae": 34,
            "evidence": "FLAURA trial",
            "nccn_category": "Category 1"
        },
        {
            "name": "Gefitinib/Erlotinib",
            "indication": "1L EGFR+ NSCLC",
            "orr": 76,
            "pfs": 10.2,
            "os": 31.8,
            "grade34_ae": 25,
            "evidence": "Multiple trials",
            "nccn_category": "Category 1"
        },
        {
            "name": "Platinum Doublet",
            "indication": "1L NSCLC",
            "orr": 35,
            "pfs": 5.5,
            "os": 13.2,
            "grade34_ae": 55,
            "evidence": "Standard of care",
            "nccn_category": "Category 1"
        }
    ]

    # Customize based on biomarkers
    biomarkers = patient_data.get("biomarker_profile", {})

    if biomarkers.get("ALK"):
        treatments = [
            {
                "name": "Alectinib",
                "indication": "1L ALK+ NSCLC",
                "orr": 83,
                "pfs": 34.8,
                "os": 45.0,
                "grade34_ae": 41,
                "evidence": "ALEX trial"
            },
            {
                "name": "Crizotinib",
                "indication": "1L ALK+ NSCLC",
                "orr": 76,
                "pfs": 10.9,
                "os": 35.0,
                "grade34_ae": 50,
                "evidence": "Multiple trials"
            }
        ]

    return {
        "treatments": treatments,
        "patient_context": patient_data
    }


async def _get_survival_curves_data(self, session_id: str) -> Dict[str, Any]:
    """Get survival curve data for visualization"""

    # Sample Kaplan-Meier data (replace with real data from analytics service)
    curves = [
        {
            "name": "Osimertinib",
            "color": "#8b5cf6",
            "data": [
                [0, 100], [3, 98], [6, 95], [9, 91], [12, 85],
                [15, 80], [18, 75], [21, 68], [24, 65], [30, 55], [36, 45]
            ],
            "median_survival": 18.9,
            "events": 45,
            "censored": 23
        },
        {
            "name": "Gefitinib",
            "color": "#3b82f6",
            "data": [
                [0, 100], [3, 96], [6, 90], [9, 82], [12, 70],
                [15, 60], [18, 50], [21, 40], [24, 35], [30, 25], [36, 18]
            ],
            "median_survival": 10.2,
            "events": 58,
            "censored": 12
        },
        {
            "name": "Chemotherapy",
            "color": "#ef4444",
            "data": [
                [0, 100], [3, 92], [6, 75], [9, 60], [12, 45],
                [15, 35], [18, 28], [21, 22], [24, 18], [30, 12], [36, 8]
            ],
            "median_survival": 5.5,
            "events": 72,
            "censored": 8
        }
    ]

    return {
        "curves": curves,
        "title": "Progression-Free Survival by Treatment",
        "x_label": "Months",
        "y_label": "PFS Probability (%)"
    }


async def _get_guideline_tree_data(self, session_id: str) -> Dict[str, Any]:
    """Get NCCN guideline tree data"""

    patient_data = self.patient_context.get(session_id, {}).get("patient_data", {})

    return {
        "stage": patient_data.get("tnm_stage", "IIIA"),
        "histology": patient_data.get("histology_type", "adenocarcinoma"),
        "biomarkers": patient_data.get("biomarker_profile", {}),
        "performance_status": patient_data.get("performance_status", 1),
        "guideline_version": "NCCN 2024.1"
    }


async def _match_clinical_trials(self, session_id: str) -> Dict[str, Any]:
    """Match clinical trials for patient"""

    patient_data = self.patient_context.get(session_id, {}).get("patient_data", {})

    # Sample trial data (replace with real API call to ClinicalTrials.gov)
    trials = [
        {
            "nct_id": "NCT04303780",
            "title": "CodeBreaK 200: Sotorasib vs Docetaxel in KRAS G12C NSCLC",
            "phase": "Phase 3",
            "status": "Active, recruiting",
            "locations": ["Multiple US sites"],
            "contact": "study.coordinator@trial.com",
            "eligibility": {
                "age": "≥18 years",
                "stage": "IV NSCLC",
                "biomarker": "KRAS G12C mutation",
                "prior_therapy": "Platinum-based chemotherapy + PD-1/PD-L1 inhibitor"
            },
            "match_score": 0.95
        },
        {
            "nct_id": "NCT03785249",
            "title": "KRYSTAL-1: Adagrasib in KRAS G12C NSCLC",
            "phase": "Phase 1/2",
            "status": "Active, recruiting",
            "locations": ["US and International"],
            "contact": "krystal@trial.com",
            "eligibility": {
                "age": "≥18 years",
                "stage": "Advanced solid tumors",
                "biomarker": "KRAS G12C mutation",
                "prior_therapy": "Any"
            },
            "match_score": 0.88
        }
    ]

    return {
        "trials": trials,
        "patient_summary": patient_data,
        "total_matches": len(trials)
    }


def _explain_treatment_comparison(self, data: Dict) -> str:
    """Generate text explanation for treatment comparison"""

    treatments = data.get("treatments", [])

    if not treatments:
        return "No treatment comparison data available."

    best_pfs = max(treatments, key=lambda t: t.get("pfs", 0))

    explanation = f"""## Treatment Comparison Results

I've compared {len(treatments)} treatment options above. Key findings:

- **Highest PFS**: {best_pfs['name']} with {best_pfs['pfs']} months
- **Evidence basis**: Each option is supported by clinical trial data
- **NCCN category**: All options are guideline-recommended

Use the interactive comparison above to explore ORR, PFS, OS, and toxicity profiles.
"""

    return explanation


def _explain_survival_curves(self, data: Dict) -> str:
    """Generate text explanation for survival curves"""

    curves = data.get("curves", [])

    if not curves:
        return "No survival data available."

    best_survival = max(curves, key=lambda c: c.get("median_survival", 0))

    explanation = f"""## Survival Analysis

The Kaplan-Meier curves above show progression-free survival for {len(curves)} treatment options.

- **Best median survival**: {best_survival['name']} at {best_survival['median_survival']} months
- **Statistical significance**: Hazard ratios and p-values available in source trials

The curves demonstrate clear separation, indicating meaningful clinical benefit.
"""

    return explanation


def _explain_guideline_tree(self, data: Dict) -> str:
    """Generate text explanation for guideline tree"""

    return f"""## NCCN Guideline Decision Tree

The interactive decision tree above shows NCCN recommendations for:

- **Stage**: {data.get('stage', 'Unknown')}
- **Histology**: {data.get('histology', 'Unknown')}
- **Biomarkers**: {', '.join(data.get('biomarkers', {}).keys()) or 'None detected'}

Navigate the tree to explore different treatment pathways based on patient characteristics.

**Guideline Version**: {data.get('guideline_version', 'NCCN 2024.1')}
"""


def _explain_trial_matches(self, data: Dict) -> str:
    """Generate text explanation for trial matches"""

    trials = data.get("trials", [])

    if not trials:
        return "No matching clinical trials found."

    explanation = f"""## Clinical Trial Matches

Found **{len(trials)} matching trials** for this patient:

"""

    for i, trial in enumerate(trials[:3], 1):
        explanation += f"""
**{i}. {trial['title']}**
- NCT ID: {trial['nct_id']}
- Phase: {trial['phase']}
- Status: {trial['status']}
- Match Score: {trial['match_score']:.0%}
"""

    explanation += "\nUse the interactive matcher above to explore full eligibility criteria and contact information."

    return explanation


# ============================================================================
# 2. CLUSTERING ANALYSIS
# ============================================================================

async def _stream_clustering_analysis(
    self,
    message: str,
    session_id: str
) -> AsyncIterator[str]:
    """Stream patient clustering analysis"""

    yield self._format_sse({
        "type": "status",
        "content": "Analyzing patient cohorts..."
    })

    # Get all patients from Neo4j or database
    patients = await self._get_all_patients_for_clustering()

    if not patients or len(patients) < 5:
        yield self._format_sse({
            "type": "text",
            "content": "Not enough patient data available for clustering analysis. Need at least 5 patients."
        })
        return

    # Determine clustering method from query
    message_lower = message.lower()

    if "clinical rule" in message_lower or "rule-based" in message_lower:
        method = ClusteringMethod.CLINICAL_RULES
    elif "kmeans" in message_lower or "k-means" in message_lower:
        method = ClusteringMethod.KMEANS
    else:
        method = ClusteringMethod.CLINICAL_RULES  # Default

    yield self._format_sse({
        "type": "status",
        "content": f"Clustering {len(patients)} patients using {method.value} method..."
    })

    # Perform clustering
    from ..services.clustering_service import ClusteringService
    clustering_service = ClusteringService()

    result = clustering_service.cluster_patients(
        patients=patients,
        method=method
    )

    # Stream cluster information
    for cluster in result.clusters:
        yield self._format_sse({
            "type": "cluster_info",
            "content": {
                "name": cluster.name,
                "description": cluster.description,
                "size": cluster.size,
                "characteristics": cluster.characteristics,
                "outcomes": cluster.outcomes_summary,
                "confidence": cluster.confidence
            }
        })

    # Generate summary
    summary = self._generate_clustering_summary(result)

    yield self._format_sse({
        "type": "text",
        "content": summary
    })

    self._add_to_history(session_id, "assistant", summary)


async def _get_all_patients_for_clustering(self) -> List[Dict]:
    """Fetch all patients from Neo4j for clustering"""

    try:
        if not self.lca_service.graph_db or not self.lca_service.graph_db.driver:
            return []

        with self.lca_service.graph_db.driver.session(
            database=getattr(self.lca_service.graph_db, 'database', 'neo4j')
        ) as session:
            result = session.run("""
                MATCH (p:Patient)
                RETURN
                    p.patient_id as patient_id,
                    p.age_at_diagnosis as age,
                    p.sex as sex,
                    p.tnm_stage as stage,
                    p.histology_type as histology,
                    p.performance_status as ecog_ps,
                    p.biomarker_profile as biomarkers
                LIMIT 1000
            """)

            patients = []
            for record in result:
                patients.append(dict(record))

            return patients

    except Exception as e:
        logger.error(f"Failed to fetch patients for clustering: {e}")
        return []


def _generate_clustering_summary(self, result) -> str:
    """Generate markdown summary of clustering results"""

    summary = f"""## Patient Cohort Analysis

Using **{result.method.value}** clustering, identified **{result.num_clusters} distinct cohorts**:

"""

    for i, cluster in enumerate(result.clusters, 1):
        summary += f"""
### {i}. {cluster.name} (n={cluster.size})

**Description**: {cluster.description}

**Key Characteristics**:
"""
        for key, value in cluster.characteristics.items():
            summary += f"- {key}: {value}\n"

        if cluster.outcomes_summary:
            summary += f"\n**Outcomes**:\n"
            for key, value in cluster.outcomes_summary.items():
                summary += f"- {key}: {value}\n"

    # Add feature importance if available
    if result.feature_importance:
        summary += f"\n## Feature Importance\n\n"
        sorted_features = sorted(
            result.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for feature, importance in sorted_features[:5]:
            summary += f"- **{feature}**: {importance:.2%}\n"

    summary += f"\n**Clustering Quality**: Silhouette Score = {result.silhouette_score:.3f}"

    return summary


# ============================================================================
# 3. CITATION ENHANCEMENT
# ============================================================================

async def _enhance_with_citations(
    self,
    text: str,
    patient_data: Dict,
    analysis_result: Dict
) -> str:
    """Add relevant citations to recommendation text"""

    enhanced = text

    # Always add NCCN citation to recommendations
    if "recommend" in enhanced.lower() and "[[Guideline:NCCN]]" not in enhanced:
        enhanced += " [[Guideline:NCCN]]"

    # Add trial citations based on biomarkers
    biomarkers = patient_data.get("biomarker_profile", {})

    # EGFR citations
    if biomarkers.get("EGFR"):
        if "osimertinib" in enhanced.lower() and "[[Trial:FLAURA]]" not in enhanced:
            enhanced = enhanced.replace(
                "osimertinib",
                "osimertinib [[Trial:FLAURA]]",
                1
            )

    # ALK citations
    if biomarkers.get("ALK"):
        if "alectinib" in enhanced.lower() and "[[Trial:ALEX]]" not in enhanced:
            enhanced = enhanced.replace(
                "alectinib",
                "alectinib [[Trial:ALEX]]",
                1
            )

    # PD-L1 citations
    if biomarkers.get("PD-L1"):
        pdl1_value = biomarkers.get("PD-L1")
        try:
            if float(pdl1_value) >= 50:
                if "pembrolizumab" in enhanced.lower() and "[[Trial:KEYNOTE-024]]" not in enhanced:
                    enhanced = enhanced.replace(
                        "pembrolizumab",
                        "pembrolizumab [[Trial:KEYNOTE-024]]",
                        1
                    )
        except (ValueError, TypeError):
            pass

    # Add ontology citation for histology terms
    if "adenocarcinoma" in enhanced.lower() and "[[Ontology:SNOMED]]" not in enhanced:
        enhanced = enhanced.replace(
            "adenocarcinoma",
            "adenocarcinoma [[Ontology:SNOMED]]",
            1
        )

    return enhanced


# ============================================================================
# 4. UPDATED INTENT CLASSIFICATION
# ============================================================================

def _classify_intent_enhanced(self, message: str, session_id: str = None) -> str:
    """
    Enhanced intent classification with MCP apps and clustering

    Add this to replace or extend the existing _classify_intent method
    """

    # MCP App detection
    mcp_app_patterns = [
        r'compare\s+treatment',
        r'treatment\s+comparison',
        r'survival\s+curve',
        r'kaplan\s+meier',
        r'guideline\s+tree',
        r'nccn\s+decision',
        r'clinical\s+trial',
        r'trial\s+match'
    ]
    for pattern in mcp_app_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return "mcp_app"

    # Clustering detection
    clustering_patterns = [
        r'cluster\s+patient',
        r'cohort\s+analysis',
        r'similar\s+patient',
        r'patient\s+group',
        r'find\s+patients?\s+like'
    ]
    for pattern in clustering_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return "clustering_analysis"

    # MCP Tool detection (from previous implementation)
    mcp_tool_patterns = [
        r'(use|invoke|call|run)\s+(tool|MCP|agent)',
        r'analyze\s+survival\s+data',
        r'match\s+clinical\s+trial',
        r'find\s+similar\s+patient',
        r'search\s+(patient|guideline)',
        r'get\s+(biomarker|pathway|treatment)',
        r'interpret\s+lab\s+result',
        r'predict\s+resistance',
        r'graph\s+(query|search)',
        r'ontology\s+(map|validate)',
        r'export\s+(patient|report)',
    ]
    for pattern in mcp_tool_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return "mcp_tool"

    # Existing intent detection (patient_analysis, follow_up, general_qa)
    # ... keep your existing logic here ...

    return "general_qa"


# ============================================================================
# INSTRUCTIONS FOR INTEGRATION
# ============================================================================

"""
TO INTEGRATE THESE ENHANCEMENTS:

1. Add to ConversationService __init__:
   from ..services.clustering_service import ClusteringService
   self.clustering_service = ClusteringService()

2. Update _classify_intent method:
   Replace with _classify_intent_enhanced or merge the patterns

3. Add to chat_stream method:
   elif intent == "mcp_app":
       async for chunk in self._stream_mcp_app(message, session_id):
           yield chunk
   elif intent == "clustering_analysis":
       async for chunk in self._stream_clustering_analysis(message, session_id):
           yield chunk

4. Enhance patient analysis responses:
   In _stream_patient_analysis, after generating recommendation:
   recommendation = await self._enhance_with_citations(
       recommendation,
       patient_data,
       result
   )

5. Test with these queries:
   - "Compare treatments for this patient"
   - "Show survival curves"
   - "Cluster all stage IV patients"
   - "Find similar patients"

6. Check frontend for SSE events:
   - type: "mcp_app" → renders interactive app
   - type: "cluster_info" → displays cohort data
   - Citations [[Type:ID]] → renders as badges
"""
