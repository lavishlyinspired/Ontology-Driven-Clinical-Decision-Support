"""
FHIR R4 Terminology API Routes
===============================

Implements FHIR R4 terminology operations as REST endpoints:
- GET /fhir/CodeSystem/$lookup — Look up a concept
- GET /fhir/ConceptMap/$translate — Translate between systems
- GET /fhir/ValueSet/$expand — Expand a value set
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from ...logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["FHIR Terminology"])


@router.get("/fhir/CodeSystem/$lookup")
async def codesystem_lookup(
    system: str = Query(
        ...,
        description="Code system URI (e.g., http://snomed.info/sct, http://loinc.org)",
        examples=["http://snomed.info/sct"],
    ),
    code: str = Query(
        ...,
        description="Concept code (e.g., 254637007 for NSCLC)",
        examples=["254637007"],
    ),
):
    """
    FHIR CodeSystem/$lookup operation.

    Looks up a concept by system and code in the loaded ontologies.

    Supported systems:
    - http://snomed.info/sct (SNOMED-CT → SNOMEDConcept nodes)
    - http://loinc.org (LOINC)
    - http://www.nlm.nih.gov/research/umls/rxnorm (RxNorm)
    - http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl (NCIt)

    Returns:
        FHIR Parameters resource with code, display name, and version.
    """
    try:
        from ...services.fhir_terminology_service import get_fhir_terminology_service

        svc = get_fhir_terminology_service()
        result = svc.lookup(system, code)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CodeSystem/$lookup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fhir/ConceptMap/$translate")
async def conceptmap_translate(
    system: str = Query(
        ...,
        description="Source code system URI",
        examples=["http://snomed.info/sct"],
    ),
    code: str = Query(
        ...,
        description="Source concept code",
        examples=["254637007"],
    ),
    target: str = Query(
        ...,
        alias="target",
        description="Target code system URI",
        examples=["http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl"],
    ),
):
    """
    FHIR ConceptMap/$translate operation.

    Translates a concept from one code system to another using the UMLS crosswalk.

    Example: Translate SNOMED 254637007 (NSCLC) → NCIt code

    Returns:
        FHIR Parameters resource with translated code.
    """
    try:
        from ...services.fhir_terminology_service import get_fhir_terminology_service

        svc = get_fhir_terminology_service()
        result = svc.translate(system, code, target)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ConceptMap/$translate failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fhir/ValueSet/$expand")
async def valueset_expand(
    url: str = Query(
        ...,
        description="Value set URL or short name (e.g., lung-cancer-histologies)",
        examples=["lung-cancer-histologies"],
    ),
):
    """
    FHIR ValueSet/$expand operation.

    Expands a predefined value set. SNOMED-based value sets query Neo4j
    for descendant concepts.

    Available value sets:
    - lung-cancer-histologies: NSCLC histology subtypes from SNOMED
    - lung-cancer-biomarkers: Actionable biomarkers for lung cancer

    Returns:
        FHIR ValueSet resource with expansion.
    """
    try:
        from ...services.fhir_terminology_service import get_fhir_terminology_service

        svc = get_fhir_terminology_service()
        result = svc.expand(url)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ValueSet/$expand failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
