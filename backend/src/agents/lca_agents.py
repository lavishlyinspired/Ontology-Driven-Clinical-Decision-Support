"""
LangGraph Agent Implementation for Lung Cancer Assistant
Uses Ollama for local LLM inference
Based on the agentic workflow from the implementation plan
"""

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Annotated, Sequence, Dict, Any, List
from pydantic import BaseModel, Field
import operator
import os

# Centralized logging
from ..logging_config import get_logger, log_agent_action

logger = get_logger(__name__)


# ========================================
# State Definitions
# ========================================

class PatientState(TypedDict):
    """State for patient classification workflow"""
    patient_id: str
    patient_data: Dict[str, Any]
    applicable_rules: List[Dict[str, Any]]
    treatment_recommendations: List[Dict[str, Any]]
    arguments: List[Dict[str, Any]]
    explanation: str
    messages: Annotated[Sequence[Any], operator.add]


# ========================================
# Structured Outputs
# ========================================

class TreatmentArgument(BaseModel):
    """Structured argument for/against a treatment"""
    claim: str = Field(description="The main assertion")
    evidence: str = Field(description="Supporting data or guideline reference")
    strength: str = Field(description="Strong, Moderate, or Weak")
    argument_type: str = Field(description="Supporting or Opposing")


class TreatmentRecommendationOutput(BaseModel):
    """Structured treatment recommendation"""
    treatment: str
    rationale: str
    evidence_level: str
    survival_impact: str
    rank: int


# ========================================
# Agent Classes
# ========================================

class PatientClassificationAgent:
    """
    Agent for classifying patients into guideline-based scenarios.
    Implements ontological inference pattern from paper.
    """

    def __init__(self, model: str = None):
        """
        Initialize classification agent.

        Args:
            model: Ollama model name (default from env: OLLAMA_MODEL)
        """
        model_name = model or os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.1,  # Low temperature for consistent clinical classification
        )

        self.system_prompt = """You are an expert clinical oncologist specializing in lung cancer treatment.

Your role is to classify patients according to NICE Lung Cancer Guidelines (CG121) and contemporary standards.

# Patient Classification Criteria

**TNM Stages:**
- Stage I (IA, IB): Localized tumor, no lymph node involvement
- Stage II (IIA, IIB): Larger tumor or limited lymph node involvement
- Stage III (IIIA, IIIB): Locally advanced, significant lymph node involvement
- Stage IV: Metastatic disease

**Histology Types:**
- NSCLC (Non-Small Cell Lung Cancer): Adenocarcinoma, Squamous Cell, Large Cell
- SCLC (Small Cell Lung Cancer): Aggressive, high metastatic potential

**WHO Performance Status:**
- 0: Fully active, no restrictions
- 1: Restricted in physically strenuous activity but ambulatory
- 2: Ambulatory, capable of self-care, unable to work
- 3: Capable of only limited self-care
- 4: Completely disabled

# Guideline Rules

R1: Chemotherapy for Stage III-IV NSCLC with PS 0-1
R2: Surgery for Stage I-II NSCLC with PS 0-1 (curative)
R3: Radiotherapy for Stage I-IIIA NSCLC with PS 0-2 (unsuitable for surgery)
R4: Palliative Care for Stage IIIB-IV with PS 3-4
R5: Chemotherapy for SCLC with PS 0-2
R6: Chemoradiotherapy for Stage IIIA/IIIB NSCLC with PS 0-1
R7: Immunotherapy for Stage IIIB-IV NSCLC with PS 0-1 (contemporary)

Analyze each patient carefully and identify ALL applicable guideline rules."""

    def classify(self, state: PatientState) -> PatientState:
        """Classify patient and update state with applicable rules"""
        patient_data = state["patient_data"]

        prompt = f"""Classify this lung cancer patient and identify applicable treatment guidelines:

**Patient Profile:**
- Age: {patient_data.get('age')} years
- Sex: {patient_data.get('sex')}
- TNM Stage: {patient_data.get('tnm_stage')}
- Histology: {patient_data.get('histology_type')}
- Performance Status: WHO {patient_data.get('performance_status')}
- FEV1: {patient_data.get('fev1_percent', 'Not recorded')}%
- Comorbidities: {', '.join(patient_data.get('comorbidities', [])) or 'None recorded'}

**Task:**
List ALL applicable guideline rules (R1-R7) with brief justification for each.
Format: "R#: [Treatment] - [One-sentence justification]"

Also identify any contraindications or special considerations."""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]

        try:
            response = self.llm.invoke(messages)
            state["messages"] = state.get("messages", []) + [
                HumanMessage(content=prompt),
                response
            ]
            logger.info(f"✓ Patient {state['patient_id']} classified")
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"Classification error: {str(e)}")
            ]

        return state


class TreatmentRecommendationAgent:
    """
    Agent for generating ranked treatment recommendations.
    Uses evidence hierarchy and clinical guidelines.
    """

    def __init__(self, model: str = None):
        model_name = model or os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.2,
        )

        self.system_prompt = """You are a clinical decision support system for lung cancer treatment.

Generate evidence-based treatment recommendations following clinical evidence hierarchy:

**Evidence Levels:**
- Grade A: High-quality randomized controlled trials, meta-analyses
- Grade B: Well-designed cohort studies, case-control studies
- Grade C: Expert opinion, case series

**Treatment Modalities:**
1. **Surgery**: Curative for early-stage (I-II) NSCLC
   - Lobectomy, pneumonectomy
   - Best outcomes: 5-year survival 60-80% for Stage I

2. **Radiotherapy**: Curative for inoperable early-stage
   - SABR/SBRT for Stage I: 40-50% 5-year survival
   - Radical RT for Stage II-III

3. **Chemotherapy**: Standard for advanced disease
   - Platinum-based doublets
   - 3-4 months median survival benefit

4. **Chemoradiotherapy**: Standard for Stage III
   - Concurrent > sequential
   - 15-20% 5-year survival

5. **Immunotherapy**: First-line for high PD-L1
   - Median survival >12 months vs 9 months chemo
   - Better long-term outcomes

6. **Palliative Care**: Focus on quality of life
   - For PS 3-4 or patient preference

For each recommendation, provide:
- Treatment type
- Evidence level (Grade A/B/C)
- Expected survival benefit
- Key contraindications
- Rank order (1 = highest priority)"""

    def recommend(self, state: PatientState) -> PatientState:
        """Generate ranked treatment recommendations"""
        applicable_rules = state.get("applicable_rules", [])
        patient_data = state["patient_data"]

        # Extract rule information
        rule_summary = "\n".join([
            f"- {r.get('rule_id')}: {r.get('recommended_treatment')} "
            f"({r.get('evidence_level')}, {r.get('treatment_intent')} intent)"
            for r in applicable_rules
        ])

        prompt = f"""Generate ranked treatment recommendations for this patient:

**Patient:**
- Stage: {patient_data.get('tnm_stage')}
- Histology: {patient_data.get('histology_type')}
- Performance Status: WHO {patient_data.get('performance_status')}
- Age: {patient_data.get('age')}

**Applicable Guidelines:**
{rule_summary if rule_summary else "No specific guidelines matched"}

**Task:**
Provide 3-5 ranked treatment recommendations. For each:
1. Treatment name
2. Rationale (2-3 sentences)
3. Evidence level
4. Expected survival benefit
5. Key contraindications

Format as a numbered list, ranked by clinical appropriateness."""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]

        try:
            response = self.llm.invoke(messages)
            state["messages"] = state.get("messages", []) + [response]
            logger.info(f"✓ Treatment recommendations generated")
        except Exception as e:
            logger.error(f"Recommendation failed: {e}")

        return state


class ArgumentationAgent:
    """
    Agent for generating clinical arguments (pro/con).
    Implements the Argument → supports/opposes → Decision pattern.
    """

    def __init__(self, model: str = None):
        model_name = model or os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.3,
        )

        self.system_prompt = """You are a clinical argumentation expert for multidisciplinary team (MDT) meetings.

Generate balanced clinical arguments for and against each treatment option.

**Argument Structure:**
- **Claim**: Main assertion (1 sentence)
- **Evidence**: Supporting data, guideline reference, or clinical trial
- **Strength**: Strong/Moderate/Weak

**Argument Types:**
1. **Supporting Arguments** (reasons to recommend):
   - Guideline compliance
   - Survival benefit
   - Quality of life improvement
   - Patient fitness/suitability

2. **Opposing Arguments** (contraindications/risks):
   - Comorbidities
   - Age-related risks
   - Treatment toxicity
   - Patient preference
   - Logistical barriers

Provide 2-3 supporting and 1-2 opposing arguments per treatment option.
Be objective and evidence-based."""

    def generate_arguments(self, state: PatientState) -> PatientState:
        """Generate arguments for each treatment recommendation"""
        recommendations = state.get("treatment_recommendations", [])
        patient_data = state["patient_data"]

        arguments_list = []

        # Generate arguments for top 3 recommendations
        for rec in recommendations[:3]:
            treatment = rec.get("recommended_treatment", "treatment")

            prompt = f"""Generate clinical arguments for {treatment} in this patient:

**Patient Profile:**
- Age: {patient_data.get('age')}
- Stage: {patient_data.get('tnm_stage')}
- PS: {patient_data.get('performance_status')}
- Histology: {patient_data.get('histology_type')}
- FEV1: {patient_data.get('fev1_percent', 'unknown')}%

**Treatment:** {treatment}
**Evidence Level:** {rec.get('evidence_level')}
**Survival Benefit:** {rec.get('survival_benefit', 'See literature')}

**Task:**
Provide:
1. 2-3 SUPPORTING arguments (why to recommend)
2. 1-2 OPPOSING arguments (contraindications/concerns)

Format each argument as:
- Claim: [assertion]
- Evidence: [data/reference]
- Strength: [Strong/Moderate/Weak]"""

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]

            try:
                response = self.llm.invoke(messages)
                arguments_list.append({
                    "treatment": treatment,
                    "arguments": response.content
                })
            except Exception as e:
                logger.error(f"Argumentation failed for {treatment}: {e}")

        state["arguments"] = arguments_list
        logger.info(f"✓ Generated arguments for {len(arguments_list)} treatments")
        return state


class ExplanationAgent:
    """
    Agent for generating MDT-ready explanations.
    Synthesizes all information into clinician-friendly summary.
    """

    def __init__(self, model: str = None):
        model_name = model or os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.4,
        )

        self.system_prompt = """You are an expert clinical oncologist preparing a case summary for a multidisciplinary team (MDT) meeting.

Generate a concise, structured case summary suitable for oncology clinicians.

**MDT Summary Structure:**

1. **Patient Summary** (2-3 lines)
   - Key demographics and diagnosis

2. **Clinical Classification** (2-3 lines)
   - Stage, histology, performance status
   - Applicable guideline rules

3. **Treatment Recommendations** (ranked list)
   - Top 3 options with brief rationale
   - Evidence level for each

4. **Key Arguments**
   - Main supporting factors
   - Main concerns/contraindications

5. **Discussion Points** (bullet points)
   - Critical decision factors
   - Questions for MDT consideration
   - Patient preference considerations

Keep the summary professional, concise, and clinically actionable.
Target length: 300-400 words."""

    def explain(self, state: PatientState) -> PatientState:
        """Generate final MDT explanation"""

        # Gather all context
        patient_data = state["patient_data"]
        rules = state.get("applicable_rules", [])
        arguments = state.get("arguments", [])

        rules_summary = ", ".join([r.get("rule_id", "") for r in rules[:3]])
        treatments_summary = "\n".join([
            f"{i+1}. {r.get('recommended_treatment')} "
            f"({r.get('evidence_level')}, {r.get('treatment_intent')})"
            for i, r in enumerate(rules[:3])
        ])

        prompt = f"""Generate an MDT case summary for this lung cancer patient:

**Patient:**
- ID: {state['patient_id']}
- Age: {patient_data.get('age')}, Sex: {patient_data.get('sex')}
- Stage: {patient_data.get('tnm_stage')}
- Histology: {patient_data.get('histology_type')}
- Performance Status: WHO {patient_data.get('performance_status')}

**Applicable Guidelines:**
{rules_summary or 'None matched'}

**Top Recommendations:**
{treatments_summary or 'See detailed analysis'}

**Task:**
Generate a structured MDT summary following the template:
1. Patient Summary
2. Clinical Classification
3. Treatment Recommendations (ranked)
4. Key Arguments (pro/con)
5. Discussion Points for MDT

Be concise and clinically focused."""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]

        try:
            response = self.llm.invoke(messages)
            state["explanation"] = response.content
            logger.info(f"✓ MDT summary generated for patient {state['patient_id']}")
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            state["explanation"] = f"Error generating explanation: {str(e)}"

        return state


# ========================================
# Workflow Definition
# ========================================

def create_lca_workflow():
    """
    Create the main LCA decision support workflow using LangGraph.

    Workflow: Classify → Recommend → Argue → Explain
    """
    logger.info("Creating LCA workflow...")

    # Initialize agents
    classifier = PatientClassificationAgent()
    recommender = TreatmentRecommendationAgent()
    argumenter = ArgumentationAgent()
    explainer = ExplanationAgent()

    # Create workflow graph
    workflow = StateGraph(PatientState)

    # Add nodes
    workflow.add_node("classify_patient", classifier.classify)
    workflow.add_node("recommend_treatment", recommender.recommend)
    workflow.add_node("generate_arguments", argumenter.generate_arguments)
    workflow.add_node("explain", explainer.explain)

    # Define edges
    workflow.set_entry_point("classify_patient")
    workflow.add_edge("classify_patient", "recommend_treatment")
    workflow.add_edge("recommend_treatment", "generate_arguments")
    workflow.add_edge("generate_arguments", "explain")
    workflow.add_edge("explain", END)

    logger.info("✓ LCA workflow created")
    return workflow.compile()


# ========================================
# Test Function
# ========================================

async def test_workflow():
    """Test the LCA workflow with a sample patient"""
    print("\n" + "=" * 80)
    print("TESTING LCA WORKFLOW WITH OLLAMA")
    print("=" * 80)

    # Sample patient
    test_patient = {
        "patient_id": "TEST_001",
        "name": "John Doe",
        "age": 68,
        "sex": "M",
        "tnm_stage": "IIIA",
        "histology_type": "Adenocarcinoma",
        "performance_status": 1,
        "fev1_percent": 65.0,
        "comorbidities": ["COPD", "Hypertension"]
    }

    print(f"\nTest Patient: {test_patient['name']}")
    print(f"  Stage: {test_patient['tnm_stage']}")
    print(f"  Histology: {test_patient['histology_type']}")
    print(f"  PS: WHO {test_patient['performance_status']}")

    # Mock rules (would come from GuidelineRuleEngine)
    mock_rules = [
        {
            "rule_id": "R6",
            "recommended_treatment": "Chemoradiotherapy",
            "evidence_level": "Grade A",
            "treatment_intent": "Curative"
        },
        {
            "rule_id": "R3",
            "recommended_treatment": "Radiotherapy",
            "evidence_level": "Grade B",
            "treatment_intent": "Curative"
        }
    ]

    # Initialize state
    initial_state: PatientState = {
        "patient_id": test_patient["patient_id"],
        "patient_data": test_patient,
        "applicable_rules": mock_rules,
        "treatment_recommendations": mock_rules,
        "arguments": [],
        "explanation": "",
        "messages": []
    }

    # Run workflow
    workflow = create_lca_workflow()

    print("\n" + "-" * 80)
    print("Running workflow...")
    print("-" * 80)

    try:
        final_state = workflow.invoke(initial_state)

        print("\n" + "=" * 80)
        print("MDT SUMMARY")
        print("=" * 80)
        print(final_state["explanation"])
        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n✗ Workflow failed: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_workflow())
