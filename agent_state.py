from typing import TypedDict


class TriageState(TypedDict):

    symptoms: str

    patient_history: dict

    labs: dict

    patient_profile: dict

    retrieved_docs: list

    stm_cases: list

    ltm_cases: list

    priority: str

    risk_score: int

    reasoning: str

    confidence: float

    needs_review: bool

    final_output: dict
