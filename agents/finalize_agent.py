from memory.stm import STM_MEMORY
from memory.ltm import save_case_to_ltm


def finalize_agent(state):

    state["final_output"] = {

        "priority": state["priority"],

        "risk_score": state["risk_score"],

        "confidence": state["confidence"],

        "needs_review": state["needs_review"],

        "reasoning": state["reasoning"]
    }

    STM_MEMORY.append({

        "symptoms": state["symptoms"],

        "priority": state["priority"],

        "risk_score": state["risk_score"]
    })

    save_case_to_ltm(state)

    return state
