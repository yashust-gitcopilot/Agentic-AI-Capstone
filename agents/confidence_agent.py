def confidence_agent(state):

    confidence = 95

    if len(state["retrieved_docs"]) < 3:
        confidence -= 20

    if state["priority"] == "P1":
        confidence -= 10

    if len(state["ltm_cases"]) == 0:
        confidence -= 5

    state["confidence"] = confidence

    state["needs_review"] = confidence < 80

    return state
