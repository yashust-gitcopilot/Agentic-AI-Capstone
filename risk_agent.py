def risk_agent(state):

    symptoms = state["symptoms"].lower()

    labs = state["labs"]

    score = 30

    # all your existing rules

    state["risk_score"] = score
    state["priority"] = priority

    return state
