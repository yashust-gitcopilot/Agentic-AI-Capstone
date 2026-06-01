def route_review(state):

    if state["priority"] == "P1":
        return "review"

    if state["confidence"] < 80:
        return "review"

    return "final"
