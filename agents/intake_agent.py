def intake_agent(state):

    state["patient_profile"] = {

        "symptoms": state["symptoms"],

        "age": state["patient_history"].get("age"),

        "gender": state["patient_history"].get("gender"),

        "conditions": state["patient_history"].get(
            "conditions",
            []
        ),

        "labs": state["labs"]
    }

    return state
