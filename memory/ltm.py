import json
import os

from config.settings import MEMORY_FILE


def save_case_to_ltm(state):

    case = {
        "symptoms": state["symptoms"],
        "priority": state["priority"],
        "risk_score": state["risk_score"],
        "reasoning": state["reasoning"]
    }

    if os.path.exists(MEMORY_FILE):

        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)

    else:
        data = []

    data.append(case)

    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=4)


def retrieve_ltm_cases():

    if not os.path.exists(MEMORY_FILE):
        return []

    with open(MEMORY_FILE, "r") as f:
        return json.load(f)[-3:]
