from memory.stm import STM_MEMORY
from memory.ltm import retrieve_ltm_cases


def memory_agent(state):

    state["stm_cases"] = list(STM_MEMORY)

    state["ltm_cases"] = retrieve_ltm_cases()

    return state
