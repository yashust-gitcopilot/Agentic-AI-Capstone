import json

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from config.llm import llm

SYSTEM_PROMPT = """
You are an expert emergency triage physician.
"""


def reasoning_agent(state):

    # your existing prompt

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    state["reasoning"] = response.content

    return state
