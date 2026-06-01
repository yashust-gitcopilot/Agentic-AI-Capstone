from langgraph.graph import (
    StateGraph,
    END
)

from graph.state import TriageState
from graph.router import route_review

from agents.intake_agent import intake_agent
from agents.memory_agent import memory_agent
from agents.rag_agent import rag_agent
from agents.risk_agent import risk_agent
from agents.reasoning_agent import reasoning_agent
from agents.confidence_agent import confidence_agent
from agents.review_agent import review_agent
from agents.finalize_agent import finalize_agent


def build_graph():

    builder = StateGraph(TriageState)

    builder.add_node("intake", intake_agent)
    builder.add_node("memory", memory_agent)
    builder.add_node("rag", rag_agent)
    builder.add_node("risk", risk_agent)
    builder.add_node("reasoning", reasoning_agent)
    builder.add_node("confidence", confidence_agent)
    builder.add_node("review", review_agent)
    builder.add_node("final", finalize_agent)

    builder.set_entry_point("intake")

    builder.add_edge("intake", "memory")
    builder.add_edge("memory", "rag")
    builder.add_edge("rag", "risk")
    builder.add_edge("risk", "reasoning")
    builder.add_edge("reasoning", "confidence")

    builder.add_conditional_edges(
        "confidence",
        route_review,
        {
            "review": "review",
            "final": "final"
        }
    )

    builder.add_edge("review", "final")
    builder.add_edge("final", END)

    return builder.compile()
