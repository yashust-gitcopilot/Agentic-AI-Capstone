from rag.load_vector_db import load_vector_db


def rag_agent(state):

    vectordb = load_vector_db()

    query = f"""
Symptoms:
{state['symptoms']}

History:
{state['patient_history']}

Labs:
{state['labs']}
"""

    docs = vectordb.similarity_search(
        query,
        k=5
    )

    state["retrieved_docs"] = [
        d.page_content
        for d in docs
    ]

    return state
