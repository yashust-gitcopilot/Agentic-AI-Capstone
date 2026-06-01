import json
import os

from graph.workflow import build_graph
from rag.build_vector_db import build_vector_db
from config.settings import FAISS_DB_PATH


if not os.path.exists(FAISS_DB_PATH):
    build_vector_db()

graph = build_graph()

patient_history = {
    "age": 65,
    "gender": "Male",
    "conditions": [
        "Diabetes",
        "Hypertension"
    ]
}

labs = {
    "troponin": 450,
    "bp_systolic": 190,
    "wbc": 18000,
    "oxygen": 88
}

result = graph.invoke({

    "symptoms":
    """
    Severe chest pain
    radiating to left arm
    with shortness of breath
    """,

    "patient_history": patient_history,

    "labs": labs
})

print(
    json.dumps(
        result["final_output"],
        indent=4
    )
)
