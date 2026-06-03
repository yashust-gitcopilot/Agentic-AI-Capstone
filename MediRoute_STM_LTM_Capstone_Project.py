# ==========================================================
# COLAB INSTALLS
# ==========================================================

!pip install -q langchain
!pip install -q langchain-community
!pip install -q langchain-groq
!pip install -q langgraph
!pip install -q sentence-transformers
!pip install -q pypdf
!pip install -q pandas
!pip install -q python-dotenv
!pip install -q langchain-huggingface
!pip install -q langchain-text-splitters
!pip install -q faiss-cpu

# ==========================================================
# IMPORTS
# ==========================================================

import os
import json

from collections import deque

from typing import TypedDict

from langchain_core.documents import Document

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from langchain_groq import ChatGroq

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    FAISS
)

from langgraph.graph import (
    StateGraph,
    END,
    START # Import START node
)

# ==========================================================
# API KEY
# ==========================================================

os.environ["GROQ_API_KEY"] = "GROQ_API_KEY"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==========================================================
# LLM
# ==========================================================

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# ==========================================================
# EMBEDDING MODEL
# ==========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================================
# PATHS
# ==========================================================

FAISS_DB_PATH = "/content/faiss_db"

MEMORY_FILE = "/content/patient_memory.json"

# ==========================================================
# STM
# ==========================================================)

STM_MEMORY = deque(maxlen=10)

# ==========================================================
# BUILD FAISS DATABASE
# ==========================================================

def build_vector_db():

    documents = []

    pdf_folder = "./"

    for file in os.listdir(pdf_folder):

        if file.endswith(".pdf"):

            print(f"Loading {file}")

            loader = PyPDFLoader(
                os.path.join(pdf_folder, file)
            )

            docs = loader.load()

            for d in docs:

                d.metadata["source_file"] = file

            documents.extend(docs)

    if len(documents) == 0:

        print("No PDFs found")

        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    vectordb = FAISS.from_documents(
        chunks,
        embedding_model
    )

    vectordb.save_local(
        FAISS_DB_PATH
    )

    print("FAISS Database Created")

    return vectordb


# ==========================================================
# LOAD FAISS
# ==========================================================

def load_vector_db():

    return FAISS.load_local(
        FAISS_DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

# ==========================================================
# SAVE CASE TO LTM
# ==========================================================

def save_case_to_ltm(state):

    case = {

        "symptoms":
            state["symptoms"],

        "priority":
            state["priority"],

        "risk_score":
            state["risk_score"],

        "reasoning":
            state["reasoning"]
    }

    if os.path.exists(MEMORY_FILE):

        with open(
            MEMORY_FILE,
            "r"
        ) as f:

            data = json.load(f)

    else:

        data = []

    data.append(case)

    with open(
            MEMORY_FILE,
            "w"
        ) as f:

            json.dump(
                data,
                f,
                indent=4
            )


# ==========================================================
# RETRIEVE LTM
# ==========================================================

def retrieve_ltm_cases(state):

    if not os.path.exists(
        MEMORY_FILE
    ):

        return []

    with open(
        MEMORY_FILE,
        "r"
    ) as f:

        data = json.load(f)

    return data[-3:]

# ==========================================================
# GRAPH STATE
# ==========================================================

class TriageState(TypedDict):

    symptoms: str

    patient_history: dict

    labs: dict

    patient_profile: dict

    retrieved_docs: list

    stm_cases: list

    ltm_cases: list

    priority: str

    risk_score: int

    reasoning: str

    confidence: float

    needs_review: bool

    final_output: dict

    query_response: str # New field for query responses
    next_step_decision: str # To store the routing decision

# ==========================================================
# ROUTER: CLASSIFY INPUT (SYMPTOM OR QUERY)
# ==========================================================

def route_input(state):
    symptoms_input = state["symptoms"].strip().lower()
    query_phrases = [
        "what is my last symptom?",
        "what is my last symptoms?",
        "what was the risk score?",
        "what should i do for this disease?",
        "explain the reasoning",
        "tell me more about"
    ]

    for phrase in query_phrases:
        if phrase in symptoms_input:
            print(f"✓ Input Classifier: Detected query '{symptoms_input}'. Routing to Query Agent.")
            return "query_agent"

    print(f"✓ Input Classifier: Detected symptoms '{symptoms_input}'. Routing to Intake Agent.")
    return "intake"


# ==========================================================
# AGENT 0: QUERY AGENT
# ==========================================================

def query_agent(state):
    query = state["symptoms"]
    response_text = "I'm sorry, I cannot answer that query from the available information."

    # Attempt to answer 'what is my last symptom?'
    if "what is my last symptom?" in query.lower() or "what is my last symptoms?" in query.lower():
        if STM_MEMORY:
            last_triage = STM_MEMORY[-1]
            if "symptoms" in last_triage:
                response_text = f"Your last recorded symptom was: {last_triage['symptoms']}"
            else:
                response_text = "No specific symptoms found in the last STM entry."
        else:
            response_text = "No previous symptoms found in short-term memory."

    # Attempt to answer 'what was the risk score?'
    elif "what was the risk score?" in query.lower():
        if STM_MEMORY:
            last_triage = STM_MEMORY[-1]
            if "risk_score" in last_triage:
                response_text = f"The risk score for the last triage was: {last_triage['risk_score']}"
            else:
                response_text = "No risk score found in the last STM entry."
        else:
            response_text = "No previous triage data found in short-term memory."

    # Attempt to answer 'explain the reasoning' or similar
    elif "explain the reasoning" in query.lower() or "tell me more about" in query.lower():
        if STM_MEMORY:
            last_triage = STM_MEMORY[-1]
            if "final_output" in last_triage and "reasoning" in last_triage["final_output"]:
                response_text = f"Here is the reasoning from the last triage: {last_triage['final_output']['reasoning']}"
            else:
                response_text = "No detailed reasoning found in the last STM entry."
        else:
            response_text = "No previous triage data found to explain reasoning."

    # General LLM query if specific patterns not matched but it's clearly a query
    elif query.strip() != "": # If it's not empty, try a general LLM response
        try:
            # Construct a prompt for the LLM to answer the query based on available STM/LTM
            stm_context = f"Recent patient symptoms: {STM_MEMORY[-1]['symptoms'] if STM_MEMORY else 'None'}"
            ltm_context = f"Historical cases: {state['ltm_cases']}"
            llm_query_prompt = f"Given the following context: {stm_context}. {ltm_context}. User asks: {query}. Provide a concise answer."

            llm_response = llm.invoke([
                SystemMessage(content="You are a helpful assistant providing information based on patient data."),
                HumanMessage(content=llm_query_prompt)
            ])
            response_text = llm_response.content
        except Exception as e:
            print(f"Error in general LLM query: {e}")
            response_text = "I encountered an error trying to process your general query."

    state["query_response"] = response_text
    print("✓ Query Agent Completed")
    return state


# ==========================================================
# AGENT 1 : INTAKE AGENT
# ==========================================================

def intake_agent(state):

    profile = {

        "symptoms":
            state["symptoms"],

        "age":
            state["patient_history"].get("age"),

        "gender":
            state["patient_history"].get("gender"),

        "conditions":
            state["patient_history"].get(
                "conditions",
                []
            ),

        "labs":
            state["labs"]
    }

    state["patient_profile"] = profile

    print("✓ Intake Agent Completed")

    return state


# ==========================================================
# AGENT 2 : MEMORY AGENT
# ==========================================================

def memory_agent(state):

    state["stm_cases"] = list(
        STM_MEMORY
    )

    state["ltm_cases"] = retrieve_ltm_cases(
        state
    )

    print(
        f"✓ STM Cases: "
        f"{len(state['stm_cases'])}"
    )

    print(
        f"✓ LTM Cases: "
        f"{len(state['ltm_cases'])}"
    )

    return state


# ==========================================================
# AGENT 3 : RAG AGENT
# ==========================================================

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

    print(
        f"✓ Retrieved "
        f"{len(state['retrieved_docs'])} "
        f"chunks"
    )

    return state

# ==========================================================
# AGENT 4 : RISK SCORING AGENT
# ==========================================================

def risk_agent(state):

    symptoms = (
        state["symptoms"]
        .lower()
    )

    labs = state["labs"]

    score = 30

    # ----------------------------------
    # SYMPTOMS
    # ----------------------------------

    if "chest pain" in symptoms:

        score += 30

    if "stroke" in symptoms:

        score += 50

    if "shortness of breath" in symptoms:

        score += 20

    if "unconscious" in symptoms:

        score += 50

    if "seizure" in symptoms:

        score += 40

    if "bleeding" in symptoms:

        score += 20

    # ----------------------------------
    # LABS
    # ----------------------------------

    if labs.get(
        "troponin",
        0
    ) > 100:

        score += 30

    if labs.get(
        "bp_systolic",
        0
    ) > 180:

        score += 20

    if labs.get(
        "oxygen",
        100
    ) < 90:

        score += 20

    if labs.get(
        "wbc",
        0
    ) > 15000:

        score += 10

    # ----------------------------------
    # PRIORITY
    # ----------------------------------

    if score >= 90:

        priority = "P1"

    elif score >= 70:

        priority = "P2"

    elif score >= 50:

        priority = "P3"

    else:

        priority = "P4"

    state["risk_score"] = score

    state["priority"] = priority

    print(
        f"✓ Risk Score: {score}"
    )

    print(
        f"✓ Priority: {priority}"
    )

    return state

# ==========================================================
# AGENT 5 : REASONING AGENT
# ==========================================================

SYSTEM_PROMPT = """
You are an expert emergency triage physician.

Analyze:

1. Symptoms
2. Medical history
3. Lab values
4. Retrieved medical guidelines
5. Similar historical patient cases

Provide:

- Clinical reasoning
- Risk assessment
- Supporting evidence
- Recommended triage action

Do not hallucinate.

Use retrieved evidence whenever possible.
"""

def reasoning_agent(state):

    docs = "\n\n".join(
        state["retrieved_docs"]
    )

    stm_cases = "\n\n".join(

        str(case)

        for case in state["stm_cases"]
    )

    ltm_cases = "\n\n".join(

        str(case)

        for case in state["ltm_cases"]
    )

    prompt = f"""
CURRENT PATIENT

=================================

Symptoms:

{state['symptoms']}

=================================

History:

{json.dumps(
    state['patient_history'],
    indent=2
)}

=================================

Labs:

{json.dumps(
    state['labs'],
    indent=2
)}

=================================

Retrieved Medical Guidelines

{docs}

=================================

Short Term Memory Cases

{stm_cases}

=================================

Long Term Memory Cases

{ltm_cases}

=================================

Assigned Priority

{state['priority']}

=================================

Provide:

1. Clinical Assessment

2. Risk Evaluation

3. Evidence Supporting Decision

4. Recommended Triage Action
"""

    response = llm.invoke([

        SystemMessage(
            content=SYSTEM_PROMPT
        ),

        HumanMessage(
            content=prompt
        )
    ])

    state["reasoning"] = response.content

    print(
        "✓ Clinical Reasoning Generated"
    )

    return state

# ==========================================================
# AGENT 6 : CONFIDENCE AGENT
# ==========================================================

def confidence_agent(state):

    confidence = 95

    # Few retrieved docs
    if len(
        state["retrieved_docs"]
    ) < 3:

        confidence -= 20

    # High risk case
    if state["priority"] == "P1":

        confidence -= 10

    # No historical cases
    if len(
        state["ltm_cases"]
    ) == 0:

        confidence -= 5

    confidence = max(
        confidence,
        0
    )

    state["confidence"] = confidence

    state["needs_review"] = (

        confidence < 80

    )

    print(
        f"✓ Confidence: {confidence}%"
    )

    return state


# ==========================================================
# AGENT 7 : HUMAN REVIEW AGENT
# ==========================================================

def review_agent(state):

    print(
        "\n⚠ HUMAN REVIEW REQUIRED\n"
    )

    state["needs_review"] = True

    return state


# ==========================================================
# AGENT 8 : FINAL AGENT
# ==========================================================

def finalize_agent(state):

    final_output_data = {
        "priority":
            state["priority"],
        "risk_score":
            state["risk_score"],
        "confidence":
            state["confidence"],
        "needs_review":
            state["needs_review"],
        "reasoning":
            state["reasoning"]
    }
    state["final_output"] = final_output_data

    # ----------------------------------
    # SAVE TO STM
    # ----------------------------------

    STM_MEMORY.append({
        "symptoms": state["symptoms"],
        "priority": state["priority"],
        "risk_score": state["risk_score"],
        "final_output": final_output_data # Store the full output
    })

    # ----------------------------------
    # SAVE TO LTM
    # ----------------------------------

    save_case_to_ltm(
        state
    )

    print(
        "✓ Saved To STM"
    )

    print(
        state["symptoms"]
    )

    print(
        "✓ Saved To LTM"
    )

    print(
        "✓ Final Output Ready"
    )

    return state

# ==========================================================
# ROUTER (FOR TRIAGE REVIEW)
# ==========================================================

def route_review(state):

    # Very High Risk

    if state["priority"] == "P1":

        return "review"

    # Low Confidence

    if state["confidence"] < 80:

        return "review"

    return "final"

# ==========================================================
# DEDICATED INITIAL ROUTER NODE
# ==========================================================

def initial_router_node(state):
    next_step = route_input(state)
    return {"next_step_decision": next_step}

# ==========================================================
# BUILD LANGGRAPH
# ==========================================================

builder = StateGraph(
    TriageState
)

# ----------------------------------
# NODES
# ----------------------------------

builder.add_node(
    "initial_router_node",
    initial_router_node
)

builder.add_node(
    "query_agent",
    query_agent
)

builder.add_node(
    "intake",
    intake_agent
)

builder.add_node(
    "memory",
    memory_agent
)

builder.add_node(
    "rag",
    rag_agent
)

builder.add_node(
    "risk",
    risk_agent
)

builder.add_node(
    "reasoning",
    reasoning_agent
)

builder.add_node(
    "confidence",
    confidence_agent
)

builder.add_node(
    "review",
    review_agent
)

builder.add_node(
    "final",
    finalize_agent
)

# ----------------------------------
# ENTRY & ROUTING
# ----------------------------------

builder.set_entry_point("initial_router_node")

builder.add_conditional_edges(
    "initial_router_node",
    lambda state: state["next_step_decision"],
    {
        "query_agent": "query_agent",
        "intake": "intake"
    }
)

# ----------------------------------
# FLOW FOR QUERY AGENT (ENDS IMMEDIATELY)
# ----------------------------------
builder.add_edge(
    "query_agent",
    END
)

# ----------------------------------
# FLOW FOR TRIAGE AGENTS
# ----------------------------------

builder.add_edge(
    "intake",
    "memory"
)

builder.add_edge(
    "memory",
    "rag"
)

builder.add_edge(
    "rag",
    "risk"
)

builder.add_edge(
    "risk",
    "reasoning"
)

builder.add_edge(
    "reasoning",
    "confidence"
)

# ----------------------------------
# CONDITIONAL ROUTING FOR TRIAGE (review or final)
# ----------------------------------

builder.add_conditional_edges(

    "confidence",

    route_review,

    {

        "review":
            "review",

        "final":
            "final"
    }
)

builder.add_edge(
    "review",
    "final"
)

builder.add_edge(
    "final",
    END
)

# ----------------------------------
# COMPILE
# ==========================================================

graph = builder.compile()

print(
    "✓ LangGraph Ready"
)


if __name__ == "__main__":

    # ----------------------------------
    # BUILD RAG DB IF NOT EXISTS
    # ----------------------------------

    if not os.path.exists(
        FAISS_DB_PATH
    ):

        print(
            "Building FAISS DB..."
        )

        build_vector_db()

    # ----------------------------------
    # SAMPLE PATIENT
    # ----------------------------------

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

    # ----------------------------------
    # RUN GRAPH
    # ----------------------------------

    # First run to populate STM_MEMORY with a full triage output
    print("\n--- First Run: Initial Symptom Triage ---")
    result1 = graph.invoke({

        "symptoms":
        "Severe chest pain radiating to left arm with shortness of breath",

        "patient_history":
            patient_history,

        "labs":
            labs
    })

    # Print the full triage output for the first run
    print("\n======================================================================")
    print("FINAL TRIAGE OUTPUT (First Run)")
    print("======================================================================")
    print(
        json.dumps(
            result1["final_output"],
            indent=4
        )
    )

    # Second run with a query
    print("\n--- Second Run: Query for last symptom ---")
    query_result_last_symptom = graph.invoke({

        "symptoms":
            "What is my last symptom?", # Pass the query string

        "patient_history":
            patient_history, # Still need to pass these, though they might not be used by query_agent

        "labs":
            labs
    })

    # Print the query response
    print("\n======================================================================")
    print("QUERY RESPONSE (What is my last symptom?)")
    print("======================================================================")
    print(query_result_last_symptom["query_response"])

    # Third run with another query
    print("\n--- Third Run: Query for risk score ---")
    query_result_risk_score = graph.invoke({

        "symptoms":
            "What was the risk score?", # Another query

        "patient_history":
            patient_history,

        "labs":
            labs
    })

    # Print the query response
    print("\n======================================================================")
    print("QUERY RESPONSE (What was the risk score?)")
    print("======================================================================")
    print(query_result_risk_score["query_response"])

    # Fourth run with a general query
    print("\n--- Fourth Run: General query ---")
    query_result_general = graph.invoke({

        "symptoms":
            "Tell me about this patient's conditions.", # General query

        "patient_history":
            patient_history,

        "labs":
            labs
    })
