# ==========================================================
# LLM AND EMBEDDING MODEL SETUP
# ==========================================================

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from config import LLM_CONFIG, EMBEDDING_MODEL_NAME

# ==========================================================
# INITIALIZE LLM
# ==========================================================

def setup_llm() -> ChatGroq:
    """
    Initialize and configure the Groq LLM.
    
    Returns:
        ChatGroq: Configured LLM instance
    """
    llm = ChatGroq(
        api_key=LLM_CONFIG["api_key"],
        model_name=LLM_CONFIG["model_name"],
        temperature=LLM_CONFIG["temperature"]
    )
    return llm


# ==========================================================
# INITIALIZE EMBEDDING MODEL
# ==========================================================

def setup_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize and configure the embedding model.
    
    Returns:
        HuggingFaceEmbeddings: Configured embedding model instance
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )
    return embedding_model


# ==========================================================
# SYSTEM PROMPT
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
