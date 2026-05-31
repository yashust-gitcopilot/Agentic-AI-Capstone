# ==========================================================
# CONFIGURATION SETTINGS
# ==========================================================

import os
from collections import deque

# ==========================================================
# API KEY
# ==========================================================

os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==========================================================
# LLM CONFIGURATION
# ==========================================================

LLM_CONFIG = {
    "api_key": GROQ_API_KEY,
    "model_name": "llama-3.3-70b-versatile",
    "temperature": 0
}

# ==========================================================
# EMBEDDING MODEL CONFIGURATION
# ==========================================================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ==========================================================
# PATHS
# ==========================================================

FAISS_DB_PATH = "/content/faiss_db"
MEMORY_FILE = "/content/patient_memory.json"

# ==========================================================
# MEMORY CONFIGURATION
# ==========================================================

STM_MAX_LENGTH = 10  # Short-Term Memory max size

# ==========================================================
# RAG CONFIGURATION
# ==========================================================

RAG_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "similarity_search_k": 5
}

# ==========================================================
# RISK SCORING CONFIGURATION
# ==========================================================

RISK_THRESHOLDS = {
    "P1": 90,  # Critical
    "P2": 70,  # Urgent
    "P3": 50,  # Semi-urgent
    "P4": 0    # Non-urgent
}

# ==========================================================
# CONFIDENCE CONFIGURATION
# ==========================================================

CONFIDENCE_CONFIG = {
    "base_score": 95,
    "min_docs_penalty": -20,
    "high_risk_penalty": -10,
    "no_history_penalty": -5,
    "review_threshold": 80
}
