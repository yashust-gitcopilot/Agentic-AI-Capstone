import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

FAISS_DB_PATH = "data/faiss_db"

MEMORY_FILE = "data/patient_memory.json"

PDF_FOLDER = "data/pdfs"

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)
