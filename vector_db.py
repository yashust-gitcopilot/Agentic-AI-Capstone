# ==========================================================
# VECTOR DATABASE (FAISS) MANAGEMENT
# ==========================================================

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from config import FAISS_DB_PATH, RAG_CONFIG
from llm_setup import setup_embedding_model

# ==========================================================
# BUILD FAISS DATABASE
# ==========================================================

def build_vector_db(pdf_folder: str = "./") -> FAISS:
    """
    Build FAISS vector database from PDF documents.
    
    Args:
        pdf_folder (str): Path to folder containing PDF files
        
    Returns:
        FAISS: Initialized vector database
    """
    embedding_model = setup_embedding_model()
    documents = []

    # Load all PDFs from folder
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            print(f"Loading {file}")
            
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            docs = loader.load()
            
            for d in docs:
                d.metadata["source_file"] = file
            
            documents.extend(docs)

    if len(documents) == 0:
        print("No PDFs found")
        return None

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CONFIG["chunk_size"],
        chunk_overlap=RAG_CONFIG["chunk_overlap"]
    )
    chunks = splitter.split_documents(documents)

    # Create and save FAISS database
    vectordb = FAISS.from_documents(chunks, embedding_model)
    vectordb.save_local(FAISS_DB_PATH)

    print("FAISS Database Created")
    return vectordb


# ==========================================================
# LOAD FAISS DATABASE
# ==========================================================

def load_vector_db() -> FAISS:
    """
    Load existing FAISS vector database.
    
    Returns:
        FAISS: Loaded vector database
        
    Raises:
        FileNotFoundError: If FAISS database does not exist
    """
    embedding_model = setup_embedding_model()
    
    return FAISS.load_local(
        FAISS_DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )


# ==========================================================
# ENSURE DATABASE EXISTS
# ==========================================================

def ensure_vector_db_exists() -> FAISS:
    """
    Ensure FAISS database exists, building it if necessary.
    
    Returns:
        FAISS: Vector database instance
    """
    if not os.path.exists(FAISS_DB_PATH):
        print("Building FAISS DB...")
        return build_vector_db()
    else:
        return load_vector_db()
