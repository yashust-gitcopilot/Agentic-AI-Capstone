from langchain_huggingface import HuggingFaceEmbeddings
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from rag.embeddings import embedding_model
from config.settings import PDF_FOLDER, FAISS_DB_PATH

from langchain_community.vectorstores import FAISS

from rag.embeddings import embedding_model
from config.settings import FAISS_DB_PATH

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def build_vector_db():

    documents = []

    for file in os.listdir(PDF_FOLDER):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(
                os.path.join(PDF_FOLDER, file)
            )

            docs = loader.load()

            for d in docs:
                d.metadata["source_file"] = file

            documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    vectordb = FAISS.from_documents(
        chunks,
        embedding_model
    )

    vectordb.save_local(FAISS_DB_PATH)

    return vectordb


def load_vector_db():

    return FAISS.load_local(
        FAISS_DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
