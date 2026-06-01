from langchain_community.vectorstores import FAISS

from rag.embeddings import embedding_model
from config.settings import FAISS_DB_PATH


def load_vector_db():

    return FAISS.load_local(
        FAISS_DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
