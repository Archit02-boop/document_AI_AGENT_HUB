import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

VECTOR_DB_PATH = "faiss_index"


def get_embeddings():
    """
    Creates a free local embedding model.

    This model runs on your own laptop and does not require OpenAI credits.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


def create_vector_store(chunks: list[str]):
    """
    Converts text chunks into embeddings and stores them in FAISS.
    """

    documents = [
        Document(
            page_content=chunk,
            metadata={"chunk_id": index}
        )
        for index, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    vector_store.save_local(VECTOR_DB_PATH)

    return len(documents)


def load_vector_store():
    """
    Loads the saved FAISS vector database.
    """

    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError("FAISS index not found. Please ingest a document first.")

    embeddings = get_embeddings()

    vector_store = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store