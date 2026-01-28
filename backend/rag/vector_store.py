import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

INDEX_PATH = "faiss_index"

embeddings = OllamaEmbeddings(model="mistral")


# -------------------------
# CREATE + SAVE
# -------------------------
def create_vector_store(chunks):

    db = FAISS.from_documents(chunks, embeddings)

    db.save_local(INDEX_PATH)   #  SAVE TO DISK

    return db


# -------------------------
# LOAD EXISTING
# -------------------------
def load_vector_store():

    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    return None
