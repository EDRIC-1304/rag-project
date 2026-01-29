import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend.rag.loader import DocumentLoader
from backend.rag.chunker import chunk_documents
from backend.rag.vector_store import create_vector_store

from langchain_ollama import OllamaLLM
from backend.rag.hybrid_retriever import HybridRetriever
import json
from fastapi.responses import StreamingResponse


# ==============================
# LLM
# ==============================

llm = OllamaLLM(model="mistral")


# ==============================
# APP
# ==============================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

vector_store = None
retriever = None


# =========================================================
# UPLOAD + INDEX
# =========================================================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    # clear old files
    for f in os.listdir(UPLOAD_DIR):
        os.remove(os.path.join(UPLOAD_DIR, f))

    path = os.path.join(UPLOAD_DIR, file.filename)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = DocumentLoader(UPLOAD_DIR)
    docs = loader.load()

    chunks = chunk_documents(docs)

    global vector_store, retriever

    vector_store = create_vector_store(chunks)

    retriever = HybridRetriever(vector_store, chunks)

    return {"message": "File uploaded and indexed"}

# =========================================================
# SEARCH
# =========================================================
def search_chunks(retriever, query, k=6):
    """
    Uses HybridRetriever (FAISS + BM25)
    Returns list of Document objects
    """

    print("\n======================")
    print("QUERY:", query)
    print("======================")

    docs = retriever.search(query, k=k)

    for i, doc in enumerate(docs):
        print(f"Chunk {i+1}")
        print(doc.page_content[:300])

    return docs

# =========================================================
# GENERATE
# =========================================================
def stream_answer(query, docs):
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Answer ONLY from the context.
If not present, say 'Not found'.

Context:
{context}

Question:
{query}
"""

    # ---- stream tokens ----
    for chunk in llm.stream(prompt):
        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

    # ---- send sources at end ----
    sources = []
    seen = set()

    for d in docs:
        key = (d.metadata["source"], d.metadata["page"], d.metadata["chunk"])

        if key not in seen:
            seen.add(key)

            sources.append({
                "file": d.metadata["source"],
                "page": d.metadata["page"],
                "chunk": d.metadata["chunk"],
                "preview": d.page_content[:120]
            })

    yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

    # ---- done signal ----
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

# =========================================================
# ASK  (PART 12 — citations)
# =========================================================
@app.post("/ask")
async def ask_question(query: str):

    global vector_store

    if vector_store is None:
        return {"answer": "Upload documents first"}

    docs = search_chunks(vector_store, query)

    return StreamingResponse(
        stream_answer(query, docs),
        media_type="text/event-stream"   # 🔥 IMPORTANT
    )

# =========================================================
# ROOT
# =========================================================
@app.get("/")
def root():
    return {"status": "API running"}
