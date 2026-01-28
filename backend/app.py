import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend.rag.loader import DocumentLoader
from backend.rag.chunker import chunk_documents
from backend.rag.vector_store import create_vector_store

from langchain_ollama import OllamaLLM


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

    global vector_store
    vector_store = create_vector_store(chunks)

    return {"message": "File uploaded and indexed"}


# =========================================================
# SEARCH
# =========================================================
def search_chunks(vector_store, query, k=6, score_threshold=1.55):

    results = vector_store.similarity_search_with_score(query, k=k)

    filtered_docs = []

    print("\n======================")
    print("QUERY:", query)
    print("======================")

    for i, (doc, score) in enumerate(results):

        print(f"Chunk {i+1} | score={score:.4f}")

        # ✅ keep only strong matches
        if score <= score_threshold:
            filtered_docs.append(doc)

    # fallback (avoid empty context crash)
    if not filtered_docs:
        filtered_docs = [results[0][0]]

    return filtered_docs

# =========================================================
# GENERATE
# =========================================================
def generate_answer(query, docs):

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Answer ONLY from the context.
If not present, say 'Not found'.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return response


# =========================================================
# ASK  (PART 12 — citations)
# =========================================================
@app.post("/ask")
async def ask_question(query: str):

    global vector_store

    if vector_store is None:
        return {"answer": "Upload documents first"}

    docs = search_chunks(vector_store, query)

    answer = generate_answer(query, docs)

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

    return {
        "answer": answer,
        "sources": sources
    }


# =========================================================
# ROOT
# =========================================================
@app.get("/")
def root():
    return {"status": "API running"}
