import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from backend.rag.loader import DocumentLoader
from backend.rag.chunker import chunk_documents
from backend.rag.vector_store import create_vector_store
from backend.rag.retriever import search_chunks
from backend.rag.generator import generate_answer
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="mistral")



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


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    path = os.path.join(UPLOAD_DIR, file.filename)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ✅ CORRECT
    loader = DocumentLoader(UPLOAD_DIR)

    docs = loader.load()

    print("Loaded docs count:", len(docs))   # debug
    print("First 300 chars:", docs[0].page_content[:300])

    chunks = chunk_documents(docs)

    global vector_store
    vector_store = create_vector_store(chunks)

    return {"message": "File uploaded and indexed"}

def search_chunks(vector_store, query, k=4):
    results = vector_store.similarity_search_with_score(query, k=k)

    docs = []

    print("\n======================")
    print("QUERY:", query)
    print("======================")

    for i, (doc, score) in enumerate(results):
        print(f"\n--- CHUNK {i+1} | score={score:.4f} ---")
        print(doc.page_content[:500])

        docs.append(doc.page_content)

    return docs

def generate_answer(query, chunks):
    context = "\n\n".join(chunks)

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


@app.post("/ask")
async def ask_question(query: str):
    if vector_store is None:
        return {"answer": "Upload documents first"}

    results = search_chunks(vector_store, query)
    answer = generate_answer(query, results)

    return {"answer": answer}
    

@app.get("/")
def root():
    return {"status": "API running"}
