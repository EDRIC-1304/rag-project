from src.loader import load_documents
from src.chunker import chunk_documents
from src.vector_store import create_vector_store


def main():
    docs = load_documents()
    chunks = chunk_documents(docs)

    vector_store = create_vector_store(chunks)

    print(f"Loaded {len(docs)} docs")
    print(f"Created {len(chunks)} chunks")
    print("Vector store created successfully!")


if __name__ == "__main__":
    main()
