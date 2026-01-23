from src.loader import load_documents
from src.chunker import chunk_documents
from src.vector_store import create_vector_store
from src.retriever import search_chunks
from src.generator import generate_answer


def main():
    docs = load_documents()
    chunks = chunk_documents(docs)
    vector_store = create_vector_store(chunks)

    print("RAG system ready!\n")

    while True:
        query = input("Ask something (or type exit): ")

        if query.lower() == "exit":
            break

        results = search_chunks(vector_store, query)

        answer = generate_answer(query, results)

        print("\nAnswer:\n")
        print(answer)
        print("-" * 50)


if __name__ == "__main__":
    main()
