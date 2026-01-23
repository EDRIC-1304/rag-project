from src.loader import load_documents
from src.chunker import chunk_documents


def main():
    docs = load_documents()
    chunks = chunk_documents(docs)

    print(f"Loaded {len(docs)} documents")
    print(f"Created {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks[:5]):  # show first 5 only
        print(f"Chunk {i+1}:")
        print(chunk.page_content)
        print("-" * 40)


if __name__ == "__main__":
    main()
