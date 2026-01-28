from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = []

    for doc in documents:

        splits = splitter.split_text(doc.page_content)

        for i, text in enumerate(splits):

            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page", 0),
                        "chunk": i
                    }
                )
            )

    return chunks
