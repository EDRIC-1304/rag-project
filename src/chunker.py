from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # max characters per chunk
        chunk_overlap=100    # overlap for context continuity
    )

    chunks = splitter.split_documents(documents)

    return chunks
