import os
from langchain_community.document_loaders import TextLoader


def load_documents(data_path="data"):
    documents = []

    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_path, filename)

            loader = TextLoader(filepath)
            docs = loader.load()

            documents.extend(docs)

    return documents
