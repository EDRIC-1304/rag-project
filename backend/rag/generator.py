from langchain_ollama import OllamaLLM


def generate_answer(query, retrieved_docs):
    llm = OllamaLLM(model="llama3")

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the provided context.
If answer not found, say "I don't know".

Context:
{context}

Question: {query}

Answer:
"""

    response = llm.invoke(prompt)

    return response
