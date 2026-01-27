def search_chunks(vector_store, query, k=3):
    results = vector_store.similarity_search(query, k=k)
    return results
