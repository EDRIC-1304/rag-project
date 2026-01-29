from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self, vector_store, documents):
        self.vector_store = vector_store
        self.documents = documents

        # build BM25 index
        tokenized = [doc.page_content.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    # -----------------------------------
    # keyword search (BM25)
    # -----------------------------------
    def keyword_search(self, query, k=4):
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked[:k]]

    # -----------------------------------
    # vector search
    # -----------------------------------
    def vector_search(self, query, k=4):
        results = self.vector_store.similarity_search(query, k=k)
        return results

    # -----------------------------------
    # HYBRID search (merge both)
    # -----------------------------------
    def search(self, query, k=4):
        v_docs = self.vector_search(query, k)
        k_docs = self.keyword_search(query, k)

        # merge + dedupe
        combined = {id(doc): doc for doc in (v_docs + k_docs)}

        return list(combined.values())[:k]
