class RAGRetriever:
    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query, top_k=5):
        print(f"\n🔎 Retrieving for query: {query}")
        query_emb = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k
        )

        retrieved_docs = []
        if results.get("documents") and results["documents"][0]:
            for i, (doc, meta, dist) in enumerate(
                zip(results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0])
            ):
                retrieved_docs.append({
                    "rank": i + 1,
                    "content": doc,
                    "metadata": meta,
                    "score": 1 - dist
                })

        print(f"✅ Retrieved {len(retrieved_docs)} relevant documents")
        return retrieved_docs
