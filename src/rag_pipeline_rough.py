from src.rag_components_rough import (
    process_all_pdfs,
    split_documents,
    EmbeddingManager,
    VectorStore,
    RAGRetriever,
    GeminiLLM
)

def run_rag_pipeline():
    print("\n🚀 Starting RAG Pipeline...")

    # Step 1: Load PDFs
    pdf_docs = process_all_pdfs("data")
    pdf_docs = process_all_pdfs("data")
    print(f"🔎 Example document content:\n{pdf_docs[0].page_content[:300] if pdf_docs else 'No text found'}")


    # Step 2: Split into chunks
    chunks = split_documents(pdf_docs)

    # Step 3: Generate embeddings
    embedder = EmbeddingManager()
    texts = [doc.page_content for doc in chunks]
    embeddings = embedder.generate_embeddings(texts)

    # Step 4: Store in vector DB
    vector_store = VectorStore()
    vector_store.add_documents(chunks, embeddings)

    # Step 5: Initialize retriever & LLM
    retriever = RAGRetriever(vector_store, embedder)
    llm = GeminiLLM()

    # Step 6: Query input
    query = input("\n🔍 Enter your question: ")
    results = retriever.retrieve(query, top_k=3)

    if not results:
        print("⚠️ No relevant documents found.")
        return

    context = "\n\n".join([r["content"] for r in results])
    print("\n💬 Generating answer...")
    answer = llm.generate_response(query, context)

    print("\n🧠 Final Answer:\n")
    print(answer)
