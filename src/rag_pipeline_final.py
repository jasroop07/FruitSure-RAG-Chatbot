from src.data_loader import process_all_pdfs
from src.text_splitter import split_documents
from src.embeddings_manager import EmbeddingManager
from src.vector_store import VectorStore
from src.retriever import RAGRetriever
from src.llm_manager import GeminiLLM

# Global instances
retriever = None
llm = None
chat_history = []

def initialize_rag():
    """Initialize and load everything once."""
    global retriever, llm

    print("\n🚀 Initializing RAG System...")

    pdf_docs = process_all_pdfs("data")
    if not pdf_docs:
        print("⚠️ No valid PDFs found in /data.")
        return

    chunks = split_documents(pdf_docs)
    embedder = EmbeddingManager()
    vector_store = VectorStore()

    if vector_store.is_empty():
        texts = [doc.page_content for doc in chunks]
        embeddings = embedder.generate_embeddings(texts)
        vector_store.add_documents(chunks, embeddings)
    else:
        print("⚡ Existing vector store detected — skipping embedding.")

    retriever = RAGRetriever(vector_store, embedder)
    llm = GeminiLLM()
    print("✅ RAG System Ready for Chat!")

def chat_with_rag(message: str) -> str:
    """Simple chat interaction."""
    global retriever, llm, chat_history

    if message.lower() in ["hi", "hello", "hey"]:
        return "👋 Hello! I’m your FruitSure assistant. Ask me anything about apples or leaves!"

    if message.lower() in ["bye", "exit", "quit"]:
        return "🍎 Goodbye! Have a fruitful day!"

    if not retriever or not llm:
        return "⚠️ System not initialized."

    results = retriever.retrieve(message, top_k=3)
    context = "\n\n".join([r["content"] for r in results]) if results else ""
    if not context:
        return "🤔 Sorry, I couldn’t find relevant info in the documents."

    response = llm.generate_response(message, context)
    chat_history.append({"user": message, "assistant": response})
    return response
