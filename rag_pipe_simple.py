"""
Simple RAG Pipeline (.py version of your working notebook)
----------------------------------------------------------
- Loads PDFs from /data
- Splits into chunks
- Embeds with SentenceTransformer
- Stores in Chroma
- Retrieves top matches
- Answers with Gemini
"""

import os
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# LangChain + Models
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables (for Gemini)
load_dotenv()

# ================================================================
# 1️⃣ Load all PDFs
# ================================================================
def process_all_pdfs(pdf_directory):
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    all_documents = []

    print(f"\n📂 Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        print(f"→ Processing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")
        except Exception as e:
            print(f"  ✗ Error loading {pdf_file.name}: {e}")

    print(f"✅ Total documents loaded: {len(all_documents)}")
    return all_documents


# ================================================================
# 2️⃣ Split documents
# ================================================================
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    split_docs = splitter.split_documents(documents)
    print(f"🧩 Split {len(documents)} docs into {len(split_docs)} chunks")
    return split_docs


# ================================================================
# 3️⃣ Embedding Manager
# ================================================================
class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"🔍 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        print(f"⚙️ Generating embeddings for {len(texts)} texts...")
        return self.model.encode(texts, show_progress_bar=True)


# ================================================================
# 4️⃣ Vector Store (Chroma)
# ================================================================
class VectorStore:
    def __init__(self, collection_name="pdf_documents", persist_dir="data/vector_store"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        print(f"\n💽 Vector store initialized → {collection_name}")
        print(f"Existing docs: {self.collection.count()}")

    def add_documents(self, documents, embeddings):
        ids, metadatas, docs_text, embeds = [], [], [], []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            metadatas.append(doc.metadata)
            docs_text.append(doc.page_content)
            embeds.append(emb.tolist())

        if not docs_text:
            print("⚠️ No text to add, skipping Chroma add.")
            return

        print(f"📥 Adding {len(docs_text)} docs to Chroma...")
        self.collection.add(
            ids=ids,
            embeddings=embeds,
            metadatas=metadatas,
            documents=docs_text,
        )
        print(f"✅ Added successfully → total docs now: {self.collection.count()}")


# ================================================================
# 5️⃣ Retriever
# ================================================================
class RAGRetriever:
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query, top_k=5):
        print(f"\n🔎 Retrieving for query: {query}")
        q_emb = self.embedder.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[q_emb.tolist()], n_results=top_k
        )

        retrieved = []
        if results.get("documents") and results["documents"][0]:
            for i, (doc, meta, dist) in enumerate(
                zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
            ):
                retrieved.append(
                    {"rank": i + 1, "content": doc, "metadata": meta, "score": 1 - dist}
                )
        print(f"✅ Retrieved {len(retrieved)} docs")
        return retrieved


# ================================================================
# 6️⃣ Gemini LLM Wrapper
# ================================================================
class GeminiLLM:
    def __init__(self, model_name="gemini-2.5-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("⚠️ GOOGLE_API_KEY not found in .env")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, google_api_key=api_key, temperature=0.2, max_output_tokens=1024
        )
        print(f"🤖 Gemini model ready → {model_name}")

    def answer(self, query, context):
        prompt = f"""Use this context to answer clearly and accurately.

Context:
{context}

Question: {query}

Answer:"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


# ================================================================
# 7️⃣ Run the Pipeline
# ================================================================
def run_rag_pipeline():
    print("\n🚀 Starting RAG Pipeline...")

    # Load and split
    pdfs = process_all_pdfs("data")
    if not pdfs:
        print("⚠️ No text found in PDFs. Exiting.")
        return
    chunks = split_documents(pdfs)

    # Embed and store
    embedder = EmbeddingManager()
    texts = [doc.page_content for doc in chunks]
    embeddings = embedder.generate_embeddings(texts)
    store = VectorStore()
    store.add_documents(chunks, embeddings)

    # Retrieve + answer
    retriever = RAGRetriever(store, embedder)
    llm = GeminiLLM()
    query = input("\n🔍 Enter your question: ")
    results = retriever.retrieve(query)
    context = "\n\n".join([r["content"] for r in results])
    if not context:
        print("⚠️ No relevant context found.")
        return

    answer = llm.answer(query, context)
    print("\n🧠 Final Answer:\n", answer)


# ================================================================
# 8️⃣ Entry point
# ================================================================
if __name__ == "__main__":
    run_rag_pipeline()
