import os
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
import os
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from pathlib import Path

def process_all_pdfs(pdf_directory: str):
    """Smart PDF processor: auto OCR fallback for image-based PDFs."""
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    all_documents = []

    print(f"\n📂 Found {len(pdf_files)} PDF files in {pdf_directory}")

    for pdf_file in pdf_files:
        print(f"→ Processing: {pdf_file.name}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()
            text_pages = [doc for doc in documents if doc.page_content.strip()]

            if not text_pages:
                print(f"⚠️ No text found in {pdf_file.name}, applying OCR...")
                ocr_loader = UnstructuredPDFLoader(str(pdf_file), strategy="ocr_only")
                documents = ocr_loader.load()
                text_pages = [doc for doc in documents if doc.page_content.strip()]
                print(f"  🔠 OCR extracted {len(text_pages)} text blocks.")

            for doc in text_pages:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(text_pages)
            print(f"  ✓ Loaded {pdf_file.name} ({len(text_pages)} text pages)")
        except Exception as e:
            print(f"  ✗ Error loading {pdf_file.name}: {e}")

    print(f"✅ Total valid documents loaded: {len(all_documents)}")
    return all_documents




# ------------------------------------------------------
# ✂️ 2. Text Splitter
# ------------------------------------------------------
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = splitter.split_documents(documents)
    print(f"\n🧩 Split {len(documents)} documents into {len(split_docs)} chunks")
    return split_docs


# ------------------------------------------------------
# 🧠 3. Embedding Manager
# ------------------------------------------------------
class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        print(f"\n🔍 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model loaded ({self.model.get_sentence_embedding_dimension()} dimensions)")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        print(f"⚙️ Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"✅ Embeddings generated: {embeddings.shape}")
        return embeddings


# ------------------------------------------------------
# 💾 4. Vector Store (ChromaDB)
# ------------------------------------------------------
class VectorStore:
    def __init__(self, collection_name="pdf_documents", persist_directory="data/vector_store"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        print(f"\n💽 Vector store ready: {collection_name}")
        print(f"Existing docs: {self.collection.count()}")

    def add_documents(self, documents, embeddings):
        ids, metadatas, texts, embeds = [], [], [], []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            metadatas.append({
                **doc.metadata,
                "length": len(doc.page_content)
            })
            texts.append(doc.page_content)
            embeds.append(emb.tolist())

        print(f"\n📥 Adding {len(documents)} documents to vector store...")
        self.collection.add(ids=ids, embeddings=embeds, metadatas=metadatas, documents=texts)
        print(f"✅ Added successfully! Total docs: {self.collection.count()}")


# ------------------------------------------------------
# 🔍 5. RAG Retriever
# ------------------------------------------------------
class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k=5, score_threshold=0.0):
        print(f"\n🔎 Retrieving for: {query}")
        query_emb = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k
        )

        retrieved_docs = []
        if results.get("documents") and results["documents"][0]:
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                sim = 1 - dist
                if sim >= score_threshold:
                    retrieved_docs.append({
                        "rank": i + 1,
                        "content": doc,
                        "metadata": meta,
                        "similarity_score": sim
                    })
        print(f"✅ Retrieved {len(retrieved_docs)} docs")
        return retrieved_docs


# ------------------------------------------------------
# 🤖 6. Gemini LLM
# ------------------------------------------------------
class GeminiLLM:
    def __init__(self, model_name="gemini-2.5-flash", api_key=None):
        load_dotenv()
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment variables")

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=0.2,
            max_output_tokens=1024
        )
        print(f"\n🤖 Gemini LLM initialized: {model_name}")

    def generate_response(self, query, context):
        prompt = f"""Use the following context to answer the question accurately and concisely.
        
Context:
{context}

Question: {query}

Answer:"""
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return response.content
