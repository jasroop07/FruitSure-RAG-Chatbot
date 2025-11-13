import os
import uuid
import chromadb

class VectorStore:
    def __init__(self, collection_name="pdf_documents", persist_directory="data/vector_store"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        print(f"\n💽 Vector store ready: {collection_name}")
        print(f"Existing docs: {self.collection.count()}")

    def is_empty(self):
        return self.collection.count() == 0

    def add_documents(self, documents, embeddings):
        if not documents or len(embeddings) == 0:
            print("⚠️ No documents or embeddings to add.")
            return

        ids, metadatas, docs_text, embeds = [], [], [], []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            metadatas.append(doc.metadata)
            docs_text.append(doc.page_content)
            embeds.append(emb.tolist())

        print(f"📥 Adding {len(docs_text)} documents to vector store...")
        self.collection.add(
            ids=ids,
            embeddings=embeds,
            metadatas=metadatas,
            documents=docs_text
        )
        print(f"✅ Added successfully → total docs now: {self.collection.count()}")
