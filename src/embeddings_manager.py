from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"\n🔍 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model loaded ({self.model.get_sentence_embedding_dimension()} dimensions)")

    def generate_embeddings(self, texts):
        print(f"⚙️ Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")
        return embeddings
