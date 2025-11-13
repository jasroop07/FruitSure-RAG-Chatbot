"""
main.py — FastAPI RAG Chatbot API
---------------------------------
✅ Loads all models and embeddings once at startup
✅ Exposes /chat endpoint for your frontend
✅ Includes greeting and goodbye logic
✅ Ready for production use
"""

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from src.rag_pipeline_final import initialize_rag, chat_with_rag

# =============================================================
#  FastAPI App Setup
# =============================================================

app = FastAPI(
    title="FruitSure RAG Chatbot API",
    version="1.0",
    description="An intelligent chatbot API using RAG + Gemini for apple and leaf quality analysis.",
)

# Enable CORS (so frontend can call the API from a browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace "*" with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================
#  Initialize the RAG System on Startup
# =============================================================

@app.on_event("startup")
def startup_event():
    print("\n🧠 Starting up — initializing RAG pipeline...")
    initialize_rag()
    print("✅ RAG system loaded and ready to chat!")


# =============================================================
#  Request and Response Models
# =============================================================

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


# =============================================================
#  Root Endpoint
# =============================================================

@app.get("/", tags=["Root"])
def root():
    return {
        "status": "✅ Running",
        "message": "Welcome to the FruitSure RAG Chatbot API!",
        "usage": "POST /chat with {'message': 'your question'}",
    }


# =============================================================
#  Chat Endpoint
# =============================================================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat_endpoint(request: ChatRequest):
    """
    Accepts a user query and returns an AI-generated response
    using Retrieval-Augmented Generation (RAG).
    """
    user_message = request.message.strip()
    response = chat_with_rag(user_message)
    return ChatResponse(reply=response)


# =============================================================
#  Run App (for local testing)
# =============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
