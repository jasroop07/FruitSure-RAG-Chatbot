import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

class GeminiLLM:
    def __init__(self, model_name="gemini-2.5-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("⚠️ GOOGLE_API_KEY not found in .env")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2,
            max_output_tokens=1024
        )
        print(f"🤖 Gemini model ready → {model_name}")

    def generate_response(self, query, context):
        prompt = f"""Use this context to answer clearly and accurately.

Context:
{context}

Question: {query}

Answer:"""
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return response.content

