# gemini_embedder.py
import os
from typing import List
import numpy as np
from google import genai
from google.genai.types import EmbedContentConfig
from langchain_core.embeddings import Embeddings

# --- Config ---
PROJECT_ID = "project-5ce21651-cf21-42e3-8ce"
LOCATION = "us-central1"

# --- Initialize Vertex AI client ---
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiEmbeddings(Embeddings):
    """Custom LangChain-compatible wrapper for Gemini embeddings."""

    def __init__(self, model_name: str = "gemini-embedding-001"):
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        resp = client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        return [e.values for e in resp.embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self.embed_documents([text])[0]
