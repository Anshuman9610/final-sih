import os
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
import requests

# === Load API key ===
load_dotenv()
twelvelabs_api_key = os.getenv("TWELVELABS_API_KEY")

if not twelvelabs_api_key:
    raise RuntimeError("TWELVELABS_API_KEY missing!")

class TwelveLabsEmbeddings(Embeddings):
    """
    LangChain-compatible wrapper for TwelveLabs Embeddings API.
    """

    def __init__(self, model="Marengo-retrieval-2.7"):
        self.model = model
        self.api_key = twelvelabs_api_key
        self.endpoint = "https://api.twelvelabs.io/v1/embeddings"

    def _embed(self, texts):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": texts}

        response = requests.post(self.endpoint, json=payload, headers=headers)

        if response.status_code != 200:
            raise RuntimeError(f"TwelveLabs API error: {response.text}")

        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, query):
        return self._embed([query])[0]
