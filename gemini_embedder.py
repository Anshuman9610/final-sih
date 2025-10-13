import os
from typing import List
from dotenv import load_dotenv

# FIXED: Correct import for Google GenAI
try:
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "google-generativeai package not installed. "
        "Install it with: pip install google-generativeai"
    )

load_dotenv()

# Get API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in environment variables!")

# Configure GenAI
genai.configure(api_key=GOOGLE_API_KEY)


class GeminiEmbeddings:
    """Wrapper for Google Gemini embeddings compatible with LangChain."""
    
    def __init__(self, model_name: str = "models/embedding-001"):
        """
        Initialize Gemini embeddings.
        
        Args:
            model_name: The Gemini embedding model to use
        """
        self.model_name = model_name
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        embeddings = []
        for text in texts:
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                print(f"⚠ Error embedding document: {e}")
                # Return zero vector on error
                embeddings.append([0.0] * 768)  # Default dimension
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"⚠ Error embedding query: {e}")
            # Return zero vector on error
            return [0.0] * 768  # Default dimension
