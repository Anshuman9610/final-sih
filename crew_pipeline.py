import os
import numpy as np
import hashlib
import time
from functools import wraps
from cachetools import LFUCache
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from BM25_INDEXER import prepare_bm25_index, bm25_search
from langchain_community.embeddings import HuggingFaceEmbeddings

# === NEW IMPORTS FOR MEMORY ===
from langchain.memory import ConversationBufferMemory

# === Load Env Vars ===
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("❌ GROQ_API_KEY missing!")

# === LLM Setup ===
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    max_tokens=800,
)

# === Embeddings ===
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
)

# === Caches ===
pdf_cache = LFUCache(maxsize=10)     # Cache per-PDF indexes
query_cache = LFUCache(maxsize=100)  # Cache per-query answers


# === Session-based Memory Manager ===
class MemoryManager:
    def __init__(self):
        self.sessions = {}  # session_id -> ConversationBufferMemory

    def get_memory(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferMemory(
                memory_key="history",
                return_messages=True
            )
        return self.sessions[session_id]

    def reset_memory(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].clear()
            print(f"🧠 Memory cleared for session: {session_id}")
        else:
            print(f"⚠️ No memory found for session: {session_id}")


memory_manager = MemoryManager()


# === Decorator for query caching (NOW INCLUDES SESSION_ID) ===
def cache_query(func):
    @wraps(func)
    def wrapper(user_query: str, pdf_path: str, session_id="default"):
        # FIXED: Include session_id in cache key
        cache_key = (user_query, pdf_path, session_id)
        if cache_key in query_cache:
            print(f"✅ Returning cached answer for: {cache_key}")
            return query_cache[cache_key]

        print(f"⚡ Executing pipeline for: {cache_key}")
        result = func(user_query, pdf_path, session_id=session_id)
        query_cache[cache_key] = result
        return result
    return wrapper


# === PDF Loader ===
def load_pdf_chunks(pdf_path):
    from pypdf import PdfReader
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return []

    try:
        reader = PdfReader(pdf_path)
        texts = [page.extract_text() or "" for page in reader.pages]
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return []
    return texts


# === Helper: Truncate long passages ===
def truncate_passages(passages, max_total_tokens=3000):
    total_tokens = 0
    truncated = []
    for p in passages:
        tokens = len(p.split())
        if total_tokens + tokens > max_total_tokens:
            break
        truncated.append(p)
        total_tokens += tokens
    return truncated


# === Get/Create Indexes ===
def get_or_create_indexes(pdf_path):
    if pdf_path in pdf_cache:
        print(f"✅ Using cached indexes for {pdf_path}")
        return pdf_cache[pdf_path]

    file_hash = hashlib.md5(pdf_path.encode("utf-8")).hexdigest()
    persist_directory = f"./chroma_db/{file_hash}"

    if os.path.exists(persist_directory):
        print(f"✅ Loading existing Chroma index for {pdf_path}")
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
        )
        texts = vectordb.get(include=["documents"])["documents"]
    else:
        print(f"⚡ Creating new indexes for {pdf_path}")
        chunks = load_pdf_chunks(pdf_path)
        texts = [doc for doc in chunks if isinstance(doc, str)]

        vectordb = Chroma.from_texts(
            texts,
            embedding_model,
            persist_directory=persist_directory,
        )

        # Build BM25 index once
        prepare_bm25_index(pdf_path)

    # Precompute embeddings
    doc_embeddings = [np.array(vec) for vec in embedding_model.embed_documents(texts)]
    doc_norms = [np.linalg.norm(vec) for vec in doc_embeddings]

    index_data = {
        "vectordb": vectordb,
        "texts": texts,
        "doc_embeddings": doc_embeddings,
        "doc_norms": doc_norms,
    }

    pdf_cache[pdf_path] = index_data
    return index_data


# === Local Rerank ===
def local_rerank(query, texts, doc_embeddings, doc_norms, top_n=5):
    query_emb = np.array(embedding_model.embed_query(query))
    query_norm = np.linalg.norm(query_emb)

    doc_matrix = np.stack(doc_embeddings)
    sims = doc_matrix @ query_emb
    sims /= np.array(doc_norms) * query_norm

    top_idx = np.argsort(-sims)[:top_n]
    return [texts[i] for i in top_idx]


# === Extract Relevant Text with Metadata ===
def extract_relevant_clause(pdf_path, user_query, k=6):
    indexes = get_or_create_indexes(pdf_path)
    vectordb = indexes["vectordb"]
    texts = indexes["texts"]
    doc_embeddings = indexes["doc_embeddings"]
    doc_norms = indexes["doc_norms"]

    metadata_results = []

    # Semantic Search
    try:
        results = vectordb.similarity_search_with_score(user_query, k=20)
        semantic_passages = []
        for doc, score in results:
            if hasattr(doc, "metadata"):
                meta = doc.metadata
                pdf_name = meta.get("source", os.path.basename(pdf_path))
                page = meta.get("page", "Unknown")
            else:
                pdf_name = os.path.basename(pdf_path)
                page = "Unknown"
            metadata_results.append({
                "content": doc.page_content,
                "pdf_name": pdf_name,
                "page": page,
                "score": score
            })
            semantic_passages.append(doc.page_content)
    except Exception as e:
        print("❌ Semantic search failed:", e)
        semantic_passages = []

    # BM25 Search
    try:
        bm25_passages = bm25_search(user_query, k=k)
    except Exception as e:
        print("❌ BM25 failed:", e)
        bm25_passages = []

    # Combine and deduplicate
    combined_passages = list(set(semantic_passages + bm25_passages))
    if not combined_passages:
        return "", []

    available_indices = [texts.index(p) for p in combined_passages if p in texts]
    if not available_indices:
        return "", []

    top_n_passages = local_rerank(
        user_query,
        [texts[i] for i in available_indices],
        [doc_embeddings[i] for i in available_indices],
        [doc_norms[i] for i in available_indices],
        top_n=5,
    )

    truncated_passages = truncate_passages(top_n_passages)

    matched_metadata = [
        m for m in metadata_results if m["content"] in truncated_passages
    ]

    return "\n\n".join(truncated_passages), matched_metadata


# === Main Pipeline with Memory (FULLY INTEGRATED) ===
@cache_query
def run_full_pipeline(user_query: str, pdf_path: str, session_id="default"):
    context, metadata_info = extract_relevant_clause(pdf_path, user_query, k=10)

    metadata_text = "\n".join(
        [f"- Source: {m['pdf_name']}, Page: {m['page']}" for m in metadata_info]
    ) or "No metadata available."

    # Get memory for session
    memory = memory_manager.get_memory(session_id)
    
    # Load conversation history
    history = memory.load_memory_variables({})
    chat_history = history.get("history", [])
    
    # Format chat history for prompt
    history_text = ""
    if chat_history:
        history_text = "\n\nConversation History:\n"
        for msg in chat_history:
            if hasattr(msg, 'type'):
                role = "User" if msg.type == "human" else "Assistant"
                history_text += f"{role}: {msg.content}\n"

    prompt_template = f"""
You are an intelligent assistant working for the Department of Higher Education (MoE).
Your role is to help officials quickly retrieve, compare, and interpret data, rules,
schemes, projects, and policies from multiple authentic documents.

Objectives:
1. Find all relevant information from provided contexts — even if scattered or overlapping.
2. Combine related clauses logically, avoid repetition, and highlight any conflicting info.
3. Maintain traceability by citing document names and page numbers.
4. Always remain factual, concise, and policy-accurate.
5. If data is insufficient or unclear, clearly state that instead of guessing.
6. Use conversation history to provide contextual answers and maintain coherent dialogue.
{history_text}

User Query:
{user_query}

Retrieved Contexts:
{context}

Source Metadata:
{metadata_text}
"""

    # Add user message to memory
    memory.chat_memory.add_user_message(user_query)

    # Call LLM
    start = time.time()
    llm_response = llm.invoke(prompt_template)
    print(f"⏱ Groq call took {time.time() - start:.2f} sec")

    # Add AI response to memory
    memory.chat_memory.add_ai_message(llm_response.content)

    # FIXED: Return metadata
    return {
        "answer": llm_response.content,
        "metadata": metadata_info
    }


# === Reset Memory Helper ===
def reset_memory(session_id="default"):
    memory_manager.reset_memory(session_id)