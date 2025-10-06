from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from crew_pipeline import run_full_pipeline, get_or_create_indexes, reset_memory, pdf_cache
import uvicorn
import os
import tempfile
import asyncio
import traceback
import logging

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === FastAPI App ===
app = FastAPI(
    title="Insurance Claim Evaluator",
    description="Backend for insurance policy evaluation with session-based memory",
    version="2.2.0"
)

# === Enable CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === PDF Storage ===
uploaded_pdfs: Dict[str, str] = {}  # pdf_id -> pdf_path

# === Health Check ===
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "Insurance Claim Evaluator API",
        "version": "2.2.0"
    }

@app.get("/health")
def health_check():
    return {
        "status": "running",
        "message": "Backend is up!",
        "uploaded_pdfs": len(uploaded_pdfs)
    }

# === Upload PDF Endpoint ===
@app.post("/pdf/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF once and return a pdf_id.
    The PDF will be indexed for vector and BM25 search.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="❌ Only PDF files allowed")
        
        # Read file content
        content = await file.read()
        
        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"❌ File too large (max {max_size // (1024*1024)}MB)"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            pdf_path = tmp.name

        # Use filename as simple ID
        pdf_id = os.path.basename(pdf_path)

        # Warm-up cache (create vector + BM25 indexes) in background
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, get_or_create_indexes, pdf_path)

        uploaded_pdfs[pdf_id] = pdf_path
        logger.info(f"✅ PDF uploaded: {pdf_id} ({file.filename})")
        
        return {
            "pdf_id": pdf_id,
            "filename": file.filename,
            "message": "✅ PDF uploaded & indexed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("PDF upload failed")
        raise HTTPException(status_code=500, detail=f"❌ Failed to upload: {str(e)}")

# === Query PDF Endpoint (WITH SESSION SUPPORT) ===
@app.post("/pdf/query")
async def query_pdf(
    pdf_id: str = Form(...),
    question: str = Form(...),
    session_id: str = Form("default")  # NEW: Session ID parameter
):
    """
    Query an already uploaded PDF using its pdf_id.
    Use session_id to maintain conversation history across multiple queries.
    
    Parameters:
    - pdf_id: The ID returned from /pdf/upload
    - question: Your question about the PDF
    - session_id: Unique identifier for this conversation (default: "default")
    """
    if pdf_id not in uploaded_pdfs:
        raise HTTPException(
            status_code=404,
            detail=f"❌ pdf_id '{pdf_id}' not found. Please upload the PDF first!"
        )

    pdf_path = uploaded_pdfs[pdf_id]
    loop = asyncio.get_running_loop()
    
    try:
        logger.info(f"📝 Query from session '{session_id}': {question[:50]}...")
        
        # Run pipeline in executor to avoid blocking (WITH SESSION ID)
        result = await loop.run_in_executor(
            None,
            run_full_pipeline,
            question,
            pdf_path,
            session_id  # Pass session_id to pipeline
        )
        
        return {
            "answer": result["answer"],
            "metadata": result.get("metadata", []),
            "session_id": session_id
        }
        
    except Exception as e:
        logger.exception(f"Query failed for session '{session_id}'")
        raise HTTPException(status_code=500, detail=f"❌ Query failed: {str(e)}")

# === Reset Memory Endpoint ===
@app.post("/memory/reset")
async def reset_conversation(session_id: str = Form("default")):
    """
    Clear conversation history for a specific session.
    Use this to start a fresh conversation.
    
    Parameters:
    - session_id: The session to reset (default: "default")
    """
    try:
        reset_memory(session_id)
        logger.info(f"🧠 Memory cleared for session: {session_id}")
        return {
            "message": f"✅ Memory cleared for session: {session_id}",
            "session_id": session_id
        }
    except Exception as e:
        logger.exception(f"Failed to reset memory for session '{session_id}'")
        raise HTTPException(status_code=500, detail=f"❌ Failed to reset memory: {str(e)}")

# === Get Session Info ===
@app.get("/memory/sessions")
async def get_sessions():
    """
    Get list of active sessions with conversation history.
    """
    from crew_pipeline import memory_manager
    
    sessions_info = {}
    for session_id, memory in memory_manager.sessions.items():
        history = memory.load_memory_variables({})
        chat_history = history.get("history", [])
        sessions_info[session_id] = {
            "message_count": len(chat_history),
            "has_history": len(chat_history) > 0
        }
    
    return {
        "active_sessions": len(sessions_info),
        "sessions": sessions_info
    }

# === Delete PDF Endpoint ===
@app.delete("/pdf/{pdf_id}")
async def delete_pdf(pdf_id: str):
    """
    Delete an uploaded PDF and free resources.
    This will remove the PDF file and clear its cache.
    """
    if pdf_id not in uploaded_pdfs:
        raise HTTPException(
            status_code=404,
            detail=f"❌ pdf_id '{pdf_id}' not found"
        )
    
    pdf_path = uploaded_pdfs.pop(pdf_id)
    
    try:
        # Remove PDF file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        # Clear from cache
        if pdf_path in pdf_cache:
            del pdf_cache[pdf_path]
        
        logger.info(f"🗑️ Deleted PDF: {pdf_id}")
        return {
            "message": f"✅ Deleted {pdf_id}",
            "pdf_id": pdf_id
        }
        
    except Exception as e:
        logger.exception(f"Failed to delete PDF '{pdf_id}'")
        raise HTTPException(status_code=500, detail=f"❌ Failed to delete: {str(e)}")

# === List Uploaded PDFs ===
@app.get("/pdf/list")
async def list_pdfs():
    """
    Get list of all uploaded PDFs.
    """
    return {
        "count": len(uploaded_pdfs),
        "pdfs": list(uploaded_pdfs.keys())
    }

# === Run Server ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )