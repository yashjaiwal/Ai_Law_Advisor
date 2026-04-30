
import sys
import os
sys.path.insert(0, "/content/ai_law_advisor_colab")

import logging
import uvicorn
import nest_asyncio
import threading
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import uuid

from memory import load, delete_session, get_session_count, init_db
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App Init ──────────────────────────────────────────────
app = FastAPI(
    title="AI Law Advisor API",
    description="Indian Constitutional Law Assistant powered by RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Readiness Flag ────────────────────────────────────────
app.state.ready = False  # server ready hai ki nahi

# ─── Schemas ───────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = Field(default=None)

class AnswerResponse(BaseModel):
    answer: str
    source_found: bool
    session_id: str

class HistoryResponse(BaseModel):
    session_id: str
    history: list[dict]
    total_messages: int

# ─── Background Init ───────────────────────────────────────
def background_init():
    try:
        # ✅ Folder nahi — actual FILE check karo
        index_file = os.path.join(config.INDEX_PATH, "index.faiss")

        if not os.path.exists(index_file):
            logger.info("🔨 index.faiss nahi mili — build kar rahe hain...")
            from vectorstore import build_index, reset_index
            reset_index()

            pdf_path = "/content/ai_law_advisor_colab/data/the_constitution_of_india.pdf"

            if not os.path.exists(pdf_path):
                logger.error(f"❌ PDF nahi mila: {pdf_path}")
                return

            build_index(pdf_path)
            logger.info("✅ FAISS index ready!")
        else:
            logger.info("✅ index.faiss already exists — skip build")

        # Model preload
        logger.info("🚀 Model load ho raha hai...")
        from model_loader import load_model
        load_model()
        logger.info("✅ Model ready!")

        app.state.ready = True
        logger.info("✅ App fully ready!")

    except Exception as e:
        logger.error(f"❌ Background init failed: {e}")
        raise e

# ─── Startup ───────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    init_db()
    logger.info("✅ DB initialized")

    # Background thread mein heavy tasks
    t = threading.Thread(target=background_init, daemon=True)
    t.start()
    logger.info("⚙️ Background init started — server is UP!")

# ─── Health Check ──────────────────────────────────────────
@app.get("/health")
async def health():
    index_file = os.path.join(config.INDEX_PATH, "index.faiss")
    return {
        "status": "ok",
        "ready": app.state.ready,
        "index_ready": os.path.exists(index_file),  # ✅ actual file check
        "message": "✅ Fully ready!" if app.state.ready else "⏳ Loading...",
    }

# ─── Ask Question ──────────────────────────────────────────
@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    # Ready nahi hai toh wait karo
    if not app.state.ready:
        raise HTTPException(
            status_code=503,
            detail="⏳ App abhi load ho rahi hai — thoda wait karo aur dobara try karo"
        )
    try:
        from rag_engine import get_answer
        session_id = req.session_id or str(uuid.uuid4())
        result = get_answer(session_id, req.question)
        return AnswerResponse(
            answer=result["answer"],
            source_found=result["source_found"],
            session_id=session_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ─── Get History ───────────────────────────────────────────
@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    try:
        history = load(session_id)
        return HistoryResponse(
            session_id=session_id,
            history=history,
            total_messages=len(history),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Clear Session ─────────────────────────────────────────
@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    try:
        delete_session(session_id)
        return {"message": f"Session {session_id[:8]}... cleared ✅"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── New Session ───────────────────────────────────────────
@app.post("/session/new")
async def new_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

# ─── Global Error Handler ──────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong. Please try again."}
    )

# ─── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)
