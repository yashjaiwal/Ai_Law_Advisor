
import logging
import os
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_PDF = "/content/ai_law_advisor_colab/data/the_constitution_of_india.pdf"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


_embeddings: Optional[HuggingFaceEmbeddings] = None

def get_embeddings() -> HuggingFaceEmbeddings:

    global _embeddings
    if _embeddings is None:
        logger.info(f"Loading embeddings: {config.EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},   # embeddings CPU pe theek hain
            encode_kwargs={"normalize_embeddings": True},  # cosine similarity ke liye
        )
        logger.info("✅ Embeddings loaded")
    return _embeddings



def build_index(pdf_path: str = DEFAULT_PDF) -> None:

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF nahi mila: {pdf_path}")

    logger.info(f"📄 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    logger.info(f"✅ {len(docs)} pages loaded")

    # Chunk karo
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],  # legal docs ke liye better splitting
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f" {len(chunks)} chunks created")

    # Empty chunks remove karo
    chunks = [c for c in chunks if c.page_content.strip()]
    logger.info(f" {len(chunks)} non-empty chunks")

    # Index banao
    logger.info("⚙️ Building FAISS index...")
    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)

    # Save karo
    os.makedirs(config.INDEX_PATH, exist_ok=True)
    db.save_local(config.INDEX_PATH)
    logger.info(f" Index saved: {config.INDEX_PATH}")


# ─── Load Index ────────────────────────────────────────────
_db: Optional[FAISS] = None

def load_index(path: Optional[str] = None) -> FAISS:
    """
    FAISS index load karo — cached, baar baar load nahi hoga.
    """
    global _db
    if _db is not None:
        return _db

    index_path = path or config.INDEX_PATH

    # Index exist karta hai?
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index nahi mila: {index_path}\n"
            f"Pehle build_index() run karo!"
        )

    logger.info(f"📂 Loading FAISS index: {index_path}")
    _db = FAISS.load_local(
        index_path,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    logger.info("✅ Index loaded")
    return _db


# ─── Search ────────────────────────────────────────────────
def search(query: str, top_k: int = config.TOP_K) -> list[Document]:
    """
    Query ke liye relevant chunks dhundo.
    """
    if not query.strip():
        raise ValueError("Query empty nahi honi chahiye")

    db = load_index()
    results = db.similarity_search(query, k=top_k)
    logger.debug(f"🔍 {len(results)} chunks found for: {query[:50]}")
    return results


def search_with_score(query: str, top_k: int = config.TOP_K) -> list[tuple[Document, float]]:
    """
    Score ke saath results — low score = better match.
    """
    db = load_index()
    return db.similarity_search_with_score(query, k=top_k)


# ─── Index Reset ───────────────────────────────────────────
def reset_index() -> None:
    """Cache clear karo — fresh load hoga next call pe."""
    global _db
    _db = None
    logger.info("🗑️ Index cache cleared")
