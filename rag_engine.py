
import sys
import os
sys.path.insert(0, "/content/ai_law_advisor_colab")
import logging
from typing import Optional
import config
from vectorstore import search, search_with_score
from model_loader import generate
from prompt import build_prompt, extract_answer
from memory import load, save, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

init_db()

MAX_SCORE = 1.2  # FAISS L2 distance — isse zyada = irrelevant chunk


# ─── Context Builder ───────────────────────────────────────
def build_context(question: str) -> tuple[str, bool]:
    """
    Relevant chunks dhundo aur context banao.
    Returns: (context_text, is_relevant)
    """
    results = search_with_score(question, top_k=config.TOP_K)

    # Score filter — irrelevant chunks hatao
    filtered = sorted(
        [(doc, score) for doc, score in results if score <= MAX_SCORE],
        key=lambda x: x[1]
    )[:5]

    if not filtered:
        logger.warning(f"⚠️ No relevant chunks found for: {question[:50]}")
        return "", False

    # Source info ke saath context banao
    context_parts = []
    for doc, score in filtered:
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Page {page}]\n{doc.page_content.strip()}"
        )

    context = "\n\n---\n\n".join(context_parts)
    logger.info(f"✅ {len(filtered)} relevant chunks found")
    return context, True


# ─── Main Answer Function ──────────────────────────────────
def get_answer(session_id: str, question: str) -> dict:
    """
    Main RAG pipeline:
    1. History load karo
    2. Relevant context dhundo
    3. Prompt banao
    4. Answer generate karo
    5. History save karo
    6. Structured response return karo
    """
    if not session_id or not question.strip():
        raise ValueError("session_id aur question empty nahi hone chahiye")

    logger.info(f"💬 Question [{session_id[:8]}...]: {question[:60]}")

    # Step 1 — History
    history = load(session_id)

    # Step 2 — Context
    context, is_relevant = build_context(question)

    # Context nahi mila toh early response
    if not is_relevant:
        fallback = "I don't have enough information in my knowledge base to answer this question."
        save(session_id, "user", question)
        save(session_id, "assistant", fallback)
        return {
            "answer": fallback,
            "source_found": False,
            "session_id": session_id,
        }

    # Step 3 — Prompt
    prompt = build_prompt(
        context=context,
        history=history,
        question=question
    )

    # Step 4 — Generate
    raw_output = generate(prompt)
    answer = extract_answer(raw_output, prompt)

    # Empty answer fallback
    if not answer.strip():
        answer = "I was unable to generate a response. Please try rephrasing your question."
        logger.warning("⚠️ Empty answer generated")

    # Step 5 — Save history
    save(session_id, "user", question)
    save(session_id, "assistant", answer)

    logger.info(f"✅ Answer generated [{session_id[:8]}...]")

    # Step 6 — Structured response
    return {
        "answer": answer,
        "source_found": True,
        "session_id": session_id,
    }
