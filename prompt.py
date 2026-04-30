
from typing import Optional

MAX_CONTEXT_CHARS = 2000
MAX_HISTORY_TURNS = 5  # last N turns rakhna


def format_history(history: list[dict]) -> str:
    """Chat history ko clean format mein convert karo."""
    if not history:
        return "No previous conversation."

    # Sirf last N turns lo
    recent = history[-(MAX_HISTORY_TURNS * 2):]

    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content'].strip()}")

    return "\n".join(lines)


def truncate_context(context: str) -> str:
    """Context ko limit ke andar rakho."""
    if len(context) > MAX_CONTEXT_CHARS:
        return context[:MAX_CONTEXT_CHARS] + "\n...[truncated]"
    return context


def build_prompt(context: str, history: list[dict], question: str) -> str:
    """
    Production-grade prompt builder.
    - Strict output control
    - Hallucination prevention
    - Clean formatting
    """

    formatted_history = format_history(history)
    truncated_context = truncate_context(context)

    return f"""<s>[INST] <<SYS>>
You are a precise and reliable Indian Constitutional Law Assistant.

STRICT RULES — MUST FOLLOW:
1. Answer ONLY from the CONTEXT provided below.
2. If the answer is NOT in the context, say exactly: "I don't have enough information in my knowledge base to answer this."
3. Do NOT hallucinate, guess, or add information not present in context.
4. Do NOT repeat the question.
5. Do NOT ask follow-up questions.
6. Do NOT add confirmations like "Is this correct?" or "Please confirm."
7. Keep answer concise — max 5-6 sentences.
8. Cite the Article/Section number if mentioned in context.
9. Use plain simple English.
<</SYS>>

PREVIOUS CONVERSATION:
{formatted_history}

LEGAL CONTEXT:
{truncated_context}

USER QUESTION:
{question.strip()}

ANSWER (based only on context above): [/INST]"""


def extract_answer(raw_output: str, prompt: str) -> str:
    """
    Model ke raw output se clean answer nikalo.
    Prompt echo aur garbage remove karo.
    """
    # Prompt part remove karo
    if "[/INST]" in raw_output:
        answer = raw_output.split("[/INST]")[-1]
    else:
        answer = raw_output.replace(prompt, "")

    # Common garbage patterns remove karo
    garbage = [
        "Is this answer correct?",
        "Please confirm",
        "Please respond",
        "Yes or No",
        "CONFIRMATION",
        "ANSWER IS CORRECT",
        "PLEASE PROCEED",
    ]

    for g in garbage:
        answer = answer.replace(g, "")

    return answer.strip()
