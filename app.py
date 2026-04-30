
import streamlit as st
import uuid
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── API Config ────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

def api_ask(session_id: str, question: str) -> dict:
    resp = requests.post(
        f"{API_URL}/ask",
        json={"question": question, "session_id": session_id},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()

def api_clear_session(session_id: str):
    requests.delete(f"{API_URL}/session/{session_id}", timeout=10)

def api_health() -> bool:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.status_code == 200
    except:
        return False

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Law Advisor ⚖️",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .source-warning {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .app-header { text-align: center; padding: 1rem 0 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── Session Init ──────────────────────────────────────────
def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "total_questions" not in st.session_state:
        st.session_state.total_questions = 0
    if "no_context_count" not in st.session_state:
        st.session_state.no_context_count = 0

init_session()

# ─── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg", width=80)
    st.title("⚖️ AI Law Advisor")
    st.caption("Powered by Indian Constitution + RAG")
    st.divider()

    # API Health
    st.subheader(" API ")
    if api_health():
        st.success("Backend Online")
    else:
        st.error("Backend Offline ❌ — FastAPI run karo pehle")

    st.divider()

    # Stats
    st.subheader("📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions", st.session_state.total_questions)
    with col2:
        st.metric("No Context", st.session_state.no_context_count)

    st.caption(f"🔑 Session: `{st.session_state.session_id[:12]}...`")
    st.divider()

    # Controls
    st.subheader("⚙️ Controls")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        api_clear_session(st.session_state.session_id)
        st.session_state.chat = []
        st.session_state.total_questions = 0
        st.session_state.no_context_count = 0
        st.success("Chat cleared!")
        st.rerun()

    if st.button("🔄 New Session", use_container_width=True):
        api_clear_session(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat = []
        st.session_state.total_questions = 0
        st.session_state.no_context_count = 0
        st.success("New session started!")
        st.rerun()

    st.divider()
    st.subheader("ℹ️ About")
    st.markdown("""
    - ✅ RAG — Constitution of India
    - ✅ FastAPI Backend
    - ✅ Chat Memory
    - ⚠️ Not a substitute for legal advice
    """)
    st.caption("Built for legal hackathon 🏆")

# ─── Header ────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h2>⚖️ AI Law Advisor</h2>
    <p style="color: gray; font-size: 0.9rem;">
        Ask anything about the Indian Constitution
    </p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ─── Welcome ───────────────────────────────────────────────
if not st.session_state.chat:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 **Namaste! I am your AI Legal Advisor.**

        I can help you with:
        - 📜 Fundamental Rights (Article 12–35)
        - 🏛️ Structure of Parliament & Government
        - ⚖️ Directive Principles & Duties
        - 🔍 Any Article or Amendment

        **Ask your legal question below!**
        """)

# ─── Chat History ──────────────────────────────────────────
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and not msg.get("source_found", True):
            st.markdown(
                '<div class="source-warning">⚠️ No relevant context found in the Constitution.</div>',
                unsafe_allow_html=True,
            )

# ─── Input ─────────────────────────────────────────────────
user_input = st.chat_input("Ask your legal question... (e.g. Explain the constitution of india?)")

if user_input:
    user_input = user_input.strip()
    if not user_input:
        st.warning("Please enter a valid question.")
        st.stop()

    st.session_state.chat.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching Constitution..."):
            try:
                result = api_ask(st.session_state.session_id, user_input)
                answer = result["answer"]
                source_found = result.get("source_found", True)

                st.markdown(answer)

                if not source_found:
                    st.markdown(
                        '<div class="source-warning">⚠️ No relevant context found.</div>',
                        unsafe_allow_html=True,
                    )

                st.session_state.total_questions += 1
                if not source_found:
                    st.session_state.no_context_count += 1

                st.session_state.chat.append({
                    "role": "assistant",
                    "content": answer,
                    "source_found": source_found,
                })

            except Exception as e:
                logger.error(f"Error: {e}")
                st.error(f"❌ Something went wrong: {str(e)}")
                st.info("FastAPI backend chal raha hai? Health check karo sidebar mein.")

    st.rerun()

# ─── Footer ────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ Disclaimer: For informational purposes only. "
    "Not a substitute for professional legal advice."
)
