import os

MODEL_NAME      = "Callmeyash11/legal-clean-model"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Colab + Docker auto-detect
BASE_DIR   = "/app" if os.path.exists("/app") else "/content/ai_law_advisor_colab"

INDEX_PATH = os.path.join(BASE_DIR, "storage/faiss_index")
DB_PATH    = os.path.join(BASE_DIR, "chat.db")

TOP_K = 5
