import os

MODEL_NAME      = "Callmeyash11/legal-clean-model"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Auto-detect — Colab + Docker + HF Spaces
if os.path.exists("/app"):
    BASE_DIR = "/app"                        # Docker
elif os.path.exists("/home/user/app"):
    BASE_DIR = "/home/user/app"              # HF Spaces
else:
    BASE_DIR = "/content/ai_law_advisor_colab"  # Colab

INDEX_PATH = os.path.join(BASE_DIR, "storage/faiss_index")
DB_PATH    = os.path.join(BASE_DIR, "chat.db")
PDF_PATH   = os.path.join(BASE_DIR, "data/the_constitution_of_india.pdf")

TOP_K = 5
