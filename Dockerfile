FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements pehle — Docker cache ke liye
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App files copy karo
COPY . .

# Folders banao
RUN mkdir -p /app/storage/faiss_index
RUN mkdir -p /app/data

# start.sh executable banao
RUN chmod +x start.sh

# HF Spaces = 7860, Normal = 8000 + 8501
EXPOSE 7860 8000 8501

# Dono services run karo
CMD ["sh", "start.sh"]