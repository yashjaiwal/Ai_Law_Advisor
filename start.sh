#!/bin/bash

echo "Starting FastAPI..."
uvicorn api:app --host 0.0.0.0 --port 8000 &

echo "Waiting for API..."
sleep 5

echo "Starting Streamlit..."
streamlit run app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true

# Wait for all processes
wait