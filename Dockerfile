FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY . /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1 libgl1 poppler-utils \
    build-essential cmake libopenblas-dev libffi-dev libssl-dev python3-dev cargo tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Whisper + Vision + Transformers + LangChain ecosystem
RUN pip install --no-cache-dir -r requirements.txt

# Debug GPU availability
RUN python3 - <<EOF
import torch
print("🔥 Torch version:", torch.__version__)
print("🔥 CUDA available:", torch.cuda.is_available())
print("🔥 Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
EOF

CMD ["python3", "handler.py"]
