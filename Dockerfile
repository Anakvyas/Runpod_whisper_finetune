
FROM runpod/serverless:gpu-cuda12.1

WORKDIR /app
COPY . /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1 libgl1 poppler-utils \
    build-essential cmake libopenblas-dev libffi-dev libssl-dev python3-dev cargo tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

RUN python3 - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

CMD ["python3", "handler.py"]
