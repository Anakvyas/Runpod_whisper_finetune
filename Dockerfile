# ✅ Official RunPod GPU Worker Image
FROM runpod/worker-torch:2.1.0-py3.10-cuda12.1

# Working directory
WORKDIR /app

# Copy project files
COPY . /app

# Disable interactive tzdata
ENV DEBIAN_FRONTEND=noninteractive

# Install system libs
RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1 libgl1 poppler-utils \
    build-essential cmake libopenblas-dev libffi-dev libssl-dev python3-dev cargo tzdata \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Debug CUDA
RUN python3 -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# RunPod serverless handler
CMD ["python3", "handler.py"]
