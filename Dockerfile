# ✅ Base image with CUDA + PyTorch
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# ✅ Working directory
WORKDIR /app

# ✅ Copy files
COPY . /app

# ✅ Install system dependencies (includes Rust, compilers, ffmpeg)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    poppler-utils \
    build-essential \
    cmake \
    libopenblas-dev \
    python3-dev \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# ✅ Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Verify CUDA & torch
RUN python3 -c "import torch; print('Torch CUDA available:', torch.cuda.is_available())"

# ✅ Default entrypoint
CMD ["python3", "handler.py"]
