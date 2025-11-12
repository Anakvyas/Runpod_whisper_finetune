# ✅ Base: Stable PyTorch + CUDA image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# ✅ Working directory
WORKDIR /app

# ✅ Copy project files
COPY . /app

# ✅ Prevent tzdata hang
ENV DEBIAN_FRONTEND=noninteractive

# ✅ Install all required system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    poppler-utils \
    build-essential \
    cmake \
    libopenblas-dev \
    libffi-dev \
    libssl-dev \
    python3-dev \
    cargo \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# ✅ Upgrade pip & core build tools
RUN pip install --upgrade pip setuptools wheel

# ✅ Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Sanity check for CUDA + Torch
RUN python3 -c "import torch; print('✅ Torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

# ✅ Reset frontend (optional)
ENV DEBIAN_FRONTEND=dialog

# ✅ Default entrypoint
CMD ["python3", "handler.py"]
