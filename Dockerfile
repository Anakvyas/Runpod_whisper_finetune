# ✅ Base: Stable PyTorch + CUDA image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# ✅ Set working directory
WORKDIR /app

# ✅ Copy all project files
COPY . /app

# ✅ Prevent tzdata from hanging (non-interactive)
ENV DEBIAN_FRONTEND=noninteractive

# ✅ Install system dependencies
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
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# ✅ Upgrade pip and core wheels
RUN pip install --upgrade pip setuptools wheel

# ✅ Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Verify CUDA + Torch after install
RUN python3 -c "import torch; print('✅ Torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

# ✅ Reset frontend mode (optional)
ENV DEBIAN_FRONTEND=dialog

# ✅ Entrypoint (start your fine-tune job)
CMD ["python3", "handler.py"]
