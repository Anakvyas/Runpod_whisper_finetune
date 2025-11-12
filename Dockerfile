# ✅ Base: Stable PyTorch + CUDA image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# ✅ Set working directory
WORKDIR /app

# ✅ Copy project files
COPY . /app

# ✅ Prevent tzdata interactive prompts (fix for slow/hanging builds)
ENV DEBIAN_FRONTEND=noninteractive

# ✅ Install system-level dependencies
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

# ✅ Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Verify CUDA and Torch availability
RUN python3 -c "import torch; print('✅ Torch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

# ✅ Reset frontend mode (optional)
ENV DEBIAN_FRONTEND=dialog

# ✅ Default entrypoint
CMD ["python3", "handler.py"]
