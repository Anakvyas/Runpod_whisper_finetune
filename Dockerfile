# ---------------------------------------------------
# 1) Use RunPod GPU Base Image (Public)
#    Includes CUDA, cuDNN, PyTorch, and Python
# ---------------------------------------------------
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1

# ---------------------------------------------------
# 2) Working directory
# ---------------------------------------------------
WORKDIR /workspace

# ---------------------------------------------------
# 3) Install system dependencies
# ---------------------------------------------------
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------
# 4) Copy your project files
# ---------------------------------------------------
COPY . .

# ---------------------------------------------------
# 5) Python dependencies
# ---------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------
# 6) Default entrypoint (optional)
# ---------------------------------------------------
CMD ["python3", "main.py"]
