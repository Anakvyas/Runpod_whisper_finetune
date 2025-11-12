# ✅ Base GPU Image (for Whisper + Fine-tune)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# ✅ Working directory
WORKDIR /app

# ✅ Copy code
COPY . /app

# ✅ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Default entrypoint
CMD ["python3", "handler.py"]

