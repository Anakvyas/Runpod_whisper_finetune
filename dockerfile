
FROM runpod/worker-torch:2.1.0-py3.10-cuda12.1
WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["python3", "runpod_handler.py"]
