import runpod
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def handler(event):
    log("🟢 Job received")
    task = event["input"].get("task")
    mode = event["input"].get("mode")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from finetune_gpu import train_gpu
    from whisper_gpu import transcribe_local, transcribe_youtube

    try:
        if task == "finetune":
            log("🚀 Fine-tune started")
            output = train_gpu(event["input"], device)
            return {"output": output}

        if mode == "youtube":
            log("🎥 YouTube transcription started")
            pdf_link = transcribe_youtube(event["input"]["video"])
            return {"output": {"pdf_link": pdf_link}}

        if mode == "file":
            log("📂 File transcription started")
            pdf_link = transcribe_local(event["input"]["path"])
            return {"output": {"pdf_link": pdf_link}}

        return {"error": "Invalid input"}

    except Exception as e:
        log(f"❌ Error: {e}")
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
