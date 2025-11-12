# import time
# import runpod

# def log(msg):
#     """Simple logger for GPU jobs."""
#     print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# def handler(event):
#     """Main RunPod entrypoint."""
#     log("🟢 Received RunPod job request")

#     # 🧠 Lazy import heavy modules only when the worker actually runs
#     import torch
#     from finetune_gpu import train_gpu
#     from whisper_gpu import transcribe_local, transcribe_youtube

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     task = event["input"].get("task", "")
#     mode = event["input"].get("mode", "")
#     log(f"🧩 Task: {task or mode}")

#     try:
#         if task == "finetune":
#             log("🚀 Starting fine-tune job...")
#             output = train_gpu(event["input"], device)
#             return {"output": output}

#         elif mode == "youtube" and event["input"].get("video"):
#             log("🎥 Processing YouTube transcription...")
#             pdf_link = transcribe_youtube(event["input"]["video"])

#         elif mode == "file" and event["input"].get("path"):
#             log("📂 Processing file transcription...")
#             pdf_link = transcribe_local(event["input"]["path"])

#         else:
#             raise ValueError("Invalid input. Expected 'task' or 'mode'.")

#         return {
#             "output": {
#                 "pdf_link": pdf_link,
#                 "persist_dir": "vectorstore/demo"
#             }
#         }

#     except Exception as e:
#         log(f"❌ Error: {str(e)}")
#         return {"error": str(e)}


# # ✅ Required by RunPod to detect the handler
# runpod.serverless.start({"handler": handler})

import runpod

def handler(event):
    return {"output": "Hello from RunPod!"}

runpod.serverless.start({"handler": handler})
