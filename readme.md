# RunPod Whisper + Fine-tune Handler

This serverless handler performs:
- Whisper transcription for video or YouTube links
- Optional fine-tune tasks on GPU

**Run Command:**
```bash
curl -X POST https://api.runpod.io/v2/<ENDPOINT_ID>/run \
  -H "Content-Type: application/json" \
  -d '{"input":{"mode":"youtube","video":"https://youtu.be/abc123"}}'
