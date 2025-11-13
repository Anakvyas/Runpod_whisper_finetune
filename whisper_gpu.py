import os
import time
import yt_dlp
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
from helpers import get_links

# -----------------------
# Device & Whisper Model
# -----------------------
device = "cuda" if (whisper.cuda.is_available() if hasattr(whisper, "cuda") else False) else "cpu"
model = whisper.load_model("base")

def log(msg: str):
    """Simple timestamped logger."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
#   LOCAL FILE TRANSCRIPTION
# ============================================================
def transcribe_local(file_path: str):
    """Transcribe an uploaded video file and return RunPod-friendly result."""
    try:
        log("🎵 Extracting audio from video...")
        clip = VideoFileClip(file_path)
        clip.audio.write_audiofile(file_path + ".wav", logger=None)
        clip.close()

        log("📝 Transcribing with Whisper...")
        result = model.transcribe(file_path + ".wav")
        text = result["text"]

        # Cleanup
        os.remove(file_path)
        os.remove(file_path + ".wav")

        log("📄 Generating PDF + vectorstore...")
        link = get_links(text)

        # MUST return list of objects to match Next.js API
        return [{
            "pdf_url": link["pdf_url"],
            "vectorstore": link["vectorstore"]
        }]

    except Exception as e:
        log(f"❌ Error in local transcription: {e}")
        return [{"error": str(e)}]


# ============================================================
#   YOUTUBE TRANSCRIPTION
# ============================================================
def transcribe_youtube(video_url: str):
    """Download and transcribe YouTube audio and return RunPod-friendly result."""
    try:
        log("🎥 Downloading YouTube audio...")
        os.makedirs("audio", exist_ok=True)

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": "audio/audio",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "quiet": True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        log("🎤 Transcribing with Whisper...")
        result = model.transcribe("audio/audio.mp3")
        text = result["text"]

        os.remove("audio/audio.mp3")

        log("📄 Generating PDF + vectorstore...")
        link = get_links(text)

        # Return correct shape for RunPod + Next.js API
        return [{
            "pdf_url": link["pdf_url"],
            "vectorstore": link["vectorstore"]
        }]

    except Exception as e:
        log(f"❌ Error in YouTube transcription: {e}")
        return [{"error": str(e)}]
