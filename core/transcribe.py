import os
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # reads from .env

_model_cache = {}

def load_model(model_name: str = WHISPER_MODEL):  # uses .env value as default
    """Load Whisper model, installing whisper package if not available."""
    
    if model_name in _model_cache:
        print(f"Model '{model_name}' already loaded, reusing cached instance.")
        return _model_cache[model_name]
    
    try:
        import whisper
    except ImportError:
        print("Whisper not found. Installing openai-whisper...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
        import whisper
        print("Whisper installed successfully.")
    
    print(f"Loading Whisper '{model_name}' model...")
    model = whisper.load_model(model_name)
    _model_cache[model_name] = model
    print(f"Model '{model_name}' loaded and cached.")
    
    return model


def transcribe_chunk(chunk_path: str, model_name: str = WHISPER_MODEL) -> str:
    """Transcribe a single WAV chunk."""
    model = load_model(model_name)
    result = model.transcribe(chunk_path, task="transcribe")
    return result["text"].strip()


def transcribe_chunks(chunks: list[str], model_name: str = WHISPER_MODEL, verbose: bool = True) -> list[str]:
    """Transcribe a list of audio chunk paths."""
    transcripts = []
    for i, chunk_path in enumerate(chunks):
        if verbose:
            print(f"Transcribing chunk {i + 1}/{len(chunks)}: {chunk_path}")
        text = transcribe_chunk(chunk_path, model_name)
        transcripts.append(text)
        if verbose:
            print(f"  ✓ Done ({len(text)} chars)")
    return transcripts


# def save_transcript(transcripts: list[str], output_path: str) -> str:
#     """Join chunk transcripts and save to a .txt file."""
#     full_text = "\n\n".join(transcripts)
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(full_text)
#     print(f"Transcript saved to: {output_path}")
#     return full_text