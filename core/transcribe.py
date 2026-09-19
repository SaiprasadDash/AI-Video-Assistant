import os
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model_cache = {}


def load_model(model_name: str = WHISPER_MODEL):
    """Load Whisper model and cache it for reuse."""

    if model_name in _model_cache:
        print(f"Model '{model_name}' already loaded, reusing cached instance.")
        return _model_cache[model_name]

    try:
        import whisper
    except ImportError:
        print("Whisper not found. Installing openai-whisper...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "openai-whisper"
        ])
        import whisper
        print("Whisper installed successfully.")

    print(f"Loading Whisper '{model_name}' model...")

    model = whisper.load_model(model_name)

    _model_cache[model_name] = model

    print(f"Model '{model_name}' loaded and cached.")

    return model


def transcribe_chunk(
    chunk_path: str,
    model_name: str = WHISPER_MODEL
) -> str:
    """Transcribe a single WAV chunk."""

    model = load_model(model_name)

    result = model.transcribe(
        chunk_path,
        task="transcribe"
    )

    return result["text"].strip()


def transcribe_chunks(
    chunks: list[str],
    model_name: str = WHISPER_MODEL,
    verbose: bool = True
) -> list[str]:
    """Transcribe a list of audio chunk paths."""

    transcripts = []

    for i, chunk_path in enumerate(chunks):

        if verbose:
            print(
                f"Transcribing chunk "
                f"{i + 1}/{len(chunks)}: {chunk_path}"
            )

        text = transcribe_chunk(
            chunk_path,
            model_name
        )

        transcripts.append(text)

        if verbose:
            print(f"  ✓ Done ({len(text)} chars)")

    return transcripts


def get_full_transcript(transcripts: list[str]) -> str:
    """
    Combine all chunk transcripts into one transcript.

    Nothing is saved to disk.
    """

    return "\n\n".join(transcripts)