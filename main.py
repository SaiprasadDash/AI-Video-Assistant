import argparse
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcribe import transcribe_chunks
from core.summarizer import process_transcript
from core.extractors import (
    extract_actionable,
    extract_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()


def get_source_from_user() -> str:
    """Prompt the user for a YouTube URL or local file path, with basic validation."""

    while True:
        source = input(
            "Enter a YouTube URL or local audio/video file path: "
        ).strip()

        if source:
            return source

        print("⚠️  Please enter a non-empty YouTube URL or file path.\n")


def run_ai_video_assistant(source: str):
    """Run the full AI video assistant pipeline for a YouTube URL or local media file."""

    print("\n=== AI Video Assistant ===")
    print(f"Source: {source}")

    # ─────────────────────────────────────────────────────────────
    # 1. Download and chunk audio
    # ─────────────────────────────────────────────────────────────

    print("\n[1/5] Downloading and chunking audio...")
    chunks = process_input(source)

    # ─────────────────────────────────────────────────────────────
    # 2. Transcribe audio
    # ─────────────────────────────────────────────────────────────

    print("\n[2/5] Transcribing audio chunks...")

    transcripts = transcribe_chunks(
        chunks,
        verbose=False
    )

    transcript = "\n\n".join(
        part.strip()
        for part in transcripts
        if part and part.strip()
    )

    # ─────────────────────────────────────────────────────────────
    # 3. Generate summary and metadata
    # ─────────────────────────────────────────────────────────────

    print("\n[3/5] Generating summary and metadata...")

    # Everything stays in memory.
    # No summary file is created.
    result = process_transcript(transcript)

    # ─────────────────────────────────────────────────────────────
    # 4. Extract additional information
    # ─────────────────────────────────────────────────────────────

    print(
        "\n[4/5] Extracting actionable items, "
        "decisions, and questions..."
    )

    actionable = extract_actionable(transcript)
    decisions = extract_decisions(transcript)
    open_questions = extract_questions(transcript)

    # ─────────────────────────────────────────────────────────────
    # 5. Build RAG chain
    # ─────────────────────────────────────────────────────────────

    print("\n[5/5] Building RAG chain for chat...")

    rag_chain = build_rag_chain(transcript)

    # ─────────────────────────────────────────────────────────────
    # Final output
    # ─────────────────────────────────────────────────────────────

    output = {
        "title": result["title"],
        "short_summary": result["short_summary"],
        "detailed_summary": result["detailed_summary"],
        "topics": result["topics"],
        "key_points": result["key_points"],
        "actionable_items": actionable,
        "decisions": decisions,
        "questions": open_questions,
        "rag_chain": rag_chain,
    }

    print("\n=== Pipeline Complete ===")

    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="AI Video Assistant"
    )

    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="YouTube URL or local file path (optional — you'll be prompted if omitted)",
    )

    args = parser.parse_args()

    # Fully user-input driven: if no source was passed on the command line,
    # ask for it interactively instead of falling back to a hardcoded default.
    source = args.source if args.source else get_source_from_user()

    data = run_ai_video_assistant(source)

    print("\nTITLE")
    print(data["title"])

    print("\nSHORT SUMMARY")
    print(data["short_summary"])

    print("\nTOPICS")
    print(data["topics"])

    print("\nKEY POINTS")
    print(data["key_points"])

    print("\nACTIONABLE ITEMS")
    print(data["actionable_items"])

    print("\nDECISIONS")
    print(data["decisions"])

    print("\nQUESTIONS")
    print(data["questions"])

    # ─────────────────────────────────────────────────────────────
    # Phase 2 — Chat with your video via RAG
    # ─────────────────────────────────────────────────────────────

    print("\n💬 Chat with your video (type 'exit' to quit)\n")

    rag_chain = data["rag_chain"]

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        answer = ask_question(
            rag_chain,
            question
        )

        print(f"\n🤖 Assistant: {answer}\n")