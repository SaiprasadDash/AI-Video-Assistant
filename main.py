import argparse
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcribe import transcribe_chunks
from core.summarizer import process_transcript
from core.extractors import extract_actionable, extract_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()


def run_ai_video_assistant(source: str, question: str | None = None):
    """Run the full AI video assistant pipeline for a YouTube URL or local media file."""
    print("\n=== AI Video Assistant ===")
    print(f"Source: {source}")

    print("\n[1/5] Downloading and chunking audio...")
    chunks = process_input(source)

    print("\n[2/5] Transcribing audio chunks...")
    transcripts = transcribe_chunks(chunks, verbose=False)
    transcript = "\n\n".join(part.strip() for part in transcripts if part and part.strip())

    print("\n[3/5] Generating summary and metadata...")
    result = process_transcript(transcript, output_path="summary_output.txt")

    print("\n[4/5] Extracting actionable items, decisions, and questions...")
    actionable = extract_actionable(transcript)
    decisions = extract_decisions(transcript)
    open_questions = extract_questions(transcript)

    output = {
        "title": result["title"],
        "short_summary": result["short_summary"],
        "detailed_summary": result["detailed_summary"],
        "topics": result["topics"],
        "key_points": result["key_points"],
        "actionable_items": actionable,
        "decisions": decisions,
        "questions": open_questions,
    }

    if question:
        print("\n[5/5] Answering question from transcript context...")
        rag_chain = build_rag_chain(transcript)
        output["rag_answer"] = ask_question(rag_chain, question)

    print("\n=== Pipeline Complete ===")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Video Assistant")
    parser.add_argument(
        "source",
        nargs="?",
        default="https://www.youtube.com/watch?v=_Q-e_nczWqM&t=223s",
        help="YouTube URL or local file path",
    )
    parser.add_argument(
        "--question",
        help="Optional question to answer using the transcript as context",
    )
    args = parser.parse_args()

    data = run_ai_video_assistant(args.source, args.question)

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

    if args.question:
        print("\nRAG ANSWER")
        print(data["rag_answer"])
