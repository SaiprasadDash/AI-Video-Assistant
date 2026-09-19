from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcribe import transcribe_chunks
from core.summarizer import summarize, generate_title
from core.extractors import (
    extract_actionable,
    extract_decisions,
    extract_questions
)


source = "https://www.youtube.com/watch?v=_Q-e_nczWqM&t=223s"


# YouTube → Audio → Chunks
chunks = process_input(source)

# Chunks → Whisper → Transcript
transcripts = transcribe_chunks(
    chunks,
    verbose=False
)

# Combine transcript
transcript = "\n\n".join(transcripts)


# Summary
title = generate_title(transcript)
summary = summarize(transcript)


# Extract information
actionable = extract_actionable(transcript)
decisions = extract_decisions(transcript)
questions = extract_questions(transcript)


# Output
print("\n📝 TRANSCRIPT")
print("-" * 60)
print(transcript[:500] + "..." if len(transcript) > 500 else transcript)

print("\n📌 TITLE")
print("-" * 60)
print(title)

print("\n📋 SUMMARY")
print("-" * 60)
print(summary)

print("\n✅ ACTION ITEMS")
print("-" * 60)
print(actionable)

print("\n🔑 KEY DECISIONS")
print("-" * 60)
print(decisions)

print("\n❓ OPEN QUESTIONS")
print("-" * 60)
print(questions)