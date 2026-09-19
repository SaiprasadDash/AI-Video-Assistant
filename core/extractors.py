# Actionable, Decisions, Questions

import os

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


# ============================================================
# LLM
# ============================================================

def get_llm():
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )


# ============================================================
# TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200
)


# ============================================================
# BUILD CHAIN
# ============================================================

def build_chain(system_prompt: str):

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])

    return (
        RunnableLambda(lambda x: {"text": x})
        | prompt
        | llm
        | StrOutputParser()
    )


# ============================================================
# GENERIC CHUNK PROCESSOR
# ============================================================

def process_chunks(transcript: str, system_prompt: str) -> str:

    chunks = text_splitter.split_text(transcript)

    print(f"Split transcript into {len(chunks)} chunks.")

    chain = build_chain(system_prompt)

    results = []

    for i, chunk in enumerate(chunks, start=1):

        print(f"Processing chunk {i}/{len(chunks)}...")

        try:
            result = chain.invoke(chunk)
            results.append(result)

        except Exception as e:
            print(f"Error processing chunk {i}: {e}")

    return "\n\n".join(results)


# ============================================================
# ACTIONABLE ITEMS
# ============================================================

def extract_actionable(transcript: str) -> str:

    prompt = """
You are an expert video transcript analyzer.

Identify all actionable items from the provided transcript section.

An actionable item is something a person is explicitly asked,
instructed, or expected to do.

Rules:
- Extract only actions supported by the transcript.
- Do not invent or assume actions.
- Make each action concise and specific.
- Include the responsible person if explicitly mentioned.
- Include deadlines if explicitly mentioned.
- Remove duplicate actions.
- If no actionable items are found, return:
  No actionable items found.

Return the result as a numbered list.

Transcript section:
"""

    return process_chunks(transcript, prompt)


# ============================================================
# DECISIONS
# ============================================================

def extract_decisions(transcript: str) -> str:

    prompt = """
You are an expert video transcript analyzer.

Identify decisions made or conclusions reached in the
provided transcript section.

A decision is a choice, conclusion, agreement, or determination
explicitly stated in the transcript.

Rules:
- Extract only decisions supported by the transcript.
- Do not invent decisions.
- Distinguish decisions from general statements or suggestions.
- Include the reason or context when explicitly provided.
- Remove duplicate decisions.
- If no decisions are found, return:
  No decisions found.

Return the result as a numbered list.

Transcript section:
"""

    return process_chunks(transcript, prompt)


# ============================================================
# QUESTIONS
# ============================================================

def extract_questions(transcript: str) -> str:

    prompt = """
You are an expert video transcript analyzer.

Identify questions present in the provided transcript section.

Extract:
1. Questions explicitly asked by the speaker.
2. Questions asked by another person, if present.
3. Important unanswered questions explicitly raised.

Rules:
- Preserve the original meaning.
- Do not invent questions.
- Do not convert statements into questions.
- Do not answer the questions.
- Remove duplicate questions.
- If no questions are found, return:
  No questions found.

Return the result as a numbered list.

Transcript section:
"""

    return process_chunks(transcript, prompt)