import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL")

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200,
)

# ── Prompts ───────────────────────────────────────────────────────────────────

map_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert video transcript analyzer.

Analyze the transcript section and extract:
1. Main topics discussed
2. Important concepts
3. Key facts
4. Examples mentioned
5. Important conclusions
6. Technical terms and their meaning

Do not invent information.
Only use information present in the transcript."""
    ),
    ("human", "Analyze this transcript section:\n\n{chunk}")
])

reduce_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert at creating high-quality summaries of video transcripts.

Create a structured summary using ONLY the information provided.
Do not introduce facts that are not present in the source.

Organize the result into:
1. Overview
2. Main Topics
3. Key Points
4. Important Details
5. Examples
6. Key Takeaways

Remove repetition between sections."""
    ),
    ("human", "Create the final summary from these section analyses:\n\n{summaries}")
])

short_summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Write a 2-3 sentence summary of the following."),
    ("human", "{summary}")
])

keypoints_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Extract the most important points from the following video summary.
Return 5-10 concise bullet points.
Do not invent information."""
    ),
    ("human", "Summary:\n\n{summary}")
])

topics_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Extract the main topics discussed in the video.
Return only a list of concise topics.
Do not include topics that are not discussed."""
    ),
    ("human", "{summary}")
])

title_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that generates concise titles."),
    ("human", "Generate a short and catchy title for the following summary. Return only the title, nothing else.\n\nSummary:\n{summary}")
])

# ── Chains ────────────────────────────────────────────────────────────────────

map_chain          = map_prompt          | llm | StrOutputParser()
reduce_chain       = reduce_prompt        | llm | StrOutputParser()
short_summary_chain = short_summary_prompt | llm | StrOutputParser()
keypoints_chain    = keypoints_prompt    | llm | StrOutputParser()
topics_chain       = topics_prompt       | llm | StrOutputParser()
title_chain        = title_prompt        | llm | StrOutputParser()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _summarize_chunk(chunk: str, index: int, total: int) -> str:
    """Summarize a single chunk with error handling."""
    try:
        print(f"  → Summarizing chunk {index}/{total}...")
        return map_chain.invoke({"chunk": chunk})
    except Exception as e:
        print(f"  ❌ Error on chunk {index}: {e}")
        return ""


def _hierarchical_reduce(summaries: list[str], group_size: int = 5) -> str:
    """
    Reduce summaries hierarchically to avoid context overflow.

    100 summaries → 20 groups → 4 groups → 1 final summary
    """
    print(f"  Reducing {len(summaries)} summaries (group size: {group_size})...")

    while len(summaries) > group_size:
        grouped = []
        for i in range(0, len(summaries), group_size):
            group = summaries[i: i + group_size]
            combined = "\n\n".join(group)
            reduced = reduce_chain.invoke({"summaries": combined})
            grouped.append(reduced)
        summaries = grouped
        print(f"  → Reduced to {len(summaries)} summaries.")

    # final reduce
    return reduce_chain.invoke({"summaries": "\n\n".join(summaries)})

# ── Public API ────────────────────────────────────────────────────────────────

def summarize(transcript: str) -> str:
    """Split → map → hierarchical reduce → final summary."""

    print("Splitting transcript...")
    chunks = text_splitter.split_text(transcript)
    print(f"Split into {len(chunks)} chunk(s).")

    print("Summarizing chunks (map)...")
    chunk_summaries = [
        _summarize_chunk(chunk, i + 1, len(chunks))
        for i, chunk in enumerate(chunks)
    ]
    chunk_summaries = [s for s in chunk_summaries if s]  # drop failed chunks

    print("Combining summaries (hierarchical reduce)...")
    final_summary = _hierarchical_reduce(chunk_summaries)

    print("Summarization complete.")
    return final_summary


def generate_short_summary(summary: str) -> str:
    """Generate a 2-3 sentence short summary."""
    print("Generating short summary...")
    return short_summary_chain.invoke({"summary": summary})


def generate_key_points(summary: str) -> str:
    """Extract 5-10 key bullet points from the summary."""
    print("Generating key points...")
    return keypoints_chain.invoke({"summary": summary})


def generate_topics(summary: str) -> str:
    """Extract main topics from the summary."""
    print("Generating topics...")
    return topics_chain.invoke({"summary": summary})


def generate_title(summary: str) -> str:
    """Generate a catchy title for the summary."""
    print("Generating title...")
    return title_chain.invoke({"summary": summary}).strip()


# def save_summary(summary: str, output_path: str) -> str:
#     """Save summary to a .txt file."""
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(summary)
#     print(f"Summary saved to: {output_path}")
#     return summary


def process_transcript(transcript: str, output_path: str = None) -> dict:
    """
    Full pipeline — returns all generated content as a dict.

    Returns:
        {
            title, short_summary, detailed_summary,
            key_points, topics
        }
    """
    detailed_summary = summarize(transcript)

    result = {
        "title":            generate_title(detailed_summary),
        "short_summary":    generate_short_summary(detailed_summary),
        "detailed_summary": detailed_summary,
        "key_points":       generate_key_points(detailed_summary),
        "topics":           generate_topics(detailed_summary),
    }

    if output_path:
        content = f"""Title: {result['title']}

Short Summary:
{result['short_summary']}

Topics:
{result['topics']}

Key Points:
{result['key_points']}

Detailed Summary:
{result['detailed_summary']}
"""
        save_summary(content, output_path)

    return result