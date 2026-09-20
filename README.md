# 🎙️ AI Video Assistant

An AI-powered meeting/video assistant that turns any **YouTube link or local audio/video file** into a searchable, chat-able knowledge base — transcription, summarization, insight extraction, and RAG-based Q&A, all in one pipeline.

No more paying for Otter.ai or Fireflies — this runs locally with free/open tooling wherever possible.

## ✨ Features

- **Flexible input** — pass a YouTube URL or a local audio/video file
- **Local transcription** with OpenAI Whisper (no external API needed for English)
- **Hindi → English translation** support for non-English audio
- **AI-generated summary** — title, short summary, detailed summary, topics, and key points
- **Insight extraction** — automatically pulls out:
  - ✅ Actionable items
  - 📌 Key decisions
  - ❓ Open questions
- **Conversational RAG chat** — ask follow-up questions about the video/meeting directly from the terminal, powered by a vector store built from the transcript
- **LLM orchestration** via LangChain (LCEL) with Mistral AI as the backing model

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Audio/video acquisition | `yt-dlp`, `pydub`, `ffmpeg-python` |
| Speech-to-text | `openai-whisper` (local, PyTorch-backed) |
| Translation | `deep-translator` |
| LLM orchestration | `langchain`, `langchain-mistralai`, `mistralai` |
| Vector store / RAG | `chromadb`, `sentence-transformers`, `langchain-huggingface` |
| UI | `streamlit` |
| Export | `reportlab`, `fpdf2` |

## 📂 Project Structure

```
AI-Video-Assistant/
├── core/
│   ├── transcribe.py       # Whisper-based audio-to-text transcription
│   ├── summarizer.py       # Title / summary / topics / key-points generation
│   ├── extractors.py       # Actionable items, decisions, open questions
│   └── rag_engine.py       # Builds the RAG chain and answers chat questions
├── utils/
│   └── audio_processor.py  # Downloads/loads input and chunks the audio
├── main.py                 # CLI entry point — runs the full pipeline
├── test.py                 # Tests
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.10
- [FFmpeg](https://ffmpeg.org/download.html) installed and available on your `PATH`
- A [Mistral AI](https://console.mistral.ai) API key (free tier available)

### Installation

```bash
git clone https://github.com/SaiprasadDash/AI-Video-Assistant.git
cd AI-Video-Assistant
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root with your API key(s):

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

### Usage

Run the assistant against a YouTube URL or a local file:

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

```bash
python main.py "/path/to/local/video-or-audio.mp4"
```

If no source is provided, a default sample YouTube URL is used.

The pipeline runs in five stages:

1. **Download & chunk audio** from the source
2. **Transcribe** audio chunks with Whisper
3. **Summarize** — generate title, short/detailed summary, topics, and key points
4. **Extract insights** — actionable items, decisions, open questions
5. **Build a RAG chain** so you can chat with the transcript

Once processing finishes, the summary and insights are printed to the console, and you're dropped into an interactive chat:

```
💬 Chat with your meeting (type 'exit' to quit)

You: What were the main decisions made?
🤖 Assistant: ...
```

Type `exit`, `quit`, or `q` to leave the chat.

## 🗺️ Roadmap

- [ ] Export full report as PDF / TXT
- [ ] Streamlit web UI
- [ ] Additional language support beyond Hindi → English

## 🤝 Contributing

Issues and pull requests are welcome! Feel free to open an issue if you run into a bug or have a feature request.

## 📄 License

No license specified yet — consider adding one (e.g. MIT) to clarify usage terms.