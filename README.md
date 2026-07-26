# ▶ YOUCHAT

Chat with any YouTube video — just paste a link, and ask questions about what was actually said in it.

YOUCHAT pulls a video's captions, indexes them with embeddings, and answers your questions using **only** that transcript — no guessing, no hallucinated facts. If the answer isn't in the video, it tells you so.

## Features

- 🔗 **Paste any YouTube link** — no manual transcript downloading or setup per video
- 🧠 **Retrieval-Augmented Generation (RAG)** — answers are grounded in the video's actual captions
- 🚫 **No hallucinations** — if the transcript doesn't cover it, YOUCHAT says so instead of making things up
- 💬 **Clean, YouTube-styled chat UI** built with Streamlit
- ⚡ **On-the-fly indexing** — no pre-built database step, everything happens in the browser session

## Tech stack

| Layer | Tool |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| Transcript fetching | [youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/) |
| Embeddings | Google Generative AI (`gemini-embedding-001`) |
| LLM | Google Generative AI (Gemini, via `langchain-google-genai`) |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss) |
| Orchestration | [LangChain](https://www.langchain.com/) |

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/youchat.git
cd youchat
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

You can get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Usage

1. Paste a YouTube video URL into the sidebar (or the box on the main screen).
2. Click **Load transcript** — YOUCHAT fetches the captions and builds a searchable index.
3. Once it says the video is loaded, ask anything about the video in the chat box.

> Note: this only works on videos that have captions (auto-generated or manual) available. Videos with captions disabled won't work.

## Project structure

```
youchat/
├── app.py              # Streamlit app (transcript fetch + RAG + chat UI)
├── requirements.txt     # Python dependencies
├── .env                 # Your API key (not committed — add to .gitignore)
└── README.md
```

## Notes / limitations

- Everything is indexed **in-memory per session** — reloading the page or loading a new video clears the previous index.
- Answer quality depends on caption quality (auto-generated captions can be noisy).
- Uses the Gemini model configured in `app.py` — swap it for any other model your API key supports if needed.

## License

MIT — do whatever you want with it.