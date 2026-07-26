"""
YOUCHAT — chat with any YouTube video's transcript.
Run with: streamlit run app.py
"""

import re
import time
from urllib.parse import urlparse, parse_qs

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="YOUCHAT",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Make the sidebar collapse arrow impossible to miss, in case a narrow
# viewport auto-collapses it on first load.
st.markdown(
    """
    <style>
    button[kind="header"] {
        background-color: #FF0033 !important;
        border-radius: 6px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
# Background   #0F0F0F   (YouTube's own dark-mode canvas)
# Surface      #181818
# Surface-2    #212121
# Border       #2E2E2E
# Accent       #FF0033   (a hair off pure YouTube red, so it reads as "ours")
# Accent-dim   #7A0019
# Text-hi      #F1F1F1
# Text-lo      #AAAAAA
# Display font: "Bebas Neue" (condensed, poster-y — feels like a channel banner)
# Body font:    "Roboto" (YouTube's actual UI font)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Roboto:wght@400;500;700&display=swap');

html, body, [class*="css"]  { font-family: 'Roboto', sans-serif; }

#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 1.2rem; max-width: 900px; }

body, .stApp { background-color: #0F0F0F; }

/* ---------- Top bar ---------- */
.yc-topbar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 4px 18px 4px;
    border-bottom: 1px solid #2E2E2E;
    margin-bottom: 22px;
}
.yc-logo {
    width: 34px; height: 24px;
    background: #FF0033;
    border-radius: 6px;
    position: relative;
    flex-shrink: 0;
}
.yc-logo::after {
    content: "";
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-45%, -50%);
    border-style: solid;
    border-width: 6px 0 6px 10px;
    border-color: transparent transparent transparent #0F0F0F;
}
.yc-wordmark {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 28px;
    letter-spacing: 1px;
    color: #F1F1F1;
    line-height: 1;
}
.yc-wordmark span { color: #FF0033; }
.yc-tagline {
    font-size: 12px;
    color: #AAAAAA;
    margin-top: -2px;
}
.yc-live {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #AAAAAA;
    padding: 4px 10px;
    border: 1px solid #2E2E2E;
    border-radius: 999px;
}
.yc-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #FF0033;
}
.yc-dot.live { animation: pulse 1.4s infinite; }
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255,0,51,0.6); }
    70% { box-shadow: 0 0 0 6px rgba(255,0,51,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,0,51,0); }
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background-color: #181818;
    border-right: 1px solid #2E2E2E;
}
section[data-testid="stSidebar"] * { color: #F1F1F1; }
.yc-side-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 20px;
    letter-spacing: 1px;
    color: #F1F1F1;
    margin-bottom: 2px;
}
.yc-side-sub { font-size: 12px; color: #AAAAAA; margin-bottom: 14px; }

.yc-thumb-wrap {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #2E2E2E;
    margin-bottom: 10px;
}
.yc-chip {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.4px;
    padding: 3px 9px;
    border-radius: 999px;
    margin-bottom: 10px;
}
.yc-chip.ok { background: rgba(60, 200, 120, 0.12); color: #3CC878; border: 1px solid rgba(60,200,120,0.35); }
.yc-chip.idle { background: rgba(170,170,170,0.10); color: #AAAAAA; border: 1px solid #2E2E2E; }

div[data-testid="stTextInput"] input {
    background-color: #0F0F0F !important;
    color: #F1F1F1 !important;
    border: 1px solid #2E2E2E !important;
    border-radius: 8px !important;
}
.stButton button {
    background-color: #FF0033;
    color: #FFFFFF;
    border: none;
    border-radius: 20px;
    font-weight: 700;
    letter-spacing: 0.3px;
    padding: 0.5rem 1.1rem;
    width: 100%;
}
.stButton button:hover { background-color: #D40029; color: #FFFFFF; }
.yc-reset button {
    background-color: transparent !important;
    color: #AAAAAA !important;
    border: 1px solid #2E2E2E !important;
}
.yc-reset button:hover { color: #F1F1F1 !important; border-color: #F1F1F1 !important; }

/* ---------- Empty state ---------- */
.yc-empty {
    text-align: center;
    padding: 90px 20px;
    color: #AAAAAA;
}
.yc-empty .yc-empty-icon {
    font-size: 40px;
    margin-bottom: 10px;
}
.yc-empty h3 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 26px;
    letter-spacing: 1px;
    color: #F1F1F1;
    margin: 0 0 6px 0;
}

/* ---------- Chat bubbles ---------- */
.yc-row { display: flex; margin-bottom: 16px; }
.yc-row.user { justify-content: flex-end; }
.yc-row.ai { justify-content: flex-start; }

.yc-avatar {
    width: 30px; height: 30px;
    border-radius: 50%;
    background: #FF0033;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    margin-right: 10px;
    font-size: 12px;
}
.yc-avatar::after {
    content: "";
    border-style: solid;
    border-width: 5px 0 5px 8px;
    border-color: transparent transparent transparent #FFFFFF;
    margin-left: 2px;
}

.yc-bubble {
    max-width: 72%;
    padding: 10px 15px;
    border-radius: 16px;
    font-size: 14.5px;
    line-height: 1.5;
}
.yc-row.user .yc-bubble {
    background: #FF0033;
    color: #FFFFFF;
    border-bottom-right-radius: 4px;
}
.yc-row.ai .yc-bubble {
    background: #212121;
    color: #F1F1F1;
    border: 1px solid #2E2E2E;
    border-bottom-left-radius: 4px;
}
.yc-name {
    font-size: 11px;
    font-weight: 700;
    color: #AAAAAA;
    margin-bottom: 3px;
}

div[data-testid="stChatInput"] textarea {
    background-color: #181818 !important;
    color: #F1F1F1 !important;
}
div[data-testid="stChatInput"] {
    border: 1px solid #2E2E2E !important;
    border-radius: 24px !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Core RAG logic
# --------------------------------------------------------------------------
def extract_video_id(url: str):
    url = url.strip()
    if "v=" in url:
        vid = parse_qs(urlparse(url).query).get("v", [None])[0]
        if vid:
            return vid
    match = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)
    match = re.search(r"^[A-Za-z0-9_-]{11}$", url)
    if match:
        return url
    return None


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.6-flash")


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an AI assistant that answers questions based ONLY on the
provided YouTube transcript.

Instructions:
1. Use only the provided context.
2. If the answer is not present in the context, say:
   "I couldn't find the answer in the provided transcript."
3. Do not make up facts.
4. Explain the answer clearly and concisely.""",
        ),
        ("human", "context :{context}\n\nQuestion :{message}"),
    ]
)


def fetch_transcript(video_id: str) -> str:
    ytt_api = YouTubeTranscriptApi()
    response = ytt_api.fetch(video_id)
    return " ".join(chunk.text for chunk in response)


def build_retriever(transcript_text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(transcript_text)
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(embedding=embeddings, texts=chunks)
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 7, "lambda_mult": 0.4},
    )


def answer_question(retriever, question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = PROMPT.invoke({"context": context, "message": question})
    response = get_llm().invoke(final_prompt)
    if isinstance(response.content, list):
        return response.content[0]["text"]
    return response.content


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("retriever", None)
st.session_state.setdefault("video_id", None)
st.session_state.setdefault("busy", False)


# --------------------------------------------------------------------------
# Top bar
# --------------------------------------------------------------------------
live_class = "live" if st.session_state.retriever else ""
live_label = "ON AIR" if st.session_state.retriever else "STANDBY"
st.markdown(
    f"""
    <div class="yc-topbar">
        <div class="yc-logo"></div>
        <div>
            <div class="yc-wordmark">YOU<span>CHAT</span></div>
            <div class="yc-tagline">Talk to any video's transcript</div>
        </div>
        <div class="yc-live"><span class="yc-dot {live_class}"></span>{live_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar — load a video
# --------------------------------------------------------------------------
def load_video(raw_url: str):
    """Fetch captions for raw_url, build the retriever, and update session state."""
    video_id = extract_video_id(raw_url) if raw_url else None
    if not video_id:
        st.error("That doesn't look like a valid YouTube URL or video ID.")
        return

    with st.spinner("Fetching captions…"):
        try:
            transcript_text = fetch_transcript(video_id)
        except (TranscriptsDisabled, NoTranscriptFound):
            transcript_text = None
            st.error("This video has no captions available.")
        except VideoUnavailable:
            transcript_text = None
            st.error("That video is unavailable.")
        except Exception as e:
            transcript_text = None
            st.error(f"Couldn't fetch that video: {e}")

    if transcript_text:
        with st.spinner("Indexing the transcript…"):
            try:
                retriever = build_retriever(transcript_text)
                st.session_state.retriever = retriever
                st.session_state.video_id = video_id
                st.session_state.messages = []
                st.success("Video loaded — ask away!")
                time.sleep(0.4)
                st.rerun()
            except Exception as e:
                st.error(f"Couldn't build the index: {e}")


with st.sidebar:
    st.markdown('<div class="yc-side-title">LOAD A VIDEO</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="yc-side-sub">Paste a YouTube link and YOUCHAT reads its captions.</div>',
        unsafe_allow_html=True,
    )

    url_input = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
        key="sidebar_url",
    )
    if st.button("Load transcript", use_container_width=True, key="sidebar_load_btn"):
        load_video(url_input)

    if st.session_state.video_id:
        st.markdown(
            f'<div class="yc-thumb-wrap"><img src="https://img.youtube.com/vi/{st.session_state.video_id}/hqdefault.jpg" style="width:100%;display:block;"></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<span class="yc-chip ok">TRANSCRIPT LOADED</span>', unsafe_allow_html=True)
        st.markdown('<div class="yc-reset">', unsafe_allow_html=True)
        if st.button("Load a different video", use_container_width=True):
            st.session_state.retriever = None
            st.session_state.video_id = None
            st.session_state.messages = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="yc-chip idle">NO VIDEO YET</span>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Main — chat
# --------------------------------------------------------------------------
if not st.session_state.retriever:
    st.markdown(
        """
        <div class="yc-empty">
            <div class="yc-empty-icon">▶</div>
            <h3>NOTHING QUEUED UP</h3>
            <div>Paste a YouTube link below (or use the sidebar) to start the chat.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([4, 1])
    with col1:
        main_url_input = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
            key="main_url",
        )
    with col2:
        if st.button("Load", use_container_width=True, key="main_load_btn"):
            load_video(main_url_input)
else:
    for msg in st.session_state.messages:
        role = msg["role"]
        if role == "user":
            st.markdown(
                f'<div class="yc-row user"><div class="yc-bubble">{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'''<div class="yc-row ai">
                        <div class="yc-avatar"></div>
                        <div>
                            <div class="yc-name">YOUCHAT</div>
                            <div class="yc-bubble">{msg["content"]}</div>
                        </div>
                    </div>''',
                unsafe_allow_html=True,
            )

    question = st.chat_input("Ask something about this video…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Watching the tape back…"):
            try:
                answer = answer_question(st.session_state.retriever, question)
            except Exception as e:
                answer = f"Something went wrong answering that: {e}"
        st.session_state.messages.append({"role": "ai", "content": answer})
        st.rerun()
