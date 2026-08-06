from dotenv import load_dotenv
load_dotenv()

import os
from urllib.parse import urlparse, parse_qs

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="YouTube RAG",
    page_icon="🎥",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown("""
<style>

.main{
    padding-top:1rem;
}

.stChatMessage{
    border-radius:15px;
    padding:12px;
}

.block-container{
    max-width:1100px;
}

.status{
    background:#262730;
    padding:12px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# MODELS
# -----------------------------

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
(
"system",
"""
You are an AI assistant.

Answer ONLY from the provided transcript.

If the answer is not found say:

"I couldn't find the answer in the data."

Never hallucinate.
"""
),

(
"human",
"""
Context:

{context}

Question:

{question}
"""
)
])

# -----------------------------
# FUNCTIONS
# -----------------------------

def get_video_id(url):
    return parse_qs(urlparse(url).query)["v"][0]


def create_database(video_id):

    transcript = YouTubeTranscriptApi().fetch(video_id)

    text = " ".join(chunk.text for chunk in transcript)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    path = f"DATABASES/{video_id}"

    os.makedirs("DATABASES", exist_ok=True)

    vector_store.save_local(path)


def load_database(video_id):

    path = f"DATABASES/{video_id}"

    if os.path.exists(path):

        status = "Loaded Existing Database"

    else:

        create_database(video_id)

        status = "Created New Database"

    vector_store = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store, status


# -----------------------------
# SESSION STATE
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# -----------------------------
# HEADER
# -----------------------------

st.title("🎥 YouTube RAG Chat")

st.caption("Ask questions about any YouTube video transcript using Mistral AI + FAISS")

# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.header("Video")

    url = st.text_input(
        "Paste YouTube URL"
    )

    if st.button("Load Video", use_container_width=True):

        if url:

            try:

                video_id = get_video_id(url)

                thumbnail = f"https://img.youtube.com/vi/{video_id}/0.jpg"

                st.image(thumbnail)

                with st.spinner("Creating / Loading Database..."):

                    vector_store, status = load_database(video_id)

                    st.session_state.retriever = vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k":4,
                            "fetch_k":15,
                            "lambda_mult":0.5
                        }
                    )

                st.success(status)

            except Exception as e:

                st.error(e)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages=[]

# -----------------------------
# CHAT HISTORY
# -----------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# -----------------------------
# CHAT
# -----------------------------

question = st.chat_input("Ask anything about the video...")

if question:

    if st.session_state.retriever is None:

        st.warning("Load a YouTube video first.")

        st.stop()

    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    docs = st.session_state.retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    final_prompt = prompt.invoke(
        {
            "context":context,
            "question":question
        }
    )

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = llm.invoke(final_prompt)

            st.markdown(response.content)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":response.content
        }
    )
