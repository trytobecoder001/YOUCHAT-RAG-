from dotenv import load_dotenv
load_dotenv()

import os
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


# Models
embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0
)

# Functions


def get_video_id(url):

    return parse_qs(
        urlparse(url).query
    )["v"][0]


def create_database(video_id):

    print("\nFetching Transcript\n")

    transcript = YouTubeTranscriptApi().fetch(video_id)

    text = " ".join(
        chunk.text
        for chunk in transcript
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    db_path = f"DATABASES/{video_id}"

    vector_store.save_local(db_path)

    print("Database Created Successfully\n")


def load_database(video_id):

    db_path = f"DATABASES/{video_id}"

    if os.path.exists(db_path):

        print("Database Found")
        print("Loading...\n")

    else:

        print("Database Not Found")
        print("Creating Database...\n")

        create_database(video_id)

    vector_store = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


# User Input

url = input("Enter YouTube URL : ").strip()

try:

    video_id = get_video_id(url)

except:

    print("Invalid URL")
    exit()

vector_store = load_database(video_id)

retriever = vector_store.as_retriever(

    search_type="mmr",

    search_kwargs={

        "k":4,

        "fetch_k":15,

        "lambda_mult":0.5
    }
)


prompt = ChatPromptTemplate.from_messages(

[
(
"system",

"""
You are an AI Assistant.

Answer ONLY using the transcript.

If the answer is not available, reply:

'I couldn't find the answer in the transcript.'

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
]
)


while True:

    query = input("You : ")

    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(

        doc.page_content

        for doc in docs
    )

    final_prompt = prompt.invoke(

        {

            "context":context,

            "question":query
        }
    )

    response = llm.invoke(final_prompt)

    print("\nAI :",response.content)
    print()