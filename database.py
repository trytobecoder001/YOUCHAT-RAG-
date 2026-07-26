from dotenv import load_dotenv
load_dotenv()
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs


url = "https://www.youtube.com/watch?v=rqTTczYv93A"

video_id = parse_qs(urlparse(url).query)["v"][0] # got the Video Id from the URL
ytt_api = YouTubeTranscriptApi()
response =ytt_api.fetch(video_id) # get the CC from the video
cc_text = " ".join(chunks.text for chunks in response)
#print(cc_text)


from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

chunking = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = chunking.split_text(cc_text)

embeddings = GoogleGenerativeAIEmbeddings( model="models/gemini-embedding-001")

vector_store = FAISS.from_texts(
    embedding=embeddings,
    texts= chunks
)

vector_store.save_local("DATABASE")