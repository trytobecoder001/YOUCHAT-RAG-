from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")

vector_store = FAISS.load_local(
    "DATABASE",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k": 3,
        "fetch_k":7,
        "lambda_mult" :0.4
    }
)

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt = ChatPromptTemplate.from_messages([
    ("system","""You are an AI assistant that answers questions based ONLY on the provided YouTube transcript.

Instructions:
1. Use only the provided context.
2. If the answer is not present in the context, say:
"I couldn't find the answer in the provided transcript."
3. Do not make up facts.
4. Explain the answer clearly and concisely."""),("human",
"""
context :{context}
Question :{message}""")
])

print("RAG System Created")
print("press 0 to exit")

while True:
    query = input("you:")
    if query == "0":
        break
    docs = retriever.invoke(query)
    
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context":context,
        "message":query
    })
    response = llm.invoke(final_prompt)

    if isinstance(response.content, list):
        print("\nAI:", response.content[0]["text"])
    else:
        print("\nAI:", response.content)