"""
RAG agent: Ask user for a question, retrieve relevant documents from Chroma vector database, and generate an answer using the retrieved documents as context.   
"""

import os
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate



load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION = "wiki_articles"    
TOP_K = 4

#1 Embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

#2 Load Chroma vector database
db = Chroma(
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION,
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": TOP_K})

#3 LLM
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from langchain_openai import ChatOpenAI

# OpenRouter — gives access to many models through one API
llm = ChatOpenAI(
    model="deepseek/deepseek-chat-v3-0324",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.2,
    default_headers={
        "HTTP-Referer": "https://github.com/deepmish83/rag-portfolio",
        "X-Title": "RAG Portfolio",
    },
)

#4 Prompt template
prompt_template = ChatPromptTemplate.from_template(
"You are a helpful assistant. Answer the question only using Context below.\n\n"
"If the Context does not contain the answer, say you don't know.\n\n"
"Context:\n{context}\n\n"
"Question:\n{question}\n\n"
"Answer:"
)

def format_context(docs):
   parts = []
   for d in docs:
       source = os.path.basename(d.metadata.get("source", "unknown"))
       parts.append(f"Source: {source}\nContent: {d.page_content}")
   return "\n\n---\n\n".join(parts)

def ask_question(question):
    #1 Retrieve relevant documents
    docs = retriever.invoke(question)
    sources = sorted({
        os.path.basename(d.metadata.get("source", "unknown")) for d in docs})
    
    context = format_context(docs)

    #2 Generate answer
    prompt = prompt_template.invoke({"context": context, "question": question})
    for attempt in range(3):
        try:
            answer = llm.invoke(prompt).content.strip()
            return answer, sources
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 30 * (attempt + 1)
                print(f"Rate limited — retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

if __name__ == "__main__":
    print("Welcome to the RAG Agent! Ask a question about the Wikipedia articles.")
    while True:
        user_input = input("\nYour question (or 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = ask_question(user_input)
        print(f"\nAnswer:\n{response}")