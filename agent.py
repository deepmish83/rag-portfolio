"""
RAG agent: Ask user for a question, retrieve relevant documents from Chroma vector database, and generate an answer using the retrieved documents as context.   
"""

import os   
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
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
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    max_output_tokens=1024,
    api_key=os.getenv("GOOGLE_API_KEY")
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
    answer = llm.invoke(prompt).content.strip()
    return answer, sources

if __name__ == "__main__":
    print("Welcome to the RAG Agent! Ask a question about the Wikipedia articles.")
    while True:
        user_input = input("\nYour question (or 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = ask_question(user_input)
        print(f"\nAnswer:\n{response}")