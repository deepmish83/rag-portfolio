"""
Ingest corpus/*.text into a loacal Chroma vector database. This is our RAG document store for today.
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

CORPUS_DIR = "corpus"
CHROMA_DIR = "chroma_db"
COLLECTION = "wiki_articles"

#1 Load documents
print("Loading documents...")
documents = []
for file_path in Path(CORPUS_DIR).glob("*.txt"):
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    documents.extend(docs)
print(f"Loaded {len(documents)} documents.")    

#2 Chunk
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=80,separators=["\n\n", "\n", " ", ""]
    )
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")

#3 Embed
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

#4 Store
print("Storing in Chroma vector database...")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=COLLECTION,
    persist_directory=CHROMA_DIR
)

print("Ingestion complete. Chroma database is ready.")