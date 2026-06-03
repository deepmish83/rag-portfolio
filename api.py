"""
Fast API for RAG agent.
RUN uvicorn api:app --reload --port 8000
Docs http://localhost:8000/docs
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent import ask_question

app = FastAPI(
    title="RAG Service API",
    description="API for Retrieval Augmented Generation (RAG) agent using FastAPI",
    version="1.0.0"
)

class Question(BaseModel):
    question:str
   
class Answer(BaseModel):
    question:str
    answer:str
    sources:list[str]

@app.get("/")
def read_root():
    return {
        "service": "RAG Service API",
        "endpoints": {
            "POST /ask": "POST endpoint to ask a question and get an answer with sources",
            "GET /health": "GET endpoint to check if the service is running",
            "GET /docs": "Interactive API documentation with Swagger UI"

        } 
    }
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "RAG Service is running"}

@app.post("/ask", response_model=Answer)
def ask(question: Question):
    answer, sources = ask_question(question.question)
    return Answer(
        question=question.question,
        answer=answer,
        sources=sources
    )

