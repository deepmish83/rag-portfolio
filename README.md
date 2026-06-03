# RAG Portfolio — Grounded Q&A Service

A small but production-shaped retrieval-augmented generation (RAG) service over a Wikipedia corpus. Built as a personal portfolio piece to demonstrate production engineering patterns — API surface, eval, observability discipline, container-ready structure — applied to a small-scale AI workload.

> **About this project**: This is a learning portfolio, not a production deployment. The corpus is small (~10 Wikipedia articles, ~700 chunks). The point is the engineering patterns, not the scale.

## Demo

![Swagger UI](screenshots/01_swagger_overview.png)
_FastAPI Swagger UI — try POST /ask with any natural-language question._

![Cross-domain answer](screenshots/03_cross_domain_synthesis.png)
_Cross-domain synthesis: retrieves from aviation and AI articles, grounds the answer in both._

## What it does

You ask a natural-language question. The agent:

1. Embeds your query (Sentence Transformer, MiniLM)
2. Retrieves the top 4 most relevant chunks from a Chroma vector store with HNSW indexing
3. Constructs a grounded prompt that constrains the LLM to answer **only** from the retrieved context
4. Returns the answer plus the source files used

## Architecture
question → embedding → Chroma (HNSW) → top-k chunks → grounded prompt → Gemini Flash → answer + sources


## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.13 |
| Orchestration | LangChain 0.3 |
| Vector store | Chroma (local, persisted) |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Indexing | HNSW (Chroma default) |
| LLM | Google Gemini 1.5 Flash |
| API | FastAPI with auto-generated OpenAPI docs |
| Server | Uvicorn |

## Eval results

20-question eval set spanning single-source retrieval, multi-source synthesis, refusal tests, and acronym / paraphrase edge cases.

| Metric | Value |
|---|---|
| Pass rate | **REPLACE_WITH_YOUR_NUMBER / 20** |
| Avg latency | **REPLACE_WITH_YOUR_NUMBER s / question** |
| Refusal-test correctness | **REPLACE_WITH_YOUR_NUMBER / 4** |

Full results in [`eval_results.json`](eval_results.json).

## Quick start

```bash
# 1. Clone and set up
git clone https://github.com/YOUR_USERNAME/rag-portfolio.git
cd rag-portfolio
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Mac/Linux
pip install -r requirements.txt

# 2. Add your Google AI Studio API key
copy .env.example .env         # Windows
# cp .env.example .env         # Mac/Linux
# Edit .env and paste your key

# 3. Fetch the sample corpus (~10 Wikipedia articles)
python fetch_corpus.py

# 4. Build the Chroma store
python ingest.py

# 5. Start the API
uvicorn api:app --reload --port 8000

# 6. Open http://localhost:8000/docs and try it
```

## Three things I learned building this

1. **Retrieval matters more than model choice.** Wrong-source retrieval produced wrong answers no matter how good the LLM was. Most production AI work lives in retrieval engineering, not prompt engineering — and my prior 20 years of data-quality and indexing instincts transfer here.

2. **The grounding rule is one line.** Loosening the system prompt from *"answer ONLY from context"* to *"answer the question"* collapses the entire safety property. Prompt files need version control, review, and regression testing like any other code.

3. **Acronyms break pure embedding retrieval.** Asking _"What is HKIA?"_ failed to retrieve the Hong Kong International Airport article because the embedding model never saw that abbreviation. Hybrid retrieval (BM25 + embedding) isn't theoretical — it solves a real failure mode I observed.

## What's next

- [ ] Hybrid retrieval (BM25 + embedding) with reranking
- [ ] LangSmith observability instead of stdout
- [ ] Dockerise + run as a container
- [ ] LLM-as-judge eval instead of keyword checks
- [ ] Larger corpus (1,000+ documents)

## About me

Senior Solution Architect, ~20 years across enterprise .NET / Azure / workflow automation, 10 of those in Hong Kong. Currently moving into AI and automation delivery. This repo is one piece of an active portfolio.

Reach me on [LinkedIn](https://www.linkedin.com/in/deepak-mishra-career) — happy to discuss.

## License

MIT.