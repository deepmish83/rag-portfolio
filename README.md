# RAG Portfolio — Grounded Q&A Service

A small but production-shaped retrieval-augmented generation (RAG) service over a Wikipedia corpus. Built as a personal portfolio piece to demonstrate production engineering patterns — API surface, eval, observability discipline, container-ready structure — applied to a small-scale AI workload.

> **About this project**: This is productio-grade deployment. Though the corpus is small (~10 Wikipedia articles, ~700 chunks). The point is the engineering patterns, not the scale.

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

## Eval framework

A 20-question eval set covers four categories: single-source retrieval, multi-source synthesis, refusal tests, and acronym / paraphrase edge cases. Each question is scored against:
- Required keywords in the answer
- Required source files in the retrieval
- Refusal markers for negative tests (must say "I don't have that information" or equivalent)

See [`eval_set.json`](eval_set.json) for the eval definitions and [`eval.py`](eval.py) for the runner.

**Smoke test (current):** 1 question verified passing end-to-end (`Q05 — What is Microsoft 365?` — 6.0s, correct source retrieved, keyword check passed). The remaining 4 questions in the smoke run errored on free-tier rate limits — see Lessons Learned below. Full 20-question eval is deferred pending paid quota.
 
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
## Run with Docker

The service is packaged as a self-contained Docker image with the embedding model pre-baked, so first-request latency is normal (no cold-download).

```bash
# Build
docker build -t rag-portfolio:latest .

# Run (with .env for keys, host-mounted Chroma store for persistence)
docker run --rm -p 8000:8000 \
  --env-file .env \
  -v "${PWD}/chroma_db:/app/chroma_db" \
  -v "${PWD}/corpus:/app/corpus" \
  rag-portfolio:latest

# Open http://localhost:8000/docs
```

Chroma store and corpus are mounted as volumes so you can swap them without rebuilding the image.

## Four things I learned building this

1. **Retrieval matters more than model choice.** Wrong-source retrieval produced wrong answers no matter how good the LLM was. Most production AI work lives in retrieval engineering, not prompt engineering — and my prior 20 years of data-quality and indexing instincts transfer here.

2. **The grounding rule is one line.** Loosening the system prompt from *"answer ONLY from context"* to *"answer the question"* collapses the entire safety property. Prompt files need version control, review, and regression testing like any other code.

3. **Acronyms break pure embedding retrieval.** Asking _"What is HKIA?"_ failed to retrieve the Hong Kong International Airport article because the embedding model never saw that abbreviation. Hybrid retrieval (BM25 + embedding) isn't theoretical — it solves a real failure mode I observed.

4. **Free LLM tiers are a brittle foundation.** Within two days of repeated eval runs I hit (a) daily quota exhaustion on Google AI Studio and (b) shared-pool rate limits on OpenRouter free models. The production lesson: real AI apps need paid quota, explicit model registry, multi-provider fallback chains, and observability on quota itself — not "point at one provider and hope." This isn't theoretical — it materially blocked my own eval run.

## What's next

- [ ] Hybrid retrieval (BM25 + embedding) with reranking
- [ ] LLM-as-judge eval instead of keyword checks
- [ ] Larger corpus (1,000+ documents)

## About me

AI and Automation Specialist, 20+ years across enterprise workflow automation/rpa/cloud integration, 10 of those in Hong Kong. This repo is one piece of an active portfolio.

Reach me on [LinkedIn](https://www.linkedin.com/in/deepak-mishra-career) — happy to discuss.

## License

MIT.
