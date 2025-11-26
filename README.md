# Customer Support FAQs – RAG Demo (LangChain + LangGraph + LangSmith + Streamlit)

An end-to-end Retrieval-Augmented Generation (RAG) demo built on the
**Customer Support FAQs** dataset (Hugging Face: `MakTek/Customer_support_faqs_dataset`).  
It uses:

- **LangChain** for chains, prompts, retrievers
- **LangGraph** for a clean graph-style app flow
- **LangSmith** for tracing and eval
- **Chroma** as a local vector database
- **Streamlit** for a simple, sleek chat UI

## Quickstart

1. **Clone / unzip** this project.
2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. **Install deps**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set env vars**: copy `.env.example` to `.env` and fill in values (at minimum `OPENAI_API_KEY`).
5. **Ingest data** (downloads the HF dataset and builds the vector DB):
   ```bash
   python ingest.py
   ```
6. **Run the app**:
   ```bash
   streamlit run app_streamlit.py
   ```

## What this app does

- Ingests ~200 Q/A pairs into a local Chroma collection
- Uses OpenAI embeddings + a chat model to answer user questions with retrieved context
- Shows a chat interface with sources and streaming tokens
- The flow is orchestrated by **LangGraph** with nodes:
  1. `retrieve` → finds top-k relevant chunks
  2. `grade` → a simple rule-based filter that makes sure we have something relevant
  3. `generate` → composes the final answer (with citations)
  4. `fallback` → graceful behavior when no context is found

## LangSmith

set `LANGSMITH_TRACING=true` and a valid `LANGSMITH_API_KEY` in `.env`,
to see rich traces for runs and to create eval datasets and tests.

## Evaluations

Run:
```bash
python eval.py
```
This computes a simple retrieval recall@k and exact match-ish metric on the tiny dataset.
