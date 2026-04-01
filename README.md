# RAG Pipeline Demo

A simple, fully-logged **Retrieval-Augmented Generation** demo using **OpenAI** + **ChromaDB** + **Flask**.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI key
cp .env.example .env
# Edit .env and paste your actual key

# 3. Run
python app.py
```

Open **http://localhost:5050** in your browser.

## How It Works

```
User Question
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Embed Query │ ──▶ │  ChromaDB    │ ──▶ │  Top-K Docs  │
│  (OpenAI)    │     │  Vector      │     │  Retrieved   │
└─────────────┘     │  Search      │     └──────┬───────┘
                    └──────────────┘            │
                                               ▼
                                    ┌──────────────────┐
                                    │  OpenAI GPT-4o   │
                                    │  (with context)  │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                        Answer + Sources
```

## Pipeline Stages (all logged)

| Stage      | What it does                                     |
|------------|--------------------------------------------------|
| **Ingest** | Receives text, splits into overlapping chunks     |
| **Embed**  | Calls OpenAI `text-embedding-3-small` for vectors |
| **Store**  | Upserts chunks + embeddings into ChromaDB         |
| **Retrieve** | Embeds the query, does cosine similarity search |
| **Generate** | Sends retrieved context + query to GPT-4o-mini  |

## Project Structure

```
├── app.py              # Flask backend + RAG pipeline
├── static/
│   └── index.html      # Single-page UI
├── requirements.txt
├── .env.example
└── README.md
```
