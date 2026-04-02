# RAG MLOps

A production-ready Retrieval-Augmented Generation (RAG) system with a full MLOps feedback loop. Users ask questions, the system retrieves relevant document chunks, generates answers via a local LLM, and collects feedback to continuously fine-tune a CrossEncoder reranker.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        React Frontend                       │
│         Chat  ·  Document Management  ·  Dashboard          │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────┐
│                      FastAPI Backend                        │
│   /query  ·  /feedback  ·  /upload  ·  /documents           │
└──────┬──────────────┬───────────────────────┬───────────────┘
       │              │                       │
┌──────▼──────┐ ┌─────▼──────┐ ┌──────────────▼─────────────┐
│   Custom    │ │ LlamaIndex │ │       PostgreSQL            │
│  Pipeline   │ │  Pipeline  │ │  feedback · documents       │
│             │ │            │ └─────────────┬───────────────┘
│ FAISS Index │ │  Vector    │               │
│ CrossEncoder│ │  Store     │ ┌─────────────▼───────────────┐
│  Reranker   │ │            │ │     Retraining Scheduler    │
└──────┬──────┘ └─────┬──────┘ │  threshold · schedule       │
       │              │        └─────────────┬───────────────┘
┌──────▼──────────────▼──────┐               │
│       Ollama (LLM)          │ ┌─────────────▼───────────────┐
│     llama3.1:8b             │ │          MLflow             │
└─────────────────────────────┘ │   experiment tracking       │
                                └─────────────────────────────┘
```

### Project Structure

```
rag-mlops/
├── data/                        # uploaded documents
├── indexes/                     # FAISS index + chunks
├── models/reranker/             # fine-tuned CrossEncoder
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI app + CORS + lifespan
│   │   ├── routes.py            # all API endpoints
│   │   └── schemas.py           # request/response models
│   ├── feedback/
│   │   ├── models.py            # SQLAlchemy table definitions
│   │   └── collector.py         # database operations
│   ├── rag/
│   │   ├── loader.py            # document loading + chunking
│   │   ├── embedder.py          # sentence-transformers embedder
│   │   ├── indexer.py           # FAISS index build/search/delete
│   │   └── pipeline.py          # full RAG pipeline + reranker
│   ├── llamaindex/
│   │   └── pipeline.py          # LlamaIndex-based pipeline
│   └── training/
│       ├── train_reranker.py    # CrossEncoder fine-tuning
│       └── scheduler.py         # automated retraining triggers
├── notebooks/     
├── tests/              
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## How the MLOps Feedback Loop Works

```
User asks a question
        ↓
FAISS retrieves top-10 candidate chunks
        ↓
CrossEncoder reranker scores and reorders chunks
        ↓
Top-3 chunks sent to Llama3.1:8b as context
        ↓
Answer returned to user
        ↓
User submits a score (1–5)
        ↓
Score + chunks saved to PostgreSQL
        ↓
Scheduler checks two triggers:
  · Threshold: every 50 new feedbacks → retrain
  · Schedule:  every 24 hours → retrain
        ↓
train_reranker.py fine-tunes CrossEncoder on feedback data:
  · score 4–5 → positive example (chunks were relevant)
  · score 1–2 → negative example (chunks were not relevant)
        ↓
New model saved to models/reranker/ and logged to MLflow
        ↓
Better reranker → better answers → better scores → more data
```

---

## Installation & Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com) running locally with `llama3.1:8b` pulled
- PostgreSQL (local or via Docker)

### 1. Clone the repo

```bash
git clone https://github.com/lantshoe/rag-mlops.git
cd rag-mlops
```

### 2. Set up the backend

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://raguser:ragpassword@localhost:5432/ragdb
MLFLOW_TRACKING_URI=http://localhost:5001
```

### 4. Pull the LLM

```bash
ollama pull llama3.1:8b
```

### 5. Set up the frontend

```bash
cd frontend
npm install
```

---

## How to Run

### Local Development

**Start the backend:**
```bash
uvicorn src.api.main:app --reload --port 8000
```

**Start the frontend:**
```bash
cd frontend
npm run dev
# → http://localhost:5173
```

**Start MLflow (optional):**
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5001
# → http://localhost:5001
```

---

### Docker (recommended for production)

Make sure Docker and Docker Compose are installed, then:

```bash
# Copy and configure environment
cp .env.example .env

# Build and start all services
docker-compose up --build
```

| Service      | URL                      |
|--------------|--------------------------|
| Frontend     | http://localhost:3000    |
| Backend API  | http://localhost:8000    |
| MLflow UI    | http://localhost:5001    |
| PostgreSQL   | localhost:5432           |

> **Note:** Ollama must be running on the host machine. The backend connects to it via `http://host.docker.internal:11434`.

---

## API Endpoints

### Health

| Method | Endpoint  | Description          |
|--------|-----------|----------------------|
| GET    | `/health` | Check if API is live |

### Query

| Method | Endpoint | Description                        |
|--------|----------|------------------------------------|
| POST   | `/query` | Ask a question, get an answer      |

**Request:**
```json
{
  "question": "What is virtual memory?",
  "pipeline": "custom"
}
```
`pipeline` can be `"custom"` (FAISS + CrossEncoder) or `"llamaindex"`.

**Response:**
```json
{
  "question": "What is virtual memory?",
  "answer": "Virtual memory is...",
  "pipeline": "custom",
  "retrieved_chunks": [
    {
      "text": "...",
      "score": 0.821,
      "reranker_score": 0.963,
      "source": "11OSproject.docx",
      "chunk_id": 4
    }
  ]
}
```

### Feedback

| Method | Endpoint    | Description                    |
|--------|-------------|--------------------------------|
| POST   | `/feedback` | Submit a score for an answer   |
| GET    | `/feedback/stats` | Get feedback statistics  |

**Request:**
```json
{
  "question": "What is virtual memory?",
  "answer": "Virtual memory is...",
  "score": 5,
  "pipeline": "custom",
  "comment": "",
  "retrieved_chunks": [...]
}
```

### Documents

| Method | Endpoint                   | Description                          |
|--------|----------------------------|--------------------------------------|
| GET    | `/documents`               | List all indexed documents           |
| POST   | `/upload`                  | Upload and index a new document      |
| DELETE | `/documents/{filename}`    | Delete a document and rebuild index  |

**Upload** uses `multipart/form-data` with a `file` field. Supported formats: `.docx`, `.pdf`, `.txt`.

---

## Manual Retraining

To trigger retraining manually (requires at least 1 feedback entry):

```bash
python -m src.training.train_reranker
```

Training runs are logged to MLflow and the model is saved to `models/reranker/`.

---

## Tech Stack

| Layer        | Technology                                      |
|--------------|-------------------------------------------------|
| LLM          | Llama 3.1 8B via Ollama                         |
| Embeddings   | `all-MiniLM-L6-v2` (sentence-transformers)      |
| Vector store | FAISS (IndexFlatIP + L2 normalization)          |
| Reranker     | `cross-encoder/ms-marco-MiniLM-L-6-v2`         |
| RAG layer    | Custom pipeline + LlamaIndex                    |
| Backend      | FastAPI + SQLAlchemy + PostgreSQL               |
| MLOps        | MLflow experiment tracking                      |
| Frontend     | React + Vite + Tailwind CSS                     |
| Deployment   | Docker + Docker Compose + Nginx                 |
