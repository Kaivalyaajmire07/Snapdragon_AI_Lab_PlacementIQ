# PlacementIQ — Snapdragon AI Lab Build

This package uses the uploaded PlacementIQ FastAPI backend as its base and adds a React frontend plus an optional local-AI path.

## 1. Prerequisites

- Python 3.12+
- Node.js 18+
- PostgreSQL
- Your existing PlacementIQ trained model artifacts
- Your existing student database/schema/data
- For final Snapdragon local-AI testing: a local inference runtime/model on the Snapdragon HP PC

## 2. Restore your existing artifacts

The uploaded `main.py` expects these model files:

backend/trained_models/placement_model_sem6.pkl
backend/trained_models/kmeans_model.pkl
backend/trained_models/kmeans_scaler.pkl
backend/trained_models/cluster_labels.pkl

Copy your existing trained_models folder into `backend/trained_models/`.

The backend also expects your existing PostgreSQL `students` dataset/schema. This package does not invent that schema or fabricate student records.

## 3. Configure

```bash
cp .env.example .env
nano .env
```

Set the PostgreSQL values.

Gemini keys are optional in this version. If they are absent, the AI assistant can use the local Ollama-compatible endpoint.

## 4. Start backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..
source backend/.venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

If your existing environment already has these packages, you can reuse it.

## 5. Start React frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints, normally http://localhost:5173.

## 6. Local AI

The backend supports an Ollama-compatible local endpoint:

`http://127.0.0.1:11434/api/chat`

Set:

```env
LOCAL_AI_ENABLED=true
LOCAL_AI_MODEL=qwen2.5:3b
```

On the final Snapdragon HP PC, use a local/open-source model and runtime that you have actually tested on that device. Do not claim NPU acceleration until verified.

## 7. Challenge demo

Recommended flow:

1. Login.
2. Open Dashboard.
3. Select a student.
4. Show placement probability/readiness.
5. Show SHAP explanation.
6. Open AI Assistant.
7. Ask why readiness is low and what skills to improve.
8. Demonstrate local inference on the Snapdragon HP PC.
9. Show architecture and deployment evidence.

## Important

This package contains your uploaded backend code as the base, but the uploaded file did not include your trained model binaries, PostgreSQL database, or existing frontend assets. Those must be copied from your original project. The package deliberately does not fabricate them.
