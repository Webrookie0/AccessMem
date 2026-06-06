# Ollama Setup Guide

This project has been migrated from OpenAI to **Ollama** for both LLM and embedding models.

## Prerequisites

1. **Install Ollama** from https://ollama.ai
2. **Start the Ollama server**:
   ```bash
   ollama serve
   ```
   The server will run on `http://localhost:11434` by default.

3. **Pull the required models** (in a new terminal):
   ```bash
   # LLM Model (Qwen 2.5 Coder)
   ollama pull qwen2.5-coder:7b

   # Embedding Model (Nomic Embed Text)
   ollama pull nomic-embed-text
   ```

## Configuration

1. **Copy `.env.example` to `.env`**:
   ```bash
   cp .env.example .env
   ```

2. **Verify the `.env` file** has the correct Ollama configuration:
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_CHAT_MODEL=qwen2.5-coder:7b
   OLLAMA_EMBED_MODEL=nomic-embed-text
   ```

## Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest documents (embeds them using Ollama)
python -m app.ingest

# Run tests (RBAC/guardrails don't need the LLM)
pytest -q

# Start the API server
uvicorn app.main:app --reload

# In another terminal, start the Streamlit UI
streamlit run ui/streamlit_app.py
```

## Models Used

- **LLM**: `qwen2.5-coder:7b` - A compact code-focused model perfect for RAG tasks
- **Embeddings**: `nomic-embed-text` - Efficient embedding model (768-dim) for semantic search

## Key Changes from OpenAI

### 1. Embeddings (`app/services/embeddings.py`)
- **Before**: Used `OpenAI()` client with `text-embedding-3-small`
- **After**: Uses HTTP requests to Ollama `/api/embed` endpoint with `nomic-embed-text`

### 2. LLM Chat (`app/services/rag.py`)
- **Before**: Used `OpenAI().chat.completions.create()` with `gpt-4o-mini`
- **After**: Uses HTTP requests to Ollama `/api/chat` endpoint with `qwen2.5-coder:7b`
- Added error handling for Ollama connection failures

### 3. Configuration (`app/config.py`)
- **Before**: Required `OPENAI_API_KEY`
- **After**: Uses `OLLAMA_BASE_URL`, `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL`

## Troubleshooting

**Q: "Failed to get embedding from Ollama"**
- Ensure Ollama is running: `ollama serve`
- Check if models are pulled: `ollama list`
- Verify OLLAMA_BASE_URL is correct

**Q: "Connection refused at localhost:11434"**
- Make sure Ollama server is started in a separate terminal
- Check if port 11434 is available

**Q: "Model not found" error**
- Pull the required models:
  ```bash
  ollama pull qwen2.5-coder:7b
  ollama pull nomic-embed-text
  ```

## Performance Notes

- **First embeddings/LLM call will be slower** (model loading into memory)
- Subsequent calls will be faster as models stay in VRAM
- On systems with <16GB RAM, consider quantized 4-bit versions:
  - `ollama pull qwen2.5-coder:4b-q4_0`
  - Models will load slower but use less memory

## Running with Custom Models

To use different Ollama models, update your `.env` file:
```env
OLLAMA_CHAT_MODEL=mistral:latest
OLLAMA_EMBED_MODEL=mxbai-embed-large:latest
```

Then restart the server. No code changes needed!
