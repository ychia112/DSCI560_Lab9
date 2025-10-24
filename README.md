# DSCI560_Lab9
This is a lab for DSCI560 doing Custom Q&amp;A Chatbot


## File Overview

| File | Description |
|------|--------------|
| `src/pdf_extraction.py` | Converts PDF → text |
| `src/chunking.py` | Splits text into chunks |
| `src/embeddings_faiss.py` | Embeds chunks and builds FAISS vector store |
| `src/retriever.py` | Retrieves top-k similar chunks for a query |
| `app.py` | Main driver — includes a placeholder section for chatbot logic |
| `config.py` | Configuration (paths, model name, chunk size, etc.) |
| `data/processed/` | Contains extracted text and chunk files |
| `vectorstore/` | Contains FAISS vector index (`faiss.index`) |

## Run the Pipeline

To build the vector database (no API key needed):

```bash
python -c "from app import prepare_pipeline; prepare_pipeline()"
```

## Next Step

I've modified the app.py, there's a marked section that you might do:
- Loading a open-source LLM
- Using retrieved chunks as input context
- Generate the chatbot's response