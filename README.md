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
| `model_engine/utils.py` | Contains utility functions such as checking vectorstore existence, managing directories, or performing cleanup before rebuild. Helps ensure stable pipeline execution. |
| `model_engine/vectorstore.py` | Rebuilds or loads the LangChain FAISS vectorstore using HuggingFace embeddings. Connects the precomputed FAISS index (from /src/embeddings_faiss.py) to the chatbot layer. |
| `model_engine/coversation.py` | Defines the chatbot’s main interaction logic. Handles user prompts, retrieves top-k context chunks from the vectorstore, builds the final prompt, and calls the selected LLM (OpenAI / HuggingFace / LLaMA). |

## Run the Pipeline

To build the vector database (no API key needed):

```bash
python -c "from app import prepare_pipeline; prepare_pipeline()"
```

## The Pipeline, LLM models and chat bot

.env settings:

```bash
# .env.example
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

### Run 
```bash
# Run with OpenAI (default)
python app.py
# or
python app.py --model openai

# Run with Hugging Face (flan-t5-small)
python app.py --model huggingface

# Run with Local LLaMA (GGUF)
python app.py --model llama
```

When started, you'll see:
```bash
Chatbot ready using OPENAI! Type 'exit' to quit.
> how to install ADS2011?
```

### Model Backends Comparison

| Backend | Library | Response Speed | Quality | Notes |
|----------|----------|----------------|----------|--------|
| **OpenAI GPT (GPT-3.5 / GPT-4)** | `langchain_openai` | ⚡ **Very fast (~2-3 s)** | **Highly accurate and context-aware** | Requires API key; ideal for final demo |
| **HuggingFace (Flan-T5-Small/Base)** | `langchain_community` | **Fairly fast (~5 s)** | **Tends to word-match instead of reason** | Free to use but less coherent |
| **Local LLaMA (Q4_K_M)** | `llama-cpp-python` | **Slow (10 s +)** | **Surprisingly coherent for offline mode** | Runs locally without API key; heavier on CPU |

---

## Open Chatbot Interface

```bash
pip install streamlit
```

```bash
streamlit run app_streamlit.py
```

Upload any pdf, press Analyze, and ask questions!