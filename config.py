PDF_DIR = "data/pdfs"
PROCESSED_DIR = "data/processed"
FAISS_PATH = "vectorstore/faiss.index"
EMBED_MODEL = "all-MiniLM-L6-v2"   # sentence-transformers
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
USE_SQLITE = True
SQLITE_PATH = "db/metadata.sqlite"


# LangChain/LLM Model Engine config
# Path to processed chunks (input for LangChain VectorStore)
CHUNKS_PATH = "data/processed/all_chunks.txt"

# LangChain vectorstore (rebuild using LangChain)
LANGCHAIN_FAISS_PATH = "vectorstore/faiss_langchain_index"

# Available LLM options: "openai", "huggingface", "llama"
DEFAULT_LLM = "openai"

# Default embedding model for LangChain
LANGCHAIN_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default prompt template for QA
PROMPT_TEMPLATE = """Answer the question using only the context below.
If the answer is not in the context, say 'I don’t have enough information.'

Context:
{context}

Question: {question}
Answer:"""