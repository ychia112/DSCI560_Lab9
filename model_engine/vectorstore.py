from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from config import CHUNKS_PATH, LANGCHAIN_FAISS_PATH, LANGCHAIN_EMBED_MODEL
from model_engine.utils import load_chunks

def build_embeddings(use_openai=False):
    if use_openai:
        print("Using OpenAIEmbeddings")
        return OpenAIEmbeddings()
    print(f"Using HF Embeddings: {LANGCHAIN_EMBED_MODEL}")
    return HuggingFaceEmbeddings(model_name=LANGCHAIN_EMBED_MODEL)


def rebuild_vectorstore(use_openai=False):
    print("Loading chunks from:", CHUNKS_PATH)
    chunks = load_chunks(CHUNKS_PATH)
    
    print("Generating embeddings...")
    embeddings = build_embeddings(use_openai=use_openai)
    
    print("Building FAISS vectorstore ...")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    save_dir = Path(LANGCHAIN_FAISS_PATH)
    save_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_dir))
    print(f"LangChain FAISS vectorstore saved at: {LANGCHAIN_FAISS_PATH}")
    
    return vectorstore

def load_vectorstore(use_openai=False):
    embeddings = build_embeddings(use_openai=use_openai)
    index_dir = Path(LANGCHAIN_FAISS_PATH)
    
    if not (index_dir.exists() and any(index_dir.glob("*"))):
        raise FileNotFoundError(f"No FAISS index found in {LANGCHAIN_FAISS_PATH}.")
    print(f"Loading FAISS vectorstore from {LANGCHAIN_FAISS_PATH}")
    return FAISS.load_local(str(index_dir), embeddings)