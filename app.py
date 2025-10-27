from pathlib import Path
from config import *
from src.pdf_extraction import batch_extract
from src.chunking import chunk_text_file
from src.embeddings_faiss import build_faiss_from_chunks, load_faiss, search
from src.retriever import load_corpus_from_chunks_file
import argparse
from model_engine.vectorstore import rebuild_vectorstore
from model_engine.conversation import run_conversation
from model_engine.utils import check_vectorstore_exists
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from model_engine.conversation import build_llm
from dotenv import load_dotenv
load_dotenv()
import shutil

CHUNKS_FILE = Path(PROCESSED_DIR) / "all_chunks.txt"

def purge_langchain_store():
    """Delete the LangChain FAISS folder so we never load stale data."""
    p = Path(LANGCHAIN_FAISS_PATH)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)

def load_vectorstore_for_chat():
    """Load (or rebuild) the LangChain FAISS vectorstore once for answering."""
    if not check_vectorstore_exists(LANGCHAIN_FAISS_PATH):
        return rebuild_vectorstore()
    embeddings = HuggingFaceEmbeddings(model_name=LANGCHAIN_EMBED_MODEL)
    return FAISS.load_local(LANGCHAIN_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

def answer_with_llm(question: str, model_choice: str = DEFAULT_LLM) -> str:
    """
    Use the freshly built FAISS via retrieve_only() for context,
    then generate the final answer with the selected LLM (OpenAI, etc.).
    """
    try:
        hits = retrieve_only(question, k=TOP_K)  # [(idx, score, chunk_text), ...]
    except Exception as e:
        return f"Retrieval error: {e}"

    if not hits:
        return "I don’t have enough information."

    context = "\n\n".join([ch[:800] for _, _, ch in hits[:3]])
    prompt = f"""
You are a helpful assistant that answers based only on the provided context.

Context:
{context}

Question: {question}
Answer:
""".strip()

    try:
        llm = build_llm(model_choice=model_choice)
        resp = llm.invoke(prompt)
        return resp.content.strip() if hasattr(resp, "content") else str(resp)
    except Exception as e:
        return f"Model error: {e}"

def prepare_pipeline():
    # 1) PDF -> text
    txt_files = batch_extract(PDF_DIR, PROCESSED_DIR)

    # 2) text -> chunks
    all_chunks = []
    for t in txt_files:
        chunks = chunk_text_file(t, PROCESSED_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(chunks)
    CHUNKS_FILE.write_text("\n\n".join(all_chunks), encoding="utf-8")

    # 3) chunks -> embeddings -> FAISS
    build_faiss_from_chunks(all_chunks, EMBED_MODEL, FAISS_PATH)
    print(f"Prepared. {len(all_chunks)} chunks -> {FAISS_PATH}")

def retrieve_only(query: str, k: int = TOP_K):
    index = load_faiss(FAISS_PATH)
    corpus = load_corpus_from_chunks_file(str(CHUNKS_FILE))
    hits = search(index, query, EMBED_MODEL, k, corpus)
    return hits  # [(idx, score, chunk_text), ...]

def main_cli(model_choice="openai"):
    if not Path(FAISS_PATH).exists():
        prepare_pipeline()
        
    # Build or load LangChain vectorstore
    print(f"Rebuilding LangChain vectorstore for [{model_choice}] ...")
    purge_langchain_store()           # <--- nuke old index
    vectorstore = rebuild_vectorstore()
        
    # # Build conversation chain
    # chain = build_conversation_chain(vectorstore, model_choice=model_choice)
    # print("LangChain chatbot initialized.\n")
        
    # Interactive CLI
    print("Starting Chatbot CLI...\n")
    run_conversation(vectorstore, model_choice=model_choice)
    # print("Type your question (or 'exit'):")
    # while True:
    #     q = input("> ").strip()
    #     if q.lower() == "exit":
    #         print("Exiting chatbot.")
    #         break
        
    #     hits = retrieve_only(q, TOP_K)
    #     print("\nTop context:")
    #     for i,(idx,score,ch) in enumerate(hits,1):
    #         print(f"[{i}] score={score:.3f}\n{ch[:500]}\n---")
            
    #     # LLM response
    #     print("\n Generating AI response...")
    #     response = chain({"question": q})
    #     print("Bot:", response["answer"], "\n")

        # ---------------------------------------------------------------------
        # NOTE:
        # for the LLM or front-end integration.
        # Here, the top-k retrieved text chunks can be passed into a local LLM
        # (e.g., using Ollama or a Hugging Face model) to generate the final 
        # chatbot response based on these context snippets.
        #
        # Example:
        #
        # from ollama import chat
        # context = "\n\n".join([c for _,_,c in hits])
        # prompt = f"Answer the following question using only the context:\n{context}\n\nQuestion: {q}\nAnswer:"
        # answer = chat(model="phi3", messages=[{"role": "user", "content": prompt}])["message"]["content"]
        # print("\nAnswer:\n", answer)
        # ---------------------------------------------------------------------

if __name__ == "__main__":
    prepare_pipeline()
    parser = argparse.ArgumentParser(description="LLM Chatbot")
    parser.add_argument(
        "--model",
        type=str,
        default="openai",
        choices=["openai", "huggingface", "llama"],
        help="Choose LLM backend (default: openai)",
    )
    args = parser.parse_args()
    
    main_cli(model_choice=args.model)