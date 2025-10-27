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
from dotenv import load_dotenv
load_dotenv()

CHUNKS_FILE = Path(PROCESSED_DIR) / "all_chunks.txt"

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
    print(f"Initializing LangChain vectorstore for [{model_choice}] ...")
    if not check_vectorstore_exists(LANGCHAIN_FAISS_PATH):
        vectorstore = rebuild_vectorstore()
    else:
        embeddings = HuggingFaceEmbeddings(model_name=LANGCHAIN_EMBED_MODEL)
        vectorstore = FAISS.load_local(LANGCHAIN_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        
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