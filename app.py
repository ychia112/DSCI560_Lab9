from pathlib import Path
from config import *
from src.pdf_extraction import batch_extract
from src.chunking import chunk_text_file
from src.embeddings_faiss import build_faiss_from_chunks, load_faiss, search
from src.retriever import load_corpus_from_chunks_file

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

def main_cli():
    if not Path(FAISS_PATH).exists():
        prepare_pipeline()

    print("Type your question (or 'exit'):")
    while True:
        q = input("> ").strip()
        if q.lower() == "exit":
            break
        hits = retrieve_only(q, TOP_K)
        print("\nTop context:")
        for i,(idx,score,ch) in enumerate(hits,1):
            print(f"[{i}] score={score:.3f}\n{ch[:500]}\n---")

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
    main_cli()