import numpy as np, faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

def build_faiss_from_chunks(chunks: list[str], model_name: str, faiss_path: str) -> None:
    model = SentenceTransformer(model_name)
    embs = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim) 
    index.add(np.asarray(embs, dtype=np.float32))
    Path(faiss_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, faiss_path)

def load_faiss(faiss_path: str):
    return faiss.read_index(faiss_path)

def search(index, query: str, model_name: str, k: int, corpus: list[str]):
    model = SentenceTransformer(model_name)
    q = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.asarray(q, dtype=np.float32), k)
    hits = [(int(i), float(d), corpus[int(i)]) for i, d in zip(I[0], D[0])]
    return hits