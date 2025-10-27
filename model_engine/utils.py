from pathlib import Path

def load_chunks(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    
    text = path.read_text(encoding="utf-8")
    chunks = [seg.strip() for seg in text.split("\n\n") if seg.strip()]
    if not chunks:
        chunks = [line.strip() for line in text.splitlines() if line.strip()]
    
    print(f"Loaded {len(chunks)} chunks from {path}")

    return chunks
    
def preivew_chunks(chunks, n=3):
    print(f"\n Previewing first {n} chunks:")
    for i,c in enumerate(chunks[:n], 1):
        print(f"[{i}] {c[:200]}...\n")
        
def check_vectorstore_exists(path):
    path = Path(path)
    exists = path.exists() and (path.is_file() or any(path.glob("*")))
    if exists:
        print(f"Vectorstore found at {path}")
    else:
        print(f"Vectorstore not found at {path}")
    
    return exists