from pathlib import Path

def load_corpus_from_chunks_file(chunks_txt_path: str) -> list[str]:
    raw = Path(chunks_txt_path).read_text(encoding="utf-8")
    return [c.strip() for c in raw.split("\n\n") if c.strip()]