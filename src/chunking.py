from langchain_text_splitters import CharacterTextSplitter
from pathlib import Path

def chunk_text_file(txt_path: str, out_dir: str, chunk_size=500, overlap=50) -> list[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    raw = Path(txt_path).read_text(encoding="utf-8")
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(raw)

    lines_path = Path(out_dir) / (Path(txt_path).stem + "_chunks.txt")
    lines_path.write_text("\n\n".join(chunks), encoding="utf-8")
    return chunks