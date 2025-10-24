from PyPDF2 import PdfReader
from pathlib import Path

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    texts = []
    for p in reader.pages:
        t = p.extract_text() or ""
        texts.append(t.replace("\x00","").strip())
    return "\n".join(texts)

def batch_extract(pdf_dir: str, out_dir: str) -> list[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    outputs = []
    for pdf in Path(pdf_dir).glob("*.pdf"):
        text = extract_text_from_pdf(str(pdf))
        out_path = Path(out_dir) / (pdf.stem + ".txt")
        out_path.write_text(text, encoding="utf-8")
        outputs.append(str(out_path))
    return outputs