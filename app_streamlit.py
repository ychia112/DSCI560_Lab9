from pathlib import Path
from typing import List
import streamlit as st

# Backend imports
from config import PDF_DIR, DEFAULT_LLM 
from app import prepare_pipeline, answer_with_llm

PDF_DIR_PATH = Path(PDF_DIR)

def save_uploaded_pdfs(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> int:
    PDF_DIR_PATH.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in files or []:
        if not f or not getattr(f, "name", "").lower().endswith(".pdf"):
            continue
        (PDF_DIR_PATH / f.name).write_bytes(f.read())
        saved += 1
    return saved

# UI
st.set_page_config(page_title="DSCI 560 Lab 9: Chatbot UI", page_icon="💬", layout="wide")
st.title("💬 DSCI 560 Lab 9: Chatbot")
st.caption("Upload PDFs -> Analyze -> Chat")

with st.sidebar:
    st.header("📄 PDF Upload")
    uploaded = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    analyze = st.button("🔧 Analyze PDFs")

if "backend_ready" not in st.session_state:
    st.session_state.backend_ready = False
if "messages" not in st.session_state:
    st.session_state.messages: List[dict] = []

# Build embeddings/index (once) so the vectorstore exists
if analyze:
    if not uploaded:
        st.warning("Please upload at least one PDF.")
    else:
        with st.status("Saving PDFs and running pipeline…", expanded=True) as s:
            n = save_uploaded_pdfs(uploaded)
            st.write(f"Saved {n} file(s) to `{PDF_DIR_PATH}`.")
            try:
                prepare_pipeline()  # PDF -> text -> chunks -> embeddings -> FAISS
                st.session_state.backend_ready = True
                s.update(label="Done", state="complete")
                st.success("Vector store prepared.")
            except Exception as e:
                st.session_state.backend_ready = False
                st.error(f"Pipeline failed: {e}")

st.subheader("Chat")

# Show history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# Input -> final response
user_input = st.chat_input("Ask a question about the uploaded PDFs...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    if not st.session_state.backend_ready:
        answer = "Please upload and analyze PDFs first from the sidebar."
    else:
        answer = answer_with_llm(user_input, model_choice=DEFAULT_LLM)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
