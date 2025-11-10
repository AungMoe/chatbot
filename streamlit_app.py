import streamlit as st
import io
import re
from typing import List

# Optional libraries for richer extraction. Install if you want PDF/DOCX support:
# pip install PyPDF2 python-docx
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx
except Exception:
    docx = None

st.set_page_config(page_title="Chatbot (file-aware, no API key)", page_icon="💬")

st.title("💬 Chatbot (file-aware, no API key)")
st.write(
    "This chatbot reads files you upload and answers questions using only the uploaded content — "
    "no OpenAI API key required. It performs a lightweight keyword-based retrieval over uploaded files "
    "and returns relevant passages. For better PDF/DOCX extraction, install PyPDF2 and python-docx."
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    # uploaded_files: list of dicts {name, text, chunks}
    st.session_state.uploaded_files = []

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if PyPDF2 is None:
        return ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    except Exception:
        return ""

def extract_text_from_docx(file_bytes: bytes) -> str:
    if docx is None:
        return ""
    try:
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in document.paragraphs)
    except Exception:
        return ""

def chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    """Split text into chunks respecting paragraphs and sentence boundaries where possible."""
    if not text:
        return []
    paragraphs = re.split(r"\n{1,}", text)
    chunks: List[str] = []
    current = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current) + len(p) + 1 > max_chars:
            if current:
                chunks.append(current.strip())
            if len(p) > max_chars:
                # split large paragraph into sentence-like pieces
                sentences = re.split(r"(?<=[.!?])\s+", p)
                buf = ""
                for s in sentences:
                    if len(buf) + len(s) + 1 > max_chars:
                        if buf:
                            chunks.append(buf.strip())
                        buf = s
                    else:
                        buf = f"{buf} {s}" if buf else s
                if buf:
                    chunks.append(buf.strip())
                current = ""
            else:
                current = p
        else:
            current = f"{current}\n{p}" if current else p
    if current:
        chunks.append(current.strip())
    return chunks

def parse_uploaded_file(uploaded_file) -> str:
    """Extract text from uploaded file. Supports txt, pdf, docx when libraries are available."""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    text = ""
    if name.endswith(".pdf"):
        text = extract_text_from_pdf(data)
    elif name.endswith(".docx"):
        text = extract_text_from_docx(data)
    else:
        # try to decode as text
        try:
            text = data.decode("utf-8")
        except Exception:
            try:
                text = data.decode("latin-1")
            except Exception:
                text = ""
    # fallback crude representation
    if not text:
        try:
            text = str(data)
        except Exception:
            text = ""
    return text

def index_uploaded_files(uploaded_files) -> None:
    """Index uploaded files into session_state.uploaded_files (name, text, chunks)."""
    for f in uploaded_files:
        name = f.name
        parsed_text = parse_uploaded_file(f)
        chunks = chunk_text(parsed_text)
        st.session_state.uploaded_files.append({
            "name": name,
            "text": parsed_text,
            "chunks": chunks,
        })

# Uploader widget
uploaded = st.file_uploader("Upload text, PDF or DOCX files", accept_multiple_files=True)
if uploaded:
    # Re-index fresh upload set
    st.session_state.uploaded_files = []
    index_uploaded_files(uploaded)

# Show uploaded files with previews
if st.session_state.uploaded_files:
    for uf in st.session_state.uploaded_files:
        with st.expander(f"{uf['name']} ({len(uf['chunks'])} chunks)"):
            preview = uf["text"][:3000]
            if not preview:
                st.write("(Could not extract a preview for this file.)")
            else:
                st.text_area("Preview", preview, height=200)
else:
    st.info("No files uploaded yet — upload files so the assistant can answer using their contents.")

# Render existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def simple_retrieve(query: str, top_k: int = 3):
    """Keyword-overlap retrieval: returns list of {file, chunk, score}."""
    tokens = re.findall(r"\w+", query.lower())
    if not tokens:
        return []
    results = []
    for uf in st.session_state.uploaded_files:
        for i, chunk in enumerate(uf["chunks"]):
            words = re.findall(r"\w+", chunk.lower())
            if not words:
                continue
            # score = number of query tokens appearing in chunk (simple)
            score = sum(words.count(t) for t in tokens)
            if score > 0:
                results.append({
                    "file": uf["name"],
                    "chunk": chunk,
                    "score": score,
                    "index": i,
                })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]

def build_answer_from_matches(matches, query):
    """Create a human-friendly answer composed of top matching snippets with simple highlighting."""
    if not matches:
        return ("I couldn't find any passages in the uploaded files that match your question. "
                "Try a different question or upload more files.")
    lines = []
    lines.append("I searched your uploaded files and found these relevant passages:")
    for m in matches:
        snippet = m["chunk"]
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "..."
        # simple highlighting: surround matched tokens with backticks
        qtokens = set(re.findall(r"\w+", query.lower()))
        def highlight(text: str) -> str:
            def repl(match):
                w = match.group(0)
                return f"`{w}`" if w.lower() in qtokens else w
            return re.sub(r"\w+", repl, text)
        snippet_high = highlight(snippet)
        lines.append(f"From {m['file']} (score={m['score']}):\n> {snippet_high}")
    lines.append("\nIf you'd like, ask me to summarize these passages, or ask a follow-up that references a file name.")
    return "\n\n".join(lines)

# Chat input (questions about uploaded files)
if prompt := st.chat_input("Ask about the uploaded files..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.uploaded_files:
        matches = simple_retrieve(prompt, top_k=5)
        assistant_response = build_answer_from_matches(matches, prompt)
    else:
        assistant_response = (
            "No files have been uploaded. This assistant can only answer based on the files you upload. "
            "Upload one or more files (txt, pdf, docx) and ask a question about their contents."
        )

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

st.markdown("---")
st.write(
    "This assistant runs locally in your Streamlit app and does not require an OpenAI API key. "
    "It uses a simple keyword-based retrieval over uploaded files. For better extraction "
    "from PDFs and DOCX, install PyPDF2 and python-docx."
)
