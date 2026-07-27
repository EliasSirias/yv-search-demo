import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

import streamlit as st

st.set_page_config(page_title="MVP (Demo)", page_icon="🔎")
# st.set_page_config(page_title="YV Search (Demo)", page_icon="🔎")
import pdfplumber
import re
from pathlib import Path
import pytesseract


from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)  # solid + cheap-ish
# --------------------------------------------------
# Future Phase:
# Optional AI Summary (currently disabled)
# --------------------------------------------------

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -------------------------------
# Config (must be first Streamlit call)
# -------------------------------

# PROJECT_CODENAME = "YV"

# -------------------------------
# UI Style
# -------------------------------
st.markdown(
    """
<style>
/*Margin For Answer Retrieved*/
.bot-header {
    margin-top: 0.2rem;
    margin-bottom: 0.4rem;
    font-weight: 600;
}
/* Preserve newlines in answer text */
.chat-bubble pre,
.chat-bubble .answer-text {
    white-space: pre-wrap;   /* keeps line breaks and wraps nicely */
    word-wrap: break-word;
}
/* Emphasized helper text */
.confidence-note {
    color: rgba(0, 0, 139, 0.95);   /* YV deep blue */
    font-size: 0.95rem;
    font-weight: 600;
    margin-top: 0.35rem;
}

/* Page width & spacing */
.block-container {
    padding-top: 2rem;
    max-width: 900px;
}

/* Primary buttons – premium touch */
.stButton > button {
    border-radius: 14px;
    padding: 0.6rem 1.1rem;
    font-weight: 600;
    border: 1px solid rgba(49, 51, 63, 0.20);
}

.stButton > button:hover {
    border: 1px solid rgba(0, 0, 139, 0.45); /* subtle YV blue */
}

/* Chat bubbles (shared) */
.chat-bubble {
    padding: 0.9rem 1.1rem;
    border-radius: 18px;
    margin: 0.6rem 0;
    border: 1px solid rgba(49, 51, 63, 0.18);
    line-height: 1.45;
}

/* User messages */
.chat-bubble.user {
    background: rgba(0, 122, 255, 0.10);
}

/* Bot messages */
.chat-bubble.bot {
    background: rgba(46, 204, 113, 0.10);
}
</style>

""",
    unsafe_allow_html=True,
)

# st.title("🔎 YV Search (Demo)")
st.title("🔎 MVP Search (Demo)")
st.caption("Hallucination-resistant RAG assistant over example technical documentation")
# st.caption(f"{PROJECT_CODENAME} • YV-hard work defines it")

# -------------------------------
# Retrieval guardrails (define BEFORE use)
# -------------------------------
NOT_FOUND_MESSAGE = (
    "This information isn’t found in the available documentation.<br>"
    "Please check your core tenant configuration and source material.<br>"
    "If needed, escalate to the appropriate team members."
)

knowledge_base_filter = st.sidebar.selectbox(
    "Search in",
    options=["All", "Generic", "Work", "Sample"],
    index=0,
)

st.sidebar.header("Settings")
k = st.sidebar.slider("Top-K retrieved chunks", 1, 4, 2)
min_hit_count = st.sidebar.slider("Min matching chunks required", 1, 4, 1)
# For LangChain FAISS, score is often L2 distance (lower = better). Tune as needed.

show_context = st.sidebar.checkbox("Show retrieved context", value=False)
# --------------------------------------------------
# Future Phase:
# Optional AI Summary (currently disabled)
# --------------------------------------------------
use_llm = False

st.sidebar.caption("Default strictness tuned for documentation accuracy (1.25)")
DEFAULT_MAX_DISTANCE = 1.25
max_distance = st.sidebar.slider(
    "Max distance allowed (lower = stricter)", 0.2, 2.0, DEFAULT_MAX_DISTANCE, 0.05
)
st.sidebar.markdown(
    """
    <div class="confidence-note">
        This assistant only responds when documentation relevance meets confidence thresholds.
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------
# Keyword overlap guardrail
# -------------------------------
def keyword_overlap_ok(question: str, kept_texts: list[str]) -> bool:
    # Extract simple keywords (>=3 chars). Keeps acronyms like SSO.
    q_terms = set(re.findall(r"[A-Za-z]{3,}", question.lower()))
    if not q_terms:
        return True

    ctx_terms = set(re.findall(r"[A-Za-z]{3,}", " ".join(kept_texts).lower()))

    # Require at least one meaningful shared term
    return len(q_terms.intersection(ctx_terms)) >= 1


# -------------------------------
# Scope coverage guardrail (NEW)
# -------------------------------
def scope_coverage_ok(question: str, kept_texts: list[str]) -> bool:
    q = question.lower()
    ctx = " ".join(kept_texts).lower()

    # If the user asks tenant-scoped questions, require tenant language in the retrieved text.
    tenant_scoped = bool(
        re.search(r"\btenant\b|\bper[-\s]?tenant\b|\bspecific tenant\b", q)
    )
    if tenant_scoped:
        return bool(
            re.search(r"\btenant\b|\bper[-\s]?tenant\b|\btenant[-\s]?specific\b", ctx)
        )

    # You can add more scopes later (version/env/role) the same way.
    return True


def multi_intent_coverage_ok(question: str, kept_texts: list[str]) -> bool:
    q = question.lower()
    ctx = " ".join(kept_texts).lower()

    asks_sync = bool(re.search(r"\bsync\b|\bsynchroniz", q))
    asks_perms = bool(re.search(r"\bpermission\b|\bpermissions\b|\bgroup\b", q))

    if asks_sync and asks_perms:
        has_sync = bool(re.search(r"\bsync\b|\bsynchroniz", ctx))
        has_perms = bool(re.search(r"\bpermission\b|\bpermissions\b|\bgroup\b", ctx))
        return has_sync and has_perms

    return True


# -------------------------------
# PDF Extraction (Text + OCR)
# -------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

            # OCR only if page is basically empty AND tesseract is available
            if len(page_text.strip()) < 20:
                try:
                    if Path(pytesseract.pytesseract.tesseract_cmd).exists():
                        image = page.to_image(resolution=300).original
                        text += pytesseract.image_to_string(image) + "\n"
                except Exception:
                    pass
    return text.strip()


# -------------------------------
# Embeddings and Vector Store
# -------------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vectorstore_from_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    embeddings = get_embeddings()

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(INDEX_PATH))

    return vs, chunks


def load_vectorstore():
    embeddings = get_embeddings()

    return FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# -------------------------------
# Build Vector Store
# -------------------------------


# -------------------------------
# Load PDFs at startup
# -------------------------------
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_PATH = BASE_DIR / "faiss_index"

INDEX_FAISS_FILE = INDEX_PATH / "index.faiss"
INDEX_METADATA_FILE = INDEX_PATH / "index.pkl"

SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md"]

DOC_PATHS = sorted(
    p
    for p in DOCS_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
)


def extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(str(path))

    if suffix in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")

    return ""


def load_full_document(relative_path: str) -> str:
    path = DOCS_DIR / relative_path

    if not path.exists():
        return "Document not found."

    return extract_text_from_file(path)


def format_source_name(filename: str) -> str:
    name = Path(filename).stem

    for prefix in ("TS_", "HT_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]

    name = name.replace("_", " ").replace("-", " ")

    name = re.sub(r"\s+", " ", name).strip()

    return name.title()


st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("chunks", None)

if st.session_state.vectorstore is None:
    with st.spinner("Loading knowledge base..."):
        documents = []

        if not DOCS_DIR.exists():
            st.error(f"Missing docs folder: {DOCS_DIR}")
            st.stop()

        if not DOC_PATHS:
            st.error(f"No supported files found in: {DOCS_DIR}")
            st.stop()

        for path in DOC_PATHS:
            text = extract_text_from_file(path)

            if text.strip():
                relative_path = path.relative_to(DOCS_DIR)
                parts = relative_path.parts

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": path.name,
                            "relative_path": str(relative_path),
                            "knowledge_base": parts[0],
                            "category": (
                                parts[1] if len(parts) > 2 else "Uncategorized"
                            ),
                        },
                    )
                )

        if not documents:
            st.error("Very little text detected in your documents.")
            st.stop()

        if INDEX_FAISS_FILE.exists() and INDEX_METADATA_FILE.exists():
            vs = load_vectorstore()
            chunks = list(vs.docstore._dict.values())

            st.success(f"Saved knowledge index loaded ✅ ({len(chunks)} chunks)")
        else:
            vs, chunks = build_vectorstore_from_documents(documents)

            st.success(
                f"Knowledge base indexed and saved ✅ "
                f"({len(chunks)} chunks from {len(documents)} readable files)"
            )

        st.session_state.vectorstore = vs
        st.session_state.chunks = chunks
# -------------------------------
# Ask Question
# -------------------------------
with st.form("search_form"):
    question = st.text_input(
        "Question", placeholder="e.g., How do I sync data on mobile?"
    )

    submitted = st.form_submit_button("Ask MVP Search")


def generate_answer(question: str, context: str) -> str:
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # good cost/quality for demos
        temperature=0.1,
    )

    system = (
        "You are a documentation assistant. "
        "Answer ONLY using the provided documentation context. "
        "If the answer is not clearly supported by the context, say you don't have enough information."
    )

    user = f"""Question:
{question}

Documentation context:
{context}

Return a concise answer (3-6 sentences). If steps exist, use bullets."""
    resp = llm.invoke([("system", system), ("user", user)])
    return resp.content.strip()


# if st.button("Ask YV Search") and question:
if submitted and question:
    if st.session_state.vectorstore is None:
        st.error("Knowledge base not loaded.")
        st.stop()
    candidate_k = max(k * 4, 10)

    docs_with_scores = st.session_state.vectorstore.similarity_search_with_score(
        question,
        k=candidate_k,
    )
    if not docs_with_scores:
        st.markdown(
            f'<div class="chat-bubble bot"><b>Bot:</b><br>{NOT_FOUND_MESSAGE}</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    kept = []

    for document, score in docs_with_scores:
        text = document.page_content.strip()
        source = document.metadata.get("source", "Unknown Source")
        knowledge_base = document.metadata.get("knowledge_base", "Unknown")
        category = document.metadata.get("category", "Uncategorized")
        relative_path = document.metadata.get("relative_path", source)

        matches_selected_base = (
            knowledge_base_filter == "All" or knowledge_base == knowledge_base_filter
        )

        if text and score <= max_distance and matches_selected_base:
            kept.append(
                (
                    text,
                    score,
                    source,
                    knowledge_base,
                    category,
                    relative_path,
                )
            )

        if len(kept) >= k:
            break

    st.markdown(
        f'<div class="chat-bubble user"><b>You Asked:</b><br>{question}</div>',
        unsafe_allow_html=True,
    )

    kept_texts = [t for t, _, _, _, _, _ in kept]
    passes_keywords = keyword_overlap_ok(question, kept_texts)
    passes_scope = scope_coverage_ok(question, kept_texts)
    passes_multi = multi_intent_coverage_ok(question, kept_texts)

    if (
        len(kept) < min_hit_count
        or not passes_keywords
        or not passes_scope
        or not passes_multi
    ):
        if not passes_scope:
            msg = (
                "I found documentation related to your topic, but it doesn’t cover the "
                "**tenant-specific** part of your question.<br>"
                "Please check tenant configuration or provide tenant-scoped documentation."
            )
        elif not passes_multi:
            msg = (
                "I found documentation related to part of your question, but it does not cover "
                "the full combination (e.g., **sync + permissions**).<br>"
                "Please provide documentation that links these topics, or rephrase to one topic."
            )
        else:
            msg = NOT_FOUND_MESSAGE

        st.markdown(
            f'<div class="chat-bubble bot"><b>Bot:</b><br>{msg}</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # Build context once
    context = "\n\n".join(t for t, _, _, _, _, _ in kept)

    # Optional LLM generation with safe fallback
    answer = None
    if use_llm:
        try:
            answer = generate_answer(question, context)
        except Exception:
            st.sidebar.warning(
                "Answer generation temporarily unavailable — showing sources only."
            )
            answer = None

    # Render answer OR retrieved context
    if answer and answer.strip():
        st.markdown(
            f'<div class="chat-bubble bot"><div class="answer-text">{answer}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="chat-bubble bot bot-header">Retrieved documentation context:</div>',
            unsafe_allow_html=True,
        )
        from collections import defaultdict

    best_score = min(score for _, score, _, _, _, _ in kept)

    if best_score <= 1.00:
        confidence_label = "High confidence"
        confidence_message = "I found documentation that closely matches your question."
    else:
        confidence_label = "Possible match"
        confidence_message = (
            "I found related documentation that may help, "
            "but it may not fully answer your question."
        )

    st.markdown(f"**{confidence_label}:** {confidence_message}")

    grouped = defaultdict(list)

    for text, score, source, knowledge_base, category, relative_path in kept:
        grouped[relative_path].append(
            {
                "text": text,
                "score": score,
                "source": source,
                "knowledge_base": knowledge_base,
                "category": category,
            }
        )

    for relative_path, items in grouped.items():
        source = items[0]["source"]
        knowledge_base = items[0]["knowledge_base"]
        category = items[0]["category"]

        display_source = format_source_name(source)
        combined_text = "\n\n".join(item["text"] for item in items)

        st.markdown(
            f"""
            <div class="chat-bubble bot">
                <b>Source:</b> {display_source}<br>
                <b>Knowledge Base:</b> {knowledge_base}<br>
                <b>Category:</b> {category}<br><br>
                <div class="answer-text">{combined_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📄 View Full Document"):
            full_document = load_full_document(relative_path)
            st.markdown(
                f'<div class="answer-text">{full_document}</div>',
                unsafe_allow_html=True,
            )

    # Optional sources
    if show_context:
        with st.expander("Sources (retrieved context)"):
            for t, s, source, knowledge_base, category, relative_path in kept:
                st.write(f"Source: {source}")
                st.write(f"Knowledge Base: {knowledge_base}")
                st.write(f"Category: {category}")
                st.write(f"Score: {s:.4f}")
                st.code(t)
                st.divider()
