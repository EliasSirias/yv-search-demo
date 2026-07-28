import hashlib
import json
from pathlib import Path
from typing import Any
import pytesseract
import pdfplumber
import pytesseract
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_PATH = BASE_DIR / "faiss_index"

INDEX_FAISS_FILE = INDEX_PATH / "index.faiss"
INDEX_METADATA_FILE = INDEX_PATH / "index.pkl"
INDEX_FINGERPRINT_FILE = INDEX_PATH / "fingerprint.json"

SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md"]

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def get_doc_paths() -> list[Path]:
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Missing docs folder: {DOCS_DIR}")

    doc_paths = sorted(
        path
        for path in DOCS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not doc_paths:
        raise FileNotFoundError(f"No supported files found in: {DOCS_DIR}")

    return doc_paths


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

            if len(page_text.strip()) < 20:
                try:
                    tesseract_path = Path(pytesseract.pytesseract.tesseract_cmd)

                    if tesseract_path.exists():
                        image = page.to_image(resolution=300).original
                        text += pytesseract.image_to_string(image) + "\n"
                except Exception:
                    pass

    return text.strip()


def extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(str(path))

    if suffix in [".txt", ".md"]:
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    return ""


def create_documents(doc_paths: list[Path]) -> list[Document]:
    documents = []

    for path in doc_paths:
        text = extract_text_from_file(path)

        if not text.strip():
            continue

        relative_path = path.relative_to(DOCS_DIR)
        parts = relative_path.parts

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "relative_path": str(relative_path),
                    "knowledge_base": parts[0],
                    "category": (parts[1] if len(parts) > 2 else "Uncategorized"),
                },
            )
        )

    if not documents:
        raise ValueError("Very little text was detected in the documents.")

    return documents


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def calculate_docs_fingerprint(
    doc_paths: list[Path],
) -> str:
    hasher = hashlib.sha256()

    for path in sorted(doc_paths):
        relative_path = path.relative_to(DOCS_DIR)

        hasher.update(str(relative_path).encode("utf-8"))

        with path.open("rb") as file:
            while chunk := file.read(8192):
                hasher.update(chunk)

    return hasher.hexdigest()


def save_fingerprint(fingerprint: str) -> None:
    INDEX_PATH.mkdir(parents=True, exist_ok=True)

    INDEX_FINGERPRINT_FILE.write_text(
        json.dumps(
            {"fingerprint": fingerprint},
            indent=2,
        ),
        encoding="utf-8",
    )


def load_saved_fingerprint() -> str | None:
    if not INDEX_FINGERPRINT_FILE.exists():
        return None

    try:
        data = json.loads(INDEX_FINGERPRINT_FILE.read_text(encoding="utf-8"))

        return data.get("fingerprint")

    except (json.JSONDecodeError, OSError):
        return None


def build_vectorstore_from_documents(
    documents: list[Document],
    fingerprint: str,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
    )

    vectorstore.save_local(str(INDEX_PATH))
    save_fingerprint(fingerprint)

    return vectorstore, chunks


def load_vectorstore() -> FAISS:
    embeddings = get_embeddings()

    return FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_stored_chunks(vectorstore: FAISS) -> list[Document]:
    chunks = []

    for doc_id in vectorstore.index_to_docstore_id.values():
        stored_doc = vectorstore.docstore.search(doc_id)

        if isinstance(stored_doc, Document):
            chunks.append(stored_doc)

    return chunks


def load_or_build_index() -> dict[str, Any]:
    doc_paths = get_doc_paths()
    documents = create_documents(doc_paths)

    current_fingerprint = calculate_docs_fingerprint(doc_paths)

    saved_fingerprint = load_saved_fingerprint()

    index_files_exist = INDEX_FAISS_FILE.exists() and INDEX_METADATA_FILE.exists()

    index_is_current = index_files_exist and saved_fingerprint == current_fingerprint

    if index_is_current:
        vectorstore = load_vectorstore()
        chunks = get_stored_chunks(vectorstore)

        status = f"Saved knowledge index loaded ✅ " f"({len(chunks)} chunks)"

        return {
            "vectorstore": vectorstore,
            "chunks": chunks,
            "status": status,
        }

    vectorstore, chunks = build_vectorstore_from_documents(
        documents,
        current_fingerprint,
    )

    if index_files_exist:
        message = "Documents changed — knowledge index rebuilt and saved"
    else:
        message = "Knowledge base indexed and saved"

    status = (
        f"{message} ✅ "
        f"({len(chunks)} chunks from "
        f"{len(documents)} readable files)"
    )

    return {
        "vectorstore": vectorstore,
        "chunks": chunks,
        "status": status,
    }


def load_full_document(relative_path: str) -> str:
    path = DOCS_DIR / relative_path

    if not path.exists():
        return "Document not found."

    return extract_text_from_file(path)
