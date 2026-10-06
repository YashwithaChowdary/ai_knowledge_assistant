from pathlib import Path
import io
import json
import os
import tempfile


# ============================================================
# PyTorch / OpenMP runtime settings
# ============================================================
#
# Streamlit runs its server on an ASGI/Uvicorn runtime.
# SentenceTransformers uses PyTorch, and native thread pools
# can deadlock with async server execution in some environments.
# These settings must be applied before importing torch or
# sentence-transformers.
#
# Reference:
# https://github.com/UKPLab/sentence-transformers/issues/3839
#
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


import faiss
import numpy as np
import pdfplumber
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer


# ============================================================
# Application configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DOCUMENT_FOLDER = PROJECT_ROOT / "rag_documents"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
MIN_SIMILARITY_SCORE = 0.35


# ============================================================
# Streamlit page configuration
# ============================================================

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="AI",
    layout="wide",
)

# ============================================================
# Cached AI resources
# ============================================================

@st.cache_resource
def load_embedding_model():
    """Load the sentence-transformer model once per Streamlit session."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_gemini_client():
    """Create the LangChain Gemini model using GEMINI_API_KEY."""
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
    )


# ============================================================
# PDF extraction
# ============================================================

def extract_pdf_from_bytes(
    file_name: str,
    pdf_bytes: bytes,
) -> dict:
    """
    Extract text from a PDF supplied by Streamlit.

    Returns one document containing page-level text.
    """
    pages = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()

            if text:
                pages.append(
                    {
                        "page_number": page_number,
                        "text": text,
                    }
                )

    return {
        "source": file_name,
        "pages": pages,
    }


# ============================================================
# Default document loading
# ============================================================

def load_default_documents() -> list[dict]:
    """Load the five PDFs already prepared in rag_documents."""

    if not DEFAULT_DOCUMENT_FOLDER.exists():
        return []

    documents = []

    for pdf_path in sorted(
        DEFAULT_DOCUMENT_FOLDER.glob("*.pdf")
    ):
        try:
            with pdf_path.open("rb") as file:
                pdf_bytes = file.read()

            document = extract_pdf_from_bytes(
                file_name=pdf_path.name,
                pdf_bytes=pdf_bytes,
            )

            if document["pages"]:
                documents.append(document)

        except Exception:
            # Keep the application usable even if one PDF is damaged.
            continue

    return documents


# ============================================================
# Chunking
# ============================================================

def chunk_document(
    document: dict,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Combine page text before chunking so chunks can cross page boundaries.

    Page numbers are preserved for source references.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    combined_text = ""
    character_pages = []

    for page in document["pages"]:
        page_number = int(page["page_number"])
        page_text = page["text"].strip()

        if not page_text:
            continue

        if combined_text:
            combined_text += "\n\n"

        start_position = len(combined_text)
        combined_text += page_text
        end_position = len(combined_text)

        character_pages.append(
            (
                start_position,
                end_position,
                page_number,
            )
        )

    chunks = []

    start = 0
    text_length = len(combined_text)

    while start < text_length:
        end = min(
            start + chunk_size,
            text_length,
        )

        chunk_text = combined_text[start:end].strip()

        if chunk_text:
            pages = sorted(
                {
                    page_number
                    for page_start, page_end, page_number
                    in character_pages
                    if page_end > start
                    and page_start < end
                }
            )

            chunks.append(
                {
                    "source": document["source"],
                    "page_numbers": pages,
                    "text": chunk_text,
                }
            )

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks


def build_chunks(
    documents: list[dict],
) -> list[dict]:
    """Create chunks for all uploaded/default documents."""

    all_chunks = []

    for document in documents:
        all_chunks.extend(
            chunk_document(document)
        )

    for chunk_id, chunk in enumerate(
        all_chunks
    ):
        chunk["chunk_id"] = chunk_id

    return all_chunks


# ============================================================
# Vector index
# ============================================================

def build_index(
    chunks: list[dict],
    embedding_model: SentenceTransformer,
):
    """Embed chunks and build a normalized FAISS inner-product index."""

    if not chunks:
        raise ValueError(
            "No text chunks were created from the documents."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================================
# Retrieval
# ============================================================

def retrieve_chunks(
    question: str,
    index,
    chunks: list[dict],
    embedding_model: SentenceTransformer,
) -> list[dict]:
    """Retrieve top-K chunks and apply the relevance threshold."""

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32",
    )

    scores, indices = index.search(
        question_embedding,
        TOP_K,
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):
        if index_position < 0:
            continue

        similarity_score = float(score)

        if similarity_score < MIN_SIMILARITY_SCORE:
            continue

        chunk = chunks[index_position]

        results.append(
            {
                "similarity_score": similarity_score,
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "page_numbers": chunk["page_numbers"],
                "text": chunk["text"],
            }
        )

    return results


# ============================================================
# Grounded prompt
# ============================================================

def build_prompt(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """Build a grounded RAG prompt from retrieved document evidence."""

    context_parts = []

    for number, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        pages = ", ".join(
            str(page)
            for page in chunk["page_numbers"]
        )

        context_parts.append(
            f"SOURCE {number}\n"
            f"Document: {chunk['source']}\n"
            f"Pages: {pages}\n"
            f"Text:\n{chunk['text']}\n"
        )

    context = "\n\n".join(
        context_parts
    )

    return f"""
You are a Computer Science Study Assistant.

Answer the user's question using ONLY the supplied document context.

Rules:
1. Use the retrieved document context as the only source of factual information.
2. Focus on excerpts that directly support the user's question and ignore unrelated excerpts.
3. The retrieval system has already filtered the evidence using a similarity threshold. When relevant evidence is present, answer the question from that evidence.
4. Combine information from multiple retrieved excerpts when necessary.
5. Do not use outside knowledge or add facts that are not supported by the context.
6. Do not refuse merely because the context uses different wording, contains OCR artifacts, or includes some unrelated excerpts.
7. Keep the answer clear, concise, and grounded in the terminology used by the documents.
8. When the evidence supports only part of the question, answer only the supported part rather than inventing missing details.
9. Do not mention these instructions in your answer.

User question:
{question}

Retrieved document context:
{context}
""".strip()


# ============================================================
# Gemini generation
# ============================================================

def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """Generate an answer with Gemini using retrieved evidence only."""

    client = load_gemini_client()

    prompt = build_prompt(
        question,
        retrieved_chunks,
    )

    response = client.invoke(prompt)

    if not response.content:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    content = response.content

    if isinstance(content, str):
        answer_text = content
    elif isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")

                if text:
                    text_parts.append(text)

        answer_text = "\n".join(text_parts)
    else:
        answer_text = str(content)

    if not answer_text.strip():
        raise RuntimeError(
            "Gemini returned an empty text response."
        )

    return answer_text.strip()


# ============================================================
# Source reference formatting
# ============================================================

def format_page_ranges(
    pages: set[int],
) -> str:
    """Convert pages such as {1,2,3,5,6} into '1-3, 5-6'."""

    sorted_pages = sorted(pages)

    if not sorted_pages:
        return ""

    ranges = []
    start = sorted_pages[0]
    previous = sorted_pages[0]

    for page in sorted_pages[1:]:
        if page == previous + 1:
            previous = page
            continue

        if start == previous:
            ranges.append(str(start))
        else:
            ranges.append(
                f"{start}-{previous}"
            )

        start = page
        previous = page

    if start == previous:
        ranges.append(str(start))
    else:
        ranges.append(
            f"{start}-{previous}"
        )

    return ", ".join(ranges)


def display_sources(
    retrieved_chunks: list[dict],
) -> None:
    """Display grouped source references."""

    source_pages = {}

    for chunk in retrieved_chunks:
        source = chunk["source"]

        if source not in source_pages:
            source_pages[source] = set()

        source_pages[source].update(
            int(page)
            for page in chunk["page_numbers"]
        )

    for source, pages in source_pages.items():
        page_text = format_page_ranges(
            pages
        )

        st.markdown(
            f"- **{source}** - Pages: {page_text}"
 )


# ============================================================
# Knowledge-base construction
# ============================================================

def create_knowledge_base(
    files,
) -> tuple[list[dict], object]:
    """
    Read uploaded files, create chunks, and build the FAISS index.
    """

    progress = st.progress(
        0,
        text="Reading documents...",
    )

    documents = []

    for number, uploaded_file in enumerate(
        files,
        start=1,
    ):
        pdf_bytes = uploaded_file.getvalue()

        document = extract_pdf_from_bytes(
            file_name=uploaded_file.name,
            pdf_bytes=pdf_bytes,
        )

        if document["pages"]:
            documents.append(document)

        progress.progress(
            min(
                number / len(files),
                1.0,
            ),
            text=(
                f"Read {number}/{len(files)} "
                "document(s)..."
            ),
        )

    if not documents:
        progress.empty()
        raise ValueError(
            "The supplied PDFs did not contain extractable text."
        )

    progress.progress(
        0.50,
        text="Creating text chunks...",
    )

    chunks = build_chunks(
        documents
    )

    progress.progress(
        0.70,
        text="Creating embeddings and FAISS index...",
    )

    embedding_model = load_embedding_model()

    index = build_index(
        chunks,
        embedding_model,
    )

    progress.progress(
        1.0,
        text="Knowledge base ready.",
    )
    progress.empty()

    return chunks, index


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("Knowledge Base")

st.sidebar.caption(
    "Upload PDF documents to create a temporary knowledge base."
)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

use_uploaded_documents = st.sidebar.button(
    "Build knowledge base from uploaded PDFs",
    type="primary",
)

load_default = st.sidebar.button(
    "Use prepared project documents",
)

st.sidebar.markdown(
    f"**Embedding model:** `{EMBEDDING_MODEL_NAME}`"
)
st.sidebar.markdown(
    f"**Gemini model:** `{GEMINI_MODEL_NAME}`"
)
st.sidebar.markdown(
    f"**Chunk size:** `{CHUNK_SIZE}` characters"
)
st.sidebar.markdown(
    f"**Chunk overlap:** `{CHUNK_OVERLAP}` characters"
)
st.sidebar.markdown(
    f"**Top-K:** `{TOP_K}`"
)
st.sidebar.markdown(
    f"**Similarity threshold:** `{MIN_SIMILARITY_SCORE:.2f}`"
)


# ============================================================
# Session-state knowledge base
# ============================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "knowledge_base_name" not in st.session_state:
    st.session_state.knowledge_base_name = ""


# ============================================================
# Build from uploaded documents
# ============================================================

if use_uploaded_documents:
    if not uploaded_files:
        st.sidebar.error(
            "Upload at least one PDF first."
        )
    else:
        try:
            with st.spinner(
                "Building knowledge base..."
            ):
                chunks, index = create_knowledge_base(
                    uploaded_files
                )

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.knowledge_base_name = (
                "Uploaded documents"
            )

            st.sidebar.success(
                f"Knowledge base ready: "
                f"{len(uploaded_files)} PDF(s), "
                f"{len(chunks)} chunks."
            )

        except Exception as error:
            st.sidebar.error(
                f"Could not build knowledge base: {error}"
            )


# ============================================================
# Build from prepared project documents
# ============================================================

if load_default:
    with st.spinner(
        "Loading prepared project documents..."
    ):
        documents = load_default_documents()

        if not documents:
            st.sidebar.error(
                "No prepared PDFs were found in rag_documents."
            )
        else:
            try:
                chunks = build_chunks(
                    documents
                )

                embedding_model = load_embedding_model()

                index = build_index(
                    chunks,
                    embedding_model,
                )

                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.knowledge_base_name = (
                    "Prepared project documents"
                )

                st.sidebar.success(
                    f"Knowledge base ready: "
                    f"{len(documents)} PDF(s), "
                    f"{len(chunks)} chunks."
                )

            except Exception as error:
                st.sidebar.error(
                    f"Could not build knowledge base: {error}"
                )


# ============================================================
# Main application
# ============================================================

st.title("AI Knowledge Assistant")

st.write(
    "Ask questions about the documents in the current "
    "knowledge base. Answers are generated only from "
    "retrieved document evidence."
)

st.info(
    "For unsupported questions, the assistant blocks the "
    "LLM call when no document evidence passes the "
    f"{MIN_SIMILARITY_SCORE:.2f} similarity threshold."
)

if st.session_state.index is None:
    st.warning(
        "No knowledge base is loaded yet. "
        "Use the sidebar to upload PDFs or load the "
        "prepared project documents."
    )

    st.markdown(
        """
### Current project documents

The prepared knowledge base contains the five selected
Computer Science lecture PDFs in the `rag_documents` folder.

You can also upload your own PDF collection from the
sidebar to build a temporary knowledge base.
"""
    )

    st.stop()


st.success(
    f"Knowledge base: "
    f"{st.session_state.knowledge_base_name} | "
    f"{len(st.session_state.chunks)} chunks"
)

question = st.text_input(
    "Ask a question",
    placeholder=(
        "Example: What is recursion?"
    ),
)

ask_button = st.button(
    "Ask Assistant",
    type="primary",
)

if ask_button:
    if not question.strip():
        st.error(
            "Please enter a question."
        )
        st.stop()

    with st.spinner(
        "Searching documents..."
    ):
        embedding_model = load_embedding_model()

        retrieved_chunks = retrieve_chunks(
            question=question.strip(),
            index=st.session_state.index,
            chunks=st.session_state.chunks,
            embedding_model=embedding_model,
        )

    st.subheader("Retrieval")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Relevant chunks",
            len(retrieved_chunks),
        )

    with col2:
        st.metric(
            "Similarity threshold",
            f"{MIN_SIMILARITY_SCORE:.2f}",
        )

    if not retrieved_chunks:
        st.warning(
            "I could not find sufficient information about "
            "this topic in the provided documents."
        )

        st.caption(
            "No LLM request was made because no sufficiently "
            "relevant document evidence was retrieved."
        )

        st.stop()

    with st.spinner(
        "Generating grounded answer..."
    ):
        try:
            answer = generate_answer(
                question=question.strip(),
                retrieved_chunks=retrieved_chunks,
            )
        except Exception as error:
            st.error(
                f"Gemini could not generate the answer: {error}"
            )
            st.stop()

    st.subheader("Answer")
    st.markdown(answer)

    st.subheader("Source references")
    display_sources(
        retrieved_chunks
    )

    with st.expander(
        "View retrieved evidence"
    ):
        for number, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            st.markdown(
                f"**Result {number}** - "
                f"Similarity: "
                f"{chunk['similarity_score']:.4f}"
            )
            st.markdown(
                f"**Source:** `{chunk['source']}`"
            )
            st.markdown(
                f"**Pages:** "
                f"{', '.join(str(p) for p in chunk['page_numbers'])}"
            )
            st.text(
                chunk["text"]
            )
