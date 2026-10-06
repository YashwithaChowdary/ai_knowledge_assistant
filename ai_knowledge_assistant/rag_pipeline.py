from pathlib import Path
import json
import os

import faiss
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INDEX_FILE = PROJECT_ROOT / "outputs" / "faiss_index.bin"
METADATA_FILE = PROJECT_ROOT / "outputs" / "chunk_metadata.json"


# ============================================================
# Model and retrieval settings
# ============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

TOP_K = 5
MIN_SIMILARITY_SCORE = 0.35


# ============================================================
# Load resources
# ============================================================

def load_resources():
    """Load FAISS, chunk metadata, and the embedding model."""
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found:\n{INDEX_FILE}\n\n"
            "Run faiss_index.py first."
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Chunk metadata not found:\n{METADATA_FILE}\n\n"
            "Run embeddings.py first."
        )

    index = faiss.read_index(str(INDEX_FILE))

    with METADATA_FILE.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # LangChain's Google Gemini integration automatically reads
    # GEMINI_API_KEY from the environment.
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        
    )

    return index, metadata, embedding_model, llm


# ============================================================
# Retrieve relevant chunks
# ============================================================

def retrieve_chunks(
    question: str,
    index,
    metadata: list[dict],
    embedding_model: SentenceTransformer,
) -> list[dict]:
    """Retrieve the most relevant document chunks for a question."""
    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
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

    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0:
            continue

        similarity_score = float(score)

        if similarity_score < MIN_SIMILARITY_SCORE:
            continue

        chunk = metadata[index_position]

        results.append(
            {
                "similarity_score": similarity_score,
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page_numbers": chunk["page_numbers"],
                "text": chunk["text"],
            }
        )

    return results


# ============================================================
# Build the RAG prompt
# ============================================================

def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build a grounded prompt using only retrieved document text.
    """
    context_parts = []

    for number, chunk in enumerate(retrieved_chunks, start=1):
        pages = ", ".join(
            str(page) for page in chunk["page_numbers"]
        )

        context_parts.append(
            f"SOURCE {number}\n"
            f"Document: {chunk['source']}\n"
            f"Pages: {pages}\n"
            f"Text:\n{chunk['text']}\n"
        )

    context = "\n\n".join(context_parts)

    return f"""
You are a Computer Science Study Assistant.

Answer the user's question using ONLY the supplied document context.

Rules:
1. Use the retrieved document context as the only source of factual information.
2. Focus on the excerpts that directly support the user's question and ignore unrelated excerpts.
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
# Generate the grounded answer
# ============================================================

def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    llm: ChatGoogleGenerativeAI,
) -> str:
    """Generate an answer from Gemini using the retrieved context."""
    prompt = build_prompt(question, retrieved_chunks)

    response = llm.invoke(prompt)

    if not response.content:
        raise RuntimeError("Gemini returned an empty response.")

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
        raise RuntimeError("Gemini returned an empty text response.")

    return answer_text.strip()

# ============================================================
# Print source references
# ============================================================

def print_sources(retrieved_chunks: list[dict]) -> None:
    """Display source documents with combined page references."""
    print("\n" + "=" * 80)
    print("SOURCE REFERENCES")
    print("=" * 80)

    source_pages: dict[str, set[int]] = {}

    for chunk in retrieved_chunks:
        source = chunk["source"]
        pages = source_pages.setdefault(source, set())
        pages.update(int(page) for page in chunk["page_numbers"])

    for source, pages in source_pages.items():
        sorted_pages = sorted(pages)
        page_text = ", ".join(str(page) for page in sorted_pages)
        print(f"- {source} | Pages: {page_text}")


# ============================================================
# Main program
# ============================================================

def main() -> None:
    print("=" * 80)
    print("AI KNOWLEDGE ASSISTANT - RAG PROTOTYPE")
    print("=" * 80)

    question = input("\nEnter your question: ").strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    index, metadata, embedding_model, llm = load_resources()

    retrieved_chunks = retrieve_chunks(
        question=question,
        index=index,
        metadata=metadata,
        embedding_model=embedding_model,
    )

    print(f"\nSimilarity threshold: {MIN_SIMILARITY_SCORE:.2f}")
    print(f"Retrieved relevant chunks: {len(retrieved_chunks)}")

    if not retrieved_chunks:
        print(
            "\nI could not find sufficient information about this topic "
            "in the provided documents."
        )
        print("\nNo LLM request was made because no sufficiently relevant "
              "document evidence was retrieved.")
        return

    answer = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        llm=llm,
    )

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print_sources(retrieved_chunks)

    print("\n" + "=" * 80)
    print("RAG RESPONSE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
