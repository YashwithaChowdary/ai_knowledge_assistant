from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INDEX_FILE = PROJECT_ROOT / "outputs" / "faiss_index.bin"
METADATA_FILE = PROJECT_ROOT / "outputs" / "chunk_metadata.json"


# ============================================================
# Retrieval settings
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

# Initial threshold based on our real retrieval tests:
# relevant questions returned scores around 0.39-0.64,
# while the unanswerable test returned a top score of 0.1376.
# This is a starting point, not a universal threshold.
MIN_SIMILARITY_SCORE = 0.35


# ============================================================
# Load resources
# ============================================================

def load_resources():
    """Load the FAISS index, metadata, and embedding model."""
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

    model = SentenceTransformer(MODEL_NAME)

    return index, metadata, model


# ============================================================
# Search
# ============================================================

def search(
    query: str,
    index,
    metadata,
    model,
    top_k: int = TOP_K,
    min_similarity: float = MIN_SIMILARITY_SCORE,
):
    """
    Search for semantically similar chunks.

    The similarity threshold is used to avoid treating weakly
    related chunks as evidence for an answer.
    """
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0:
            continue

        similarity_score = float(score)

        if similarity_score < min_similarity:
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
# Main program
# ============================================================

def main() -> None:
    print("=" * 80)
    print("WEEK 3 - RAG VECTOR SEARCH TEST")
    print("=" * 80)

    query = input("\nEnter your question: ").strip()

    if not query:
        raise ValueError("Question cannot be empty.")

    index, metadata, model = load_resources()

    results = search(
        query=query,
        index=index,
        metadata=metadata,
        model=model,
    )

    print("\n" + "=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)

    print(f"Similarity threshold: {MIN_SIMILARITY_SCORE:.2f}")

    if not results:
        print("\nNo sufficiently relevant information was found")
        print("in the selected document collection.")

        print("\nThis is important for hallucination control:")
        print("the system should not use weakly related chunks as evidence.")

    else:
        print(f"\nRelevant chunks found: {len(results)}")

        for rank, result in enumerate(results, start=1):
            print(f"\nResult {rank}")
            print("-" * 80)
            print(f"Similarity score: {result['similarity_score']:.4f}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Source: {result['source']}")
            print(f"Pages: {result['page_numbers']}")
            print("\nText:")
            print(result["text"][:700])

    print("\n" + "=" * 80)
    print("VECTOR SEARCH COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
