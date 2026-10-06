from pathlib import Path

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent

EMBEDDINGS_FILE = PROJECT_ROOT / "outputs" / "chunk_embeddings.npy"
INDEX_FILE = PROJECT_ROOT / "outputs" / "faiss_index.bin"


def main() -> None:
    print("=" * 80)
    print("WEEK 3 - FAISS VECTOR INDEX")
    print("=" * 80)

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"\nEmbedding file not found:\n{EMBEDDINGS_FILE}\n\n"
            "Run embeddings.py first."
        )

    embeddings = np.load(EMBEDDINGS_FILE)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, got shape {embeddings.shape}."
        )

    embeddings = np.asarray(embeddings, dtype="float32")

    number_of_vectors, embedding_dimension = embeddings.shape

    print(f"\nVectors loaded: {number_of_vectors}")
    print(f"Embedding dimension: {embedding_dimension}")

    # Embeddings were normalized during generation, so inner product
    # gives cosine-similarity ranking for our vectors.
    index = faiss.IndexFlatIP(embedding_dimension)

    index.add(embeddings)

    print(f"Vectors stored in FAISS: {index.ntotal}")

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))

    print("\nFAISS index saved to:")
    print(INDEX_FILE)

    print("\n" + "=" * 80)
    print("FAISS INDEX CREATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
