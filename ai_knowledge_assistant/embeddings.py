from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = PROJECT_ROOT / "outputs" / "chunked_documents.json"
EMBEDDINGS_FILE = PROJECT_ROOT / "outputs" / "chunk_embeddings.npy"
METADATA_FILE = PROJECT_ROOT / "outputs" / "chunk_metadata.json"


# ============================================================
# Embedding model
# ============================================================

# A compact, general-purpose sentence embedding model.
# It is a practical baseline for a small local RAG prototype.
MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# Load chunks
# ============================================================

def load_chunks() -> list[dict]:
    """Load the chunks created by chunking.py."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}\n\n"
            "Run chunking.py first."
        )

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    chunks = data.get("chunks", [])

    if not chunks:
        raise ValueError("No chunks were found in chunked_documents.json.")

    return chunks


# ============================================================
# Create embeddings
# ============================================================

def main() -> None:
    print("=" * 80)
    print("WEEK 3 - RAG EMBEDDING GENERATION")
    print("=" * 80)

    chunks = load_chunks()

    texts = [chunk["text"] for chunk in chunks]

    print(f"\nChunks to embed: {len(texts)}")
    print(f"Embedding model: {MODEL_NAME}")

    # Load the pretrained embedding model.
    # The first run may download the model files.
    model = SentenceTransformer(MODEL_NAME)

    # Convert every chunk into a numerical vector.
    # normalize_embeddings=True makes cosine similarity equivalent
    # to inner-product similarity, which will be useful with FAISS.
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, got shape {embeddings.shape}."
        )

    # Save the numerical vectors separately from their metadata.
    OUTPUT_DIR = EMBEDDINGS_FILE.parent
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDINGS_FILE, embeddings)

    metadata = []

    for index, chunk in enumerate(chunks):
        metadata.append(
            {
                "embedding_index": index,
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page_numbers": chunk["page_numbers"],
                "text": chunk["text"],
                "character_count": chunk["character_count"],
            }
        )

    with METADATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 80)
    print(f"Chunks embedded: {len(texts)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embeddings saved to: {EMBEDDINGS_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
