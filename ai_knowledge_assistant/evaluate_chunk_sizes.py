from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import chunking
from evaluate_retrieval import EXPECTED_SOURCES


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
MIN_SIMILARITY_SCORE = 0.35

CONFIGS = [
    (500, 100),
    (1000, 200),
    (1500, 300),
]


def load_documents():
    """Load the already extracted project documents."""
    return chunking.load_extracted_documents()


def create_index(chunks: list[dict], embedding_model):
    """Create an in-memory normalized FAISS index."""
    texts = [chunk["text"] for chunk in chunks]

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

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def retrieve(
    question: str,
    index,
    chunks: list[dict],
    embedding_model,
) -> list[dict]:
    """Retrieve top-K chunks above the final similarity threshold."""
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
                "page_numbers": chunk["page_numbers"],
            }
        )

    return results


def evaluate_question(
    question: str,
    results: list[dict],
) -> bool:
    """Apply the same pass/fail logic as evaluate_retrieval.py."""
    expected_sources = EXPECTED_SOURCES.get(question, set())

    retrieved_sources = {
        result["source"]
        for result in results
    }

    if not expected_sources:
        return len(results) == 0

    return bool(
        retrieved_sources.intersection(expected_sources)
    )


def main():
    print("=" * 80)
    print("WEEK 3 - CHUNK SIZE RETRIEVAL EXPERIMENT")
    print("=" * 80)

    data = load_documents()

    questions = [
        question
        for question in EXPECTED_SOURCES.keys()
    ]

    embedding_model = SentenceTransformer(MODEL_NAME)

    print(f"\nQuestions: {len(questions)}")
    print(f"Top-K: {TOP_K}")
    print(f"Similarity threshold: {MIN_SIMILARITY_SCORE:.2f}")

    for chunk_size, chunk_overlap in CONFIGS:
        chunking.CHUNK_SIZE = chunk_size
        chunking.CHUNK_OVERLAP = chunk_overlap

        chunks = chunking.create_all_chunks(data)
        index = create_index(
            chunks,
            embedding_model,
        )

        passed = 0

        print("\n" + "=" * 80)
        print(
            f"CONFIGURATION: "
            f"chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}"
        )
        print(f"Chunks created: {len(chunks)}")
        print("-" * 80)

        for number, question in enumerate(
            questions,
            start=1,
        ):
            results = retrieve(
                question=question,
                index=index,
                chunks=chunks,
                embedding_model=embedding_model,
            )

            question_passed = evaluate_question(
                question=question,
                results=results,
            )

            if question_passed:
                passed += 1

            status = "PASS" if question_passed else "FAIL"

            top_result = (
                results[0]
                if results
                else None
            )

            if top_result:
                top_text = (
                    f"{top_result['source']} "
                    f"(score={top_result['similarity_score']:.4f})"
                )
            else:
                top_text = "None"

            print(
                f"{number:2}. {status} | "
                f"chunks={len(results)} | "
                f"top1={top_text} | "
                f"{question}"
            )

        pass_rate = passed / len(questions) * 100

        print("-" * 80)
        print(
            f"PASS RATE: {passed}/{len(questions)} "
            f"({pass_rate:.1f}%)"
        )

    print("\n" + "=" * 80)
    print("CHUNK SIZE EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()