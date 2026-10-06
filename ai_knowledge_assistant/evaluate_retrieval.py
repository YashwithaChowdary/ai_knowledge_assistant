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
QUESTIONS_FILE = PROJECT_ROOT / "evaluation_questions.txt"


# ============================================================
# Retrieval settings
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
MIN_SIMILARITY_SCORE = 0.35


# ============================================================
# Evaluation expectations
# ============================================================
#
# These expected sources are based on the five PDF documents
# selected for this project.
#
# quantum entanglement is intentionally marked as unanswerable
# because that topic is not part of the selected document set.
#

EXPECTED_SOURCES = {
    "What is decomposition in computational thinking?":
        {"01_decomposition_abstraction_functions.pdf"},

    "What is abstraction in computer science?":
        {"01_decomposition_abstraction_functions.pdf"},

    "What is recursion?":
        {"02_recursion_dictionaries.pdf"},

    "What is a base case in recursion?":
        {"02_recursion_dictionaries.pdf"},

    "What is the purpose of testing and debugging?":
        {"03_testing_debugging_exceptions_assertions.pdf"},

    "What is defensive programming?":
        {"03_testing_debugging_exceptions_assertions.pdf"},

    "How does binary search work?":
        {"05_searching_and_sorting_algorithms.pdf"},

    "What is the time complexity of selection sort?":
        {"05_searching_and_sorting_algorithms.pdf"},

    "What is Big O notation?":
        {"04_understanding_program_efficiency_1.pdf"},

    "What is quantum entanglement?":
        set(),
}


# ============================================================
# Load project resources
# ============================================================

def load_resources():
    """Load the FAISS index, metadata, questions, and embedding model."""

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

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation questions file not found:\n{QUESTIONS_FILE}"
        )

    index = faiss.read_index(str(INDEX_FILE))

    with METADATA_FILE.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        questions = [
            line.strip()
            for line in file
            if line.strip()
        ]

    embedding_model = SentenceTransformer(MODEL_NAME)

    return index, metadata, questions, embedding_model


# ============================================================
# Retrieval
# ============================================================

def retrieve(
    question: str,
    index,
    metadata: list[dict],
    embedding_model: SentenceTransformer,
) -> list[dict]:
    """Retrieve the top K chunks and keep only relevant chunks."""

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
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "page_numbers": chunk["page_numbers"],
            }
        )

    return results


# ============================================================
# Evaluate one question
# ============================================================

def evaluate_question(
    question: str,
    results: list[dict],
) -> tuple[bool, str]:
    """Evaluate whether retrieval behaved as expected."""

    expected_sources = EXPECTED_SOURCES.get(question)

    if expected_sources is None:
        return False, "Question is not defined in EXPECTED_SOURCES."

    retrieved_sources = {
        result["source"]
        for result in results
    }

    # Expected unanswerable question:
    # no chunk should pass the similarity threshold.
    if not expected_sources:
        passed = len(results) == 0

        if passed:
            return True, "No relevant chunks retrieved, as expected."

        return (
            False,
            "Relevant chunks were retrieved for an intentionally "
            "unanswerable question."
        )

    # Expected answerable question:
    # at least one relevant result should come from the expected document.
    matched_sources = retrieved_sources.intersection(expected_sources)

    if matched_sources:
        return (
            True,
            "Expected source was retrieved above the similarity threshold.",
        )

    return (
        False,
        "Expected source was not retrieved above the similarity threshold.",
    )


# ============================================================
# Main evaluation
# ============================================================

def main() -> None:
    print("=" * 80)
    print("WEEK 3 - RAG RETRIEVAL EVALUATION")
    print("=" * 80)

    index, metadata, questions, embedding_model = load_resources()

    print(f"\nQuestions loaded: {len(questions)}")
    print(f"Top K: {TOP_K}")
    print(f"Similarity threshold: {MIN_SIMILARITY_SCORE:.2f}")

    passed_count = 0
    failed_count = 0

    for number, question in enumerate(questions, start=1):
        results = retrieve(
            question=question,
            index=index,
            metadata=metadata,
            embedding_model=embedding_model,
        )

        passed, reason = evaluate_question(
            question=question,
            results=results,
        )

        if passed:
            passed_count += 1
            status = "PASS"
        else:
            failed_count += 1
            status = "FAIL"

        print("\n" + "-" * 80)
        print(f"Question {number}: {question}")
        print(f"Status: {status}")
        print(f"Relevant chunks: {len(results)}")

        if results:
            print("Retrieved sources:")

            for rank, result in enumerate(results, start=1):
                print(
                    f"  {rank}. "
                    f"{result['source']} | "
                    f"score={result['similarity_score']:.4f} | "
                    f"pages={result['page_numbers']}"
                )
        else:
            print("Retrieved sources: None")

        print(f"Evaluation: {reason}")

    total = passed_count + failed_count
    pass_rate = (passed_count / total * 100) if total else 0.0

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total questions: {total}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Pass rate: {pass_rate:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
