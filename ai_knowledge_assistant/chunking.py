from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_ROOT / "outputs" / "extracted_documents.json"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "chunked_documents.json"

# Baseline settings for our RAG prototype.
# We will test other values later.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def chunk_document(document: dict) -> list[dict]:
    """
    Combine all pages of one document into one text stream and
    create overlapping chunks across page boundaries.

    Page numbers are retained so the RAG system can later show
    source references for each retrieved chunk.
    """
    source = document.get("source", "")
    pages = document.get("pages", [])

    parts = []
    page_ranges = []
    current_position = 0

    for page in pages:
        page_number = page.get("page_number")
        text = page.get("text", "").strip()

        if not text:
            continue

        if parts:
            separator = "\n\n"
            parts.append(separator)
            current_position += len(separator)

        page_start = current_position
        parts.append(text)
        current_position += len(text)
        page_end = current_position

        page_ranges.append(
            {
                "page_number": page_number,
                "start": page_start,
                "end": page_end,
            }
        )

    full_text = "".join(parts)

    if not full_text:
        return []

    chunks = []
    start = 0
    text_length = len(full_text)

    while start < text_length:
        end = min(start + CHUNK_SIZE, text_length)
        chunk_text = full_text[start:end].strip()

        if chunk_text:
            covered_pages = [
                item["page_number"]
                for item in page_ranges
                if item["end"] > start and item["start"] < end
            ]

            chunks.append(
                {
                    "source": source,
                    "page_numbers": covered_pages,
                    "text": chunk_text,
                    "character_count": len(chunk_text),
                }
            )

        if end >= text_length:
            break

        start = end - CHUNK_OVERLAP

    return chunks


def load_extracted_documents() -> dict:
    """Load the JSON produced by document_ingestion.py."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run document_ingestion.py first."
        )

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_all_chunks(data: dict) -> list[dict]:
    """Create chunks for every source document."""
    all_chunks = []

    for document in data.get("documents", []):
        for chunk in chunk_document(document):
            chunk["chunk_id"] = len(all_chunks) + 1
            all_chunks.append(chunk)

    return all_chunks


def main() -> None:
    print("=" * 80)
    print("WEEK 3 - RAG DOCUMENT CHUNKING")
    print("=" * 80)

    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than 0.")

    if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE."
        )

    data = load_extracted_documents()
    chunks = create_all_chunks(data)

    if not chunks:
        raise ValueError("No chunks were created.")

    output_data = {
        "project": data.get("project", "AI Knowledge Assistant"),
        "domain": data.get("domain", "Computer Science Study Assistant"),
        "source_document_count": data.get("document_count", 0),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, ensure_ascii=False, indent=2)

    print(f"Source documents: {output_data['source_document_count']}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")
    print(f"Chunks created: {len(chunks)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nFirst chunk preview:")
    print("-" * 80)
    print(chunks[0]["text"][:500])
    print("-" * 80)
    print("Source pages:", chunks[0]["page_numbers"])

    print("\nChunking complete.")


if __name__ == "__main__":
    main()
