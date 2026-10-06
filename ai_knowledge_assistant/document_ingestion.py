from pathlib import Path
import json
import re

import pdfplumber


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# The five selected PDFs are stored directly in this folder.
DOCUMENTS_DIR = PROJECT_ROOT / "rag_documents"

# Extracted text will be saved here.
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "extracted_documents.json"


# ============================================================
# Text cleaning
# ============================================================

def clean_text(text: str) -> str:
    """Clean basic whitespace while preserving readable line breaks."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# Extract one PDF
# ============================================================

def extract_pdf(pdf_path: Path) -> dict:
    """Extract text from every page of one PDF and keep page metadata."""
    pages = []
    total_characters = 0

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)

        for page_number, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned = clean_text(raw_text)
            character_count = len(cleaned)
            total_characters += character_count

            pages.append(
                {
                    "page_number": page_number,
                    "text": cleaned,
                    "character_count": character_count,
                }
            )

    return {
        "source": pdf_path.name,
        "page_count": page_count,
        "total_character_count": total_characters,
        "pages": pages,
    }


# ============================================================
# Main program
# ============================================================

def main() -> None:
    print("=" * 80)
    print("WEEK 3 - RAG DOCUMENT INGESTION")
    print("=" * 80)

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"\nDocument folder not found:\n{DOCUMENTS_DIR}\n\n"
            "Expected folder: ai_knowledge_assistant\\rag_documents"
        )

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"\nNo PDF files were found in:\n{DOCUMENTS_DIR}\n\n"
            "Place the selected PDF files directly inside rag_documents."
        )

    print(f"\nDocuments found: {len(pdf_files)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extracted_documents = []

    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"\n[{index}/{len(pdf_files)}] Processing:")
        print(f"    {pdf_path.name}")

        try:
            document_data = extract_pdf(pdf_path)
            extracted_documents.append(document_data)

            pages_with_text = sum(
                1 for page in document_data["pages"] if page["text"]
            )

            print(f"    Pages: {document_data['page_count']}")
            print(f"    Pages with text: {pages_with_text}")
            print(
                "    Characters extracted: "
                f"{document_data['total_character_count']:,}"
            )

        except Exception as error:
            print(f"    ERROR: {error}")
            raise

    output_data = {
        "project": "AI Knowledge Assistant",
        "domain": "Computer Science Study Assistant",
        "document_count": len(extracted_documents),
        "documents": extracted_documents,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, ensure_ascii=False, indent=2)

    total_pages = sum(
        document["page_count"] for document in extracted_documents
    )
    total_characters = sum(
        document["total_character_count"] for document in extracted_documents
    )

    print("\n" + "=" * 80)
    print("INGESTION COMPLETE")
    print("=" * 80)
    print(f"Documents processed: {len(extracted_documents)}")
    print(f"Total pages processed: {total_pages}")
    print(f"Total characters extracted: {total_characters:,}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
