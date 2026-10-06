# AI Knowledge Assistant

A Retrieval-Augmented Generation (RAG) study assistant that answers Computer Science questions using a selected collection of PDF documents.

The system retrieves relevant document passages with semantic vector search and uses Google Gemini through LangChain to generate grounded answers. Source document names and page references are displayed with responses.

## Project goal

The project demonstrates a complete RAG workflow:

PDF documents
-> text extraction
-> text chunking
-> embeddings
-> FAISS vector index
-> user question
-> semantic retrieval
-> grounded prompt
-> Gemini answer
-> source references

The application is designed for Computer Science study material and is intentionally limited to the currently loaded documents.

## Main technologies

- Python
- Streamlit
- pdfplumber
- SentenceTransformers
- FAISS
- Google Gemini
- LangChain
- NumPy
- PyTorch

## Current configuration

### Embedding model

`all-MiniLM-L6-v2`

Embedding dimension: `384`

### LLM

`gemini-3.5-flash-lite`

The model can be overridden using the `GEMINI_MODEL` environment variable.

### Retrieval

- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Top-K: 5
- Similarity threshold: 0.35

## Knowledge base

The prepared knowledge base contains five selected Computer Science lecture PDFs in:

`rag_documents`

Current measured collection:

- 5 PDF documents
- 205 total pages
- 205 pages containing extractable text
- 59,662 extracted characters
- 77 chunks
- 384-dimensional embeddings

Selected documents:

1. `01_decomposition_abstraction_functions.pdf`
2. `02_recursion_dictionaries.pdf`
3. `03_testing_debugging_exceptions_assertions.pdf`
4. `04_understanding_program_efficiency_1.pdf`
5. `05_searching_and_sorting_algorithms.pdf`

## RAG pipeline

### 1. PDF text extraction

`document_ingestion.py` uses `pdfplumber` to extract text page by page.

The extracted document data contains:

- source filename
- page number
- extracted text

Output:

`outputs/extracted_documents.json`

### 2. Chunking

`chunking.py` combines page text and creates overlapping character-based chunks.

Current configuration:

- chunk size: 1000
- overlap: 200

Page numbers are preserved for source references.

Output:

`outputs/chunked_documents.json`

The current prepared collection creates 77 chunks.

### 3. Embeddings

`embeddings.py` uses:

`SentenceTransformer("all-MiniLM-L6-v2")`

Each chunk is converted into a dense vector.

Embeddings are normalized before FAISS indexing.

Outputs:

- `outputs/chunk_embeddings.npy`
- `outputs/chunk_metadata.json`

### 4. FAISS vector search

`faiss_index.py` creates:

`faiss.IndexFlatIP`

The embeddings are normalized, so inner-product ranking provides cosine-similarity-style retrieval.

Output:

`outputs/faiss_index.bin`

### 5. Query retrieval

For each question:

1. The question is embedded.
2. FAISS retrieves the top 5 chunks.
3. Results below similarity `0.35` are removed.
4. Remaining chunks become the evidence supplied to the LLM.

### 6. Grounded answer generation

The retrieved passages are placed into a controlled prompt.

The prompt instructs Gemini to:

- use only retrieved document evidence
- ignore unrelated passages
- combine relevant passages when necessary
- avoid outside knowledge
- avoid unsupported claims
- answer only what the evidence supports

Gemini is accessed through LangChain's Google Gemini integration.

### 7. Hallucination control

When no retrieved chunk reaches the `0.35` threshold:

- the LLM is not called
- the user receives an insufficient-information message
- no unsupported answer is generated

## Streamlit application

The main application is:

`app.py`

Run it with:

```cmd
"C:\Users\ACER\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m streamlit run app.py --server.fileWatcherType none
```

The application supports:

- prepared project documents
- uploaded PDF collections
- PDF text extraction
- chunking
- embeddings
- FAISS retrieval
- grounded Gemini generation through LangChain
- source references
- retrieved-evidence inspection
- unsupported-question blocking

## Command-line RAG pipeline

`rag_pipeline.py` provides the non-UI RAG workflow.

Run it with:

```cmd
"C:\Users\ACER\AppData\Local\Python\pythoncore-3.14-64\python.exe" rag_pipeline.py
```

The program:

1. accepts a user question
2. retrieves relevant document chunks
3. applies the 0.35 similarity threshold
4. blocks unsupported questions when no evidence is retrieved
5. generates a grounded Gemini answer
6. prints source references

## Retrieval testing

`retrieval_test.py` is an interactive vector-search smoke test.

Run it with:

```cmd
"C:\Users\ACER\AppData\Local\Python\pythoncore-3.14-64\python.exe" retrieval_test.py
```

Example tested queries include:

- `What is recursion?`
- `How does binary search work?`
- `What is Big O notation?`
- `What is quantum entanglement?`

The quantum-entanglement question is intentionally outside the prepared knowledge base and is expected to produce no sufficiently relevant chunks.

## Retrieval evaluation

`evaluate_retrieval.py` evaluates the 10 questions in:

`evaluation_questions.txt`

The evaluation includes:

1. decomposition
2. abstraction
3. recursion
4. recursion base cases
5. testing and debugging
6. defensive programming
7. binary search
8. selection sort complexity
9. Big O notation
10. an intentionally unsupported quantum-entanglement question

### Measured result

- Total questions: 10
- Passed: 10
- Failed: 0
- Retrieval pass rate: 100.0%
- Similarity threshold: 0.35

The 100.0% result is a retrieval-source metric on this evaluation set. It is not a claim that every generated answer is 100% correct.

## Answer-quality evaluation

`answer_evaluation.md` defines a manual answer-quality rubric.

The six criteria are:

1. Relevance
2. Grounding
3. Correctness
4. Completeness
5. Hallucination avoidance
6. Source references

Each criterion is scored from 0 to 2, for a maximum of 12 points per answer.

One manually evaluated decomposition answer received:

`11/12`

This is a manual evaluation example, not a statistically significant benchmark.

## Chunk-size experiment

`evaluate_chunk_sizes.py` compares three chunk configurations using the same retrieval evaluation.

Measured results:

| Chunk size | Overlap | Chunks created | Retrieval pass rate |
|---:|---:|---:|---:|
| 500 | 100 | 152 | 100.0% |
| 1000 | 200 | 77 | 100.0% |
| 1500 | 300 | 51 | 100.0% |

All three configurations passed the current 10-question retrieval evaluation.

### Production decision

The project keeps:

`chunk size = 1000`

`chunk overlap = 200`

This configuration is already integrated and tested end-to-end in the CLI and Streamlit applications.

Detailed results are documented in:

`chunk_size_experiment.md`

## Source references

The application groups retrieved pages belonging to the same document and displays page ranges.

Example:

`02_recursion_dictionaries.pdf - Pages: 1-9, 11-18`

The Streamlit application also exposes the retrieved evidence used for generation.

## Project structure

```text
ai_knowledge_assistant/
|
|-- app.py
|-- document_ingestion.py
|-- chunking.py
|-- embeddings.py
|-- faiss_index.py
|-- retrieval_test.py
|-- rag_pipeline.py
|-- evaluate_retrieval.py
|-- evaluate_chunk_sizes.py
|-- evaluation_questions.txt
|-- answer_evaluation.md
|-- chunk_size_experiment.md
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- rag_documents/
|   |-- 01_decomposition_abstraction_functions.pdf
|   |-- 02_recursion_dictionaries.pdf
|   |-- 03_testing_debugging_exceptions_assertions.pdf
|   |-- 04_understanding_program_efficiency_1.pdf
|   |-- 05_searching_and_sorting_algorithms.pdf
|
|-- outputs/
|   |-- extracted_documents.json
|   |-- chunked_documents.json
|   |-- chunk_embeddings.npy
|   |-- chunk_metadata.json
|   |-- faiss_index.bin
|
|-- data/
|-- documents/
|-- knowledge_base/
```

Generated files in `outputs/` are excluded from version control.

## Installation

The project was tested with Python 3.14.2.

Install the pinned dependencies with:

```cmd
python -m pip install -r requirements.txt
```

Tested interpreter:

```text
C:\Users\ACER\AppData\Local\Python\pythoncore-3.14-64\python.exe
```

## Environment variables

The application requires:

`GEMINI_API_KEY`

Optional model override:

`GEMINI_MODEL`

Do not place API keys directly in source code.

Example:

```cmd
set GEMINI_MODEL=gemini-3.5-flash-lite
```

The API key should be supplied through the environment rather than hardcoded in Python files.

## Dependencies

Pinned main dependencies:

- streamlit 1.64.0
- sentence-transformers 6.1.0
- faiss-cpu 1.15.1
- pdfplumber 0.11.10
- google-genai 2.28.0
- numpy 2.4.0
- torch 2.10.0
- transformers 5.18.0
- tokenizers 0.23.2
- scikit-learn 1.8.0
- scipy 1.17.0
- huggingface-hub 1.33.0
- langchain 1.4.3
- langchain-core 1.6.6
- langchain-google-genai 4.4.0

The tested environment reports:

`No broken requirements found.`

## Security

Never commit:

- API keys
- authentication tokens
- private credentials
- `.env` files

The project's `.gitignore` excludes common environment-secret files and generated RAG outputs.

## Limitations

### PDF extraction

Some PDFs can contain text-extraction artifacts. The current implementation uses the extracted text as provided by the extraction step.

### Retrieval sensitivity

Semantic search can return related but less directly relevant chunks.

### Similarity threshold

The `0.35` threshold is an engineering choice based on the current evaluation set.

A different dataset may require recalibration.

### Evaluation size

The current retrieval evaluation contains only 10 questions.

A 100.0% result on this set does not guarantee identical performance on unseen questions.

### LLM dependence

Gemini is an external service. Availability, latency, API limits, model behavior, and service errors can affect the application.

### Knowledge scope

The assistant is limited to the currently loaded documents.

It should not be expected to reliably answer topics that are absent from the knowledge base.

### PDF quality

The quality of retrieval depends partly on the quality of extracted PDF text.

Complex layouts, scanned pages, tables, diagrams, or unusual formatting may reduce extraction quality.

## Future improvements

Possible improvements include:

- better PDF extraction
- semantic or token-aware chunking
- retrieval reranking
- configurable Top-K
- configurable similarity threshold
- larger evaluation datasets
- automated answer-quality scoring
- persistent vector storage
- metadata filtering
- improved citation presentation
- conversation history
- hosted deployment

## Project status

The prototype has demonstrated:

- PDF document ingestion
- text extraction
- overlapping chunking
- semantic embeddings
- FAISS vector search
- similarity-threshold filtering
- grounded Gemini generation
- LangChain integration
- Streamlit UI
- uploaded-PDF support
- source references
- retrieved-evidence inspection
- unsupported-question blocking
- retrieval evaluation
- chunk-size comparison

Final retrieval evaluation:

`10/10`

at a similarity threshold of:

`0.35`