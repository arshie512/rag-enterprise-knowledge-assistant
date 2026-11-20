# Retrieval-Augmented Generation Enterprise Knowledge Assistant

A local-first RAG project built from the assignment brief as its sole knowledge-base source. It uses LlamaIndex for document loading, chunking, retrieval, and chat memory; ChromaDB for a persistent local vector store; Hugging Face for local embeddings; Ollama for local answer generation; and Streamlit for the user interface.

> **Corpus scope:** `data/assignment_brief/AIML_Major_Project_Brief.pdf` is the only indexed source. The screenshots contain substantially the same information and are not indexed separately. The brief describes a RAG project; it is not an enterprise policy corpus, so the assistant should answer questions about the assignment rather than invent organization policies.

## What the project covers

- Reads the assignment PDF and preserves source/page metadata where available.
- Splits the document into overlapping sentence-aware chunks.
- Embeds chunks with `BAAI/bge-small-en-v1.5` and stores them in a persistent Chroma collection.
- Retrieves relevant chunks and answers with a configurable local Ollama model.
- Keeps conversational memory within a Streamlit session and provides a reset control.
- Displays retrieved source names and passages so users can check answers.
- Measures retrieval with Precision@k, Recall@k, and Mean Reciprocal Rank (MRR), and provides a human-review workflow for answer accuracy and abstention.

## Requirements

- Python 3.12 or newer is recommended.
- Ollama installed and running locally, with a chat model pulled (default: `llama3.2:3b`).
- Internet access for the first dependency and embedding-model downloads.
- Enough disk space for Python packages, the embedding model, the Ollama model, and the local Chroma database.

If the installed Python version cannot install one of the dependencies, use Python 3.12 in a virtual environment. The virtual environment keeps project packages separate from the system Python.

## Setup (Windows PowerShell)

From this project folder:

```powershell
python -m venv .venv

# If pip is available in this Python installation:
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# If pip is unavailable, use uv instead:
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
```

Install Ollama using the [official Windows installer](https://ollama.com/download/windows), then download the configured model:

```powershell
ollama pull llama3.2:3b
```

The default configuration keeps embeddings and text generation local. No API key is needed. If you choose a hosted provider later, keep its key in `.env` and never commit that file.

## Build the index

1. The assignment PDF is already stored under `data/assignment_brief/`.
2. The ingestion path is configured to use only that directory. Do not add the duplicate screenshots or unused sample files from `data/source_documents/`.
3. Activate the virtual environment and run:

```powershell
python -m src.ingest
```

Rebuilding replaces this project's named Chroma collection with the current files in `data/assignment_brief/`. The database is local at `storage/chroma/`.

## Run the assistant

```powershell
streamlit run src/app.py --server.fileWatcherType none
```

Ask questions that can be answered from the assignment brief. Use the source panel beneath each answer to verify the evidence. Use **Clear conversation** to reset chat memory. The assistant should decline to invent details that the brief does not state.

The file watcher is disabled for startup compatibility with the local text-embedding stack. Restart the app manually after code changes.

## Evaluate retrieval

`data/evaluation_questions.csv` has ten answerable questions labelled with the relevant PDF page(s) and two unanswerable questions for manual abstention review. Run:

```powershell
python -m src.evaluate --top-k 3
```

The command prints aggregate Precision@k, Recall@k, and MRR and writes per-question results to `reports/retrieval_results.csv`. Metrics are only meaningful when the relevance labels are reviewed against the source.

This submission includes a completed manual review of all 12 cases in reports/answer_review_template.csv, with a summary in reports/EVALUATION_REPORT.md. Repeat the review if you change the corpus, retrieval settings, or generation model. Retrieval metrics do not prove answer truthfulness.

## Project structure

```text
data/
  assignment_brief/           # The one PDF to index
  evaluation_questions.csv    # Questions, expected answers, and source labels
notebooks/
  RAG_Enterprise_Assistant_Walkthrough.ipynb
src/
  app.py                      # Streamlit chat UI
  config.py                   # Paths and environment configuration
  ingest.py                   # Loading, chunking, embedding, Chroma indexing
  rag_core.py                 # Model, index, retrieval, and chat setup
  evaluate.py                 # Retrieval metrics and per-question output
storage/chroma/               # Generated local vector database; include in ZIP if required
reports/                      # Evaluation and answer-review templates
```

## Prepare a submission ZIP

Include the project source, notebook, assignment brief, evaluation CSVs, reports, requirements, and the generated `storage/chroma/` database if the instructor expects the built index. Exclude `.venv/`, `.uv_cache/`, `storage/models/`, `storage/tmp/`, `.env`, and the unused `data/source_documents/` folder. The embedding and chat model downloads can be recreated by following Setup.

## Troubleshooting

- **No indexed documents:** confirm the PDF exists under `data/assignment_brief/` and run ingestion before launching the app.
- **Ollama connection error:** start Ollama and confirm `ollama list` includes the model named in `.env`.
- **Model not found:** run `ollama pull <model-name>` using the value from `OLLAMA_MODEL`.
- **Very slow first run:** the first ingestion downloads the embedding model; Ollama may also download its model the first time.
- **Poor answers:** inspect the retrieved page excerpts, add clearer evaluation questions from the brief, and tune chunk size/overlap and `TOP_K` one change at a time.

## Source documentation

- [LlamaIndex chat engine and memory guide](https://docs.llamaindex.ai/en/v0.10.19/examples/chat_engine/chat_engine_context.html)
- [LlamaIndex Chroma vector store guide](https://docs.llamaindex.ai/en/v0.10.17/community/integrations/vector_stores.html)
- [LlamaIndex Hugging Face embeddings guide](https://docs.llamaindex.ai/en/v0.10.17/module_guides/models/embeddings.html)
- [Chroma persistent client](https://docs.trychroma.com/docs/run-chroma/cloud-client)
- [Streamlit getting started](https://docs.streamlit.io/get-started)








