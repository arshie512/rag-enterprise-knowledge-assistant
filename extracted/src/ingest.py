"""Load files, split them into semantic chunks, embed, and store in Chroma."""

from __future__ import annotations

from pathlib import Path

from src.config import (
    CHROMA_DB_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    MODEL_CACHE_DIR,
    SOURCE_DOCUMENTS_DIR,
)

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


def discover_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".md", ".txt"}
    )


def main() -> None:
    files = discover_files(SOURCE_DOCUMENTS_DIR)
    if not files:
        raise SystemExit(
            f"No PDF, Markdown, or TXT files found under {SOURCE_DOCUMENTS_DIR}."
        )

    print(f"Loading {len(files)} source files from {SOURCE_DOCUMENTS_DIR}")
    reader = SimpleDirectoryReader(
        input_dir=str(SOURCE_DOCUMENTS_DIR),
        recursive=True,
        required_exts=[".pdf", ".md", ".txt"],
    )
    documents = reader.load_data()
    if not documents:
        raise SystemExit("The supported files were found, but no readable text was extracted.")

    for document in documents:
        source_path = document.metadata.get("file_path") or document.metadata.get("file_name")
        if source_path:
            document.metadata["source"] = Path(source_path).name

    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        device=EMBEDDING_DEVICE,
        cache_folder=str(MODEL_CACHE_DIR),
    )
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        embed_model=embed_model,
        insert_batch_size=64,
        show_progress=True,
    )
    print(f"Indexed {len(documents)} pages/files into Chroma collection '{COLLECTION_NAME}'.")
    print(f"Persistent vector database: {CHROMA_DB_DIR}")


if __name__ == "__main__":
    main()
