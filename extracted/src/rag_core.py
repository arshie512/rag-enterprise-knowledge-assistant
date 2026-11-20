"""Index loading and RAG chat-engine construction."""

from __future__ import annotations

from src.config import (
    CHAT_MEMORY_TOKENS,
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    MODEL_CACHE_DIR,
    OLLAMA_MODEL,
    OLLAMA_REQUEST_TIMEOUT,
    SYSTEM_PROMPT,
    TOP_K,
)

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


def make_embedding_model() -> HuggingFaceEmbedding:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        device=EMBEDDING_DEVICE,
        cache_folder=str(MODEL_CACHE_DIR),
    )


def make_llm() -> Ollama:
    return Ollama(
        model=OLLAMA_MODEL,
        request_timeout=OLLAMA_REQUEST_TIMEOUT,
        temperature=0,
    )


def load_index() -> VectorStoreIndex:
    """Open the persistent Chroma collection and wrap it as a LlamaIndex index."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "The vector index is missing. Add source files and run `python -m src.ingest` first."
        ) from exc

    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=make_embedding_model(),
    )


def make_chat_engine(index: VectorStoreIndex):
    """Create an isolated chat engine with bounded session memory."""
    memory = ChatMemoryBuffer.from_defaults(token_limit=CHAT_MEMORY_TOKENS)
    return index.as_chat_engine(
        chat_mode="condense_plus_context",
        llm=make_llm(),
        memory=memory,
        similarity_top_k=TOP_K,
        context_prompt=SYSTEM_PROMPT,
        verbose=False,
    )
