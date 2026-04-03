from __future__ import annotations
import os
import time
import yaml
import logging
import warnings
warnings.filterwarnings("ignore")

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# ─────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# LOAD SETTINGS
# ─────────────────────────────────────────────────────────────
with open("config/settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_documents"


# ─────────────────────────────────────────────────────────────
# CREATE VECTOR STORE
# ─────────────────────────────────────────────────────────────

def create_vectorstore(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings
) -> Chroma:
    """
    Embeds all chunks and stores in ChromaDB.
    Persists to ./chroma_db folder on disk.
    Only needs to run once.
    """

    logger.info(f"Creating ChromaDB vector store with {len(chunks)} chunks...")
    start_time = time.time()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )

    # chromadb==0.4.x requires explicit persist() call
    vectorstore.persist()

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"✅ ChromaDB vector store created in {elapsed}s")
    logger.info(f"   Chunks stored : {len(chunks)}")
    logger.info(f"   Saved to      : {CHROMA_DIR}")

    return vectorstore


# ─────────────────────────────────────────────────────────────
# LOAD VECTOR STORE
# ─────────────────────────────────────────────────────────────

def load_vectorstore(
    embeddings: HuggingFaceEmbeddings
) -> Chroma | None:
    """
    Loads existing ChromaDB store from disk.
    Fast — no re-embedding needed.
    Returns None if no store exists yet.
    """

    if not os.path.exists(CHROMA_DIR):
        logger.warning(f"No ChromaDB store found at {CHROMA_DIR}")
        return None

    logger.info(f"Loading ChromaDB from disk: {CHROMA_DIR}")
    start_time = time.time()

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    elapsed = round(time.time() - start_time, 2)
    count = vectorstore._collection.count()

    logger.info(f"✅ ChromaDB loaded in {elapsed}s")
    logger.info(f"   Vectors in store : {count}")

    return vectorstore


# ─────────────────────────────────────────────────────────────
# GET OR CREATE — smart loader
# ─────────────────────────────────────────────────────────────

def get_or_create_vectorstore(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    force_recreate: bool = False
) -> Chroma:
    """
    Smart function:
    → chroma_db/ exists + force_recreate=False → load from disk (fast)
    → chroma_db/ missing or force_recreate=True → create fresh (slow)

    Use force_recreate=True when you add new documents.
    """

    if force_recreate and os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        logger.info("Old ChromaDB deleted — rebuilding")

    if os.path.exists(CHROMA_DIR):
        logger.info("Existing ChromaDB found — loading from disk")
        vectorstore = load_vectorstore(embeddings)
        if vectorstore is not None:
            return vectorstore

    logger.info("No existing store — creating fresh ChromaDB")
    return create_vectorstore(chunks, embeddings)


# ─────────────────────────────────────────────────────────────
# TEST SEARCH
# ─────────────────────────────────────────────────────────────

def test_search(
    vectorstore: Chroma,
    query: str = "What is attention mechanism?"
) -> None:
    """
    Runs test similarity search.
    Prints top 3 results with scores.
    """

    logger.info(f"Testing search: '{query}'")

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=3
    )

    print(f"\n--- Test Search Results ---")
    print(f"Query: {query}\n")

    for i, (doc, score) in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Score  : {round(score, 4)}")
        print(f"  Source : {doc.metadata.get('source', 'unknown')}")
        print(f"  Page   : {doc.metadata.get('page', 'N/A')}")
        print(f"  Text   : {doc.page_content[:200]}...")
        print()