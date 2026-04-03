from __future__ import annotations
import time
import os
import yaml
import logging
import warnings
warnings.filterwarnings("ignore")

from langchain.schema import Document
from sentence_transformers import CrossEncoder




os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
# ─────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# LOAD SETTINGS
# ─────────────────────────────────────────────────────────────
with open("config/settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

RERANKER_MODEL = settings["retrieval"]["reranker_model"]
# → "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K = settings["retrieval"]["top_k"]
# → 5


# ─────────────────────────────────────────────────────────────
# LOAD RERANKER MODEL
# ─────────────────────────────────────────────────────────────

def load_reranker() -> CrossEncoder:
    """
    Loads the cross-encoder reranker model.
    First run: downloads ~80MB model to cache
    After that: loads from cache instantly

    CrossEncoder reads query + chunk TOGETHER
    → much more accurate than bi-encoder similarity
    """

    logger.info(f"Loading reranker: {RERANKER_MODEL}")
    start_time = time.time()

    # CrossEncoder takes (query, chunk) pairs as input
    # Outputs a relevance score for each pair
    # Unlike embeddings which encode separately,
    # cross-encoder sees both texts simultaneously
    # This makes it slower but significantly more accurate
    reranker = CrossEncoder(
        RERANKER_MODEL,
        max_length=512  # max tokens per (query, chunk) pair
    )

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"✅ Reranker loaded in {elapsed}s")
    logger.info(f"   Model: {RERANKER_MODEL}")

    return reranker


# ─────────────────────────────────────────────────────────────
# RERANK CHUNKS
# ─────────────────────────────────────────────────────────────

def rerank_chunks(
    query: str,
    chunks: list[Document],
    reranker: CrossEncoder,
    top_k: int = TOP_K
) -> list[tuple[Document, float]]:
    """
    Reranks chunks using cross-encoder.

    Process:
    1. Build (query, chunk_text) pairs
    2. CrossEncoder scores each pair together
    3. Sort by score descending
    4. Return top_k with scores

    Returns list of (Document, score) tuples
    Score closer to 1.0 = more relevant
    """

    if not chunks:
        logger.warning("No chunks to rerank")
        return []

    logger.info(f"Reranking {len(chunks)} chunks for: '{query}'")
    start_time = time.time()

    # Build input pairs — (query, chunk_text) for each chunk
    # CrossEncoder needs BOTH texts together in each pair
    # This is what makes it a "cross" encoder — crosses both inputs
    pairs = [
        [query, chunk.page_content]
        for chunk in chunks
    ]

    # predict() scores all pairs in one batch
    # Returns a numpy array of scores — one per pair
    # Score range: roughly -10 to +10 (not a probability)
    # Higher = more relevant
    scores = reranker.predict(pairs)

    # Zip chunks with their scores
    chunk_score_pairs = list(zip(chunks, scores))

    # Sort by score descending — most relevant first
    chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)

    elapsed = round(time.time() - start_time, 2)

    # Take only top_k after reranking
    top_results = chunk_score_pairs[:top_k]

    logger.info(f"✅ Reranking complete in {elapsed}s")
    logger.info(f"   Input chunks  : {len(chunks)}")
    logger.info(f"   Output chunks : {len(top_results)}")

    # Log scores for transparency
    for i, (doc, score) in enumerate(top_results):
        logger.info(
            f"   Rank {i+1}: score={round(float(score), 4)} | "
            f"source={doc.metadata.get('source', 'unknown')}"
        )

    return top_results