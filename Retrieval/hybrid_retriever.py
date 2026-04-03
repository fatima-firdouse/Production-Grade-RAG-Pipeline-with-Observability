from __future__ import annotations
import yaml
import logging
import warnings
warnings.filterwarnings("ignore")

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi

# ─────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# LOAD SETTINGS
# ─────────────────────────────────────────────────────────────
with open("config/settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

TOP_K = settings["retrieval"]["top_k"]  # 5


# ─────────────────────────────────────────────────────────────
# BM25 INDEX BUILDER
# ─────────────────────────────────────────────────────────────

def build_bm25_index(chunks: list[Document]) -> tuple[BM25Okapi, list[Document]]:
    """
    Builds a BM25 index from all chunks.
    BM25 needs tokenized text — we split on whitespace.

    Returns:
        bm25    : the searchable BM25 index
        chunks  : same chunks list (needed to retrieve text later)
    """

    # Tokenize each chunk's text into a list of words
    # BM25 works on token lists — not raw strings
    # .lower() ensures "Attention" and "attention" match equally
    tokenized_chunks = [
        chunk.page_content.lower().split()
        for chunk in chunks
    ]

    # BM25Okapi is the most common BM25 variant
    # "Okapi" refers to the Okapi BM25 formula from 1994
    # It handles term frequency + inverse document frequency
    bm25 = BM25Okapi(tokenized_chunks)

    logger.info(f"✅ BM25 index built with {len(chunks)} chunks")
    return bm25, chunks


# ─────────────────────────────────────────────────────────────
# BM25 SEARCH
# ─────────────────────────────────────────────────────────────

def bm25_search(
    query: str,
    bm25: BM25Okapi,
    chunks: list[Document],
    top_k: int = 10
) -> list[tuple[Document, float]]:
    """
    Searches BM25 index for query.
    Returns top_k chunks with their BM25 scores.
    """

    # Tokenize the query the same way we tokenized chunks
    tokenized_query = query.lower().split()

    # get_scores() returns a score for every chunk in the index
    # Higher score = more relevant to query
    scores = bm25.get_scores(tokenized_query)

    # Pair each chunk with its score
    chunk_score_pairs = list(zip(chunks, scores))

    # Sort by score descending — best matches first
    chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # Return top_k results
    top_results = chunk_score_pairs[:top_k]

    logger.info(f"BM25 search returned {len(top_results)} results")
    return top_results


# ─────────────────────────────────────────────────────────────
# VECTOR SEARCH
# ─────────────────────────────────────────────────────────────

def vector_search(
    query: str,
    vectorstore: Chroma,
    top_k: int = 10
) -> list[tuple[Document, float]]:
    """
    Searches ChromaDB for semantically similar chunks.
    Returns top_k chunks with their similarity scores.
    """

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k
    )

    logger.info(f"Vector search returned {len(results)} results")
    return results


# ─────────────────────────────────────────────────────────────
# RECIPROCAL RANK FUSION
# ─────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    bm25_results: list[tuple[Document, float]],
    vector_results: list[tuple[Document, float]],
    k: int = 60
) -> list[Document]:
    """
    Merges BM25 and vector search results using RRF.

    RRF formula: score = 1 / (rank + k)
    k=60 is standard — prevents top-ranked items dominating too much

    Chunks appearing in BOTH result lists get combined scores
    → naturally rewarded for being relevant to both search types
    """

    # Dictionary to accumulate RRF scores per chunk
    # Key: chunk text (unique identifier)
    # Value: running RRF score
    rrf_scores: dict[str, float] = {}

    # Map chunk text → Document object for retrieval later
    chunk_map: dict[str, Document] = {}

    # ── Score BM25 results ────────────────────────────────────
    for rank, (doc, score) in enumerate(bm25_results):
        chunk_id = doc.page_content

        # RRF formula — rank is 0-indexed so rank+1
        rrf_score = 1.0 / (rank + 1 + k)

        # Add to existing score (might also appear in vector results)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score
        chunk_map[chunk_id] = doc

    # ── Score vector results ──────────────────────────────────
    for rank, (doc, score) in enumerate(vector_results):
        chunk_id = doc.page_content

        rrf_score = 1.0 / (rank + 1 + k)

        # If chunk was already scored by BM25, scores ADD together
        # This is how hybrid search rewards cross-method relevance
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score
        chunk_map[chunk_id] = doc

    # ── Sort by combined RRF score ────────────────────────────
    sorted_chunks = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True  # highest score first
    )

    # Return Document objects in RRF order
    merged = [chunk_map[chunk_id] for chunk_id, score in sorted_chunks]

    logger.info(f"RRF merged {len(bm25_results)} BM25 + "
                f"{len(vector_results)} vector → {len(merged)} unique chunks")

    return merged


# ─────────────────────────────────────────────────────────────
# HYBRID RETRIEVER — MAIN FUNCTION
# ─────────────────────────────────────────────────────────────

def hybrid_retrieve(
    query: str,
    vectorstore: Chroma,
    bm25: BM25Okapi,
    chunks: list[Document],
    top_k: int = TOP_K
) -> list[Document]:
    """
    Full hybrid retrieval pipeline:
    1. BM25 keyword search → top 10
    2. Vector semantic search → top 10
    3. RRF merge → ranked unique list
    4. Return top_k for reranking

    Returns list of top_k most relevant Document chunks
    """

    logger.info(f"Hybrid retrieval for: '{query}'")

    # ── Step 1: BM25 search ───────────────────────────────────
    bm25_results = bm25_search(
        query=query,
        bm25=bm25,
        chunks=chunks,
        top_k=10  # get more than needed — reranker will filter
    )

    # ── Step 2: Vector search ─────────────────────────────────
    vector_results = vector_search(
        query=query,
        vectorstore=vectorstore,
        top_k=10
    )

    # ── Step 3: RRF merge ─────────────────────────────────────
    merged_chunks = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        vector_results=vector_results
    )

    # ── Step 4: Return top_k for reranker ─────────────────────
    # Reranker gets top 10, returns top 5
    # We pass top 10 here to give reranker enough to work with
    final_chunks = merged_chunks[:10]

    logger.info(f"✅ Hybrid retrieval complete → {len(final_chunks)} chunks")
    return final_chunks