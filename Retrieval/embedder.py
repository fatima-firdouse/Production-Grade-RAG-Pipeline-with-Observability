from __future__ import annotations
import os
import time
import yaml
import logging
import warnings
warnings.filterwarnings("ignore")

from langchain_huggingface import HuggingFaceEmbeddings

# ─────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# LOAD SETTINGS
# ─────────────────────────────────────────────────────────────
with open("config/settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

EMBEDDING_MODEL = settings["embeddings"]["model"]
# → "BAAI/bge-small-en-v1.5"


# ─────────────────────────────────────────────────────────────
# LOAD EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────

def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    Loads the embedding model from HuggingFace.
    First run: downloads model to local cache (~130MB)
    After that: loads from cache instantly — no internet needed

    Returns HuggingFaceEmbeddings object ready to embed text
    """

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")

    # Track how long model loading takes
    start_time = time.time()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,

        # model_kwargs — passed directly to the underlying model
        # device: "cpu" means run on CPU — no GPU needed
        # change to "cuda" if you have a GPU for faster embedding
        model_kwargs={"device": "cpu"},

        # encode_kwargs — controls how text gets encoded
        # normalize_embeddings: True → scales all vectors to length 1
        # This makes cosine similarity more accurate and stable
        # Standard practice for retrieval models
        encode_kwargs={"normalize_embeddings": True}
    )

    # Calculate and log load time
    load_time = round(time.time() - start_time, 2)
    logger.info(f"✅ Embedding model loaded in {load_time}s")
    logger.info(f"   Model     : {EMBEDDING_MODEL}")
    logger.info(f"   Device    : CPU")
    logger.info(f"   Normalized: True")

    return embeddings


# ─────────────────────────────────────────────────────────────
# TEST EMBEDDING — verify model works
# ─────────────────────────────────────────────────────────────

def test_embedding(embeddings: HuggingFaceEmbeddings) -> None:
    """
    Quick sanity check — embeds one sentence and prints vector info.
    Run this to verify the model loaded correctly.
    """

    test_text = "The transformer model uses self-attention mechanisms."

    # embed_query() converts one string into a vector (list of floats)
    vector = embeddings.embed_query(test_text)

    logger.info(f"✅ Test embedding successful")
    logger.info(f"   Text        : {test_text[:50]}...")
    logger.info(f"   Vector size : {len(vector)} dimensions")
    logger.info(f"   First 5 vals: {[round(v, 4) for v in vector[:5]]}")


## What Is `normalize_embeddings: True`?

# Without normalization vectors can have different lengths (magnitudes):
# ```
# Vector A: [2.0, 4.0, 6.0]  → length = 7.48
# Vector B: [0.1, 0.2, 0.3]  → length = 0.37
# ```

# Cosine similarity gets distorted when vectors have different lengths. Normalizing scales every vector to length 1:
# ```
# Vector A normalized: [0.267, 0.534, 0.802]  → length = 1.0
# Vector B normalized: [0.267, 0.534, 0.802]  → length = 1.0
