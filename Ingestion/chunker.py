from __future__ import annotations 
import yaml
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ─────────────────────────────────────────────────────────────
# LOGGER SETUP
# ─────────────────────────────────────────────────────────────

# We don't call basicConfig here again — loader.py already set it up
# Just get a logger named after this file: "ingestion.chunker"
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# LOAD SETTINGS
# ─────────────────────────────────────────────────────────────

with open("config/settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

# Pull chunking values from settings
CHUNK_SIZE = settings["ingestion"]["chunk_size"]       # 800
CHUNK_OVERLAP = settings["ingestion"]["chunk_overlap"] # 100


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────

def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Takes a list of LangChain Document objects from loader.py
    Splits each document's text into overlapping chunks
    Preserves and enriches metadata on every chunk
    Returns a new list of LangChain Document objects
    """

    # ── Build the splitter ────────────────────────────────────

    # RecursiveCharacterTextSplitter tries each separator in order
    # Only moves to the next separator if chunks are still too big
    # This guarantees natural language boundaries are respected
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,          # max 800 chars per chunk
        chunk_overlap=CHUNK_OVERLAP,    # 100 chars repeated between chunks
        separators=["\n\n", "\n", ". ", " ", ""],  # priority order
        length_function=len,            # measure by character count
    )

    all_chunks = []   # final list of all chunks across all documents
    total_docs = len(documents)

    # ── Process each document ─────────────────────────────────
    for doc_index, doc in enumerate(documents):

        # split_documents() takes a list of Documents
        # Returns a new list of Documents — one per chunk
        # IMPORTANT: it automatically carries over the metadata
        # from the parent document to every child chunk
        chunks = splitter.split_documents([doc])

        # ── Enrich metadata on each chunk ─────────────────────
        for chunk_index, chunk in enumerate(chunks):

            # Add extra metadata that wasn't in the original doc
            # chunk_index: which chunk number within this document
            # total_chunks: how many chunks this document produced
            # chunk_size_setting: what setting was used (for debugging)
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["total_chunks"] = len(chunks)
            chunk.metadata["chunk_size_setting"] = CHUNK_SIZE

            all_chunks.append(chunk)

        # Log progress every document
        logger.info(
            f"Document {doc_index + 1}/{total_docs} → "
            f"{len(chunks)} chunks | "
            f"Source: {doc.metadata.get('source', 'unknown')}"
        )

    # ── Final summary ─────────────────────────────────────────
    logger.info(f"✅ Chunking complete")
    logger.info(f"   Input documents : {total_docs}")
    logger.info(f"   Output chunks   : {len(all_chunks)}")
    logger.info(f"   Chunk size      : {CHUNK_SIZE} chars")
    logger.info(f"   Chunk overlap   : {CHUNK_OVERLAP} chars")

    return all_chunks


# ─────────────────────────────────────────────────────────────
# HELPER — INSPECT CHUNKS
# ─────────────────────────────────────────────────────────────

def print_chunk_stats(chunks: list[Document]) -> None:
    """
    Prints a summary of chunk size distribution.
    Useful for verifying chunking quality during development.
    """
    lengths = [len(c.page_content) for c in chunks]

    print(f"\n--- Chunk Statistics ---")
    print(f"Total chunks : {len(chunks)}")
    print(f"Min length   : {min(lengths)} chars")
    print(f"Max length   : {max(lengths)} chars")
    print(f"Avg length   : {int(sum(lengths)/len(lengths))} chars")
    print(f"\n--- Sample Chunk ---")
    print(f"Source  : {chunks[0].metadata.get('source')}")
    print(f"Page    : {chunks[0].metadata.get('page')}")
    print(f"Chunk # : {chunks[0].metadata.get('chunk_index')}")
    print(f"Text    : {chunks[0].page_content[:300]}...")



## How Data Flows Through This File
# ```
# loader.py returns:
# [
#   Document(page_content="Transformers are...", metadata={"source": "paper.pdf", "page": 1}),
#   Document(page_content="The attention mechanism...", metadata={"source": "paper.pdf", "page": 2}),
#   ... (92 documents total)
# ]
#           ↓
# chunk_documents() runs
#           ↓
# Page 1 (1200 chars) → split into 2 chunks:
#   Chunk 0: "Transformers are... [800 chars]"
#            metadata: {source, page:1, chunk_index:0, total_chunks:2}
#   Chunk 1: "...are neural networks [800 chars]"
#            metadata: {source, page:1, chunk_index:1, total_chunks:2}

# Page 2 (600 chars) → 1 chunk (fits in 800):
#   Chunk 0: "The attention mechanism... [600 chars]"
#            metadata: {source, page:2, chunk_index:0, total_chunks:1}
#           ↓
# Returns:
# [
#   Document(chunk 1 of page 1),
#   Document(chunk 2 of page 1),
#   Document(chunk 1 of page 2),
#   ... (200+ chunks total)
# ]
