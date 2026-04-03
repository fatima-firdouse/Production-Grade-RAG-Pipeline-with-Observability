from __future__ import annotations
from Generation.prompt_manager import load_prompts, build_prompt
from Generation.generator import load_llm_client, generate_answer
from Retrieval.hybrid_retriever import build_bm25_index, hybrid_retrieve
from Retrieval.reranker import load_reranker, rerank_chunks
import os
import json
import logging
from dotenv import load_dotenv
from Ingestion.loader import load_documents
from Ingestion.chunker import chunk_documents
from Retrieval.embedder import load_embedding_model, test_embedding
from Retrieval.vectorstore import get_or_create_vectorstore, test_search
import time
from Evaluation.evaluator import (
    load_golden_dataset,
    build_ragas_dataset,
    run_ragas_evaluation,
    print_scorecard
)
from Retrieval.hybrid_retriever import build_bm25_index
from Observability.langfuse_tracker import init_langfuse, trace_rag_query, log_ragas_scores
import yaml
with open("config/settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

    
# ─────────────────────────────────────────────────────────────
# LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# PHASE 1 — INGESTION
# ─────────────────────────────────────────────────────────────

def run_ingestion(
    data_dir: str = "./data/papers",
    urls: list = [],
    wikipedia_topics: list = []
) -> list:
    """
    Step 1 — Load documents from all sources
    Step 2 — Chunk into overlapping pieces
    Returns list of LangChain Document chunks
    """

    print("\n" + "="*55)
    print("   RAG PIPELINE — PHASE 1: INGESTION")
    print("="*55)

    # ── Step 1: Load ──────────────────────────────────────────
    print("\n📂 Step 1: Loading documents...")
    documents = load_documents(
        data_dir=data_dir,
        urls=urls,
        wikipedia_topics=wikipedia_topics
    )

    if not documents:
        print("\n❌ No documents loaded.")
        print("   → Add files to data/papers/")
        return []

    print(f"   ✅ Loaded {len(documents)} raw documents")

    # ── Step 2: Chunk ─────────────────────────────────────────
    print("\n✂️  Step 2: Chunking documents...")
    chunks = chunk_documents(documents)

    if not chunks:
        print("\n❌ Chunking failed — no chunks created")
        return []

    print(f"   ✅ Created {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────────────────────
# PHASE 2 — EMBEDDINGS + VECTOR STORE
# ─────────────────────────────────────────────────────────────

def run_embedding(chunks: list, force_recreate: bool = False):
    """
    Step 3 — Load embedding model
    Step 4 — Store chunks as vectors in ChromaDB
    Returns vectorstore object ready for search
    """

    print("\n" + "="*55)
    print("   RAG PIPELINE — PHASE 2: EMBEDDINGS")
    print("="*55)

    # ── Step 3: Load embedding model ──────────────────────────
    # First run downloads model (~130MB) — takes 1-2 minutes
    # After that loads from cache — takes seconds
    print("\n🧠 Step 3: Loading embedding model...")
    embeddings = load_embedding_model()

    # Quick sanity check — embeds one sentence to verify model works
    test_embedding(embeddings)
    print(f"   ✅ Embedding model ready")

    # ── Step 4: Create/load vector store ─────────────────────
    # force_recreate=False → load from disk if exists (fast)
    # force_recreate=True  → rebuild from scratch (slow)
    print("\n🗄️  Step 4: Setting up vector store...")
    vectorstore = get_or_create_vectorstore(
        chunks=chunks,
        embeddings=embeddings,
        force_recreate=force_recreate
    )
    print(f"   ✅ Vector store ready")

    return vectorstore, embeddings


# ─────────────────────────────────────────────────────────────
# SHOW INGESTION RESULTS
# ─────────────────────────────────────────────────────────────

def show_ingestion_results(chunks: list) -> None:
    """
    Prints summary of ingestion — chunks, sources, sample text.
    """
    if not chunks:
        return

    lengths = [len(c.page_content) for c in chunks]

    print("\n" + "="*55)
    print("   INGESTION RESULTS")
    print("="*55)

    print(f"\n📊 Chunk Statistics:")
    print(f"   Total chunks : {len(chunks)}")
    print(f"   Min length   : {min(lengths)} chars")
    print(f"   Max length   : {max(lengths)} chars")
    print(f"   Avg length   : {int(sum(lengths)/len(lengths))} chars")

    # Count chunks per source file
    source_counts = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"\n📁 Chunks per source:")
    for source, count in source_counts.items():
        print(f"   {os.path.basename(source):<40} {count} chunks")

    print(f"\n🔍 Sample Chunk Preview:")
    print(f"   Source : {chunks[0].metadata.get('source', 'unknown')}")
    print(f"   Page   : {chunks[0].metadata.get('page', 'N/A')}")
    print(f"   Text   : {chunks[0].page_content[:300]}...")

    print(f"\n🏷️  Metadata on first chunk:")
    for key, value in chunks[0].metadata.items():
        print(f"   {key:<25} : {value}")


# ─────────────────────────────────────────────────────────────
# SHOW EMBEDDING RESULTS
# ─────────────────────────────────────────────────────────────

def show_embedding_results(vectorstore) -> None:
    """
    Prints vector store summary + runs a test search.
    """
    print("\n" + "="*55)
    print("   EMBEDDING RESULTS")
    print("="*55)

    # Get total vector count from ChromaDB
    count = vectorstore._collection.count()
    print(f"\n🗄️  Vector Store:")
    print(f"   Total vectors stored : {count}")
    print(f"   Persist directory    : ./chroma_db")
    print(f"   Collection name      : rag_documents")

    # Run test search to verify retrieval works
    print(f"\n🔍 Running test search...")
    test_search(
        vectorstore=vectorstore,
        query="how to treat missing values?"
    )

    print("\n" + "="*55)
    print("   ✅ DAY 2 COMPLETE — Ready for Day 3")
    print("   Next: Hybrid Search + Reranking")
    print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────
# PHASE 3 — HYBRID RETRIEVAL + RERANKING
# ─────────────────────────────────────────────────────────────


def run_retrieval(
    query: str,
    vectorstore,
    chunks: list,
    reranker
) -> tuple:
    """
    Returns (reranked_chunks, retrieval_latency_ms, reranking_latency_ms)
    """
    print("\n" + "="*55)
    print("   RAG PIPELINE — PHASE 3: RETRIEVAL")
    print("="*55)
    print(f"\n🔍 Query: {query}\n")

    # ── Step 5: BM25 index ────────────────────────────────────
    print("📑 Step 5: Building BM25 index...")
    bm25, _ = build_bm25_index(chunks)
    print(f"   ✅ BM25 index ready ({len(chunks)} chunks indexed)")

    # ── Step 6: Hybrid retrieve ───────────────────────────────
    print("\n🔀 Step 6: Hybrid retrieval (BM25 + Vector + RRF)...")
    t1 = time.time()
    retrieved_chunks = hybrid_retrieve(
        query=query,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks=chunks,
        top_k=10
    )
    retrieval_latency = round((time.time() - t1) * 1000)
    print(f"   ✅ Retrieved {len(retrieved_chunks)} candidates")

    # ── Step 7: Rerank ────────────────────────────────────────
    print("\n🎯 Step 7: Reranking candidates...")
    t2 = time.time()
    reranked = rerank_chunks(
        query=query,
        chunks=retrieved_chunks,
        reranker=reranker,
        top_k=5
    )
    reranking_latency = round((time.time() - t2) * 1000)
    print(f"   ✅ Reranked to top {len(reranked)} chunks")

    return reranked, retrieved_chunks, retrieval_latency, reranking_latency






def show_retrieval_results(
    query: str,
    reranked_chunks: list
) -> None:
    """
    Prints final reranked chunks with scores.
    """
    print("\n" + "="*55)
    print("   RETRIEVAL RESULTS")
    print("="*55)
    print(f"\nQuery: {query}\n")

    for i, (doc, score) in enumerate(reranked_chunks):
        print(f"Rank {i+1} | Score: {round(float(score), 4)}")
        print(f"  Source : {doc.metadata.get('source', 'N/A')}")
        print(f"  Page   : {doc.metadata.get('page', 'N/A')}")
        print(f"  Text   : {doc.page_content[:200]}...")
        print()

    print("="*55)
    print("   ✅ DAY 3 COMPLETE — Ready for Day 4")
    print("   Next: LLM Generation + Citations")
    print("="*55 + "\n")

# ─────────────────────────────────────────────────────────────
# PHASE 4 — GENERATION
# ─────────────────────────────────────────────────────────────

def run_generation(
    question: str,
    reranked_chunks: list,
    llm_client,
    prompts: dict
) -> dict:
    """
    Step 8 — Build prompt from chunks + question
    Step 9 — Call Mistral-7B via HF API
    Step 10 — Return cited answer
    """

    print("\n" + "="*55)
    print("   RAG PIPELINE — PHASE 4: GENERATION")
    print("="*55)

    # ── Step 8: Build prompt ──────────────────────────────────
    # Combines system instructions + context chunks + question
    print("\n📝 Step 8: Building prompt...")
    prompt = build_prompt(
        question=question,
        chunks=reranked_chunks,
        prompts=prompts
    )
    print(f"   ✅ Prompt built ({len(prompt)} chars)")
    print(f"   Prompt version: {prompts['rag_prompt']['version']}")

    # ── Step 9: Generate answer ───────────────────────────────
    print("\n🤖 Step 9: Calling Mistral-7B...")
    print(f"   Model: {settings['generation']['model']}")
    print("   (This takes 10-30 seconds)...")

    result = generate_answer(
        prompt=prompt,
        client=llm_client,
        prompts=prompts
    )

    return result


def show_generation_results(
    question: str,
    result: dict,
    reranked_chunks: list
) -> None:
    """
    Prints the final answer with citations.
    """
    print("\n" + "="*55)
    print("   FINAL ANSWER")
    print("="*55)

    print(f"\n❓ Question: {question}\n")
    print(f"💬 Answer:\n")
    print(f"   {result['answer']}")

    print(f"\n📊 Generation Stats:")
    print(f"   Model        : {result['model']}")
    print(f"   Latency      : {result['latency_ms']}ms")
    print(f"   Prompt ver   : {result.get('prompt_version', 'N/A')}")
    print(f"   Sources used : {len(reranked_chunks)} chunks")

    print(f"\n📚 Source Chunks Used:")
    for i, (doc, score) in enumerate(reranked_chunks):
        source = os.path.basename(doc.metadata.get('source', 'N/A'))
        page = doc.metadata.get('page', 'N/A')
        print(f"   [{i+1}] {source} — Page {page} "
              f"(reranker score: {round(float(score), 2)})")

    print("\n" + "="*55)
    print("   ✅ DAY 4 COMPLETE — Ready for Day 5")
    print("   Next: Ragas Evaluation")
    print("="*55 + "\n")



# ─────────────────────────────────────────────────────────────
# PHASE 5 — RAGAS EVALUATION
# ─────────────────────────────────────────────────────────────

def run_evaluation(
    vectorstore,
    chunks: list,
    llm_client,
    prompts: dict,
    reranker
) -> dict:
    """
    Step 10 — Load golden QA dataset
    Step 11 — Run pipeline on each question
    Step 12 — Compute Ragas metrics
    Step 13 — Print scorecard
    """

    print("\n" + "="*55)
    print("   RAG PIPELINE — PHASE 5: EVALUATION")
    print("="*55)

    # ── Step 10: Load golden dataset ──────────────────────────
    print("\n📋 Step 10: Loading golden dataset...")
    golden_dataset = load_golden_dataset()
    print(f"   ✅ {len(golden_dataset)} QA pairs loaded")

    # ── Build BM25 for evaluation retrieval ──────────────────
    bm25, _ = build_bm25_index(chunks)

    # ── Step 11: Build Ragas dataset ─────────────────────────
    print(f"\n🔄 Step 11: Running pipeline on {len(golden_dataset)} questions...")
    print("   (This takes 1-2 minutes)...")

    ragas_dataset = build_ragas_dataset(
        golden_dataset=golden_dataset,
        vectorstore=vectorstore,
        chunks=chunks,
        llm_client=llm_client,
        prompts=prompts,
        reranker=reranker,
        bm25=bm25
    )
    print(f"   ✅ Ragas dataset built")

    # ── Step 12: Run evaluation ───────────────────────────────
    print(f"\n📊 Step 12: Computing Ragas metrics...")
    scores = run_ragas_evaluation(ragas_dataset)
    print(f"   ✅ Evaluation complete")

    # ── Step 13: Print scorecard ──────────────────────────────
    print_scorecard(scores)

    return scores













# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Phase 1: Ingestion ────────────────────────────────────
    chunks = run_ingestion(data_dir="./data/papers")
    if not chunks:
        exit()
    show_ingestion_results(chunks)

    # ── Phase 2: Embeddings ───────────────────────────────────
    vectorstore, embeddings = run_embedding(
        chunks=chunks,
        force_recreate=False
    )
    show_embedding_results(vectorstore)

    # ── Initialize Langfuse ───────────────────────────────────
    print("\n📡 Initializing Langfuse observability...")
    langfuse = init_langfuse()
    if langfuse:
        print("   ✅ Langfuse ready — traces will appear in dashboard")
    else:
        print("   ⚠️  Langfuse disabled — add keys to .env to enable")

    # ── Phase 3: Retrieval ────────────────────────────────────
    print("\n🤖 Loading reranker model...")
    reranker = load_reranker()
    print("   ✅ Reranker ready")

    question = "What are the techniques to handle missing values?"

    reranked_chunks, retrieved_chunks, ret_latency, rerank_latency = run_retrieval(
        query=question,
        vectorstore=vectorstore,
        chunks=chunks,
        reranker=reranker
    )
    show_retrieval_results(question, reranked_chunks)

    # ── Phase 4: Generation ───────────────────────────────────
    print("\n📚 Loading prompts...")
    prompts = load_prompts()
    print("   ✅ Prompts loaded")

    print("\n🔌 Connecting to LLM...")
    llm_client = load_llm_client()

    result = run_generation(
        question=question,
        reranked_chunks=reranked_chunks,
        llm_client=llm_client,
        prompts=prompts
    )
    show_generation_results(question, result, reranked_chunks)

    # ── Send trace to Langfuse ────────────────────────────────
    print("\n📡 Sending trace to Langfuse...")
    trace_id = trace_rag_query(
        langfuse=langfuse,
        query=question,
        retrieved_chunks=retrieved_chunks,
        reranked_chunks=reranked_chunks,
        answer=result["answer"],
        generation_result=result,
        retrieval_latency_ms=ret_latency,
        reranking_latency_ms=rerank_latency
    )
    if trace_id:
        print(f"   ✅ Trace sent: {trace_id}")
    else:
        print("   ⚠️  Tracing skipped")

    # ── Phase 5: Evaluation ───────────────────────────────────
    run_eval = False

    if run_eval:
        scores = run_evaluation(
            vectorstore=vectorstore,
            chunks=chunks,
            llm_client=llm_client,
            prompts=prompts,
            reranker=reranker
        )
        log_ragas_scores(langfuse, scores)
    else:
        scores_path = "Evaluation/ragas_results.json"
        if os.path.exists(scores_path):
            with open(scores_path) as f:
                scores = json.load(f)
            print("\n📊 Saved Ragas scores:")
            print_scorecard(scores)
        else:
            print("\n⚠️  No saved scores. Set run_eval=True once.")

    # ── Final Langfuse flush ──────────────────────────────────
    if langfuse:
        langfuse.flush()
        print("\n✅ All traces flushed to Langfuse dashboard")