from __future__ import annotations
import os
import time
import datetime
import logging
import warnings
warnings.filterwarnings("ignore")

from langfuse import Langfuse

# ─────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# INITIALIZE LANGFUSE CLIENT
# ─────────────────────────────────────────────────────────────

def init_langfuse() -> Langfuse | None:
    """
    Initializes Langfuse client using keys from .env.
    Returns None if keys not configured — pipeline still works.
    Observability should NEVER break the main pipeline.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.warning("Langfuse keys not found — tracing disabled")
        return None

    try:
        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        logger.info("✅ Langfuse initialized")
        logger.info(f"   Host: {host}")
        return client

    except Exception as e:
        logger.error(f"❌ Langfuse init failed: {str(e)}")
        return None


# ─────────────────────────────────────────────────────────────
# TRACE A FULL RAG QUERY
# ─────────────────────────────────────────────────────────────

def trace_rag_query(
    langfuse: Langfuse | None,
    query: str,
    retrieved_chunks: list,
    reranked_chunks: list,
    answer: str,
    generation_result: dict,
    retrieval_latency_ms: float,
    reranking_latency_ms: float,
) -> str | None:
    """
    Creates a full trace in Langfuse for one RAG query.

    Structure:
    Trace: rag-query (full duration)
    ├── Span 1: hybrid-retrieval
    ├── Span 2: cross-encoder-reranking
    └── Span 3: llm-generation

    Returns trace_id for linking scores later.
    """

    if langfuse is None:
        return None

    try:
        total_latency = (
            retrieval_latency_ms +
            reranking_latency_ms +
            generation_result.get("latency_ms", 0)
        )

        # Calculate real start/end times
        # Langfuse uses these to show correct latency on timeline
        end_time = datetime.datetime.utcnow()
        start_time = end_time - datetime.timedelta(
            milliseconds=total_latency
        )

        # ── Root trace ────────────────────────────────────────
        # One trace = one complete user query
        trace = langfuse.trace(
            name="rag-query",
            input={"question": query},
            output={"answer": answer},
            start_time=start_time,
            end_time=end_time,
            metadata={
                "chunks_retrieved": len(retrieved_chunks),
                "chunks_reranked": len(reranked_chunks),
                "model": generation_result.get("model", "unknown"),
                "prompt_version": generation_result.get("prompt_version", "v1"),
                "total_latency_ms": total_latency,
                "tokens_used": generation_result.get("tokens_used", 0)
            }
        )

        # ── Span 1: Hybrid Retrieval ──────────────────────────
        # Starts at trace start, ends after retrieval
        ret_start = start_time
        ret_end = start_time + datetime.timedelta(
            milliseconds=retrieval_latency_ms
        )

        retrieval_span = trace.span(
            name="hybrid-retrieval",
            start_time=ret_start,
            end_time=ret_end,
            input={"query": query},
            output={
                "chunks_retrieved": len(retrieved_chunks),
                "sources": list(set([
                    os.path.basename(
                        c.metadata.get("source", "unknown")
                    )
                    for c in retrieved_chunks
                ]))
            },
            metadata={
                "retrieval_type": "BM25 + Vector + RRF",
                "latency_ms": retrieval_latency_ms
            }
        )
        retrieval_span.end()

        # ── Span 2: Reranking ─────────────────────────────────
        # Starts where retrieval ends
        rerank_start = ret_end
        rerank_end = rerank_start + datetime.timedelta(
            milliseconds=reranking_latency_ms
        )

        rerank_results = [
            {
                "source": os.path.basename(
                    doc.metadata.get("source", "unknown")
                ),
                "page": doc.metadata.get("page", "N/A"),
                "score": round(float(score), 4)
            }
            for doc, score in reranked_chunks
        ]

        reranking_span = trace.span(
            name="cross-encoder-reranking",
            start_time=rerank_start,
            end_time=rerank_end,
            input={
                "query": query,
                "candidates": len(retrieved_chunks)
            },
            output={"top_chunks": rerank_results},
            metadata={
                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "latency_ms": reranking_latency_ms
            }
        )
        reranking_span.end()

        # ── Span 3: LLM Generation ────────────────────────────
        # Starts where reranking ends, finishes at trace end
        gen_start = rerank_end
        gen_end = end_time

        generation_span = trace.span(
            name="llm-generation",
            start_time=gen_start,
            end_time=gen_end,
            input={
                "prompt_length": generation_result.get("prompt_len", 0)
            },
            output={"answer": answer},
            metadata={
                "model": generation_result.get("model", "unknown"),
                "latency_ms": generation_result.get("latency_ms", 0),
                "tokens_used": generation_result.get("tokens_used", 0),
                "prompt_version": generation_result.get("prompt_version", "v1")
            }
        )
        generation_span.end()

        # Send everything to Langfuse server
        langfuse.flush()

        logger.info(f"✅ Trace sent to Langfuse: {trace.id}")
        return trace.id

    except Exception as e:
        # NEVER crash pipeline because of observability
        logger.error(f"❌ Langfuse trace failed: {str(e)}")
        return None


# ─────────────────────────────────────────────────────────────
# LOG RAGAS SCORES TO LANGFUSE
# ─────────────────────────────────────────────────────────────

def log_ragas_scores(
    langfuse: Langfuse | None,
    scores: dict,
    trace_id: str | None = None
) -> None:
    """
    Logs Ragas evaluation scores to Langfuse.
    Creates a new trace specifically for evaluation metrics.
    Visible in dashboard under Tracing as 'ragas-evaluation'.
    """

    if langfuse is None:
        return

    try:
        import math

        # Create a dedicated trace for evaluation scores
        # This keeps eval separate from query traces
        eval_trace = langfuse.trace(
            name="ragas-evaluation",
            input={"type": "batch_evaluation"},
            output=scores,
            metadata={"evaluation_type": "ragas"}
        )

        # Log each metric as a score on this trace
        for metric, score in scores.items():
            if score is None:
                continue
            try:
                score_float = float(score)
                if not math.isnan(score_float):
                    # score() attaches a numeric metric to a trace
                    langfuse.score(
                        trace_id=eval_trace.id,
                        name=f"ragas_{metric}",
                        value=score_float,
                        comment=f"Ragas metric: {metric}"
                    )
                    logger.info(f"   Logged {metric}: {score_float:.3f}")
            except (ValueError, TypeError):
                logger.warning(f"   Skipped {metric}: not numeric")

        langfuse.flush()
        logger.info("✅ Ragas scores logged to Langfuse")

    except Exception as e:
        logger.error(f"❌ Failed to log scores: {str(e)}")
