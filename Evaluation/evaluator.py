from __future__ import annotations
import asyncio
import nest_asyncio
nest_asyncio.apply()  # fixes Python 3.8 event loop conflict with Ragas
import os
import json
import time
import yaml
import logging
import warnings
warnings.filterwarnings("ignore")

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ─────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# LOAD SETTINGS
# ─────────────────────────────────────────────────────────────
with open("config/settings.yaml", "r") as f:
    settings = yaml.safe_load(f)




def get_ragas_llm_and_embeddings():
    """
    Configures Ragas to use Groq + BGE embeddings
    instead of OpenAI.
    """
    # LLM for Ragas evaluation
    groq_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Embeddings for answer relevancy metric
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return (
        LangchainLLMWrapper(groq_llm),
        LangchainEmbeddingsWrapper(hf_embeddings)
    )



# ─────────────────────────────────────────────────────────────
# LOAD GOLDEN DATASET
# ─────────────────────────────────────────────────────────────

def load_golden_dataset(
    path: str = "Evaluation/golden_dataset.json"
) -> list[dict]:
    """
    Loads golden QA pairs from JSON file.
    Each pair has: question + ground_truth answer.
    Returns list of dicts.
    """
    with open(path, "r") as f:
        dataset = json.load(f)

    logger.info(f"✅ Loaded {len(dataset)} golden QA pairs")
    return dataset


# ─────────────────────────────────────────────────────────────
# BUILD RAGAS DATASET
# ─────────────────────────────────────────────────────────────

def build_ragas_dataset(
    golden_dataset: list[dict],
    vectorstore,
    chunks: list[Document],
    llm_client,
    prompts: dict,
    reranker,
    bm25,
) -> Dataset:
    """
    Runs the full RAG pipeline on each golden question.
    Collects: question, answer, contexts, ground_truth.
    Returns HuggingFace Dataset ready for Ragas evaluation.
    """
    from Retrieval.hybrid_retriever import hybrid_retrieve
    from Retrieval.reranker import rerank_chunks
    from Generation.prompt_manager import build_prompt
    from Generation.generator import generate_answer

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    total = len(golden_dataset)

    for i, item in enumerate(golden_dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info(f"Evaluating {i+1}/{total}: {question[:50]}...")

        try:
            # ── Retrieve relevant chunks ──────────────────────
            retrieved = hybrid_retrieve(
                query=question,
                vectorstore=vectorstore,
                bm25=bm25,
                chunks=chunks,
                top_k=10
            )

            # ── Rerank ────────────────────────────────────────
            reranked = rerank_chunks(
                query=question,
                chunks=retrieved,
                reranker=reranker,
                top_k=5
            )

            # ── Extract context strings ───────────────────────
            # Ragas needs contexts as list of strings
            context_strings = [
                doc.page_content
                for doc, score in reranked
            ]

            # ── Generate answer ───────────────────────────────
            prompt = build_prompt(
                question=question,
                chunks=reranked,
                prompts=prompts
            )

            result = generate_answer(
                prompt=prompt,
                client=llm_client,
                prompts=prompts
            )

            answer = result["answer"]

            # Collect for dataset
            questions.append(question)
            answers.append(answer)
            contexts.append(context_strings)
            ground_truths.append(ground_truth)

            logger.info(f"   ✅ Answer: {answer[:80]}...")

            # Small delay to avoid rate limiting on Groq
            time.sleep(1)

        except Exception as e:
            logger.error(f"❌ Failed on question {i+1}: {str(e)}")
            # Add placeholder so dataset stays aligned
            questions.append(question)
            answers.append("Error generating answer")
            contexts.append([""])
            ground_truths.append(ground_truth)

    # Build HuggingFace Dataset — required format for Ragas
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    logger.info(f"✅ Built Ragas dataset with {len(questions)} samples")
    return Dataset.from_dict(dataset_dict)


# ─────────────────────────────────────────────────────────────
# RUN RAGAS EVALUATION
# ─────────────────────────────────────────────────────────────
def run_ragas_evaluation(
    dataset: Dataset,
    save_path: str = "Evaluation/ragas_results.json"
) -> dict:
    logger.info("Running Ragas evaluation...")
    ragas_llm, ragas_embeddings = get_ragas_llm_and_embeddings()



    os.environ["RAGAS_MAX_WORKERS"] = "1"  # one call at a time → no rate limit


    start_time = time.time()
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False
    )
    elapsed = round(time.time() - start_time, 2)
    logger.info(f"✅ Ragas evaluation complete in {elapsed}s")
    
    
    
    # Only compute mean on numeric columns (the metric scores)
    df = result.to_pandas()
    numeric_cols = df.select_dtypes(include='number').columns
    scores = df[numeric_cols].mean().to_dict()

    with open(save_path, "w") as f:
        json.dump(scores, f, indent=2)
    logger.info(f"✅ Results saved to {save_path}")
    return scores


# ─────────────────────────────────────────────────────────────
# PRINT SCORECARD
# ─────────────────────────────────────────────────────────────
def print_scorecard(scores: dict) -> None:
    """
    Prints formatted evaluation scorecard.
    Handles NaN values gracefully.
    """
    import math

    print("\n" + "="*55)
    print("   RAGAS EVALUATION SCORECARD")
    print("="*55)

    metrics = {
        "faithfulness": "Are answers grounded in chunks?",
        "answer_relevancy": "Do answers address the question?",
        "context_precision": "Were retrieved chunks relevant?",
        "context_recall": "Did chunks cover the answer?"
    }

    valid_scores = {}

    for metric, description in metrics.items():
        score = scores.get(metric, None)

        # Handle NaN or None
        if score is None or (isinstance(score, float) and math.isnan(score)):
            print(f"\n{metric.upper()}")
            print(f"  {description}")
            print(f"  [--------------------] N/A  ⚠️  Timed out")
            continue

        score_pct = round(score * 100, 1)
        filled = int(score * 20)
        bar = "█" * filled + "░" * (20 - filled)

        if score >= 0.8:
            rating = "✅ Excellent"
        elif score >= 0.6:
            rating = "⚠️  Good"
        else:
            rating = "❌ Needs work"

        print(f"\n{metric.upper()}")
        print(f"  {description}")
        print(f"  [{bar}] {score_pct}%  {rating}")

        valid_scores[metric] = score

    print("\n" + "─"*55)

    if valid_scores:
        avg = sum(valid_scores.values()) / len(valid_scores)
        print(f"  OVERALL SCORE: {round(avg * 100, 1)}%")
    else:
        print(f"  OVERALL SCORE: N/A (all metrics timed out)")

    print("─"*55)

    print("\n📋 Resume Metrics:")
    for metric, score in scores.items():
        if score is None or (isinstance(score, float) and math.isnan(score)):
            print(f"   {metric}: N/A")
        else:
            print(f"   {metric}: {round(score, 2)}")

    print("\n" + "="*55)
    print("   ✅ DAY 5 COMPLETE — Ready for Day 6")
    print("   Next: Langfuse Observability")
    print("="*55 + "\n")
