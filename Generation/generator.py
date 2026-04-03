from __future__ import annotations
import os
import time
import yaml
import logging
import warnings
warnings.filterwarnings("ignore")

from groq import Groq

# ─────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# LOAD SETTINGS
# ─────────────────────────────────────────────────────────────
with open("config/settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

MODEL = settings["generation"]["model"]
# → "llama3-8b-8192"

MAX_NEW_TOKENS = settings["generation"]["max_new_tokens"]
# → 512

TEMPERATURE = settings["generation"]["temperature"]
# → 0.2


# ─────────────────────────────────────────────────────────────
# LOAD LLM CLIENT
# ─────────────────────────────────────────────────────────────

def load_llm_client() -> Groq:
    """
    Creates Groq API client.

    Groq is a free LLM inference platform.
    It runs Llama 3, Mistral, and Gemma on their hardware.
    Your code sends HTTP requests — no model downloads needed.

    Llama 3 8B on Groq:
    - Free tier: 30 requests/minute, 14,400/day
    - Speed: ~500 tokens/second (much faster than HF)
    - Quality: Better than Zephyr-7B for instruction following
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in .env file. "
            "Get it free from console.groq.com"
        )

    # Groq() creates the client — no network call yet
    # Actual calls happen in generate_answer()
    client = Groq(api_key=api_key)

    logger.info(f"✅ Groq LLM client ready")
    logger.info(f"   Model    : {MODEL}")
    logger.info(f"   Provider : Groq (free tier)")
    logger.info(f"   Tokens   : {MAX_NEW_TOKENS}")
    logger.info(f"   Temp     : {TEMPERATURE}")

    return client


# ─────────────────────────────────────────────────────────────
# GENERATE ANSWER
# ─────────────────────────────────────────────────────────────

def generate_answer(
    prompt: str,
    client: Groq,
    prompts: dict
) -> dict:
    """
    Sends prompt to Llama 3 via Groq API.

    Groq uses OpenAI-compatible chat format:
    messages = [{"role": "user", "content": prompt}]

    This is different from HuggingFace's text_generation format.
    Groq returns: response.choices[0].message.content

    Returns dict with answer, model, latency_ms, prompt_len
    """

    decline_message = prompts["decline_message"]

    logger.info(f"Calling {MODEL} via Groq...")
    start_time = time.time()

    try:
        # chat.completions.create() sends prompt to Groq
        # Uses OpenAI-compatible format — industry standard
        # messages list allows multi-turn conversations
        # We use single-turn: just the full RAG prompt
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )

        elapsed_ms = round((time.time() - start_time) * 1000)

        # Extract answer from response
        # response.choices[0].message.content → the generated text
        answer = response.choices[0].message.content.strip()

        if not answer:
            answer = decline_message

        logger.info(f"✅ Generation complete in {elapsed_ms}ms")
        logger.info(f"   Answer length : {len(answer)} chars")
        logger.info(f"   Tokens used   : {response.usage.total_tokens}")

        return {
            "answer": answer,
            "model": MODEL,
            "latency_ms": elapsed_ms,
            "prompt_len": len(prompt),
            "prompt_version": prompts["rag_prompt"]["version"],
            "tokens_used": response.usage.total_tokens
        }

    except Exception as e:
        elapsed_ms = round((time.time() - start_time) * 1000)
        logger.error(f"❌ Generation failed: {str(e)}")
        return {
            "answer": decline_message,
            "model": MODEL,
            "latency_ms": elapsed_ms,
            "prompt_len": len(prompt),
            "error": str(e)
        }



## Why Groq + Llama 3 Is Better Than HF Free Tier
# ```
# HuggingFace free tier:
# → Limited models available
# → Slow (shared CPU servers)
# → Frequent timeouts
# → 404 errors on good models

# Groq free tier:
# → Llama 3 8B + 70B available
# → ~500 tokens/second (extremely fast)
# → Reliable
# → OpenAI-compatible API
# → 14,400 requests/day free
