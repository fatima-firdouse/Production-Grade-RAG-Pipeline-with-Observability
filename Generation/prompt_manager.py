from __future__ import annotations
import os
import yaml
import logging
import warnings
warnings.filterwarnings("ignore")

from langchain.schema import Document

# ─────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# LOAD PROMPTS
# ─────────────────────────────────────────────────────────────

def load_prompts(prompt_file: str = "config/prompts.yaml") -> dict:
    """
    Loads prompt templates from YAML file.
    Returns dict with all prompt configs.
    """
    with open(prompt_file, "r") as f:
        prompts = yaml.safe_load(f)

    version = prompts["rag_prompt"]["version"]
    logger.info(f"✅ Loaded prompts — version: {version}")
    return prompts


# ─────────────────────────────────────────────────────────────
# FORMAT CONTEXT
# ─────────────────────────────────────────────────────────────

def format_context(
    chunks: list[tuple[Document, float]],
    prompts: dict
) -> str:
    """
    Formats retrieved chunks into a context string for the LLM.

    Each chunk gets formatted as:
    [Chunk 1 | Source: NLP.docx | Page: 3]
    chunk text here...
    ---

    This structured format helps the LLM:
    1. Know where each piece of information came from
    2. Generate accurate citations in its answer
    """

    chunk_template = prompts["chunk_template"]
    context_parts = []

    for i, (doc, score) in enumerate(chunks):
        # Extract source filename — strip full path
        # "./data/papers\NLP.docx" → "NLP.docx"
        source = os.path.basename(
            doc.metadata.get("source", "unknown")
        )

        # Get page number — N/A for docx files
        page = doc.metadata.get("page", "N/A")

        # Format this chunk using template from YAML
        formatted_chunk = chunk_template.format(
            index=i + 1,
            source=source,
            page=page,
            text=doc.page_content.strip()
        )
        context_parts.append(formatted_chunk)

    # Join all formatted chunks into one context string
    context = "\n".join(context_parts)
    logger.info(f"Formatted {len(chunks)} chunks into context")
    return context


# ─────────────────────────────────────────────────────────────
# BUILD PROMPT
# ─────────────────────────────────────────────────────────────
def build_prompt(
    question: str,
    chunks: list,
    prompts: dict,
    chat_history: list = []
) -> str:
    """
    Builds the final prompt with context, question and chat history.
    """
    context = format_context(chunks, prompts)

    # Format chat history for the prompt
    history_text = ""
    if chat_history:
        for msg in chat_history[-6:]:  # last 6 messages only
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
    else:
        history_text = "No previous conversation."

    prompt = prompts["rag_prompt"]["template"].format(
        context=context,
        question=question,
        chat_history=history_text
    )

    logger.info(f"Built prompt — version: {prompts['rag_prompt']['version']}")
    return prompt