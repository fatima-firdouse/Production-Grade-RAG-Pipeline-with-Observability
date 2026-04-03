from __future__ import annotations
import os
import yaml
import logging
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    BSHTMLLoader,
    WebBaseLoader,
    WikipediaLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    JSONLoader,
)
from langchain.schema import Document

# ─────────────────────────────────────────────────────────────
# LOGGER SETUP
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# LOAD SETTINGS
# ─────────────────────────────────────────────────────────────
with open("config/settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

DATA_DIR = settings["ingestion"]["data_dir"]

# ─────────────────────────────────────────────────────────────
# EXTENSION → LOADER TYPE MAP
# ─────────────────────────────────────────────────────────────
EXTENSION_LOADER_MAP = {
    ".pdf"  : "pdf",
    ".txt"  : "txt",
    ".md"   : "markdown",
    ".csv"  : "csv",
    ".json" : "json",
    ".html" : "html",
    ".htm"  : "html",
    ".docx" : "docx",
    ".pptx" : "pptx",
}


# ─────────────────────────────────────────────────────────────
# PDF LOADER — using fitz directly (bypasses PyMuPDFLoader bug)
# ─────────────────────────────────────────────────────────────

def load_pdf_with_fitz(filepath: str) -> list[Document]:
    """
    Reads PDF using fitz (PyMuPDF) directly.
    Bypasses PyMuPDFLoader's Python 3.8 import issue.
    Returns one LangChain Document per page.
    """
    import fitz  # PyMuPDF

    documents = []
    doc = fitz.open(filepath)
    filename = os.path.basename(filepath)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Skip blank pages
        if not text.strip():
            continue

        documents.append(Document(
            page_content=text,
            metadata={
                "source": filepath,
                "page": page_num + 1,
                "total_pages": len(doc),
                "filename": filename
            }
        ))

    doc.close()
    return documents


# ─────────────────────────────────────────────────────────────
# BUILD LOADER
# ─────────────────────────────────────────────────────────────

def get_loader(filepath: str, loader_type: str):
    """
    Builds correct LangChain loader for each file type.
    PDF uses fitz directly — all others use LangChain loaders.
    """
    try:
        if loader_type == "pdf":
            # Return None — PDF handled separately by load_pdf_with_fitz()
            return None

        elif loader_type == "txt":
            return TextLoader(filepath, encoding="utf-8")

        elif loader_type == "markdown":
            return UnstructuredMarkdownLoader(filepath)

        elif loader_type == "csv":
            return CSVLoader(
                filepath,
                encoding="utf-8",
                csv_args={"delimiter": ","}
            )

        elif loader_type == "json":
            return JSONLoader(
                filepath,
                jq_schema=".[]",
                text_content=False
            )

        elif loader_type == "html":
            return BSHTMLLoader(filepath, open_encoding="utf-8")

        elif loader_type == "docx":
            return Docx2txtLoader(filepath)

        elif loader_type == "pptx":
            return UnstructuredPowerPointLoader(filepath)

        else:
            logger.warning(f"No loader found for type: {loader_type}")
            return None

    except Exception as e:
        logger.error(f"❌ Failed to build loader for {filepath}: {str(e)}")
        return None


# ─────────────────────────────────────────────────────────────
# SINGLE FILE LOADER
# ─────────────────────────────────────────────────────────────

def load_single_file(filepath: str) -> list:
    """
    Loads one file using correct loader for its type.
    PDFs use fitz directly.
    All other types use lazy_load() via LangChain.
    """
    ext = os.path.splitext(filepath)[1].lower()
    loader_type = EXTENSION_LOADER_MAP.get(ext)

    if loader_type is None:
        logger.warning(f"Skipping unsupported file: {filepath}")
        return []

    # ── PDF: use fitz directly ────────────────────────────────
    if loader_type == "pdf":
        try:
            documents = load_pdf_with_fitz(filepath)
            logger.info(
                f"✅ Loaded {len(documents)} page(s) ← "
                f"{os.path.basename(filepath)}"
            )
            return documents
        except Exception as e:
            logger.error(f"❌ Failed to load PDF {filepath}: {str(e)}")
            return []

    # ── All other types: use LangChain lazy_load() ────────────
    loader = get_loader(filepath, loader_type)
    if loader is None:
        return []

    documents = []
    try:
        for doc in loader.lazy_load():
            documents.append(doc)

        logger.info(
            f"✅ Loaded {len(documents)} doc(s) ← "
            f"{os.path.basename(filepath)}"
        )
        return documents

    except Exception as e:
        logger.error(f"❌ Failed to load {filepath}: {str(e)}")
        return []


# ─────────────────────────────────────────────────────────────
# WEB PAGE LOADER
# ─────────────────────────────────────────────────────────────

def load_from_urls(urls: list[str]) -> list:
    """
    Fetches live web pages from URLs.
    Strips HTML, returns clean text.
    One Document per URL.
    """
    if not urls:
        return []

    try:
        loader = WebBaseLoader(urls)
        documents = []
        for doc in loader.lazy_load():
            documents.append(doc)
        logger.info(f"✅ Loaded {len(documents)} web pages")
        return documents

    except Exception as e:
        logger.error(f"❌ Failed to load URLs: {str(e)}")
        return []


# ─────────────────────────────────────────────────────────────
# WIKIPEDIA LOADER
# ─────────────────────────────────────────────────────────────

def load_from_wikipedia(
    topics: list[str],
    lang: str = "en"
) -> list:
    """
    Loads Wikipedia articles by topic name.
    One Document per topic.
    """
    if not topics:
        return []

    all_docs = []
    for topic in topics:
        try:
            loader = WikipediaLoader(
                query=topic,
                lang=lang,
                load_max_docs=1,
                doc_content_chars_max=10000
            )
            for doc in loader.lazy_load():
                all_docs.append(doc)
            logger.info(f"✅ Loaded Wikipedia: {topic}")

        except Exception as e:
            logger.error(f"❌ Wikipedia failed for '{topic}': {e}")

    return all_docs


# ─────────────────────────────────────────────────────────────
# DIRECTORY SCANNER
# ─────────────────────────────────────────────────────────────

def load_entire_directory(data_dir: str = DATA_DIR) -> list:
    """
    Scans folder, loads ALL supported file types.
    """
    if not os.path.exists(data_dir):
        logger.error(f"Directory not found: {data_dir}")
        return []

    all_files = os.listdir(data_dir)
    supported_files = [
        f for f in all_files
        if os.path.splitext(f)[1].lower() in EXTENSION_LOADER_MAP
    ]

    if not supported_files:
        logger.warning(f"No supported files found in: {data_dir}")
        return []

    logger.info(f"Found {len(supported_files)} supported files in {data_dir}")

    all_documents = []
    for filename in supported_files:
        filepath = os.path.join(data_dir, filename)
        docs = load_single_file(filepath)
        all_documents.extend(docs)

    return all_documents


# ─────────────────────────────────────────────────────────────
# MASTER LOADER
# ─────────────────────────────────────────────────────────────

def load_documents(
    data_dir: str = DATA_DIR,
    urls: list[str] = [],
    wikipedia_topics: list[str] = []
) -> list:
    """
    Master function — combines all sources.
    """
    all_documents = []

    if data_dir:
        file_docs = load_entire_directory(data_dir)
        all_documents.extend(file_docs)
        logger.info(f"From local files : {len(file_docs)} documents")

    if urls:
        web_docs = load_from_urls(urls)
        all_documents.extend(web_docs)
        logger.info(f"From URLs        : {len(web_docs)} documents")

    if wikipedia_topics:
        wiki_docs = load_from_wikipedia(wikipedia_topics)
        all_documents.extend(wiki_docs)
        logger.info(f"From Wikipedia   : {len(wiki_docs)} documents")

    logger.info(f"✅ TOTAL documents loaded: {len(all_documents)}")
    return all_documents