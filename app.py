from __future__ import annotations
import os
import tempfile
import atexit
import shutil
import re
import streamlit as st
from dotenv import load_dotenv
import warnings
import markdown  

warnings.filterwarnings("ignore")
load_dotenv()

# ────────────────────────────────────────────────────────────────
# Session & file management
# ────────────────────────────────────────────────────────────────

def get_session_dir():
    if "session_dir" not in st.session_state:
        try:
            temp_base = tempfile.gettempdir()
            for folder in os.listdir(temp_base):
                if folder.startswith("rag_session_"):
                    shutil.rmtree(os.path.join(temp_base, folder), ignore_errors=True)
        except:
            pass
        session_dir = tempfile.mkdtemp(prefix="rag_session_")
        st.session_state.session_dir = session_dir
        def cleanup():
            try:
                shutil.rmtree(session_dir, ignore_errors=True)
            except:
                pass
        atexit.register(cleanup)
    return st.session_state.session_dir

SESSION_DIR = get_session_dir()
DATA_DIR = os.path.join(SESSION_DIR, "data", "papers")
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="DocMind", page_icon="🤖", layout="wide")

# ────────────────────────────────────────────────────────────────
# CSS (bubble styles)
# ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
body { background: linear-gradient(135deg, #0f172a 0%, #1a1f35 100%); font-family: 'Segoe UI', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.95) 100%); border-right: 2px solid rgba(59,130,246,0.3); }
.chat-user { display: flex; justify-content: flex-end; margin-bottom: 12px; }
.chat-user-bubble { background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%); color: white; border-radius: 18px 18px 4px 18px; padding: 12px 18px; max-width: 70%; box-shadow: 0 8px 32px rgba(59,130,246,0.4); }
.chat-ai { display: flex; flex-direction: column; align-items: flex-start; margin-bottom: 12px; }
.chat-ai-bubble { background: rgba(255,255,255,0.08); border: 1px solid rgba(59,130,246,0.3); border-radius: 18px 18px 18px 4px; padding: 12px 18px; max-width: 70%; }
/* Style for Markdown content inside bubbles */
.chat-ai-bubble strong, .chat-ai-bubble b { color: #60a5fa; }
.chat-ai-bubble code { background: rgba(0,0,0,0.3); padding: 2px 4px; border-radius: 4px; }
.chat-ai-bubble pre { background: rgba(0,0,0,0.4); padding: 8px; border-radius: 8px; overflow-x: auto; }
.resources-bar { background: rgba(59,130,246,0.1); border-left: 4px solid #3b82f6; padding: 8px 12px; border-radius: 8px; margin-top: 6px; font-size: 0.85rem; max-width: 70%; }
.header-title { background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# Helper: Convert Markdown to HTML for safe embedding
# ────────────────────────────────────────────────────────────────
def markdown_to_html(text: str) -> str:
    """Convert Markdown text to HTML, with code block support."""
    return markdown.markdown(text, extensions=['fenced_code', 'tables'])

# ────────────────────────────────────────────────────────────────
# Imports (your existing pipeline)
# ────────────────────────────────────────────────────────────────
try:
    from Ingestion.loader import load_documents
    from Ingestion.chunker import chunk_documents
    from Retrieval.embedder import load_embedding_model
    from Retrieval.vectorstore import get_or_create_vectorstore
    from Retrieval.hybrid_retriever import build_bm25_index, hybrid_retrieve
    from Retrieval.reranker import load_reranker, rerank_chunks
    from Generation.prompt_manager import load_prompts, build_prompt
    from Generation.generator import load_llm_client, generate_answer
    from Observability.langfuse_tracker import init_langfuse, trace_rag_query
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.stop()

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
def extract_citations_and_clean(answer_text: str) -> tuple[str, list[str]]:
    pattern = r'\[CITATIONS\](.*?)\[/CITATIONS\]'
    match = re.search(pattern, answer_text, re.DOTALL | re.IGNORECASE)
    if not match:
        return answer_text, []
    citations_block = match.group(1).strip()
    cleaned = re.sub(pattern, '', answer_text, flags=re.DOTALL | re.IGNORECASE).strip()
    citations = []
    for line in citations_block.splitlines():
        line = line.strip()
        if line.startswith('- Source:'):
            citations.append(line[9:].strip())
    return cleaned, citations

def process_uploaded_files(uploaded_files):
    if not uploaded_files:
        return False
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    count = 0
    for file in uploaded_files:
        if file.name not in [f["name"] for f in st.session_state.uploaded_files]:
            save_path = os.path.join(DATA_DIR, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
            st.session_state.uploaded_files.append({"name": file.name, "path": save_path, "size": file.size})
            count += 1
    if count > 0:
        st.session_state.files_updated = True
        st.session_state.uploader_key = str(os.urandom(16))
    return count > 0

def remove_file(filename):
    for f in st.session_state.uploaded_files[:]:
        if f["name"] == filename:
            if os.path.exists(f["path"]):
                os.remove(f["path"])
            st.session_state.uploaded_files.remove(f)
            st.session_state.files_updated = True
            st.rerun()
            break

def is_greeting(question):
    greetings = ["hi", "hello", "hey", "how are you", "what's up", "greetings", "yo", "sup"]
    q = question.lower().strip()
    return q in greetings or any(q.startswith(g) for g in greetings)

def get_greeting_response():
    import random
    return random.choice([
        "Hey there! Upload documents and ask me anything!",
        "Hello! I'm your RAG Assistant. Ready to answer from your files.",
        "Hi! Upload some documents and let's get started."
    ])

# ────────────────────────────────────────────────────────────────
# Pipeline (cached)
# ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚡ Loading...")
def load_pipeline(_force=False):
    documents = load_documents(data_dir=DATA_DIR)
    chunks = chunk_documents(documents)
    if not chunks:
        return {"chunks": [], "vectorstore": None, "bm25": None, "reranker": load_reranker(),
                "llm_client": load_llm_client(), "prompts": load_prompts(), "langfuse": init_langfuse(),
                "sources": [], "chunk_count": 0}
    embeddings = load_embedding_model()
    vectorstore = get_or_create_vectorstore(chunks, embeddings, False)
    bm25, _ = build_bm25_index(chunks)
    return {"chunks": chunks, "vectorstore": vectorstore, "bm25": bm25,
            "reranker": load_reranker(), "llm_client": load_llm_client(),
            "prompts": load_prompts(), "langfuse": init_langfuse(),
            "sources": list(set([os.path.basename(c.metadata.get("source", "")) for c in chunks])),
            "chunk_count": len(chunks)}

def get_pipeline():
    if "files_updated" not in st.session_state:
        st.session_state.files_updated = False
    if st.session_state.files_updated:
        st.cache_resource.clear()
        st.session_state.files_updated = False
        return load_pipeline(_force=True)
    return load_pipeline()

def get_trimmed_history(messages: list, max_messages: int = 6):
    return [m for m in messages if m["role"] in ("user", "assistant")][-max_messages:]

# ────────────────────────────────────────────────────────────────
# Query handler
# ────────────────────────────────────────────────────────────────
def run_query(question: str, pipeline: dict, chat_history: list) -> dict:
    if is_greeting(question):
        return {"cleaned_answer": get_greeting_response(), "citations": []}
    if pipeline["chunk_count"] == 0:
        return {"cleaned_answer": "📭 No documents uploaded. Upload files using the sidebar.", "citations": []}
    
    retrieved = hybrid_retrieve(question, pipeline["vectorstore"], pipeline["bm25"], pipeline["chunks"], top_k=10)
    reranked = rerank_chunks(question, retrieved, pipeline["reranker"], top_k=5)
    prompt = build_prompt(question, reranked, pipeline["prompts"], get_trimmed_history(chat_history))
    result = generate_answer(prompt, pipeline["llm_client"], pipeline["prompts"])
    
    cleaned, citations = extract_citations_and_clean(result["answer"])
    if not cleaned:
        cleaned = "I cannot find relevant information in the provided documents."
        citations = []
    
    trace_rag_query(pipeline["langfuse"], question, retrieved, reranked, result["answer"], result, 0, 0)
    return {"cleaned_answer": cleaned, "citations": citations}

# ────────────────────────────────────────────────────────────────
# Sidebar (no duplication)
# ────────────────────────────────────────────────────────────────
def sidebar(pipeline: dict):
    with st.sidebar:
        st.markdown("### DocMind")
        st.caption("Production-Grade RAG Pipeline with Observability")
        st.divider()
        
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = "initial"
        uploaded = st.file_uploader(
            "Upload Documents",
            type=["pdf", "docx", "txt", "md", "csv"],
            accept_multiple_files=True,
            key=st.session_state.uploader_key
        )
        if uploaded:
            process_uploaded_files(uploaded)
            st.rerun()
        
        st.divider()
        st.markdown("**📁 Files**")
        if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
            for f in st.session_state.uploaded_files:
                col1, col2 = st.columns([0.75, 0.25])
                with col1:
                    st.caption(f"📄 {f['name']}")
                with col2:
                    if st.button("✖", key=f"del_{f['name']}", help="Delete"):
                        remove_file(f["name"])
        else:
            st.info("No files uploaded")
        
        st.divider()
        if pipeline["chunk_count"] == 0:
            st.warning("📭 Upload files to start")
        else:
            st.success(f"✅ {pipeline['chunk_count']} chunks ready")

# ────────────────────────────────────────────────────────────────
# Main app (with HTML bubble + Markdown conversion)
# ────────────────────────────────────────────────────────────────
def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []      # each: {"role": "assistant", "content": str, "citations": list}
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "files_updated" not in st.session_state:
        st.session_state.files_updated = False
    
    pipeline = get_pipeline()
    sidebar(pipeline)
    
    st.markdown("<h1 class='header-title'>RAG Assistant</h1>", unsafe_allow_html=True)
    st.caption("Ask anything about your documents • Answers are cited & traced")
    st.divider()
    
    # Display chat history with bubbles and resources bar
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user"><div class="chat-user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            # Convert assistant's Markdown to HTML and embed inside bubble
            html_content = markdown_to_html(msg["content"])
            st.markdown(f'<div class="chat-ai-bubble">{html_content}</div>', unsafe_allow_html=True)
            # Resources bar (citations)
            if msg.get("citations"):
                citations_html = "<br>".join(f"• {c}" for c in msg["citations"])
                st.markdown(f'<div class="resources-bar">📚 <strong>Sources</strong><br>{citations_html}</div>', unsafe_allow_html=True)
    
    st.divider()
    
    question = st.chat_input("Ask a question...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.markdown(f'<div class="chat-user"><div class="chat-user-bubble">{question}</div></div>', unsafe_allow_html=True)
        
        with st.spinner("💭 Thinking..."):
            response = run_query(question, pipeline, st.session_state.messages[:-1])
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["cleaned_answer"],
            "citations": response["citations"]
        })
        
        # Display new assistant response
        html_content = markdown_to_html(response["cleaned_answer"])
        st.markdown(f'<div class="chat-ai-bubble">{html_content}</div>', unsafe_allow_html=True)
        if response["citations"]:
            citations_html = "<br>".join(f"• {c}" for c in response["citations"])
            st.markdown(f'<div class="resources-bar">📚 <strong>Sources</strong><br>{citations_html}</div>', unsafe_allow_html=True)
        
        st.rerun()

if __name__ == "__main__":
    main()