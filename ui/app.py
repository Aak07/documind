import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import requests
import tempfile
import time

# FastAPI backend URL
API_BASE = "http://127.0.0.1:8000"

def check_backend() -> bool:
    """Check if FastAPI backend is running."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def query_backend(question: str) -> dict:
    """Send query to FastAPI and return response."""
    response = requests.post(
        f"{API_BASE}/query",
        json={"question": question},
        timeout=60,
    )
    if response.status_code == 429:
        data = response.json()
        raise Exception(f"RATE_LIMIT:{data['detail']['message']}")
    response.raise_for_status()
    return response.json()

def ingest_file(file_bytes: bytes, filename: str) -> dict:
    """Upload file to FastAPI ingest endpoint."""
    response = requests.post(
        f"{API_BASE}/ingest",
        files={"file": (filename, file_bytes)},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()

# ---- Page Config & Custom CSS ----
st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner, production-like UI
st.markdown("""
<style>
    /* Hide Streamlit default menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Adjust top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Subtle styling for the metadata captions */
    .meta-text {
        color: #6B7280;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 DocuMind")
st.caption("Intelligent document Q&A — powered by RAG")

# ---- Backend Status ----
if not check_backend():
    st.error(
        "⚠️ **Backend not running.** Start FastAPI first:\n\n"
        "`uvicorn src.api.main:app --reload --port 8000`"
    )
    st.stop()

# ---- Session State Init ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Sidebar ----
with st.sidebar:
    st.header("📄 Document Management")
    
    uploaded_files = st.file_uploader(
        "Upload PDFs or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload your documents to make them available for querying."
    )
    
    if uploaded_files:
        if st.button("🔄 Process Documents", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        result = ingest_file(
                            uploaded_file.getvalue(),
                            uploaded_file.name,
                        )
                        # Use toast for non-blocking notifications
                        st.toast(f"✅ {uploaded_file.name} — {result.get('chunks', '?')} chunks indexed")
                    except Exception as e:
                        st.error(f"❌ Error with {uploaded_file.name}: {e}")
                progress_bar.progress((i + 1) / len(uploaded_files))
            st.success("All documents processed successfully!")
            time.sleep(2)
            st.rerun()

    st.divider()
    
    # Collection stats from backend (using columns for better layout)
    st.subheader("📊 Database Stats")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vectors", health.get("collection_vectors", 0))
        with col2:
            st.metric("Status", str(health.get("status", "unknown")).capitalize())
    except Exception:
        st.info("Could not fetch collection stats")

    st.divider()
    
    # Clear chat functionality
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---- Chat Interface ----

# Empty State
if not st.session_state.messages:
    st.info("👋 **Welcome to DocuMind!**\n\nTo get started:\n1. Upload your documents in the sidebar.\n2. Click 'Process Documents'.\n3. Start asking questions below!")

# Render existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 View Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")
        if message.get("meta"):
            st.markdown(f"<div class='meta-text'>{message['meta']}</div>", unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Append user prompt and show it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            try:
                result = query_backend(prompt)
                st.markdown(result["answer"])
                
                # Format Sources
                sources = []
                for doc in result.get("sources", []):
                    src = doc.get("source", "Unknown")
                    page = doc.get("page", "")
                    score = doc.get("score", 0)
                    line = f"**{src}**"
                    if page:
                        line += f" (Page {page})"
                    line += f" 🔹 *Relevance: {score:.2f}*"
                    sources.append(line)
                
                if sources:
                    with st.expander("📎 View Sources"):
                        for s in sources:
                            st.markdown(f"- {s}")
                
                # Format Metadata nicely
                latency = result.get("latency_ms", {})
                total_ms = sum(latency.values())
                cost = result.get("cost_usd", 0)
                hallucination = result.get("is_hallucination", False)
                retries = result.get("retry_count", 0)
                
                meta_parts = [
                    f"⏱️ **{total_ms:.0f}ms**",
                    f"💰 **${cost:.5f}**"
                ]
                if hallucination:
                    meta_parts.append("⚠️ **Low confidence**")
                if retries > 0:
                    meta_parts.append(f"🔁 **{retries} retries**")
                
                meta_string = " &nbsp;|&nbsp; ".join(meta_parts)
                st.markdown(f"<div class='meta-text'>{meta_string}</div>", unsafe_allow_html=True)
                
                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": sources,
                    "meta": meta_string,
                })
                
            except Exception as e:
                error_str = str(e)
                if error_str.startswith("RATE_LIMIT:"):
                    st.warning(f"⚠️ **Rate Limit Reached:** {error_str.replace('RATE_LIMIT:', '')}")
                else:
                    st.error(f"❌ **Something went wrong:** {error_str}")