try:
    import streamlit as st
    import os
    import tempfile
    import time

    from src.generation.graph import query
    from src.ingestion.loader import load_document
    from src.ingestion.chunker import create_chunks
    from src.ingestion.store import upsert_chunks, get_collection_info

    st.success("✅ All imports successful")

except Exception as e:
    st.error("❌ Import failure detected")
    st.exception(e)
    st.stop()


# Page config
st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide",
)

st.title("📚 DocuMind")
st.caption("Ask questions about your documents — powered by RAG")


# Sidebar: Document upload
with st.sidebar:
    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("🔄 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    try:
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=os.path.splitext(uploaded_file.name)[1]
                        ) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        # Load → Chunk → Store
                        st.write("🔵 Loading document...")
                        docs = load_document(tmp_path)

                        st.write("🟡 Chunking...")
                        chunks = create_chunks(docs)

                        st.write("🟢 Upserting into Qdrant...")
                        upsert_chunks(chunks)

                        # Cleanup
                        os.unlink(tmp_path)

                        st.success(
                            f"✅ {uploaded_file.name} — {len(chunks)} chunks indexed"
                        )

                    except Exception as e:
                        st.error("❌ Error during ingestion")
                        st.code(traceback.format_exc())


    # Collection info
    st.divider()
    try:
        info = get_collection_info()
        st.metric("Vectors in DB", info["points_count"])
    except Exception as e:
        st.warning("⚠️ Could not fetch collection info")
        st.code(traceback.format_exc())


# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 Sources"):
                for source in message["sources"]:
                    st.text(f"• {source}")
        if message.get("latency"):
            st.caption(f"⏱️ {message['latency']}")


# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                start = time.time()
                result = query(prompt)
                total_time = time.time() - start

            st.markdown(result["generation"])

            # Sources
            sources = []
            for doc in result.get("documents", []):
                src = doc["metadata"].get("source", "Unknown")
                page = doc["metadata"].get("page", "")
                score = doc.get("score", 0)

                source_str = f"{src}"
                if page:
                    source_str += f" (p.{page})"
                source_str += f" — relevance: {score:.2f}"

                sources.append(source_str)

            if sources:
                with st.expander("📎 Sources"):
                    for s in sources:
                        st.text(f"• {s}")

            # Latency info
            latency_str = f"Total: {total_time:.1f}s"
            if result.get("is_hallucination"):
                latency_str += " ⚠️ Hallucination detected — answer may be unreliable"
            if result.get("retry_count", 0) > 0:
                latency_str += f" (retried {result['retry_count']}x)"

            st.caption(f"⏱️ {latency_str}")

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.get("generation", "No response"),
                "sources": sources,
                "latency": latency_str,
            })

        except Exception as e:
            st.error("❌ Error during query execution")
            st.code(traceback.format_exc())