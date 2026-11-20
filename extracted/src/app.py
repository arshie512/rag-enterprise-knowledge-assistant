"""Streamlit chat interface for the enterprise knowledge assistant."""

from __future__ import annotations

import streamlit as st

from src.rag_core import load_index, make_chat_engine


st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="📚")
st.title("Enterprise Knowledge Assistant")
st.caption("Ask questions about the indexed documents. Check the source passages under each answer.")

with st.sidebar:
    st.subheader("Conversation")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.pop("chat_engine", None)
        st.session_state.messages = []
        st.rerun()
    st.caption("Answers are grounded in the indexed files. The app may say when it cannot find evidence.")

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = make_chat_engine(load_index())
except Exception as exc:
    st.error(str(exc))
    st.info("Add approved source files, build the index with `python -m src.ingest`, and refresh this page.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved sources"):
                for source in message["sources"]:
                    st.markdown(source)

question = st.chat_input("Ask a question about the documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and preparing an answer..."):
            response = st.session_state.chat_engine.chat(question)
        answer = str(response)
        st.markdown(answer)

        sources: list[str] = []
        seen: set[tuple[str, str]] = set()
        for item in getattr(response, "source_nodes", []) or []:
            node = item.node
            metadata = node.metadata or {}
            source = metadata.get("source") or metadata.get("file_name") or "Unknown source"
            page = metadata.get("page_label") or metadata.get("page_number")
            key = (str(source), str(page or ""))
            if key in seen:
                continue
            seen.add(key)
            location = f", page {page}" if page else ""
            score = getattr(item, "score", None)
            score_text = f" (similarity {score:.3f})" if isinstance(score, (int, float)) else ""
            excerpt = node.get_content().replace("\n", " ").strip()
            if len(excerpt) > 600:
                excerpt = excerpt[:600].rstrip() + "..."
            sources.append(f"**{source}{location}**{score_text}\n\n> {excerpt}")

        if sources:
            with st.expander("Retrieved sources"):
                for source in sources:
                    st.markdown(source)
        else:
            st.caption("No source passages were returned for this response.")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
