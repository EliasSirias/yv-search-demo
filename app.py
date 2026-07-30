import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

import streamlit as st
from rag_core import search_docs
from index_manager import load_or_build_index, load_full_document
from ui_helpers import format_source_name, group_results_by_path
from agent import build_support_agent
from feedback import save_feedback
from config import MAX_DISTANCE

st.set_page_config(page_title="MVP (Demo)", page_icon="🔎")
# st.set_page_config(page_title="YV Search (Demo)", page_icon="🔎")

# Tesseract path  moved to index_manager.py

# ------------------------------
# PROJECT_CODENAME = "YV"
# ------------------------------

# -------------------------------
# UI Style
# -------------------------------
st.markdown(
    """
<style>
/*Margin For Answer Retrieved*/
.bot-header {
    margin-top: 0.2rem;
    margin-bottom: 0.4rem;
    font-weight: 600;
}
/* Preserve newlines in answer text */
.chat-bubble pre,
.chat-bubble .answer-text {
    white-space: pre-wrap;   /* keeps line breaks and wraps nicely */
    word-wrap: break-word;
}
/* Emphasized helper text */
.confidence-note {
    color: rgba(0, 0, 139, 0.95);   /* YV deep blue */
    font-size: 0.95rem;
    font-weight: 600;
    margin-top: 0.35rem;
}

/* Page width & spacing */
.block-container {
    padding-top: 2rem;
    max-width: 900px;
}

/* Primary buttons – premium touch */
.stButton > button {
    border-radius: 14px;
    padding: 0.6rem 1.1rem;
    font-weight: 600;
    border: 1px solid rgba(49, 51, 63, 0.20);
}

.stButton > button:hover {
    border: 1px solid rgba(0, 0, 139, 0.45); /* subtle YV blue */
}

/* Chat bubbles (shared) */
.chat-bubble {
    padding: 0.9rem 1.1rem;
    border-radius: 18px;
    margin: 0.6rem 0;
    border: 1px solid rgba(49, 51, 63, 0.18);
    line-height: 1.45;
}

/* User messages */
.chat-bubble.user {
    background: rgba(0, 122, 255, 0.10);
}

/* Bot messages */
.chat-bubble.bot {
    background: rgba(46, 204, 113, 0.10);
}
</style>

""",
    unsafe_allow_html=True,
)

# st.title("🔎 YV Search (Demo)")
st.title("🔎 MVP Search (Demo)")
st.caption("Hallucination-resistant RAG assistant over example technical documentation")
# st.caption(f"{PROJECT_CODENAME} • YV-hard work defines it")

# -------------------------------
# Retrieval guardrails moved to rag_core.py
# -------------------------------

knowledge_base_filter = st.sidebar.selectbox(
    "Search in",
    options=["All", "Generic", "Work", "Sample"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.subheader("How It Works")

st.sidebar.markdown("""
- 🔍 Searches indexed documentation
- 🤖 Uses an AI agent to decide when to search
- 📚 Returns supporting documentation
- ✅ Keeps answers grounded in source material
""")

with st.sidebar.expander("Developer Settings"):
    k = st.slider("Top-K retrieved chunks", 1, 4, 2)
    min_hit_count = st.slider("Min matching chunks required", 1, 4, 1)
    # For LangChain FAISS, score is often L2 distance (lower = better). Tune as needed.

    show_context = st.checkbox("Show retrieved context", value=False)
    st.caption("Default strictness tuned for documentation accuracy (1.25)")
    DEFAULT_MAX_DISTANCE = 1.25
    max_distance = st.slider(
        "Max distance allowed (lower = stricter)", 0.2, 2.0, DEFAULT_MAX_DISTANCE, 0.05
    )
    st.markdown(
        """
        <div class="confidence-note">
            This assistant only responds when documentation relevance meets confidence thresholds.
        </div>
        """,
        unsafe_allow_html=True,
    )
    show_diagnostics = st.checkbox("Show Retrieval Diagnostics", value=False)
use_llm = False


# -------------------------------
# Keyword overlap guardrail
# -------------------------------

# -------------------------------
# Scope coverage guardrail (NEW)
# -------------------------------

# -------------------------------
# PDF extraction moved to index_manager.py
# -------------------------------

# -------------------------------
# Embeddings and vector store moved to index_manager.py
# -------------------------------

# -------------------------------
# Build Vector Store
# -------------------------------

# -------------------------------
# Load PDFs at startup
# -------------------------------

st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("chunks", None)

if st.session_state.vectorstore is None:
    with st.spinner("Loading knowledge base..."):
        try:
            index_result = load_or_build_index()
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
            st.stop()

        st.session_state.vectorstore = index_result["vectorstore"]
        st.session_state.chunks = index_result["chunks"]

        st.success(index_result["status"])
# -------------------------------
# Ask Question
# -------------------------------
with st.form("search_form"):
    question = st.text_input(
        "Question", placeholder="e.g., How do I sync data on mobile?"
    )

    submitted = st.form_submit_button("Ask MVP Search")
# --------------------------------------------------
# Phase 2:
# Agent-driven answer generation will be added here.
# The agent will decide when to call search_docs()
# and generate grounded responses from retrieved context.
# --------------------------------------------------
# if "support_agent" not in st.session_state:
#     st.session_state.support_agent = build_support_agent(
#         vectorstore=st.session_state.vectorstore,
#         knowledge_base_filter=knowledge_base_filter,
#         k=k,
#         min_hit_count=min_hit_count,
#         max_distance=max_distance,
#     )

# if st.button("Test Agent") and question:
#     response = st.session_state.support_agent.invoke(
#         {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": question,
#                 }
#             ]
#         }
#     )

#     final_message = response["messages"][-1]
#     answer = final_message.content

#     if not isinstance(answer, str):
#         answer = str(answer)

#     st.markdown(
#         f"""
# <div class="chat-bubble bot">
# <b>Assistant:</b><br>
# <div class="answer-text">{answer}</div>
# </div>
# """,
#         unsafe_allow_html=True,
#   )


# if st.button("Ask YV Search") and question:
if submitted and question:
    if st.session_state.vectorstore is None:
        st.error("Knowledge base not loaded.")
        st.stop()

    results = search_docs(
        query=question,
        vectorstore=st.session_state.vectorstore,
        knowledge_base_filter=knowledge_base_filter,
        k=k,
        min_hit_count=min_hit_count,
        max_distance=max_distance,
    )
    st.session_state["last_question"] = question
    st.session_state["last_results"] = results
if "last_results" in st.session_state:
    results = st.session_state["last_results"]
    question = st.session_state["last_question"]
    st.markdown(
        f'<div class="chat-bubble user"><b>You Asked:</b><br>{question}</div>',
        unsafe_allow_html=True,
    )

    if not results["found"]:
        st.markdown(
            f'<div class="chat-bubble bot"><b>Bot:</b><br>{results["message"]}</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    kept = results["results"]
    confidence = results["confidence"]
    # Build context once
    # Phase 2:
    # Add agent-driven answer generation on top of search_docs().
    st.markdown(
        """
            <div class="chat-bubble bot bot-header">
                Supporting Documentation
            </div>
            """,
        unsafe_allow_html=True,
    )
    if confidence["label"] == "High confidence":
        confidence_icon = "🟢"
    elif confidence["label"] == "Possible match":
        confidence_icon = "🟡"
    else:
        confidence_icon = "🔴"

    st.markdown(f"**{confidence['label']}:** {confidence['message']}")

    grouped = group_results_by_path(kept)
    for index, (relative_path, items) in enumerate(grouped.items(), start=1):

        source = items[0]["source"]
        knowledge_base = items[0]["knowledge_base"]
        category = items[0]["category"]

        display_source = format_source_name(source)
        combined_text = "\n\n".join(item["text"] for item in items)
        st.markdown(
            f"""
        <div class="chat-bubble bot">
            <b>📄Source:</b> {display_source}<br>
            <b>📚Knowledge Base:</b> {knowledge_base}<br>
            <b>🏷 Category:</b> {category}<br><br>
            <div class="answer-text">{combined_text}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        feedback_key = f"feedback_{relative_path}_{index}"

        if feedback_key not in st.session_state:
            st.session_state[feedback_key] = False

        if not st.session_state[feedback_key]:
            feedback_col1, feedback_col2 = st.columns(2)

            with feedback_col1:
                if st.button(
                    "👍 Helpful",
                    key=f"helpful_{relative_path}_{index}",
                ):
                    save_feedback(
                        question=question,
                        source=source,
                        relative_path=relative_path,
                        rank=index,
                        confidence=confidence["label"],
                        helpful=True,
                    )

                    st.session_state[feedback_key] = True

            with feedback_col2:
                if st.button(
                    "👎 Not helpful",
                    key=f"not_helpful_{relative_path}_{index}",
                ):
                    save_feedback(
                        question=question,
                        source=source,
                        relative_path=relative_path,
                        rank=index,
                        confidence=confidence["label"],
                        helpful=False,
                    )

                    st.session_state[feedback_key] = True
            if show_diagnostics:
                st.markdown("### 🔧 Retrieval Diagnostics")

                best_score = min(item["score"] for item in items)

                st.write(f"**Result Index:** {index}")
                st.write(f"**Best Score:** {best_score:.3f}")
                st.write(f"**Max Distance:** {MAX_DISTANCE:.2f}")
                st.write(f"**Confidence:** {confidence['label']}")
                st.write(f"**Items Retrieved:** {len(items)}")

                for chunk_index, item in enumerate(items, start=1):
                    st.write(f"**Chunk Scores:** {chunk_index}: {item['score']:.3f}")

        with st.expander("📄 View Full Document"):
            full_document = load_full_document(relative_path)

            st.markdown(
                f'<div class="answer-text">{full_document}</div>',
                unsafe_allow_html=True,
            )
    # Render answer OR retrieved context

    # Optional sources
    if show_context:
        with st.expander("Sources (retrieved context)"):
            for item in kept:
                st.write(f'Source: {item["source"]}')
                st.write(f'Knowledge Base: {item["knowledge_base"]}')
                st.write(f'Category: {item["category"]}')
                st.write(f'Score: {item["score"]:.4f}')
                st.code(item["text"])
                st.divider()
