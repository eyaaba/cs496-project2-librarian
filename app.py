
import hashlib
import os
import urllib.request

import streamlit as st
from src.rag import answer, INDEX_PATH


# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="The Librarian",
    page_icon="📚",
    layout="wide",
)

# --------------------------------------------------
# Prebuilt documentation index
# --------------------------------------------------

INDEX_URL = (
    "https://github.com/eyaaba/cs496-project2-librarian/"
    "releases/download/v1.0-index/index.json"
)

EXPECTED_SHA256 = (
    "bb530f0d5f5402bc6584d692120833abd7891f1c5ccb8fde9af83a952b5a618f"
)


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def ensure_index():
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Reuse the existing index if it matches the release.
    if INDEX_PATH.exists():
        return

    temp_path = INDEX_PATH.with_suffix(".download")

    try:
        digest = hashlib.sha256()

        with urllib.request.urlopen(INDEX_URL, timeout=120) as response:
            with temp_path.open("wb") as output:
                while True:
                    block = response.read(1024 * 1024)

                    if not block:
                        break

                    output.write(block)
                    digest.update(block)

        if digest.hexdigest() != EXPECTED_SHA256:
            raise ValueError(
                "Downloaded index failed SHA-256 verification."
            )

        temp_path.replace(INDEX_PATH)

    finally:
        temp_path.unlink(missing_ok=True)


# --------------------------------------------------
# Load index
# --------------------------------------------------

if not INDEX_PATH.exists():
    with st.spinner(
        "Downloading the prebuilt Python documentation index..."
    ):
        try:
            ensure_index()
        except Exception as exc:
            st.error(f"Index download failed: {exc}")
            st.stop()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📚 The Librarian")
st.caption(
    "CS496 AI Engineering • Project 2 • Python Documentation RAG"
)

st.info(
    "Ask questions about official Python documentation. "
    "Answers are generated using retrieved sources with "
    "citation validation. Hybrid retrieval is recommended."
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Document index")

    st.success("Prebuilt index loaded")
    st.caption(
        "Official Python documentation • "
        "8,772 indexed passages"
    )

    st.divider()

    mode = st.selectbox(
        "Retrieval mode",
        [
            "hybrid",
            "hybrid_rerank",
            "bm25",
            "vector",
        ],
        format_func=lambda x: {
            "hybrid": "Hybrid (Recommended)",
            "hybrid_rerank": "Hybrid + Cross-Encoder",
            "bm25": "BM25",
            "vector": "Vector",
        }[x],
    )

    st.caption(
        "Hybrid combines BM25 and semantic search "
        "using Reciprocal Rank Fusion."
    )

    st.divider()

    st.subheader("About")
    st.write(
        "The Librarian is a source-grounded RAG "
        "assistant built for CS496."
    )

    st.caption(
        "The system uses OpenAI embeddings, "
        "hybrid retrieval, and optional reranking."
    )


# --------------------------------------------------
# Question and answer
# --------------------------------------------------

question = st.text_input(
    "Ask a question about Python",
    placeholder="Example: What is the difference between a list and a tuple?",
)

if st.button("Ask", type="primary") and question.strip():

    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "OPENAI_API_KEY is not configured. "
            "Set it in Streamlit Cloud secrets."
        )
        st.stop()

    try:
        with st.spinner("Searching documentation and generating answer..."):
            response, passages = answer(
                question.strip(),
                mode=mode,
            )

        st.subheader("Answer")
        st.write(response)

        st.divider()

        st.subheader("Retrieved evidence")

        if not passages:
            st.warning("No supporting passages were retrieved.")

        for passage in passages:
            with st.expander(
                f"{passage['id']} — "
                f"{passage['source']} "
                f"(page {passage['page']})"
            ):
                st.write(passage["text"])

    except Exception as exc:
        st.error(f"Unable to process the question: {exc}")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "The Librarian • CS496 Project 2 • "
    "Answers may be incomplete. Verify important information "
    "against the cited Python documentation."
)
