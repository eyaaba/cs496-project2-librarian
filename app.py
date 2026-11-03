import streamlit as st
from src.rag import answer, build_index, INDEX_PATH

st.set_page_config(page_title='The Librarian', page_icon='📚')
st.title('📚 The Librarian')
st.caption('CS496 Project 2 • source-grounded RAG prototype')
st.info(
    "RAG system with hybrid retrieval, optional Cross-Encoder reranking, "
    "question rewriting, and citation validation. "
    "100-question retrieval evaluation completed. "
    "Deployment, full answer-quality evaluation, and final optimization "
    "validation are still pending."
)
with st.sidebar:
    st.subheader('Document index')
    if st.button('Build / rebuild index', type='primary'):
        try:
            with st.spinner('Embedding document chunks through OpenAI…'):
                count = build_index()
            st.success(f'Indexed {count} passages')
        except Exception as e: st.error(str(e))
    st.write('Status:', 'Ready' if INDEX_PATH.exists() else 'Not indexed')
    
    mode = st.selectbox(
        "Retrieval mode",
        ["hybrid", "hybrid_rerank", "bm25", "vector"],
        format_func=lambda x: {
            "hybrid": "Hybrid (Recommended)",
            "hybrid_rerank": "Hybrid + Cross-Encoder",
            "bm25": "BM25",
            "vector": "Vector",
        }[x],
    )

    question = st.text_input('Ask a question about your documents')
if st.button('Ask') and question.strip():
    try:
        with st.spinner('Searching and answering…'):
            response, citations = answer(question,mode)
        st.markdown('### Answer')
        st.write(response)
        st.markdown('### Retrieved evidence')
        for item in citations:
            with st.expander(f"{item['id']} — {item['source']} (page {item['page']})"):
                st.write(item['text'])
    except Exception as e: st.error(str(e))
