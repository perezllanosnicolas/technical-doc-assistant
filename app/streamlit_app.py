import streamlit as st
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ingestion import build_vectorstore, load_vectorstore
from src.memory import get_conversational_chain
from src.chains import extract_requirements, generate_summary

st.set_page_config(
    page_title="Technical Doc Assistant",
    layout="wide"
)

st.title("Technical Documentation Assistant")
st.caption("Upload technical PDFs and query them with natural language")

# ── Sidebar: Upload & Index ──────────────────────────────────────────
with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Index Documents", type="primary"):
        with st.spinner("Indexing..."):
            paths = []
            for f in uploaded_files:
                p = f"data/uploads/{f.name}"
                os.makedirs("data/uploads", exist_ok=True)
                with open(p, "wb") as out:
                    out.write(f.read())
                paths.append(p)
            vs = build_vectorstore(paths)
            st.session_state["vectorstore"] = vs
            st.session_state["chain"] = get_conversational_chain(vs)
        st.success(f"{len(paths)} document(s) indexed")
    
    st.divider()
    
    # Select mode
    mode = st.radio("Mode", ["Q&A Chat", "Extract Requirements", "Generate Summary"])
    
# Select model
    model = st.selectbox("Model", ["openai/gpt-oss-20b", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"])

# ── Main area ────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.info("Upload and index your technical documents to get started")
    st.stop()

vs = st.session_state["vectorstore"]

if mode == "Q&A Chat":
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if prompt := st.chat_input("Ask anything about your documents..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = st.session_state["chain"]
                result = chain.invoke({"question": prompt})
                answer = result["answer"]
                sources = list({d.metadata["source"] for d in result["source_documents"]})
            
            st.write(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.caption(f"• {os.path.basename(s)}")
            
            st.session_state["messages"].append({"role": "assistant", "content": answer})

elif mode == "Extract Requirements":
    if st.button("Extract Requirements", type="primary"):
        with st.spinner("Analyzing document..."):
            result = extract_requirements(vs)
        st.markdown(result.content)
        st.download_button("Download as .md", result.content, "requirements.md")

elif mode == "Generate Summary":
    if st.button("Generate Summary", type="primary"):
        with st.spinner("Generating summary..."):
            result = generate_summary(vs)
        st.markdown(result.content)
        st.download_button("Download as .md", result.content, "summary.md")