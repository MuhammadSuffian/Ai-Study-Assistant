import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from groq import Groq
import PyPDF2
import io
import re

# Initialize Groq client with environment variable
import os
groq_client = Groq(api_key="API KEY HERE")

# Remove hidden reasoning from model outputs
THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

def hide_thinking(text: str) -> str:
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    text = THINK_TAG_RE.sub("", text)
    drop_prefixes = ("reasoning:", "thought:", "chain-of-thought:", "scratchpad:")
    lines = [ln for ln in text.splitlines() if not ln.strip().lower().startswith(drop_prefixes)]
    return "\n".join(lines).strip()

# Custom CSS for better styling
st.markdown("""
<style>
    .query-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .query-title {
        color: black;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.95);
        border: 2px solid #000000;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 16px;
        color: #000000;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        height: 35px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4CAF50;
        box-shadow: 0 4px 20px rgba(76,175,80,0.3);
        outline: none;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76,175,80,0.3);
        height: 48px;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #45a049, #4CAF50);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76,175,80,0.4);
    }
    
    .stButton > button:disabled {
        background: #cccccc;
        color: #666666;
        transform: none;
        box-shadow: none;
        cursor: not-allowed;
    }
    
    .query-input-row {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .upload-section {
        background: rgba(248, 249, 250, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px dashed #dee2e6;
        margin: 1rem 0;
    }
    
    .status-ready {
        color: #28a745;
        font-weight: 600;
    }
    
    .status-waiting {
        color: #6c757d;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("AI STUDY BUDDY")

# App state
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "num_docs" not in st.session_state:
    st.session_state["num_docs"] = 0
if "num_chunks" not in st.session_state:
    st.session_state["num_chunks"] = 0
if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""

# Query UI (enabled once an index exists)
vectorstore_ready = st.session_state.get("vectorstore") is not None

# Enhanced Query Section
# st.markdown('<div class="query-container">', unsafe_allow_html=True)
st.markdown('<div class="query-title">💬 Ask Your Question</div>', unsafe_allow_html=True)

# Create columns for text input and send button with proper alignment
col1, col2 = st.columns([4, 1], gap="small")

with col1:
    query = st.text_input(
        "Enter your question:",
        value=st.session_state.query_input,
        disabled=not vectorstore_ready,
        placeholder="Type your question about the uploaded documents...",
        label_visibility="collapsed",
        key="query_input_field"
    )

with col2:
    # Remove the manual spacing and let CSS handle alignment
    send_clicked = st.button("🚀 Send", disabled=not vectorstore_ready or not query.strip(), key="send_button")

st.markdown('</div>', unsafe_allow_html=True)

# Update session state
if query != st.session_state.query_input:
    st.session_state.query_input = query

# Process query when send button is clicked or Enter is pressed
should_process_query = (send_clicked or query) and vectorstore_ready and query.strip()

if should_process_query:
    with st.spinner("Retrieving context and generating answer..."):
        try:
            retrieved_docs = st.session_state["vectorstore"].similarity_search(query, k=3)
            if not retrieved_docs:
                st.warning("No relevant context found in the indexed documents.")
            else:
                context = "\n".join([d.page_content for d in retrieved_docs])
                prompt = f"""Instruction: Provide only the final answer. Do not include chain-of-thought, hidden reasoning, or <think> blocks.
                Answer the question using only the context below:
                Context: {context}
                Question: {query}
                """

                response = groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": prompt}]
                )

                answer = response.choices[0].message.content
                answer = hide_thinking(answer)
                
                # Enhanced answer display
                st.markdown("### 💡 **Answer**")
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    padding: 1.5rem;
                    border-radius: 12px;
                    border-left: 4px solid #28a745;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    margin: 1rem 0;
                ">
                    {answer}
                </div>
                """, unsafe_allow_html=True)

                with st.expander("📋 Show retrieved context"):
                    for i, d in enumerate(retrieved_docs, start=1):
                        st.markdown(f"""
                        <div style="
                            background: #f8f9fa;
                            padding: 1rem;
                            border-radius: 8px;
                            margin: 0.5rem 0;
                            border-left: 3px solid #007bff;
                        ">
                            <strong>Chunk {i}</strong> — {d.metadata.get('filename', 'unknown')}
                        </div>
                        """, unsafe_allow_html=True)
                        st.write(d.page_content[:1200] + ("..." if len(d.page_content) > 1200 else ""))
        except Exception as e:
            st.error(f"Query failed: {e}")

# Sidebar status
with st.sidebar:
    # Document Upload Section
    st.markdown("### 📁 **Upload Documents**")
    uploaded_files = st.file_uploader("Upload documents", type=["txt", "pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        st.info("Files selected. Click 'Process documents' to build the index.")
        if st.button("Process documents"):
            with st.spinner("Processing and indexing documents..."):
                try:
                    docs = []
                    for file in uploaded_files:
                        if file.type == "text/plain":
                            text = file.read().decode("utf-8", errors="ignore")
                        elif file.type == "application/pdf":
                            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
                            text = ""
                            for page in pdf_reader.pages:
                                page_text = page.extract_text() or ""
                                text += page_text + "\n"
                        else:
                            st.error(f"Unsupported file type: {file.type}")
                            continue

                        text = (text or "").strip()
                        if not text:
                            st.warning(f"No extractable text in {file.name}; skipping.")
                            continue

                        docs.append(Document(page_content=text, metadata={"filename": file.name}))

                    if not docs:
                        st.error("No valid documents to index.")
                    else:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        chunks = splitter.split_documents(docs)

                        if not chunks:
                            st.error("Documents produced zero chunks; try different files.")
                        else:
                            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                            st.session_state["vectorstore"] = FAISS.from_documents(chunks, embeddings)
                            st.session_state["num_docs"] = len(docs)
                            st.session_state["num_chunks"] = len(chunks)
                            st.success(f"✅ Indexed {len(docs)} documents into {len(chunks)} chunks.")
                            st.rerun()  # Refresh the app to enable the text input
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
    
    st.markdown("---")
    
    # Index Status Section
    st.markdown("### 📊 **Index Status**")
    if vectorstore_ready:
        st.markdown(
            f'<div class="status-ready">✅ Ready: {st.session_state["num_docs"]} docs / {st.session_state["num_chunks"]} chunks</div>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-waiting">⏳ No index yet. Upload files and click Process.</div>', 
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    st.markdown("### 🔧 **How to Use**")
    st.markdown("""
    1. **Upload** your documents (TXT or PDF)
    2. **Process** them to create the index
    3. **Ask** questions using the query box
    4. **Click** the 🚀 Send button or press Enter
    """)
