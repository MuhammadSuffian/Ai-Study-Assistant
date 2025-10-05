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

st.set_page_config(
    page_title="AI STUDY BUDDY",
    page_icon="🤖",
)
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

# Custom CSS for chat-like interface
st.markdown("""
<style>
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 8rem;
        max-width: 800px;
    }
    
    /* Chat container */
    .chat-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        min-height: 400px;
        max-height: 500px;
        overflow-y: auto;
    }
    
    /* User message bubble */
    .user-message {
        background: #e3f2fd;
        color: #1976d2;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px auto;
        max-width: 70%;
        width: fit-content;
        margin-left: auto;
        display: block;
        text-align: right;
    }
    
    /* AI message bubble */
    .ai-message {
        background: #f5f5f5;
        color: #333333;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px auto 8px 0;
        max-width: 70%;
        width: fit-content;
        margin-right: auto;
        display: block;
    }
    
    /* Fixed bottom input */
    .fixed-bottom-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 999;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background: #f8f9fa;
        border: 2px solid #e0e0e0;
        border-radius: 25px;
        padding: 12px 20px;
        font-size: 16px;
        color: #333;
        transition: all 0.3s ease;
        height: 48px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1976d2;
        box-shadow: 0 0 0 3px rgba(25,118,210,0.1);
        outline: none;
    }
    
    /* Send button styling */
    .stButton > button {
        background: #1976d2;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        height: 48px;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #1565c0;
        transform: translateY(-1px);
    }
    
    .stButton > button:disabled {
        background: #cccccc;
        color: #666666;
        transform: none;
        cursor: not-allowed;
    }
    
    /* Upload section in sidebar */
    .upload-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-ready {
        color: #28a745;
        font-weight: 600;
    }
    
    .status-waiting {
        color: #6c757d;
        font-style: italic;
    }
    
    /* Hide streamlit elements */
    .stDeployButton {
        display: none;
    }
    
    footer {
        display: none;
    }
    
    .stApp > header {
        display: none;
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
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Query UI (enabled once an index exists)
vectorstore_ready = st.session_state.get("vectorstore") is not None

# Chat Interface
st.markdown("### 💬 AI Study Assistant")

# Chat container
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        # Welcome message
        st.markdown(f'<div class="ai-message">👋 Hi! I\'m your AI Study Assistant. Upload some documents in the sidebar and start asking questions!</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Fixed bottom input area
st.markdown('<div class="fixed-bottom-input">', unsafe_allow_html=True)
col1, col2 = st.columns([5, 1], gap="small")

with col1:
    query = st.text_input(
        "Type your message...",
        value="",
        disabled=not vectorstore_ready,
        placeholder="Ask anything about your documents...",
        label_visibility="collapsed",
        key="chat_input"
    )

with col2:
    send_clicked = st.button("Send", disabled=not vectorstore_ready or not query.strip(), key="send_chat")

st.markdown('</div>', unsafe_allow_html=True)

# Process query when send button is clicked or Enter is pressed
if send_clicked and query.strip() and vectorstore_ready:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    with st.spinner("Thinking..."):
        try:
            retrieved_docs = st.session_state["vectorstore"].similarity_search(query, k=3)
            if not retrieved_docs:
                answer = "I couldn't find relevant information in the uploaded documents. Please try rephrasing your question."
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
            
            # Add AI response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            
            # Clear the input and rerun to show new messages
            st.rerun()
            
        except Exception as e:
            error_message = f"Sorry, I encountered an error: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            st.rerun()

# Sidebar 
with st.sidebar:
    # Document Upload Section
    st.markdown("### 📁 **Upload Documents**")
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload documents", type=["txt", "pdf"], accept_multiple_files=True, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
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
    
    # Chat controls
    st.markdown("### 💬 **Chat Controls**")
    if st.button("🗑️ Clear Chat History", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()
    
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
