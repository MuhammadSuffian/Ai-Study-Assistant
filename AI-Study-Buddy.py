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

# Load Groq API key securely from Streamlit secrets or environment variable.
# Preferred: st.secrets['auth_token'] (set via `streamlit secrets`), fallback to GROQ_API_KEY env var.
groq_api_key = None
try:
    groq_api_key = st.secrets.get("auth_token") if hasattr(st, "secrets") else None
except Exception:
    groq_api_key = None

if not groq_api_key:
    groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("Groq API key not found. Please set `st.secrets['auth_token']` or the GROQ_API_KEY environment variable.")
    st.stop()

groq_client = Groq(api_key=groq_api_key)

st.set_page_config(
    page_title="AI STUDY BUDDY",
    page_icon="🤖",
    InitialSideBarState = "expanded",
)
# Force light theme visuals (overrides Streamlit dark theme selectors).
st.markdown("""
<style>
    /* Make backgrounds and text use light-theme colors */
    html, body, .stApp, .main, .block-container, .css-1outpf7, .css-1d391kg, .stSidebar {
        background-color: #FFFFFF !important;
        color: #31333F !important;
    }
    /* Inputs and buttons */
    .stTextInput>div>div>input, .stButton>button, textarea {
        background-color: #f8f9fa !important;
        color: #31333F !important;
        border-color: #e0e0e0 !important;
    }
    /* Sidebar and containers */
    .css-1d391kg, .css-1v3fvcr, .stSidebar, .stApp, .main {
        background-color: #FFFFFF !important;
        color: #31333F !important;
    }
    /* Ensure message bubbles stay light */
    .user-message, .ai-message, .chat-container, .chat-input-container {
        background-color: inherit !important;
        color: inherit !important;
    }
    /* Hide Streamlit header/settings that might allow theme changes */
    .stApp > header, [data-testid="stToolbar"] { display: none !important; }
</style>
<script>
    // Encourage Streamlit to keep the global theme as light in localStorage (best-effort)
    try {
        if (window && window.localStorage) {
            var current = window.localStorage.getItem('globalTheme');
            try {
                window.localStorage.setItem('globalTheme', JSON.stringify({base:'light'}));
            } catch(e) {}
        }
    } catch(e) {}
</script>
""", unsafe_allow_html=True)

# Make sidebar expand/collapse button more visible (black circle with white icon)
# Note: keep this at top-level so it's not indented inside another block
st.markdown("""
<style>
    /* Target the floating sidebar toggle - Streamlit renders it with role=button and title attribute */
    button[title="Expand"] , button[title="Collapse"], button[aria-label="Expand Sidebar"], button[aria-label="Collapse Sidebar"] {
        background: #000000 !important;
        color: #ffffff !important;
        border-radius: 999px !important;
        width: 40px !important;
        height: 40px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25) !important;
        border: 2px solid rgba(255,255,255,0.08) !important;
        transition: transform 0.12s ease-in-out !important;
    }
    button[title="Expand"]:hover, button[title="Collapse"]:hover, button[aria-label="Expand Sidebar"]:hover, button[aria-label="Collapse Sidebar"]:hover {
        transform: translateY(-2px) !important;
    }
    /* Ensure the chevron/icon inside stays white */
    button[title="Expand"] svg, button[title="Collapse"] svg, button[aria-label="Expand Sidebar"] svg, button[aria-label="Collapse Sidebar"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Additional fallback selectors and position fix for Streamlit versions where the button uses different attributes

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
    /* Main app styling - Full height layout */
    .main .block-container {
        padding: 0rem;
        max-width: 100%;
        height: 100vh;
        display: flex;
        flex-direction: column;
    }
    
    /* Remove default streamlit spacing */
    .element-container {
        margin-bottom: 0rem !important;
    }
    
    /* App header */
    .app-header {
        background: #ffffff;
        padding: 1rem;
        border-bottom: 1px solid #e0e0e0;
        flex-shrink: 0;
        text-align: center;
    }
    
    /* Title styling */
    h1 {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 1.5rem;
        color: #333;
    }
    
    /* Hide only the main content subtitle, not sidebar titles */
    .main h3 {
        margin: 0 !important;
        padding: 0 !important;
        display: none; /* Hide the subtitle in main content */
    }
    
    /* Keep sidebar titles visible */
    .sidebar h3 {
        margin: 0.5rem 0 !important;
        padding: 0 !important;
        display: block !important;
    }
    
    /* Main chat area */
    .chat-main {
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    /* Chat container */
    .chat-container {
        background: #ffffff;
        padding: 1rem;
        flex: 1;
        overflow-y: auto;
        overflow-x: hidden;
        scroll-behavior: smooth;
    }
    
    /* Scroll to bottom automatically */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* User message bubble */
    .user-message {
        background: #e3f2fd;
        color: #1976d2;
        padding: 8px 12px;
        border-radius: 18px 18px 4px 18px;
        margin: 4px 0 4px auto;
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
        padding: 8px 12px;
        border-radius: 18px 18px 18px 4px;
        margin: 4px auto 4px 0;
        max-width: 70%;
        width: fit-content;
        margin-right: auto;
        display: block;
    }
    
    /* Fixed bottom input */
    .chat-input-container {
        background: white;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
        flex-shrink: 0;
    }
    
    .fixed-bottom-input {
        position: relative;
        bottom: auto;
        left: auto;
        right: auto;
        background: transparent;
        padding: 0;
        border-top: none;
        box-shadow: none;
        z-index: auto;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background: #f8f9fa;
        border: 2px solid #e0e0e0;
        border-radius: 25px;
        padding: 8px 16px;
        font-size: 14px;
        color: #333;
        transition: all 0.3s ease;
        height: 40px;
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
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        height: 40px;
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
    
    /* Hide streamlit elements and prevent page scroll */
    .stDeployButton {
        display: none;
    }
    
    footer {
        display: none;
    }
    
    .stApp > header {
        display: none;
    }
    
    /* Prevent main page scrolling */
    .main {
        overflow: hidden;
        height: 100vh;
    }
    
    .stApp {
        overflow: hidden;
        height: 100vh;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit UI - Header
st.markdown('<div class="app-header">', unsafe_allow_html=True)
st.title("AI STUDY BUDDY")
st.markdown('</div>', unsafe_allow_html=True)

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

# Main chat area
st.markdown('<div class="chat-main">', unsafe_allow_html=True)

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

# Auto-scroll to bottom script
st.markdown("""
<script>
    setTimeout(function() {
        var chatContainer = parent.document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }, 100);
</script>
""", unsafe_allow_html=True)

# Input area
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
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
st.markdown('</div>', unsafe_allow_html=True)

# Close main chat area
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
    # st.markdown('<div class="upload-section">', unsafe_allow_html=True)
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
