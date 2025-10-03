import streamlit as st
import groq
import os
from dotenv import load_dotenv
import PyPDF2
import numpy as np
from typing import List, Dict
import json
import tempfile

# Load environment variables
load_dotenv()

# Initialize Groq client
client = groq.Groq(api_key="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")

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, file_path: str) -> str:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text() + '\n'
        return text

    def create_chunks(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            # If this is not the first chunk, back up to include overlap
            if start > 0:
                start = start - self.chunk_overlap
            chunk = text[start:end]
            chunks.append(chunk)
            start = end
        return chunks

class GroqVectorStore:
    def __init__(self, client):
        self.client = client
        self.documents: List[Dict] = []
        
    def embed_text(self, text: str) -> List[float]:
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a text embedding system. Output ONLY a JSON array of 384 floating point numbers representing the semantic embedding of the input text. No other text or explanation."},
                {"role": "user", "content": text}
            ],
            temperature=0.0
        )
        embedding = json.loads(response.choices[0].message.content)
        return embedding

    def add_documents(self, chunks: List[str]):
        for chunk in chunks:
            embedding = self.embed_text(chunk)
            self.documents.append({
                "text": chunk,
                "embedding": embedding
            })

    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        query_embedding = self.embed_text(query)
        
        # Calculate similarities
        similarities = []
        for doc in self.documents:
            similarity = self.cosine_similarity(query_embedding, doc["embedding"])
            similarities.append((similarity, doc["text"]))
        
        # Sort by similarity and return top k
        similarities.sort(reverse=True)
        return [text for _, text in similarities[:k]]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Set page configuration
st.set_page_config(page_title="AI Study Assistant", page_icon="🤖", layout="wide")
st.title("AI Study Assistant with RAG 📚")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = GroqVectorStore(client)

# Initialize document processor
doc_processor = DocumentProcessor()

# Sidebar for document upload
with st.sidebar:
    st.markdown("## Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])
    
    if uploaded_file is not None:
        try:
            # Create a temporary file to save the uploaded PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Process the document
            text = doc_processor.extract_text_from_pdf(tmp_path)
            chunks = doc_processor.create_chunks(text)
            
            # Add to vector store
            with st.spinner("Processing document..."):
                st.session_state.vectorstore.add_documents(chunks)
            
            st.success(f"Document '{uploaded_file.name}' processed successfully!")
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

    st.markdown("""
    # How to use
    1. Upload PDF documents using the file uploader above
    2. Ask questions about the uploaded documents
    3. Get AI-powered responses based on your documents
    """)

# Main chat interface
st.markdown("### Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Retrieve relevant documents
            relevant_docs = st.session_state.vectorstore.similarity_search(prompt)
            context = "\n\n".join(relevant_docs) if relevant_docs else "No relevant documents found."

            # Create messages for the API call
            messages = [
                {"role": "system", "content": "You are a helpful and knowledgeable study assistant. Use the following context to answer questions accurately. If the question cannot be answered from the context, say so clearly.\n\nContext:\n" + context},
                {"role": "user", "content": prompt}
            ]

            # Make API call to Groq
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                stream=True
            )

            # Stream the response
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Display document context
with st.expander("View Document Context"):
    st.info("This section shows the relevant document chunks used to answer your questions.")
    if st.session_state.messages and len(st.session_state.messages) > 0:
        last_user_message = None
        for message in reversed(st.session_state.messages):
            if message["role"] == "user":
                last_user_message = message["content"]
                break
        
        if last_user_message:
            relevant_chunks = st.session_state.vectorstore.similarity_search(last_user_message)
            for i, chunk in enumerate(relevant_chunks):
                st.markdown(f"**Relevant Chunk {i+1}:**")
                st.markdown(chunk)
                st.markdown("---")