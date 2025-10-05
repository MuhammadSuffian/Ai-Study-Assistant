# RAG Chatbot with Groq + Streamlit

A Retrieval-Augmented Generation (RAG) chatbot that uses Groq API for fast inference and Streamlit for the web interface.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Groq API key as an environment variable:
```bash
# Windows
set GROQ_API_KEY=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

# Linux/Mac
export GROQ_API_KEY=your_groq_api_key_here
```

**Important**: Replace `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` with your actual Groq API key from https://console.groq.com/

3. Run the application:
```bash
streamlit run test.py
```

## Features

- Upload multiple documents (TXT and PDF)
- Automatic text chunking and vector indexing
- Fast inference with Groq API
- Interactive web interface

## Security Note

⚠️ **Important**: Never commit your API keys to version control. Use environment variables instead.
