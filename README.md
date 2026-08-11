# 📚 PDF Question Answering System

A simple **Retrieval-Augmented Generation (RAG)** application that allows users to upload a PDF and ask questions about its contents.

The application extracts text from the uploaded PDF, splits it into smaller chunks, converts the chunks into embeddings, stores them in a FAISS vector database, retrieves the most relevant chunks, and uses a Groq LLM to generate an answer.

---

## 🚀 Features

- 📄 Upload PDF documents
- 🔍 Extract text from PDFs
- ✂️ Split text into smaller chunks
- 🧠 Generate embeddings using Hugging Face
- 🗄️ Store embeddings using FAISS
- 🔎 Perform similarity search
- 🤖 Generate answers using Groq LLM
- 💬 Interactive Streamlit interface
- 🛡️ Answers are generated only from the uploaded PDF context

---

## 🏗️ RAG Architecture

The application follows this workflow:

```text
                    PDF
                     │
                     ▼
              ┌─────────────┐
              │    PyPDF    │
              │ Text Extract│
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ Text Splitter│
              │   Chunking   │
              └──────┬──────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Hugging Face        │
          │ Embeddings          │
          │ all-MiniLM-L6-v2    │
          └──────────┬──────────┘
                     │
                     ▼
              ┌─────────────┐
              │    FAISS    │
              │Vector Store │
              └──────┬──────┘
                     │
              User Question
                     │
                     ▼
              Similarity Search
                     │
                     ▼
              Relevant Chunks
                     │
                     ▼
              ┌─────────────┐
              │  Groq LLM   │
              │    Llama    │
              └──────┬──────┘
                     │
                     ▼
                Final Answer
