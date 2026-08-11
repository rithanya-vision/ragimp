import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# ==========================================
# 1. GROQ API KEY
# ==========================================

GROQ_API_KEY = ""


# ==========================================
# 2. PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="PDF RAG",
    page_icon="📚"
)

st.title("📚 PDF Question Answering System")

st.write(
    "Upload a PDF and ask questions based only on the PDF."
)


# ==========================================
# 3. UPLOAD PDF
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# ==========================================
# 4. PROCESS PDF
# ==========================================

if uploaded_file:

    st.success("✅ PDF uploaded successfully!")


    # ==========================================
    # 5. TEXT EXTRACTION
    # ==========================================

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"


    if not text.strip():

        st.error("❌ No text found in this PDF.")

        st.stop()


    st.success("✅ PDF text extracted successfully!")

    st.write(
        "Characters extracted:",
        len(text)
    )


    # ==========================================
    # 6. CHUNKING
    # ==========================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    st.success("✅ Text chunking completed!")

    st.write(
        "Number of chunks:",
        len(chunks)
    )


    # ==========================================
    # 7. EMBEDDING
    # ==========================================

    with st.spinner("Creating embeddings..."):

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    st.success("✅ Embeddings created!")


    # ==========================================
    # 8. VECTOR DATABASE
    # ==========================================

    with st.spinner("Creating vector database..."):

        vector_db = FAISS.from_texts(
            chunks,
            embedding=embeddings
        )

    st.success("✅ FAISS vector database created!")


    # ==========================================
    # 9. USER QUESTION
    # ==========================================

    question = st.text_input(
        "💬 Ask a question about your PDF:"
    )


    if question:

        # ==========================================
        # 10. RETRIEVER
        # ==========================================

        with st.spinner("Searching the PDF..."):

            documents = vector_db.similarity_search(
                question,
                k=4
            )


        # ==========================================
        # 11. CREATE CONTEXT
        # ==========================================

        context = "\n\n".join(
            document.page_content
            for document in documents
        )


        # ==========================================
        # 12. PROMPT
        # ==========================================

        prompt = f"""
You are a PDF Question Answering Assistant.

Answer the user's question ONLY using the
information provided in the PDF context.

Rules:

1. Do not use your own knowledge.
2. Do not make up information.
3. Do not guess.
4. If the answer is available in the context,
   answer clearly.
5. If the answer is NOT available in the context,
   respond exactly:

"The answer is not found in the uploaded PDF."

PDF CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""


        # ==========================================
        # 13. GROQ LLM
        # ==========================================

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=GROQ_API_KEY
        )


        # ==========================================
        # 14. GENERATE ANSWER
        # ==========================================

        with st.spinner("Generating answer..."):

            response = llm.invoke(prompt)


        # ==========================================
        # 15. DISPLAY ANSWER
        # ==========================================

        st.subheader("🤖 Answer")

        st.write(response.content)