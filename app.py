import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile

st.set_page_config(page_title="Enterprise Document AI Assistant", page_icon="📄")
st.title("📄 Enterprise Document AI Assistant")
st.markdown("Upload a PDF and ask questions about it!")

groq_api_key = os.environ.get("GROQ_API_KEY", "")
if not groq_api_key:
    groq_api_key = st.text_input("Enter your Groq API Key", type="password")

if groq_api_key:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=groq_api_key
    )

    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            temp_path = f.name

        with st.spinner("Processing document..."):
            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            prompt = PromptTemplate.from_template("""
            You are a helpful AI assistant. Use the following context to answer
            the question accurately. If you don't know the answer from the context,
            say "I don't have enough information to answer this."

            Context: {context}
            Question: {question}
            Answer:
            """)

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            qa_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

        st.success(f"Document processed! {len(chunks)} chunks indexed.")

        question = st.text_input("Ask a question about your document")

        if question:
            with st.spinner("Thinking..."):
                answer = qa_chain.invoke(question)
            st.markdown("### Answer")
            st.write(answer)
