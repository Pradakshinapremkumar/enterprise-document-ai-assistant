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

st.sidebar.title("📄 Document AI Assistant")
st.sidebar.markdown("Upload PDFs and ask questions!")

groq_api_key = os.environ.get("GROQ_API_KEY", "")
if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

st.title("📄 Enterprise Document AI Assistant")
st.markdown("Powered by LLaMA 3.3 70B + FAISS Vector Search")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if groq_api_key:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=groq_api_key
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing documents..."):
            all_chunks = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(uploaded_file.read())
                    temp_path = f.name

                loader = PyPDFLoader(temp_path)
                documents = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(documents)
                all_chunks.extend(chunks)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vector_store = FAISS.from_documents(all_chunks, embeddings)
            st.session_state.retriever = vector_store.as_retriever(
                search_kwargs={"k": 3}
            )

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

            st.session_state.qa_chain = (
                {"context": st.session_state.retriever | format_docs,
                 "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

        st.success(f"✅ {len(uploaded_files)} document(s) processed! {len(all_chunks)} chunks indexed.")

    # Chat interface
    if st.session_state.qa_chain:
        st.markdown("### 💬 Ask a Question")
        question = st.chat_input("Ask anything about your documents...")

        if question:
            with st.spinner("Thinking..."):
                # Get answer
                answer = st.session_state.qa_chain.invoke(question)
                
                # Get source chunks
                source_docs = st.session_state.retriever.invoke(question)
                sources = list(set([
                    f"Page {doc.metadata.get('page', 'N/A') + 1}"
                    for doc in source_docs
                ]))

            # Save to chat history
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "sources": sources
            })

        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])
                st.caption(f"📚 Sources: {', '.join(chat['sources'])}")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
