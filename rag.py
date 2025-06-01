import streamlit as st
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

st.set_page_config(page_title="PDF QA Bot", layout="wide")
st.title("Gibbs AI Bot with Ollama")

uploaded_file = st.file_uploader("📎 Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner("🔍 Processing PDF..."):
        # Load and chunk PDF
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Create embedding
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        db = FAISS.from_documents(chunks, embeddings)

        # Set up LLM
        retriever = db.as_retriever()
        llm = Ollama(model="llama3")
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # Ask a question
    question = st.text_input("💬 Ask a question about the PDF")

    if question:
        with st.spinner("🧠 Thinking..."):
            result = qa.invoke(question)
        st.subheader("Answer")

        if result["result"].strip():
            st.write(result["result"])

            with st.expander("Sources"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.code(doc.page_content[:1000] + "...")
        else:
            st.warning("No relevant information found in the document for this question. Please try asking something else.")



