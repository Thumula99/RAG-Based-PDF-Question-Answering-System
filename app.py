import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

st.title("📚 Chat With Your PDF (RAG App)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.write("Processing document...")

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever()

    st.success("Document ready! Ask your question 👇")

    query = st.text_input("Enter your question")

    if query:
        # Retrieve relevant chunks
        docs = retriever.get_relevant_documents(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        # Create prompt manually
        prompt = f"""
        Use the following context to answer the question.

        Context:
        {context}

        Question:
        {query}
        """

        llm = ChatOpenAI(temperature=0)
        response = llm.invoke(prompt)

        st.write("### Answer:")
        st.write(response.content)
