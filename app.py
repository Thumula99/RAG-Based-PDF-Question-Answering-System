import streamlit as st
import tempfile
import os
import json
from datetime import datetime
from io import StringIO

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Page config
st.set_page_config(
    page_title="PDF Chat - RAG App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-message {padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex}
    .chat-message.user {background-color: #e3f2fd}
    .chat-message.assistant {background-color: #f3e5f5}
    .source-box {background-color: #fafafa; padding: 0.75rem; border-left: 3px solid #2196F3; margin: 0.5rem 0}
</style>
""", unsafe_allow_html=True)

# ==================== CONFIGURATION ====================
MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS_DISPLAY = 4000

# ==================== VALIDATION ====================
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ **OPENAI_API_KEY** not configured. Please set your API key as an environment variable.")
    st.stop()

# ==================== SESSION STATE INITIALIZATION ====================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "token_count" not in st.session_state:
    st.session_state.token_count = {"input": 0, "output": 0}
if "selected_pages" not in st.session_state:
    st.session_state.selected_pages = None
if "total_pages" not in st.session_state:
    st.session_state.total_pages = None

# ==================== UTILITY FUNCTIONS ====================
def count_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters)"""
    return len(text) // 4

def export_chat_history() -> str:
    """Export chat history as formatted text"""
    export_text = f"PDF Chat History - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += f"File: {st.session_state.file_name}\n"
    export_text += "=" * 80 + "\n\n"
    
    for i, msg in enumerate(st.session_state.chat_history, 1):
        export_text += f"Q{i}: {msg['question']}\n"
        export_text += f"A{i}: {msg['answer']}\n"
        export_text += f"Tokens used: {msg['tokens']}\n"
        export_text += "-" * 80 + "\n\n"
    
    return export_text

def extract_pages_from_pdf(pdf_path: str) -> list:
    """Extract number of pages from PDF"""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        pages = set(doc.metadata.get('page', 0) for doc in documents)
        return sorted(list(pages))
    except Exception as e:
        st.error(f"Error reading PDF metadata: {str(e)}")
        return []

# ==================== MAIN UI ====================
st.title("📚 Chat With Your PDF (RAG App)")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        retrieval_k = st.slider("Retrieved Chunks", 1, 10, 3, help="Number of relevant chunks to retrieve")
    with col2:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.1, help="Control response creativity")
    
    st.divider()
    st.subheader("📊 Statistics")
    if st.session_state.chat_history:
        total_input = sum(count_tokens(msg["question"]) for msg in st.session_state.chat_history)
        total_output = sum(count_tokens(msg["answer"]) for msg in st.session_state.chat_history)
        st.metric("Total Questions", len(st.session_state.chat_history))
        st.metric("Approx. Tokens Used", total_input + total_output)
    else:
        st.info("No queries yet")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a PDF", type="pdf")

if uploaded_file:
    # Only reprocess if different file is uploaded
    if st.session_state.file_name != uploaded_file.name:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            with st.spinner("🔄 Processing document..."):
                # Load PDF
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                
                if not documents:
                    st.error("❌ No text found in PDF. Try a different file.")
                    st.stop()
                
                # Get page information
                pages = extract_pages_from_pdf(tmp_path)
                st.session_state.total_pages = len(pages) if pages else len(set(doc.metadata.get('page', 0) for doc in documents))

                # Split text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                chunks = text_splitter.split_documents(documents)

                # Create embeddings
                embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
                st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.file_name = uploaded_file.name
                st.session_state.chat_history = []  # Reset chat history for new file

                # Cleanup temp file
                os.unlink(tmp_path)

            st.success(f"✅ Document ready! {len(chunks)} chunks created.")

        except Exception as e:
            st.error(f"❌ Error processing PDF: {str(e)}")
            st.stop()
    
    # ==================== PAGE SELECTION ====================
    with st.expander("📄 Select specific pages (optional)", expanded=False):
        if st.session_state.total_pages and st.session_state.total_pages > 1:
            page_range = st.slider(
                "Pages to query from",
                0, st.session_state.total_pages - 1,
                (0, st.session_state.total_pages - 1),
                help="Narrow down which pages to search"
            )
            st.session_state.selected_pages = page_range
        else:
            st.info("Single page PDF - no filtering needed")

    # ==================== CHAT INTERFACE ====================
    st.divider()
    st.subheader("💬 Chat with your PDF")

    # Display chat history
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(msg["question"])
            with st.chat_message("assistant"):
                st.write(msg["answer"])
                with st.expander("📖 View sources"):
                    for i, source in enumerate(msg.get("sources", []), 1):
                        st.markdown(f"""
                        <div class="source-box">
                        <strong>Source {i}</strong> (Page {source['page']})<br>
                        {source['content'][:300]}...
                        </div>
                        """, unsafe_allow_html=True)

    # Query input
    query = st.chat_input("Ask a question about your PDF...", key="query_input")

    if query:
        try:
            # Add user message to history
            st.chat_message("user").write(query)
            
            # Retrieve relevant documents
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": retrieval_k})
            all_docs = retriever.get_relevant_documents(query)
            
            # Filter by page if selected
            if st.session_state.selected_pages:
                start_page, end_page = st.session_state.selected_pages
                docs = [doc for doc in all_docs if start_page <= doc.metadata.get('page', 0) <= end_page]
            else:
                docs = all_docs

            if not docs:
                st.warning("⚠️ No relevant content found. Try rephrasing your question.")
                st.stop()

            # Prepare context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Check token count
            context_tokens = count_tokens(context)
            if context_tokens > MAX_TOKENS_DISPLAY:
                st.warning(f"⚠️ Context is large (~{context_tokens} tokens). Response may be truncated.")

            # Create enhanced prompt
            system_prompt = """You are a helpful assistant that answers questions based on the provided PDF context. 
- Always cite the context when answering
- If the answer is not in the context, clearly say "I don't have enough information in the PDF to answer this question"
- Be concise but thorough
- Format your response with clear sections when appropriate"""

            prompt = f"""{system_prompt}

Context from PDF:
{context}

User Question: {query}

Answer:"""

            # Generate response with streaming
            llm = ChatOpenAI(model=MODEL, temperature=temperature, streaming=True)
            
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                
                # Stream the response
                for chunk in llm.stream(prompt):
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)

            # Calculate tokens
            query_tokens = count_tokens(query)
            answer_tokens = count_tokens(full_response)
            total_tokens = query_tokens + answer_tokens

            # Store in history
            st.session_state.chat_history.append({
                "question": query,
                "answer": full_response,
                "sources": [
                    {
                        "page": doc.metadata.get('page', 'Unknown'),
                        "content": doc.page_content
                    }
                    for doc in docs
                ],
                "tokens": f"~{total_tokens} tokens"
            })

            # Show token usage
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"Query tokens: ~{query_tokens}")
            with col2:
                st.caption(f"Answer tokens: ~{answer_tokens}")
            with col3:
                st.caption(f"Total: ~{total_tokens}")

            # Show sources
            with st.expander("📖 View source documents"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"""
                    <div class="source-box">
                    <strong>Source {i}</strong> (Page {doc.metadata.get('page', 'N/A')})<br>
                    {doc.page_content[:500]}...
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error generating answer: {str(e)}")

    # ==================== EXPORT & CLEAR ====================
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.chat_history:
            export_data = export_chat_history()
            st.download_button(
                label="📥 Export Chat History",
                data=export_data,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset Everything"):
            st.session_state.clear()
            st.success("App reset!")
            st.rerun()

else:
    st.info("👆 Please upload a PDF file to get started!")
    st.markdown("""
    ### How to use:
    1. **Upload** a PDF file using the uploader above
    2. **Configure** retrieval settings in the sidebar (optional)
    3. **Ask questions** about your PDF in the chat box
    4. **View sources** to see which parts of the PDF were used
    5. **Export** your chat history when done
    """)
