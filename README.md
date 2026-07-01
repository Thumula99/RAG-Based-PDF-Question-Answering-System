---
title: RAG-Based PDF Question-Answering System
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.28.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 📚 RAG-Based PDF Question-Answering System

A powerful Retrieval-Augmented Generation (RAG) application with an intuitive web UI that allows users to upload PDFs and ask questions using advanced LLM capabilities with semantic vector search.

## ✨ Features

### Core RAG Features
- 🔍 **Intelligent Document Retrieval** - Semantic search using FAISS vector store and OpenAI embeddings
- 🤖 **AI-Powered Responses** - GPT-3.5 Turbo with configurable temperature and retrieval parameters
- 📄 **Multi-Document Support** - Upload and query any PDF file
- ⚡ **Real-time Streaming** - Watch responses generate in real-time with streaming output
- 🎯 **Source Attribution** - View exact PDF excerpts used to generate answers

### Advanced Features
- 💬 **Chat History** - Persistent conversation history within the session
- 📊 **Token Counting** - Approximate token usage tracking for cost management
- 📥 **Export Functionality** - Download chat history as formatted text files
- 📄 **Page Selection** - Filter queries to specific page ranges (optional)
- ⚙️ **Configurable Settings** - Adjust retrieval parameters and model temperature on-the-fly
- 🛡️ **Error Handling** - Robust error management with user-friendly messages
- 💾 **Session Caching** - Automatic caching to avoid reprocessing PDFs
- 📱 **Responsive UI** - Clean, modern interface built with Streamlit

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API Key

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Thumula99/RAG-Based-PDF-Question-Answering-System.git
cd RAG-Based-PDF-Question-Answering-System
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-openai-api-key
```

4. **Run the application:**
```bash
streamlit run app.py
```

5. **Open in browser:**
```
http://localhost:8501
```

## 📖 Usage Guide

### Basic Workflow

1. **Upload PDF**: Click the file uploader to select your PDF file
   - Supported format: PDF only
   - No file size limits enforced

2. **Configure Settings** (Optional):
   - Adjust **Retrieved Chunks** (1-10) - number of relevant sections to fetch
   - Adjust **Temperature** (0.0-1.0) - control response creativity (0 = deterministic, 1 = creative)

3. **Ask Questions**: Type your question in the chat input box
   - Questions are automatically processed through the RAG pipeline
   - Real-time streaming shows the response as it's generated

4. **View Results**:
   - **Answer**: Main response from the LLM
   - **Source Documents**: Expandable section showing referenced PDF excerpts with page numbers
   - **Token Usage**: Approximate tokens consumed for the query

5. **Manage Conversation**:
   - View full chat history in the main interface
   - **Export Chat History**: Download conversation as a `.txt` file
   - **Clear Chat History**: Reset conversation for current PDF
   - **Reset Everything**: Clear all sessions and start fresh

### Advanced Options

#### Page Selection
- Expand **"Select specific pages"** in the left sidebar
- Use the slider to choose page range (if PDF has multiple pages)
- Queries will only search within selected pages

#### Session Statistics
- Sidebar shows real-time statistics:
   - Total questions asked
   - Approximate total tokens used
   - Helps track API usage and costs

## 🏗️ Architecture

### Technology Stack
```
Frontend: Streamlit (Web UI)
LLM: OpenAI GPT-3.5 Turbo
Embeddings: OpenAI text-embedding-3-small
Vector Store: FAISS (Facebook AI Similarity Search)
Document Processing: LangChain PyPDFLoader
Text Splitting: RecursiveCharacterTextSplitter
```

### RAG Pipeline Flow

```
[PDF Upload]
     ↓
[PyPDFLoader] → Extract text & metadata
     ↓
[RecursiveCharacterTextSplitter] → Split into 1000-char chunks (200 overlap)
     ↓
[OpenAI Embeddings] → Convert chunks to vectors
     ↓
[FAISS Vector Store] → Index and store embeddings
     ↓
[User Query]
     ↓
[Semantic Search] → Retrieve top-k relevant chunks
     ↓
[RAG Prompt] → Combine context + question
     ↓
[GPT-3.5 Turbo] → Generate contextual answer
     ↓
[Stream Response] → Display answer in real-time
```

## ⚙️ Configuration

### Customizable Parameters

Edit these constants in `app.py` to adjust behavior:

```python
MODEL = "gpt-3.5-turbo"              # LLM model
EMBEDDING_MODEL = "text-embedding-3-small"  # Embedding model
CHUNK_SIZE = 1000                    # Characters per chunk
CHUNK_OVERLAP = 200                  # Overlap between chunks
MAX_TOKENS_DISPLAY = 4000            # Warning threshold for large contexts
```

### Sidebar Controls

| Setting | Range | Default | Purpose |
|---------|-------|---------|---------|
| Retrieved Chunks | 1-10 | 3 | Number of relevant sections to fetch |
| Temperature | 0.0-1.0 | 0.0 | Response creativity (0=deterministic) |

## 📊 Token Usage & Costs

The app estimates token usage:
- **Query tokens**: ~1 token per 4 characters
- **Answer tokens**: ~1 token per 4 characters
- Total shown in chat interface

### Cost Estimation (as of Feb 2025)
- GPT-3.5 Turbo: $0.0005 / 1K input tokens, $0.0015 / 1K output tokens
- Embeddings: $0.02 / 1M tokens

Monitor usage in the sidebar statistics panel.

## 📁 Project Structure

```
RAG-Based-PDF-Question-Answering-System/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── .env.example           # Environment variables template
```

## 🔒 Security & Best Practices

- ✅ **API Key Validation**: Application checks for OPENAI_API_KEY at startup
- ✅ **Temp File Cleanup**: Temporary PDF files are automatically deleted after processing
- ✅ **Session Caching**: Avoids reprocessing PDFs unnecessarily
- ✅ **Error Handling**: Comprehensive try-catch blocks with user-friendly error messages
- ✅ **No Data Persistence**: Chat history is session-only (cleared on app reset)

## 🐛 Troubleshooting

### "OPENAI_API_KEY not configured"
**Solution**: Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

### "No text found in PDF"
**Solution**: Ensure your PDF contains selectable text (not scanned image). OCR is not currently supported.

### "No relevant content found"
**Solution**: 
- Try rephrasing your question
- Expand "Retrieved Chunks" in sidebar (increase k value)
- Check that your question relates to PDF content

### Slow Response Time
**Solution**:
- Reduce "Retrieved Chunks" (fewer documents to process)
- Use a smaller PDF or select specific pages
- Check OpenAI API status

### Memory Issues with Large PDFs
**Solution**:
- Split large PDFs into smaller files
- Use "Select specific pages" to query portions
- Restart the app and process one PDF at a time

## 🚀 Future Enhancements

- [ ] Support for multiple document types (DOCX, TXT, images with OCR)
- [ ] Custom prompt templates
- [ ] Multiple LLM model selection
- [ ] Persistent database for long-term chat history
- [ ] User authentication and session management
- [ ] Advanced filtering (by date, author, document section)
- [ ] Summarization features
- [ ] Citation formatting (APA, MLA, Chicago)
- [ ] Batch processing multiple PDFs
- [ ] Integration with other vector stores (Pinecone, Weaviate)

## 📝 Example Use Cases

1. **Research Papers**: Quickly find key insights from academic papers
2. **Legal Documents**: Extract relevant clauses and information
3. **Technical Manuals**: Get specific instructions without reading entire docs
4. **Financial Reports**: Query earnings, metrics, and insights
5. **Meeting Notes**: Search meeting transcripts for decisions and action items

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 💬 Support

For issues, questions, or feature requests, please open an issue on GitHub.

## 👨‍💻 Author

Created by [Thumula99](https://github.com/Thumula99)

---

**Last Updated**: February 2026
**Status**: Active Development ✨
