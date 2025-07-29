# 🗡️ Zoro - GitHub API Assistant (v2)

A modular, production-grade RAG (Retrieval-Augmented Generation) system for GitHub API documentation, built with LangChain, Groq, ChromaDB, and Streamlit. Now with custom sliding window chunking and a ChatGPT-style conversational UI.

---

## 🚀 Features
- **Modular RAG System**: Clean separation of concerns for embeddings, vectorstore, memory, prompts, and evaluation
- **Custom Sliding Window Chunking**: Robust, context-preserving chunking for all documents
- **Fast LLM Integration**: Uses Groq's Llama3-70B for rapid, high-quality responses
- **Vector Database**: ChromaDB for efficient semantic retrieval
- **Conversational Memory**: Maintains context across chat sessions
- **Evaluation System**: Built-in metrics (F1, ROUGE, keyword coverage)
- **Modern UI**: Streamlit app with ChatGPT-style chat experience and analytics

---

## 🛠️ Technology Stack
- **Framework**: LangChain
- **LLM**: Groq (llama3-70b-8192)
- **Vector Store**: ChromaDB
- **Embeddings**: BAAI/bge-large-en-v1.5
- **Frontend**: Streamlit
- **Evaluation**: ROUGE, F1 Score, Keyword Coverage

---

## 📋 Prerequisites
- Python 3.8+
- Groq API key
- Git

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Balajiyoganantham/github_documentation_RAG.git
cd github_documentation_RAG
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```
github_documentation_RAG/
├── app.py                 # Streamlit UI (ChatGPT-style)
├── rag_system/            # Modular RAG system components
│   ├── __init__.py        # Main RAG system class (modular)
│   ├── config.py          # Configuration constants
│   ├── embeddings.py      # Embedding model loader
│   ├── vectorstore.py     # Document loading & vectorstore logic
│   ├── memory.py          # Conversation memory setup
│   ├── prompts.py         # Prompt template logic
│   ├── conversational_chain.py # Conversational chain setup
│   ├── response.py        # Confidence scoring
│   ├── stats.py           # Stats & memory summary
├── evaluation.py          # Evaluation and testing framework
├── documents/             # GitHub API documentation files
├── chroma_db/             # Vector database storage
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── README.md              # This file
```

---

## 🔧 Configuration
- **Chunk Size**: 400 characters
- **Chunk Overlap**: 200 characters (sliding window)
- **Model**: llama3-70b-8192
- **Embeddings**: BAAI/bge-large-en-v1.5
- All settings in `rag_system/config.py`

---

## 🧩 Architecture
- **Modularized**: Each RAG component is in its own file for maintainability and extensibility.
- **Custom Chunking**: Uses a custom sliding window chunker (based on RecursiveCharacterTextSplitter) for robust, context-preserving document splitting.
- **ChatGPT-style UI**: Chat input at the bottom, conversation history above, real-time metrics, and expandable source/context sections.

---

## 🧪 Evaluation
- **F1 Score**: Answer accuracy and completeness
- **ROUGE Metrics**: Text similarity and overlap
- **Keyword Coverage**: Important term identification
- **Success Rate**: Overall system reliability

Run evaluation from the Streamlit interface or use:
```bash
python evaluation.py
```

---

## 🎯 Use Cases
- **GitHub API Documentation**: Get instant answers about GitHub API endpoints
- **Code Examples**: Retrieve relevant code snippets and examples
- **Authentication Help**: Understand OAuth flows and token management
- **Error Troubleshooting**: Get guidance on common API issues
- **Best Practices**: Learn recommended patterns and approaches

---

## 🔒 Security
- API keys are stored securely in `.env` files
- `.env` files are excluded from version control
- No sensitive data is exposed in the frontend

---

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author
**Balaji** - *GitHub API Documentation RAG System*

---

## 🙏 Acknowledgments
- LangChain team for the excellent framework
- Groq for lightning-fast LLM inference
- ChromaDB for efficient vector storage
- Streamlit for the beautiful UI framework

---

⭐ **Star this repository if you find it helpful!** 