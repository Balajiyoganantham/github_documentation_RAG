# 🗡️ Zoro - GitHub API Assistant

A sophisticated RAG (Retrieval-Augmented Generation) system built with LangChain, Groq, and ChromaDB to provide intelligent assistance for GitHub API documentation.

## 🚀 Features

- **Advanced RAG System**: Powered by LangChain framework with step-back prompting
- **Fast LLM Integration**: Uses Groq's lightning-fast LLM inference
- **Vector Database**: ChromaDB for efficient document retrieval
- **Step-Back Prompting**: Enhanced reasoning capabilities for complex queries
- **Auto-Initialization**: Seamless system setup and configuration
- **Conversation Memory**: Maintains context across chat sessions
- **Evaluation System**: Comprehensive performance metrics and testing
- **Modern UI**: Beautiful Streamlit interface with real-time feedback

## 🛠️ Technology Stack

- **Framework**: LangChain
- **LLM**: Groq (llama3-70b-8192)
- **Vector Store**: ChromaDB
- **Embeddings**: BAAI/bge-large-en-v1.5
- **Frontend**: Streamlit
- **Evaluation**: ROUGE, F1 Score, Keyword Coverage

## 📋 Prerequisites

- Python 3.8+
- Groq API key
- Git

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

## 📁 Project Structure

```
github_documentation_RAG/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings
├── rag_system.py         # Core RAG system implementation
├── evaluation.py         # Evaluation and testing framework
├── prompt_generator.py   # Step-back prompting logic
├── documents/            # GitHub API documentation files
├── chroma_db/           # Vector database storage
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
└── README.md           # This file
```

## 🔧 Configuration

The system is configured through `config.py` with the following key settings:

- **Chunk Size**: 400 tokens for focused document chunks
- **Chunk Overlap**: 200 tokens for context preservation
- **Model**: llama3-70b-8192 for optimal performance
- **Embeddings**: BAAI/bge-large-en-v1.5 for semantic search

## 🧪 Evaluation

The system includes a comprehensive evaluation framework that tests:

- **F1 Score**: Answer accuracy and completeness
- **ROUGE Metrics**: Text similarity and overlap
- **Keyword Coverage**: Important term identification
- **Success Rate**: Overall system reliability

Run evaluation from the Streamlit interface or use:
```bash
python evaluation.py
```

## 🎯 Use Cases

- **GitHub API Documentation**: Get instant answers about GitHub API endpoints
- **Code Examples**: Retrieve relevant code snippets and examples
- **Authentication Help**: Understand OAuth flows and token management
- **Error Troubleshooting**: Get guidance on common API issues
- **Best Practices**: Learn recommended patterns and approaches

## 🔒 Security

- API keys are stored securely in `.env` files
- `.env` files are excluded from version control
- No sensitive data is exposed in the frontend

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Balaji** - *GitHub API Documentation RAG System*

## 🙏 Acknowledgments

- LangChain team for the excellent framework
- Groq for lightning-fast LLM inference
- ChromaDB for efficient vector storage
- Streamlit for the beautiful UI framework

---

⭐ **Star this repository if you find it helpful!** 