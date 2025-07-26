import streamlit as st
import logging
from datetime import datetime
import json
from rag_system import LangChainRAGSystem, GROQ_API_KEY, GROQ_MODEL_NAME, EMBEDDING_MODEL_NAME, DOCUMENTS_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="🗡️ Zoro - GitHub API Assistant",
    page_icon="🗡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'initialization_attempted' not in st.session_state:
    st.session_state.initialization_attempted = False

def auto_initialize_system():
    """Auto-initialize the system on app startup"""
    if not st.session_state.initialization_attempted and GROQ_API_KEY:
        st.session_state.initialization_attempted = True
        with st.spinner("🚀 Auto-initializing RAG system..."):
            try:
                rag_system = LangChainRAGSystem()
                if not rag_system.initialize_components(GROQ_API_KEY):
                    st.error("Failed to auto-initialize components!")
                    return False
                if not rag_system.create_vectorstore(DOCUMENTS_FOLDER):
                    st.error("Failed to create vectorstore!")
                    return False
                if not rag_system.setup_qa_chain():
                    st.error("Failed to setup QA chain!")
                    return False
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success("✅ System auto-initialized successfully!")
                return True
            except Exception as e:
                st.error(f"Auto-initialization failed: {str(e)}")
                return False
    return st.session_state.initialized

def main():
    st.title("🗡️ Zoro - GitHub API Assistant")
    st.markdown("*Enhanced with Step-Back Prompting, Auto-initialization, and Detailed Chunk Inspection*")
    auto_initialize_system()
    with st.sidebar:
        st.header("⚙️ Configuration")
        if GROQ_API_KEY:
            st.success("✅ Groq API Key loaded from .env file")
        else:
            st.error("❌ No API key found in .env file")
            st.info("💡 Create a .env file in the project root with: GROQ_API_KEY=your_key_here")
            st.stop()
        st.markdown("---")
        st.header("🚀 System Setup")
        if st.button("Re-initialize System", type="secondary"):
            if not groq_api_key:
                st.error("Please provide your Groq API key first!")
                return
            with st.spinner("Re-initializing RAG system..."):
                try:
                    rag_system = LangChainRAGSystem()
                    if not rag_system.initialize_components(groq_api_key):
                        st.error("Failed to initialize components!")
                        return
                    if not rag_system.create_vectorstore(DOCUMENTS_FOLDER):
                        st.error("Failed to create vectorstore!")
                        return
                    if not rag_system.setup_qa_chain():
                        st.error("Failed to setup QA chain!")
                        return
                    st.session_state.rag_system = rag_system
                    st.session_state.initialized = True
                    st.success("✅ System re-initialized successfully!")
                except Exception as e:
                    st.error(f"Re-initialization failed: {str(e)}")
        if st.session_state.initialized:
            st.success("🟢 System Ready")
            st.info("💡 Step-back prompting enabled")
        else:
            st.warning("🟡 System Not Initialized")
        st.markdown("---")
        if st.button("Clear Conversation"):
            if st.session_state.rag_system:
                st.session_state.rag_system.clear_memory()
            st.session_state.conversation_history = []
            st.success("Conversation cleared!")
    if not st.session_state.initialized:
        if not GROQ_API_KEY:
            st.info("👈 Please create a .env file with your Groq API key or enter it in the sidebar.")
            st.code("GROQ_API_KEY=your_actual_api_key_here", language="bash")
        else:
            st.info("🔄 System initialization in progress or failed. Check the sidebar for status.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 System Configuration")
            st.write(f"**Embedding Model:** {EMBEDDING_MODEL_NAME}")
            st.write(f"**Documents Folder:** {DOCUMENTS_FOLDER}")
            st.write(f"**Chunk Size:** {CHUNK_SIZE}")
            st.write(f"**Chunk Overlap:** {CHUNK_OVERLAP}")
        with col2:
            st.subheader("📝 Enhanced Features")
            st.write("✅ LangChain Framework")
            st.write("✅ Groq LLM Integration")
            st.write("✅ ChromaDB Vector Store")
            st.write("✅ Step-Back Prompting")
            st.write("✅ Auto-Initialization")
            st.write("✅ Chunk Inspection")
            st.write("✅ Conversation Memory")
            st.write("✅ Evaluation System")
        return
    st.header("💬 Chat with Zoro")
    # Suggested questions for the user
    st.info("**Try asking one of these questions:**\n\n"
            "1. How do I authenticate with the GitHub API?\n"
            "2. How can I list all repositories for a user?\n"
            "3. What should I do if I get a 404 error from the API?\n"
            "4. How do webhooks work in GitHub?", icon="💡")
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                if "step_back_question" in message:
                    with st.expander("🔄 Step-Back Question Used"):
                        st.write(message["step_back_question"])
                if "retrieved_chunks" in message:
                    with st.expander("📄 Retrieved Document Chunks"):
                        for i, chunk in enumerate(message["retrieved_chunks"]):
                            chunk_type = "🎯 Original Query" if chunk["type"] == "original" else "🔄 Step-Back"
                            st.write(f"**{chunk_type} - Source: {chunk['source']}**")
                            st.write(chunk["content"])
                            if i < len(message["retrieved_chunks"]) - 1:
                                st.markdown("---")
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"• {source}")
    if prompt := st.chat_input("Ask me anything about GitHub API..."):
        st.session_state.conversation_history.append({
            "role": "user",
            "content": prompt
        })
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking with step-back reasoning..."):
                try:
                    response_data = st.session_state.rag_system.get_response(prompt)
                    st.write(response_data["answer"])
                    assistant_message = {
                        "role": "assistant",
                        "content": response_data["answer"],
                        "sources": response_data.get("sources", []),
                        "step_back_question": response_data.get("step_back_question", ""),
                        "retrieved_chunks": response_data.get("retrieved_chunks", [])
                    }
                    st.session_state.conversation_history.append(assistant_message)
                    if response_data.get("step_back_question"):
                        with st.expander("🔄 Step-Back Question Used"):
                            st.write(response_data["step_back_question"])
                    if response_data.get("retrieved_chunks"):
                        with st.expander("📄 Retrieved Document Chunks"):
                            for i, chunk in enumerate(response_data["retrieved_chunks"]):
                                chunk_type = "🎯 Original Query" if chunk["type"] == "original" else "🔄 Step-Back"
                                st.write(f"**{chunk_type} - Source: {chunk['source']}**")
                                st.write(chunk["content"])
                                if i < len(response_data["retrieved_chunks"]) - 1:
                                    st.markdown("---")
                    sources = response_data.get("sources", [])
                    if sources:
                        with st.expander("📚 Sources"):
                            for source in sources:
                                st.write(f"• {source}")
                except Exception as e:
                    error_message = f"I apologize, but I encountered an error: {str(e)}"
                    st.error(error_message)
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": error_message
                    })
    st.markdown("---")
    st.header("📊 System Evaluation")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Evaluate the RAG system performance using predefined Q&A pairs with ground truth answers.")
    with col2:
        if st.button("🧪 Run Evaluation", type="primary"):
            if st.session_state.rag_system:
                with st.spinner("Running comprehensive evaluation..."):
                    try:
                        evaluation_results = st.session_state.rag_system.run_evaluation()
                        st.subheader("📈 Aggregate Metrics")
                        if "error" not in evaluation_results["aggregate_metrics"]:
                            metrics = evaluation_results["aggregate_metrics"]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Average F1 Score",
                                    f"{metrics['f1_score']['mean']:.3f}",
                                    f"±{metrics['f1_score']['std']:.3f}"
                                )
                            with col2:
                                st.metric(
                                    "Average ROUGE-1 F1",
                                    f"{metrics['rouge1_f']['mean']:.3f}",
                                    f"±{metrics['rouge1_f']['std']:.3f}"
                                )
                            with col3:
                                st.metric(
                                    "Keyword Coverage",
                                    f"{metrics['keyword_coverage']['mean']:.3f}",
                                    f"±{metrics['keyword_coverage']['std']:.3f}"
                                )
                            success_rate = evaluation_results["successful_evaluations"] / evaluation_results["total_questions"]
                            st.metric("Success Rate", f"{success_rate:.1%}")
                        st.subheader("📋 Individual Results")
                        for i, result in enumerate(evaluation_results["individual_results"]):
                            if "error" not in result:
                                with st.expander(f"Q{i+1}: {result['question'][:100]}..."):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**Metrics:**")
                                        st.write(f"F1 Score: {result['f1_metrics']['f1']:.3f}")
                                        st.write(f"ROUGE-1 F1: {result['rouge_scores']['rouge1_f']:.3f}")
                                        st.write(f"Keyword Coverage: {result['keyword_coverage']:.3f}")
                                    with col2:
                                        st.write("**Keywords:**")
                                        st.write(f"Found: {len(result['found_keywords'])}/{len(result['expected_keywords'])}")
                                        if result['found_keywords']:
                                            st.write("✅ " + ", ".join(result['found_keywords']))
                                    st.write("**System Response:**")
                                    st.write(result['predicted_response'])
                                    st.write("**Ground Truth:**")
                                    st.write(result['ground_truth'])
                                    if "rag_metadata" in result:
                                        st.write("**RAG Metadata:**")
                                        metadata = result["rag_metadata"]
                                        st.write(f"Sources used: {len(metadata.get('sources', []))}")
                                        st.write(f"Documents retrieved: {metadata.get('source_documents', 0)}")
                                        if metadata.get('step_back_question'):
                                            st.write(f"Step-back question: {metadata['step_back_question']}")
                            else:
                                with st.expander(f"❌ Q{i+1}: Error occurred"):
                                    st.error(result.get("error", "Unknown error"))
                        results_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        try:
                            with open(results_file, 'w') as f:
                                json.dump(evaluation_results, f, indent=2)
                            st.success(f"📁 Evaluation results saved to: {results_file}")
                        except Exception as e:
                            st.warning(f"Could not save results to file: {e}")
                    except Exception as e:
                        st.error(f"Evaluation failed: {str(e)}")
            else:
                st.error("Please initialize the system first!")
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🗡️ Zoro - GitHub API Assistant | Enhanced with LangChain & Step-Back Prompting<br>
        Created by Balaji | Powered by Groq & ChromaDB
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()