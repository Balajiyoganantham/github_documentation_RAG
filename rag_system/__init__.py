from .config import *
from .embeddings import get_embeddings
from .vectorstore import load_documents, create_vectorstore
from .memory import get_conversation_memory
from .prompts import get_custom_prompt
from .conversational_chain import setup_conversational_chain
from .response import calculate_confidence
from .stats import get_conversation_stats, get_memory_summary

import logging
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from evaluation import EvaluationDataset, RAGEvaluator
from datetime import datetime

logger = logging.getLogger(__name__)

def sliding_window_chunks(text, chunk_size, chunk_overlap):
    """Custom sliding window chunker using RecursiveCharacterTextSplitter for base splitting."""
    # First, split into base chunks (no overlap, large size)
    base_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    base_chunks = base_splitter.split_text(text)
    # Now, apply sliding window over the base chunks
    windowed_chunks = []
    i = 0
    while i < len(base_chunks):
        window = ''
        j = i
        chars = 0
        while j < len(base_chunks) and chars + len(base_chunks[j]) <= chunk_size:
            window += base_chunks[j]
            chars += len(base_chunks[j])
            j += 1
        windowed_chunks.append(window)
        # Move window forward by chunk_size - chunk_overlap
        chars_advanced = 0
        while i < len(base_chunks) and chars_advanced < (chunk_size - chunk_overlap):
            chars_advanced += len(base_chunks[i])
            i += 1
    return windowed_chunks

class EnhancedRAGSystem:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.conversational_chain = None
        self.memory = None
        self.text_splitter = None
        self.evaluator = None
        self.evaluation_dataset = EvaluationDataset()
        self.conversation_history = []
        self.response_times = []
        self.confidence_scores = []

    def initialize_components(self, groq_api_key: str = None):
        try:
            api_key = groq_api_key or GROQ_API_KEY
            if not api_key:
                raise ValueError("GROQ_API_KEY not found. Please check your .env file.")
            self.embeddings = get_embeddings()
            self.llm = ChatGroq(
                groq_api_key=api_key,
                model_name=GROQ_MODEL_NAME,
                temperature=0.05,
                max_tokens=1500
            )
            # Use a custom sliding window chunker based on RecursiveCharacterTextSplitter
            self.text_splitter = lambda text: sliding_window_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
            self.memory = get_conversation_memory(k=10)
            logger.info("All components initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False

    def create_vectorstore(self, folder_path: str):
        try:
            documents = load_documents(folder_path)
            if not documents:
                logger.warning("No documents found to process")
                return False
            self.vectorstore = create_vectorstore(documents, self.text_splitter, self.embeddings)
            return True
        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            return False

    def setup_conversational_chain(self):
        try:
            custom_prompt = get_custom_prompt()
            self.conversational_chain = setup_conversational_chain(
                self.llm, self.vectorstore, self.memory, custom_prompt
            )
            return self.conversational_chain is not None
        except Exception as e:
            logger.error(f"Error setting up conversational chain: {e}")
            return False

    def get_response(self, question: str):
        start_time = datetime.now()
        try:
            result = self.conversational_chain({"question": question})
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self.response_times.append(response_time)
            answer = result.get("answer", "")
            source_documents = result.get("source_documents", [])
            confidence = calculate_confidence(question, answer, source_documents)
            self.confidence_scores.append(confidence)
            sources = list(set([
                doc.metadata.get('source_file', 'Unknown') 
                for doc in source_documents
            ]))
            self.conversation_history.append({
                'question': question,
                'answer': answer,
                'timestamp': datetime.now().isoformat(),
                'sources': sources,
                'confidence': confidence,
                'response_time': response_time
            })
            return {
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
                "source_documents": len(source_documents),
                "response_time": response_time,
                "retrieved_chunks": [
                    {
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "source": doc.metadata.get('source_file', 'Unknown'),
                        "relevance_score": getattr(doc, 'relevance_score', 0.0)
                    }
                    for doc in source_documents
                ],
                "memory_context": len(self.memory.chat_memory.messages)
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "source_documents": 0,
                "response_time": 0.0,
                "retrieved_chunks": [],
                "memory_context": 0
            }

    def get_conversation_stats(self):
        return get_conversation_stats(self.conversation_history, self.confidence_scores, self.response_times, self.memory)

    def clear_memory(self):
        if self.memory:
            self.memory.clear()
        self.conversation_history = []
        self.response_times = []
        self.confidence_scores = []
        logger.info("Memory and conversation history cleared")

    def get_memory_summary(self):
        return get_memory_summary(self.memory)

    def initialize_evaluator(self):
        self.evaluator = RAGEvaluator(self)
        logger.info("Evaluation system initialized")

    def run_evaluation(self):
        if not self.evaluator:
            self.initialize_evaluator()
        qa_pairs = self.evaluation_dataset.get_qa_pairs()
        results = []
        for i, qa_pair in enumerate(qa_pairs):
            question = qa_pair["question"]
            try:
                rag_response = self.get_response(question)
                predicted_response = rag_response["answer"]
                evaluation_result = self.evaluator.evaluate_response(
                    question, predicted_response, qa_pair
                )
                evaluation_result["rag_metadata"] = {
                    "sources": rag_response.get("sources", []),
                    "source_documents": rag_response.get("source_documents", 0),
                    "confidence": rag_response.get("confidence", 0.0),
                    "response_time": rag_response.get("response_time", 0.0)
                }
                results.append(evaluation_result)
            except Exception as e:
                logger.error(f"Error evaluating question '{question}': {e}")
                results.append({
                    "question": question,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        valid_results = [r for r in results if "error" not in r]
        import statistics
        if valid_results:
            f1_scores = [r["f1_metrics"]["f1"] for r in valid_results]
            rouge1_scores = [r["rouge_scores"]["rouge1_f"] for r in valid_results]
            keyword_coverages = [r["keyword_coverage"] for r in valid_results]
            aggregate_metrics = {
                "f1_score": {
                    "mean": statistics.mean(f1_scores),
                    "std": statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0,
                    "min": min(f1_scores),
                    "max": max(f1_scores)
                },
                "rouge1_f": {
                    "mean": statistics.mean(rouge1_scores),
                    "std": statistics.stdev(rouge1_scores) if len(rouge1_scores) > 1 else 0.0
                },
                "keyword_coverage": {
                    "mean": statistics.mean(keyword_coverages),
                    "std": statistics.stdev(keyword_coverages) if len(keyword_coverages) > 1 else 0.0
                }
            }
        else:
            aggregate_metrics = {"error": "No valid results to aggregate"}
        return {
            "total_questions": len(qa_pairs),
            "successful_evaluations": len(valid_results),
            "failed_evaluations": len([r for r in results if "error" in r]),
            "aggregate_metrics": aggregate_metrics,
            "individual_results": results,
            "evaluation_timestamp": datetime.now().isoformat()
        } 