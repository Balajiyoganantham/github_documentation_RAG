# rag_system.py
# Enhanced RAG system with conversational memory and improved retrieval

import os
import glob
import logging
from datetime import datetime
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from evaluation import EvaluationDataset, RAGEvaluator
import statistics
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration constants
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL_NAME = "llama3-70b-8192"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DOCUMENTS_FOLDER = "documents"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 200
COLLECTION_NAME = "github_api_docs"

logger = logging.getLogger(__name__)

class EnhancedRAGSystem:
    """Enhanced RAG System with Conversational Memory using LangChain"""
    
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
        """Initialize all RAG components"""
        try:
            api_key = groq_api_key or GROQ_API_KEY
            if not api_key:
                raise ValueError("GROQ_API_KEY not found. Please check your .env file.")
            
            # Initialize embeddings with fallback
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL_NAME,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                logger.warning(f"Failed to load {EMBEDDING_MODEL_NAME}, using fallback: {e}")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            
            # Initialize Groq LLM
            self.llm = ChatGroq(
                groq_api_key=api_key,
                model_name=GROQ_MODEL_NAME,
                temperature=0.05,
                max_tokens=1500
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Initialize conversation memory with window
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=10,  # Remember last 10 exchanges
                output_key='answer'
            )
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
    
    def load_documents(self, folder_path: str) -> List[Document]:
        """Load documents from folder"""
        documents = []
        if not os.path.exists(folder_path):
            logger.error(f"Documents folder not found: {folder_path}")
            return documents
            
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        for file_path in txt_files:
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                for doc in docs:
                    doc.metadata.update({
                        'source_file': os.path.basename(file_path),
                        'source_path': file_path,
                        'loaded_at': datetime.now().isoformat()
                    })
                    documents.extend([doc])
                logger.info(f"Loaded document: {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        return documents
    
    def create_vectorstore(self, folder_path: str):
        """Create and persist vector store"""
        try:
            documents = self.load_documents(folder_path)
            if not documents:
                logger.warning("No documents found to process")
                return False
                
            texts = self.text_splitter.split_documents(documents)
            
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory="./chroma_db"
            )
            
            self.vectorstore.persist()
            logger.info(f"Created vectorstore with {len(texts)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            return False
    
    def setup_conversational_chain(self):
        """Setup conversational retrieval chain"""
        try:
            # Custom prompt template for GitHub API assistant
            custom_template = """You are Zoro, an expert GitHub API assistant created by Balaji. 
            You have access to comprehensive GitHub API documentation and conversation history.

            INSTRUCTIONS:
            1. Use EXACT terminology from the context (endpoints, parameters, status codes)
            2. Include specific API endpoints like "GET /user/repos" or "POST /repos"
            3. Reference previous conversation when relevant
            4. Mention authentication requirements when relevant
            5. Include query parameters and HTTP methods
            6. Be concise but comprehensive
            7. If information is not in context, say "I don't have that information in my knowledge base"
            8. Remember previous questions and build upon them

            Context: {context}
            Chat History: {chat_history}
            Question: {question}

            Answer (be specific and reference previous conversation when relevant):"""

            # Create custom prompt
            custom_prompt = PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=custom_template
            )
            
            # Setup conversational retrieval chain
            self.conversational_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 8}
                ),
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": custom_prompt},
                verbose=True
            )
            
            logger.info("Conversational chain setup successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up conversational chain: {e}")
            return False
    
    def get_response(self, question: str) -> Dict[str, Any]:
        """Get response with conversational memory"""
        start_time = datetime.now()
        
        try:
            # Get response from conversational chain
            result = self.conversational_chain({
                "question": question
            })
            
            # Calculate response time
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self.response_times.append(response_time)
            
            # Extract information
            answer = result.get("answer", "")
            source_documents = result.get("source_documents", [])
            
            # Calculate confidence score based on various factors
            confidence = self._calculate_confidence(question, answer, source_documents)
            self.confidence_scores.append(confidence)
            
            # Extract sources
            sources = list(set([
                doc.metadata.get('source_file', 'Unknown') 
                for doc in source_documents
            ]))
            
            # Store conversation
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
    
    def _calculate_confidence(self, question: str, answer: str, source_documents: List) -> float:
        """Calculate confidence score based on multiple factors"""
        try:
            confidence = 0.0
            
            # Factor 1: Number of source documents (0-0.3)
            doc_count = len(source_documents)
            if doc_count >= 5:
                confidence += 0.3
            elif doc_count >= 3:
                confidence += 0.2
            elif doc_count >= 1:
                confidence += 0.1
            
            # Factor 2: Answer length and detail (0-0.2)
            if len(answer) > 200 and "I don't" not in answer:
                confidence += 0.2
            elif len(answer) > 100:
                confidence += 0.1
            
            # Factor 3: Specific API terms present (0-0.3)
            api_terms = ['GET ', 'POST ', 'PUT ', 'DELETE ', '/api/', 'endpoint', 'parameter', 'header', 'token']
            api_terms_found = sum(1 for term in api_terms if term.lower() in answer.lower())
            confidence += min(0.3, api_terms_found * 0.05)
            
            # Factor 4: No uncertainty phrases (0-0.2)
            uncertainty_phrases = ["I don't know", "I'm not sure", "I don't have", "unclear"]
            if not any(phrase.lower() in answer.lower() for phrase in uncertainty_phrases):
                confidence += 0.2
            
            return min(1.0, confidence)
            
        except Exception:
            return 0.5
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        if not self.conversation_history:
            return {}
        
        return {
            "total_questions": len(self.conversation_history),
            "avg_confidence": statistics.mean(self.confidence_scores) if self.confidence_scores else 0,
            "avg_response_time": statistics.mean(self.response_times) if self.response_times else 0,
            "memory_size": len(self.memory.chat_memory.messages),
            "unique_sources": len(set([
                source for conv in self.conversation_history 
                for source in conv.get('sources', [])
            ]))
        }
    
    def clear_memory(self):
        """Clear conversation memory and history"""
        if self.memory:
            self.memory.clear()
        self.conversation_history = []
        self.response_times = []
        self.confidence_scores = []
        logger.info("Memory and conversation history cleared")
    
    def get_memory_summary(self) -> str:
        """Get a summary of current conversation memory"""
        if not self.memory or not self.memory.chat_memory.messages:
            return "No conversation history"
        
        messages = self.memory.chat_memory.messages
        return f"Conversation contains {len(messages)} messages. Recent topics discussed: {', '.join([msg.content[:50] + '...' for msg in messages[-3:] if hasattr(msg, 'content')])}"
    
    def initialize_evaluator(self):
        """Initialize evaluation system"""
        self.evaluator = RAGEvaluator(self)
        logger.info("Evaluation system initialized")
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
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
        
        # Calculate aggregate metrics
        valid_results = [r for r in results if "error" not in r]
        
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