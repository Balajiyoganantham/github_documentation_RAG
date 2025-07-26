# rag_system.py
# Handles the core RAG system logic for the Streamlit RAG app

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
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from evaluation import EvaluationDataset, RAGEvaluator
import statistics
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration constants
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
# Don't raise error at import time - let the class handle it

GROQ_MODEL_NAME = "llama3-70b-8192"  # More efficient model
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"  # Reverted to working model
DOCUMENTS_FOLDER = "documents"
CHUNK_SIZE = 400  # Reduced for more focused chunks - better for keyword matching
CHUNK_OVERLAP = 200  # Increased overlap for better context preservation
COLLECTION_NAME = "github_api_docs"

logger = logging.getLogger(__name__)

class LangChainRAGSystem:
    """RAG System using LangChain with Groq and ChromaDB"""
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.memory = None
        self.text_splitter = None
        self.evaluator = None
        self.evaluation_dataset = EvaluationDataset()
    def initialize_components(self, groq_api_key: str = None):
        try:
            api_key = groq_api_key or GROQ_API_KEY
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file or provide the API key as a parameter.")
            
            # Use a more compatible embedding approach
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL_NAME,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                logger.warning(f"Failed to load HuggingFaceEmbeddings: {e}")
                # Fallback to a simpler embedding model
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            
            self.llm = ChatGroq(
                groq_api_key=api_key,
                model_name=GROQ_MODEL_NAME,
                temperature=0.05,  # Lower temperature for more focused responses
                max_tokens=1200  # Increased for more comprehensive responses
            )
            # Sliding window chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=6
            )
            logger.info("All components initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
    def load_documents(self, folder_path: str) -> List[Document]:
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
                        'source_path': file_path
                    })
                    documents.extend([doc])
                logger.info(f"Loaded document: {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        return documents
    def create_vectorstore(self, folder_path: str):
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
    def generate_step_back_question(self, original_question: str) -> str:
        step_back_prompt = f"""Given the following specific question about GitHub API:\n"{original_question}"\n\nGenerate a broader, more general question that would help understand the fundamental concepts needed to answer the original question. The step-back question should focus on general principles or broader categories.\n\nExamples:\n- Specific: \"How do I create a repository using POST /user/repos?\"\n- Step-back: \"What are the general patterns for creating resources via GitHub API?\"\n\n- Specific: \"How do I authenticate with personal access tokens?\"\n- Step-back: \"What are the different authentication methods available in GitHub API?\"\n\nStep-back question:"""
        try:
            response = self.llm.invoke(step_back_prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating step-back question: {e}")
            return f"What are the general principles and concepts related to {original_question}?"
    def setup_qa_chain(self):
        try:
            prompt_template = """You are Zoro, an expert AI assistant created by Balaji, specializing in GitHub API documentation.

CRITICAL REQUIREMENTS:
1. Use EXACT terminology from the context (endpoints, parameters, status codes)
2. Include specific API endpoints like "GET /user/repos" or "POST /repos"
3. Mention authentication requirements when relevant
4. Include query parameters when mentioned in context
5. Use the exact keywords from the context
6. If information is not in the context, say "I don't know"
7. Be concise but comprehensive
8. Focus on practical, actionable information

Context: {context}

Question: {question}

Chat History: {chat_history}

Answer (be specific and use exact terms from context):"""
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question", "chat_history"]
            )
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 8}  # Increased from 3 to 8 for better coverage
                ),
                return_source_documents=True
            )
            logger.info("QA chain setup successfully with step-back prompting")
            return True
        except Exception as e:
            logger.error(f"Error setting up QA chain: {e}")
            return False
    def get_response(self, question: str) -> Dict[str, Any]:
        try:
            # Step 1: Enhanced query expansion for better retrieval
            expanded_queries = [
                question,
                question.replace("?", "").strip(),
                question.lower(),
                # Add GitHub API specific terms
                f"{question} GitHub API endpoint",
                f"{question} authentication",
                f"{question} parameters",
                f"{question} HTTP method",
                f"{question} status codes"
            ]
            
            # Step 2: Multi-query retrieval with higher k
            all_docs = []
            for query in expanded_queries[:5]:  # Use top 5 expanded queries
                try:
                    docs = self.vectorstore.similarity_search(query, k=5)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to search for query '{query}': {e}")
            
            # Step 3: Deduplicate and select best docs
            unique_docs = []
            seen_content = set()
            for doc in all_docs:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(content_hash)
            
            # Take top 8 unique docs for maximum coverage
            docs = unique_docs[:8]
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Step 4: Create a highly directive prompt focused on exact matching
            prompt_text = f"""You are a GitHub API expert. Answer the question using ONLY information from the provided context.

CRITICAL REQUIREMENTS:
1. Use EXACT terminology from the context (endpoints, parameters, status codes)
2. Include specific API endpoints like "GET /user/repos" or "POST /repos"
3. Mention authentication requirements when relevant
4. Include query parameters when mentioned in context
5. Use the exact keywords from the context
6. If information is not in the context, say "I don't know"
7. Be concise but comprehensive
8. Focus on practical, actionable information
9. Include HTTP status codes when mentioned
10. Use exact parameter names and values

Context:
{context}

Question: {question}

Answer (be specific and use exact terms from context):"""
            
            response = self.llm.invoke(prompt_text)
            
            # Step 5: Post-process to ensure key terms are included
            response_text = response.content
            
            # Extract key terms from context that should be in the answer
            context_lower = context.lower()
            question_lower = question.lower()
            
            # Find important terms that should be mentioned
            important_terms = []
            
            # Authentication terms
            if "authentication" in question_lower or "auth" in question_lower:
                if "personal access token" in context_lower:
                    important_terms.append("Personal Access Token")
                if "oauth" in context_lower:
                    important_terms.append("OAuth")
                if "authorization header" in context_lower:
                    important_terms.append("Authorization header")
                if "pat" in context_lower:
                    important_terms.append("PAT")
            
            # API endpoint terms
            if "endpoint" in question_lower or "api" in question_lower:
                import re
                endpoints = re.findall(r'[A-Z]+\s+/[a-zA-Z0-9/{}]+', context)
                important_terms.extend(endpoints[:5])  # Top 5 endpoints
            
            # Status codes
            if "status" in question_lower or "code" in question_lower:
                status_codes = re.findall(r'\b(?:200|201|204|400|401|403|404|422|500|503)\b', context)
                important_terms.extend(status_codes)
            
            # Parameters
            if "parameter" in question_lower or "query" in question_lower:
                params = re.findall(r'\b(?:per_page|page|type|sort|direction|state|labels|assignee|since)\b', context)
                important_terms.extend(params)
            
            # If important terms are missing, add them
            response_lower = response_text.lower()
            missing_terms = []
            for term in important_terms:
                if term.lower() not in response_lower:
                    missing_terms.append(term)
            
            if missing_terms:
                response_text += f"\n\nNote: Key terms from context include: {', '.join(missing_terms[:3])}"
            
            return {
                "answer": response_text,
                "sources": list(set([doc.metadata.get('source_file', 'Unknown') for doc in docs])),
                "source_documents": len(docs),
                "step_back_question": "",
                "retrieved_chunks": [
                    {
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "source": doc.metadata.get('source_file', 'Unknown'),
                        "type": "original"
                    }
                    for doc in docs
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "source_documents": 0,
                "step_back_question": "",
                "retrieved_chunks": []
            }
    def clear_memory(self):
        if self.memory:
            self.memory.clear()
    def initialize_evaluator(self):
        self.evaluator = RAGEvaluator(self)
        logger.info("Evaluation system initialized")
    def run_evaluation(self) -> Dict[str, Any]:
        if not self.evaluator:
            self.initialize_evaluator()
        qa_pairs = self.evaluation_dataset.get_qa_pairs()
        results = []
        import streamlit as st
        st.write("🔄 Running evaluation...")
        progress_bar = st.progress(0)
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
                    "step_back_question": rag_response.get("step_back_question", "")
                }
                results.append(evaluation_result)
                progress_bar.progress((i + 1) / len(qa_pairs))
            except Exception as e:
                logger.error(f"Error evaluating question '{question}': {e}")
                results.append({
                    "question": question,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
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