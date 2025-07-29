import os
import glob
import logging
from datetime import datetime
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from .config import COLLECTION_NAME

logger = logging.getLogger(__name__)

def load_documents(folder_path: str):
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

def sliding_window_chunk_documents(documents, chunk_size, chunk_overlap):
    """Custom sliding window chunker for Document objects using RecursiveCharacterTextSplitter for base splitting."""
    base_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    all_chunks = []
    for doc in documents:
        base_chunks = base_splitter.split_text(doc.page_content)
        i = 0
        while i < len(base_chunks):
            window = ''
            j = i
            chars = 0
            while j < len(base_chunks) and chars + len(base_chunks[j]) <= chunk_size:
                window += base_chunks[j]
                chars += len(base_chunks[j])
                j += 1
            # Create a new Document for the chunk, preserving metadata
            all_chunks.append(Document(page_content=window, metadata=doc.metadata.copy()))
            # Move window forward by chunk_size - chunk_overlap
            chars_advanced = 0
            while i < len(base_chunks) and chars_advanced < (chunk_size - chunk_overlap):
                chars_advanced += len(base_chunks[i])
                i += 1
    return all_chunks

def create_vectorstore(documents, text_splitter, embeddings, persist_directory="./chroma_db"):
    # Use the custom sliding window chunker
    texts = sliding_window_chunk_documents(documents, chunk_size=400, chunk_overlap=200)
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    logger.info(f"Created vectorstore with {len(texts)} chunks")
    return vectorstore 