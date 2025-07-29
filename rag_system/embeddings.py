from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
from .config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

def get_embeddings():
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        logger.warning(f"Failed to load {EMBEDDING_MODEL_NAME}, using fallback: {e}")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        ) 