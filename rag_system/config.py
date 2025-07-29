import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL_NAME = "llama3-70b-8192"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DOCUMENTS_FOLDER = "documents"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 200
COLLECTION_NAME = "github_api_docs" 