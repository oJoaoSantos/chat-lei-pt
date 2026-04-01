import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Chroma Cloud
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
COLLECTION_NAME = "legislacao_penal"

# RAG
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 5

# Paths
DATA_RAW_PATH = "./data/raw"
DATA_PROCESSED_PATH = "./data/processed"

# App
APP_TITLE = "Assistente Jurídico — CP & CPP"
APP_SUBTITLE = "Pesquisa para forças de segurança"