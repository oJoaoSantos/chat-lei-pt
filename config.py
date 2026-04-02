import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Chroma Cloud
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
COLLECTION_NAME = "legislacao_penal"

# RAG
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 10

# Paths
DATA_RAW_PATH = "./data/raw"
DATA_PROCESSED_PATH = "./data/processed"

# App
APP_TITLE = "Assistente Jurídico — CP & CPP"
APP_SUBTITLE = "Pesquisa para forças de segurança"