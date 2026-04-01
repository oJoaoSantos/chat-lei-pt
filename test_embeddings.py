import sys
sys.path.append(".")
from config import EMBEDDING_MODEL, CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE, COLLECTION_NAME
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

print("[1/2] A testar embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
result = embeddings.embed_query("teste de embedding jurídico")
print(f"  OK — vetor gerado com {len(result)} dimensões")

print("[2/2] A testar ligação ao Chroma Cloud...")
client = chromadb.HttpClient(
    host="api.trychroma.com",
    ssl=True,
    headers={"x-chroma-token": CHROMA_API_KEY},
    settings=chromadb.config.Settings(
        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        chroma_client_auth_credentials=CHROMA_API_KEY,
    ),
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
)
collections = client.list_collections()
print(f"  OK — ligado ao Chroma Cloud. Collections: {[c.name for c in collections]}")

print("\nTudo OK — podes correr o ingestion.py!")