import json
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    OPENAI_API_KEY,
    CHROMA_API_KEY,
    CHROMA_TENANT,
    CHROMA_DATABASE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K_RESULTS,
)
from src.prompts import CLASSIFICATION_PROMPT


# ============================================================
# CHROMA CLIENT
# ============================================================
def get_chroma_client():
    return chromadb.HttpClient(
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


# ============================================================
# VECTORSTORE
# ============================================================
def get_vectorstore() -> Chroma:
    """Devolve o vectorstore ligado ao Chroma Cloud."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        client=get_chroma_client(),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


# ============================================================
# CLASSIFICAÇÃO DA QUESTÃO
# ============================================================
def classify_query(query: str, history: str) -> dict:
    """
    Classifica a questão para determinar filtros de pesquisa.
    Devolve dict com 'diploma' e 'tema'.
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    chain = CLASSIFICATION_PROMPT | llm | StrOutputParser()

    try:
        result = chain.invoke({"query": query, "history": history})
        # Limpa possível markdown à volta do JSON
        result = result.strip().replace("```json", "").replace("```", "").strip()
        classification = json.loads(result)
        print(f"  Classificação: diploma={classification.get('diploma')} | tema={classification.get('tema')}")
        return classification
    except Exception as e:
        print(f"  Classificação falhou ({e}) — sem filtros")
        return {"diploma": "AMBOS", "tema": "outro"}


# ============================================================
# CONSTRUÇÃO DO FILTRO CHROMA
# ============================================================
def build_chroma_filter(classification: dict) -> dict | None:
    """
    Constrói o filtro de metadados para o Chroma
    com base na classificação da questão.
    """
    filters = []
    diploma = classification.get("diploma", "AMBOS").upper()

    if diploma in ("CP", "CPP"):
        filters.append({"source_doc": {"$eq": diploma}})

    if len(filters) == 0:
        return None
    elif len(filters) == 1:
        return filters[0]
    else:
        return {"$and": filters}


# ============================================================
# PESQUISA
# ============================================================
def retrieve(
    query: str,
    history: str = "",
    vectorstore: Chroma = None,
) -> list:
    """
    Pipeline completo de retrieval:
    1. Classifica a questão (diploma + tema)
    2. Constrói filtro de metadados
    3. Pesquisa vetorial no Chroma
    4. Devolve lista de documentos relevantes
    """
    print("\n[RETRIEVER]")

    # 1. Classificar
    classification = classify_query(query, history)

    # 2. Filtro
    chroma_filter = build_chroma_filter(classification)
    print(f"  Filtro Chroma: {chroma_filter}")

    # 3. Pesquisa
    if vectorstore is None:
        vectorstore = get_vectorstore()

    if chroma_filter:
        results = vectorstore.similarity_search(
            query, k=TOP_K_RESULTS, filter=chroma_filter
        )
    else:
        results = vectorstore.similarity_search(query, k=TOP_K_RESULTS)

    print(f"  Documentos encontrados: {len(results)}")
    for doc in results:
        artigo = doc.metadata.get("artigo", "s/artigo")
        diploma = doc.metadata.get("source_doc", "")
        tema = doc.metadata.get("tema", "")
        print(f"    — {diploma} | {artigo} | tema: {tema}")

    return results