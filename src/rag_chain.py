import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

from config import OPENAI_API_KEY, LLM_MODEL, TOP_K_RESULTS
from src.prompts import RAG_PROMPT, SYSTEM_PROMPT
from src.retriever import get_vectorstore, retrieve, classify_query


# ============================================================
# HISTÓRICO DE CONVERSA
# ============================================================
def format_chat_history(history: list, max_turns: int = 5) -> str:
    if not history:
        return "Sem histórico de conversa."

    recent = history[-(max_turns * 2):]
    formatted = []
    for msg in recent:
        role = "Agente" if msg["role"] == "user" else "Assistente"
        formatted.append(f"{role}: {msg['content']}")

    return "\n".join(formatted)


# ============================================================
# FORMATAR CONTEXTO DOS DOCUMENTOS
# ============================================================
def format_context(documents: list) -> str:
    if not documents:
        return "Nenhum artigo relevante encontrado."

    context_parts = []
    seen_content = set()

    for i, doc in enumerate(documents, 1):
        # Evita chunks duplicados
        content_hash = doc.page_content[:100]
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)

        diploma = doc.metadata.get("source_doc", "")
        artigo = doc.metadata.get("artigo", "")
        tema = doc.metadata.get("tema", "")

        header = f"[Documento {i}]"
        if diploma:
            header += f" {diploma}"
        if artigo:
            header += f" — {artigo}"
        if tema:
            header += f" ({tema})"

        context_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(context_parts)


# ============================================================
# RAG CHAIN
# ============================================================
def create_rag_chain():
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )
    return RAG_PROMPT | llm | StrOutputParser()


# ============================================================
# INFERÊNCIA PRINCIPAL
# ============================================================
def inference(
    query: str,
    chat_history: list = None,
    vectorstore: Chroma = None,
) -> dict:

    print("=" * 50)
    print("INFERÊNCIA — ASSISTENTE JURÍDICO")
    print("=" * 50)

    # 1. Histórico
    print("\n[1/4] A formatar histórico...")
    if chat_history is None:
        chat_history = []
    formatted_history = format_chat_history(chat_history)

    # 2. Classificação
    print("\n[2/4] A classificar questão...")
    classification = classify_query(query, formatted_history)
    is_greeting = classification.get("is_greeting", False)

    # 3. Retrieval
    print("\n[3/4] A pesquisar artigos relevantes...")
    if vectorstore is None:
        vectorstore = get_vectorstore()

    if is_greeting:
        documents = []
        print("  Cumprimento detetado — sem pesquisa jurídica")
    else:
        documents = retrieve(
            query=query,
            history=formatted_history,
            vectorstore=vectorstore,
            classification=classification,
        )

    # 4. Geração
    print("\n[4/4] A gerar resposta...")
    context = format_context(documents)
    chain = create_rag_chain()
    response = chain.invoke({
        "system_prompt": SYSTEM_PROMPT,
        "history": formatted_history,
        "context": context,
        "query": query,
    })

    # Extrai pills APENAS dos artigos mencionados na resposta gerada
    sources = _extract_sources_from_response(response)
    has_legal_context = len(sources) > 0 and not is_greeting

    print(f"\nFONTES EXTRAÍDAS: {sources}")

    return {
        "response": response,
        "sources": sources,
        "documents": documents,
        "has_legal_context": has_legal_context,
    }

# Helper
def _extract_sources_from_response(response: str) -> list:
    """
    Extrai as pills diretamente do texto da resposta gerada.
    Procura padrões como 'Artigo 131.º do CP' ou 'Artigo 202.º do CPP'.
    Devolve lista ordenada e sem duplicados.
    """
    import re

    pattern = r"Artigo\s+(\d+\.º(?:-[A-Z])?)\s+do\s+(CP|CPP)"
    matches = re.findall(pattern, response)

    seen = set()
    sources = []
    for artigo, diploma in matches:
        key = f"{diploma} — Artigo {artigo}"
        if key not in seen:
            sources.append(key)
            seen.add(key)

    return sources

# ============================================================
# SINGLETON DO VECTORSTORE
# ============================================================
_vectorstore = None

def get_shared_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        print("A carregar vectorstore...")
        _vectorstore = get_vectorstore()
        print("Vectorstore pronto.")
    return _vectorstore