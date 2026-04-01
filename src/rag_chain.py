import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

from config import OPENAI_API_KEY, LLM_MODEL, TOP_K_RESULTS
from src.prompts import RAG_PROMPT, SYSTEM_PROMPT
from src.retriever import get_vectorstore, retrieve


# ============================================================
# HISTÓRICO DE CONVERSA
# ============================================================
def format_chat_history(history: list, max_turns: int = 5) -> str:
    """
    Converte o histórico de conversa num string legível para o LLM.
    Limita aos últimos N turnos para não exceder o contexto.

    history: lista de dicts {"role": "user"/"assistant", "content": "..."}
    """
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
    """
    Formata os documentos recuperados em contexto estruturado
    para o LLM, incluindo a referência legal de cada chunk.
    """
    if not documents:
        return "Nenhum artigo relevante encontrado."

    context_parts = []
    for i, doc in enumerate(documents, 1):
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
    """Constrói e devolve a RAG chain."""
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
    """
    Pipeline completo de inferência:
    1. Formata o histórico de conversa
    2. Recupera artigos relevantes (retriever)
    3. Formata o contexto
    4. Gera resposta com o LLM

    Devolve dict com:
        - response: resposta gerada
        - sources: lista de fontes usadas
        - documents: documentos recuperados
    """
    print("=" * 50)
    print("INFERÊNCIA — ASSISTENTE JURÍDICO")
    print("=" * 50)

    # 1. Histórico
    print("\n[1/4] A formatar histórico...")
    if chat_history is None:
        chat_history = []
    formatted_history = format_chat_history(chat_history)
    print(f"  {len(chat_history)} mensagens no histórico")

    # 2. Retrieval
    print("\n[2/4] A pesquisar artigos relevantes...")
    if vectorstore is None:
        vectorstore = get_vectorstore()

    documents = retrieve(
        query=query,
        history=formatted_history,
        vectorstore=vectorstore,
    )

    # 3. Contexto
    print("\n[3/4] A formatar contexto...")
    context = format_context(documents)

    # 4. Geração
    print("\n[4/4] A gerar resposta...")
    chain = create_rag_chain()
    response = chain.invoke({
        "system_prompt": SYSTEM_PROMPT,
        "history": formatted_history,
        "context": context,
        "query": query,
    })

    # Fontes únicas para apresentar ao utilizador
    sources = []
    seen = set()
    for doc in documents:
        diploma = doc.metadata.get("source_doc", "")
        artigo = doc.metadata.get("artigo", "")
        if artigo and (diploma, artigo) not in seen:
            sources.append(f"{diploma} — {artigo}")
            seen.add((diploma, artigo))

    print("\n" + "=" * 50)
    print("RESPOSTA GERADA")
    print("=" * 50)
    print(response)
    print("\nFONTES:")
    for s in sources:
        print(f"  {s}")

    return {
        "response": response,
        "sources": sources,
        "documents": documents,
    }


# ============================================================
# SINGLETON DO VECTORSTORE
# ============================================================
_vectorstore = None

def get_shared_vectorstore() -> Chroma:
    """
    Devolve uma instância partilhada do vectorstore.
    Carrega apenas uma vez — evita recarregar o modelo
    de embeddings a cada pergunta.
    """
    global _vectorstore
    if _vectorstore is None:
        print("A carregar vectorstore...")
        _vectorstore = get_vectorstore()
        print("Vectorstore pronto.")
    return _vectorstore