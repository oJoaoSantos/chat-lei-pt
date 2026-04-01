import os
import re
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_RAW_PATH,
)

# ============================================================
# TEMAS JURÍDICOS — lista fechada para classificação
# ============================================================
TEMAS_JURIDICOS = [
    "homicídio", "ofensa à integridade física", "ameaça", "coação",
    "sequestro", "rapto", "tráfico de pessoas", "escravidão",
    "violação", "abuso sexual", "pornografia de menores",
    "furto", "roubo", "extorsão", "burla", "abuso de confiança",
    "recetação", "dano", "incêndio", "explosivos",
    "falsificação", "contrafação", "corrupção", "peculato",
    "tráfico de droga", "associação criminosa", "terrorismo",
    "detenção ilegal de arma", "condução perigosa",
    "desobediência", "resistência à autoridade",
    "injúria", "difamação",
    "processo penal", "prisão preventiva", "arguido", "detenção",
    "julgamento", "recurso", "prova", "inquérito", "instrução",
    "medidas de coação", "buscas e apreensões",
    "outro"
]

# ============================================================
# CHROMA CLIENT
# ============================================================
def get_chroma_client():
    """Cria e devolve o cliente Chroma Cloud (API 1.0.x)."""
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
# CARREGAMENTO DE PDFs
# ============================================================
def load_pdfs(directory: str) -> list:
    documents = []
    pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"Nenhum PDF encontrado em: {directory}")

    for filename in sorted(pdf_files):
        filepath = os.path.join(directory, filename)
        print(f"  A carregar: {filename}")

        loader = PyPDFLoader(filepath)
        docs = loader.load()

        source_tag = _identify_source(filename)
        for doc in docs:
            doc.metadata["source_doc"] = source_tag
            doc.metadata["filename"] = filename

        documents.extend(docs)
        print(f"    {len(docs)} páginas carregadas")

    return documents


def _identify_source(filename: str) -> str:
    name = filename.lower()
    if "processo" in name or "cpp" in name:
        return "CPP"
    elif "penal" in name or "cp" in name:
        return "CP"
    return "OUTRO"


# ============================================================
# LIMPEZA DE TEXTO
# ============================================================
def clean_text(text: str) -> str:
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================
# EXTRAÇÃO DE METADADOS ESTRUTURAIS (regex)
# ============================================================
def extract_legal_metadata(text: str) -> dict:
    """
    Extrai Título, Capítulo, Secção e Artigo do texto por regex.
    Funciona para a estrutura do CP e CPP portugueses.
    """
    metadata = {
        "titulo": "",
        "capitulo": "",
        "seccao": "",
        "artigo": "",
    }

    titulo = re.search(
        r"(TÍTULO\s+[IVXLCDM]+[^\n]*)", text, re.IGNORECASE
    )
    if titulo:
        metadata["titulo"] = titulo.group(1).strip()

    capitulo = re.search(
        r"(CAPÍTULO\s+[IVXLCDM]+[^\n]*)", text, re.IGNORECASE
    )
    if capitulo:
        metadata["capitulo"] = capitulo.group(1).strip()

    seccao = re.search(
        r"(SECÇÃO\s+[IVXLCDM]+[^\n]*)", text, re.IGNORECASE
    )
    if seccao:
        metadata["seccao"] = seccao.group(1).strip()

    artigo = re.search(
        r"(Artigo\s+\d+\.º[^\n]*)", text, re.IGNORECASE
    )
    if artigo:
        metadata["artigo"] = artigo.group(1).strip()

    return metadata


# ============================================================
# CLASSIFICAÇÃO DE TEMA (LLM)
# ============================================================
def build_classification_chain():
    """Constrói a chain de classificação de tema jurídico."""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    temas_str = ", ".join(TEMAS_JURIDICOS)

    prompt = ChatPromptTemplate.from_template(
        """És um especialista em direito penal português.
Analisa o seguinte excerto de lei e classifica-o com o tema mais adequado.

Responde APENAS com uma palavra ou expressão curta da lista abaixo.
Não expliques, não uses pontuação extra.

Lista de temas permitidos:
{temas}

Excerto:
{texto}

Tema:"""
    )

    return prompt | llm | StrOutputParser()


def classify_tema(chain, text: str) -> str:
    """Classifica o tema de um chunk. Em caso de erro devolve 'outro'."""
    try:
        result = chain.invoke({
            "temas": ", ".join(TEMAS_JURIDICOS),
            "texto": text[:800],
        })
        tema = result.strip().lower()
        # Valida que o tema está na lista
        if tema not in TEMAS_JURIDICOS:
            tema = "outro"
        return tema
    except Exception:
        return "outro"


# ============================================================
# DIVISÃO EM CHUNKS
# ============================================================
def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\nArtigo ",
            "\nCAPÍTULO ",
            "\nTÍTULO ",
            "\nSECÇÃO ",
            "\n\n",
            "\n",
            " ",
        ],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)

    chunks = [c for c in chunks if len(c.page_content) > 100]

    print(f"  Total de chunks gerados: {len(chunks)}")
    return chunks


# ============================================================
# ENRIQUECIMENTO DE METADADOS
# ============================================================
def enrich_metadata(chunks: list) -> list:
    """
    Enriquece cada chunk com:
    - metadados estruturais (regex): titulo, capitulo, seccao, artigo
    - tema jurídico (LLM): classificação semântica
    """
    print("  A construir chain de classificação...")
    classification_chain = build_classification_chain()

    total = len(chunks)
    for i, chunk in enumerate(chunks):

        # 1. Extração estrutural por regex
        legal_meta = extract_legal_metadata(chunk.page_content)
        chunk.metadata.update(legal_meta)

        # 2. Classificação de tema com LLM
        chunk.metadata["tema"] = classify_tema(
            classification_chain, chunk.page_content
        )

        if (i + 1) % 50 == 0:
            print(f"  Metadados enriquecidos: {i + 1}/{total}")

    print(f"  Metadados enriquecidos: {total}/{total}")
    return chunks


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def ingest(directory: str = DATA_RAW_PATH) -> None:
    print("=" * 50)
    print("INGESTÃO DE DADOS — CP & CPP")
    print("=" * 50)

    # 1. Carregar PDFs
    print("\n[1/4] A carregar PDFs...")
    documents = load_pdfs(directory)
    print(f"  Total de páginas: {len(documents)}")

    # 2. Dividir em chunks
    print("\n[2/4] A dividir em chunks...")
    chunks = split_documents(documents)

    # 3. Enriquecer metadados
    print("\n[3/4] A enriquecer metadados...")
    print("  (regex + LLM — pode demorar alguns minutos)")
    chunks = enrich_metadata(chunks)

    # Mostra exemplo de metadados
    print("\n  Exemplo de metadados do chunk 0:")
    for k, v in chunks[0].metadata.items():
        print(f"    {k}: {v}")

    # 4. Embeddings + Chroma Cloud
    print("\n[4/4] A gerar embeddings e a indexar no Chroma Cloud...")
    print("  A carregar modelo local (primeira vez demora ~1 min)...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    chroma_client = get_chroma_client()

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    batch_size = 100
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i: i + batch_size]
        vectorstore.add_documents(batch)
        print(f"  Indexados {min(i + batch_size, total)}/{total} chunks")

    print("\nIngestão concluída com sucesso!")
    print(f"Collection '{COLLECTION_NAME}' disponível no Chroma Cloud.")
    print("=" * 50)


if __name__ == "__main__":
    ingest()