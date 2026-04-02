import os
import re
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CHROMA_API_KEY,
    CHROMA_TENANT,
    CHROMA_DATABASE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_RAW_PATH,
)


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
# EXTRAÇÃO DE METADADOS (regex)
# ============================================================
def extract_legal_metadata(text: str) -> dict:
    """Extrai apenas source_doc e artigo — o essencial para filtros e pills."""
    metadata = {"artigo": ""}

    artigo = re.search(r"(Artigo\s+\d+\.º(?:-[A-Z])?)", text, re.IGNORECASE)
    if artigo:
        metadata["artigo"] = artigo.group(1).strip()

    return metadata


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
        legal_meta = extract_legal_metadata(chunk.page_content)
        chunk.metadata.update(legal_meta)

    chunks = [c for c in chunks if len(c.page_content) > 100]

    print(f"  Total de chunks gerados: {len(chunks)}")
    return chunks


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def ingest(directory: str = DATA_RAW_PATH) -> None:
    print("=" * 50)
    print("INGESTÃO DE DADOS — CP & CPP")
    print("=" * 50)

    # 1. Carregar PDFs
    print("\n[1/3] A carregar PDFs...")
    documents = load_pdfs(directory)
    print(f"  Total de páginas: {len(documents)}")

    # 2. Dividir em chunks + metadados
    print("\n[2/3] A dividir em chunks e extrair metadados...")
    chunks = split_documents(documents)

    # Mostra exemplo
    print("\n  Exemplo de metadados do chunk 10:")
    for k, v in chunks[10].metadata.items():
        print(f"    {k}: {v}")

    # 3. Embeddings + Chroma Cloud
    print("\n[3/3] A gerar embeddings e a indexar no Chroma Cloud...")
    print("  A carregar modelo (primeira vez demora ~2 min)...")

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