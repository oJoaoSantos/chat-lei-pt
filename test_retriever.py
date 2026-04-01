import sys
sys.path.append(".")

from src.retriever import get_vectorstore, retrieve

print("=" * 50)
print("TESTE DO RETRIEVER")
print("=" * 50)

# Carrega o vectorstore uma vez
print("\nA carregar vectorstore...")
vectorstore = get_vectorstore()
print("Vectorstore carregado!")

# ============================================================
# TESTES
# ============================================================
queries = [
    "Qual é a pena para homicídio simples?",
    "O que é a prisão preventiva e quando se aplica?",
    "Quais são os requisitos para efetuar uma busca domiciliária?",
]

for query in queries:
    print("\n" + "=" * 50)
    print(f"QUESTÃO: {query}")
    print("=" * 50)

    results = retrieve(query=query, history="", vectorstore=vectorstore)

    print(f"\nCONTEÚDO DO 1º RESULTADO:")
    print("-" * 40)
    print(results[0].page_content[:400])
    print("...")