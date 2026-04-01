import sys
sys.path.append(".")

from src.rag_chain import inference, get_shared_vectorstore

# Carrega vectorstore uma vez
vectorstore = get_shared_vectorstore()

# Simula uma conversa com histórico
chat_history = []

queries = [
    "Qual é a pena para homicídio simples?",
    "E para o homicídio qualificado?",  # testa o histórico
    "Quais os requisitos para uma busca domiciliária?",
]

for query in queries:
    print(f"\nAGENTE: {query}")

    result = inference(
        query=query,
        chat_history=chat_history,
        vectorstore=vectorstore,
    )

    # Atualiza histórico
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": result["response"]})

    print(f"\nASSISTENTE: {result['response']}")
    print(f"\nFONTES: {result['sources']}")
    print("\n" + "=" * 60)