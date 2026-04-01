# ⚖️ Chat Lei PT

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3.25-green?logo=chainlink)
![Streamlit](https://img.shields.io/badge/Streamlit-1.44.1-red?logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.0.9-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-purple?logo=openai)

Assistente jurídico especializado em **Direito Penal Português**, desenvolvido com RAG (Retrieval-Augmented Generation). Apoia agentes da **PSP e GNR** na pesquisa de legislação para elaboração de expediente policial.

---

## Funcionalidades

- Pesquisa semântica no **Código Penal (CP)** e **Código de Processo Penal (CPP)**
- Respostas com citação do artigo e diploma legal
- Classificação automática da questão (CP vs CPP)
- Histórico de conversa com memória contextual
- Metadados enriquecidos por artigo (título, capítulo, tema)
- Interface de chat intuitiva para uso operacional

---

## Arquitetura
```
chat-lei-pt/
├── src/
│   ├── ingestion.py      # Carrega PDFs, gera chunks e indexa no ChromaDB
│   ├── retriever.py      # Pesquisa vetorial com filtros por diploma e tema
│   ├── rag_chain.py      # Pipeline RAG com histórico de conversa
│   └── prompts.py        # Templates de prompt especializados
├── assets/               # Imagens e logos
├── data/
│   └── raw/              # PDFs do CP e CPP (não incluídos no repositório)
├── app.py                # Interface Streamlit
├── config.py             # Configurações e constantes
└── requirements.txt      # Dependências Python
```

---

## Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| Framework RAG | LangChain 0.3 |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | sentence-transformers (multilingual) |
| Base de dados vetorial | ChromaDB Cloud |
| Observabilidade | LangSmith |
| Interface | Streamlit |
| Deploy | Streamlit Community Cloud |

---

## Instalação Local

### Pré-requisitos
- Python 3.10+
- Conta OpenAI com créditos disponíveis
- Conta ChromaDB Cloud
- PDFs do CP e CPP em `data/raw/`

### Passos
```bash
# 1. Clonar o repositório
git clone https://github.com/SEU_USERNAME/chat-lei-pt.git
cd chat-lei-pt

# 2. Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar variáveis de ambiente
cp .env.example .env
# Edita o .env com as tuas credenciais

# 5. Correr a ingestão (apenas uma vez)
python src/ingestion.py

# 6. Lançar a aplicação
streamlit run app.py
```

---

## Variáveis de Ambiente

Cria um ficheiro `.env` na raiz do projeto:
```env
# OpenAI
OPENAI_API_KEY=sk-proj-...

# ChromaDB Cloud
CHROMA_API_KEY=ck-...
CHROMA_TENANT=...
CHROMA_DATABASE=chat-lei

# LangSmith (opcional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=Chat Lei PT
```

---

## Exemplos de Questões

- *"Qual a pena para homicídio simples?"*
- *"Quais os requisitos para uma busca domiciliária?"*
- *"Quando se aplica a prisão preventiva?"*
- *"O que é o crime de resistência à autoridade?"*
- *"Em que condições pode o arguido pedir indemnização após prisão preventiva?"*

---

## Desenvolvido com

- [LangChain](https://langchain.com)
- [ChromaDB](https://trychroma.com)
- [OpenAI](https://openai.com)
- [Streamlit](https://streamlit.io)
- [Hugging Face](https://huggingface.co) — modelos de embeddings

---