# Chat Lei PT

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3.25-green?logo=chainlink)
![Streamlit](https://img.shields.io/badge/Streamlit-1.44.1-red?logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.0.9-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-purple?logo=openai)
![LangSmith](https://img.shields.io/badge/LangSmith-Observability-yellow)

Assistente jurídico especializado em **Direito Penal Português**, desenvolvido com RAG (Retrieval-Augmented Generation). Apoia agentes da **PSP e GNR** na pesquisa de legislação para elaboração de expediente policial.

---

## Aplicação

🔗 [chat-lei-pt.streamlit.app](https://chat-lei-pt.streamlit.app)

---

## Funcionalidades

- Pesquisa semântica no **Código Penal (CP)** e **Código de Processo Penal (CPP)**
- Respostas com citação do artigo e diploma legal, sem repetições
- Classificação automática da questão por diploma (CP vs CPP) e tema
- Deteção automática de cumprimentos — sem forçar legislação em conversa informal
- Quando não encontra legislação relevante, afirma-o claramente — nunca inventa
- Histórico de conversa com memória contextual (até 5 turnos)
- Metadados enriquecidos por artigo (título, capítulo, secção, tema)
- Interface de chat intuitiva para uso operacional

---

## Arquitetura
```
chat-lei-pt/
├── src/
│   ├── __init__.py       # Módulo Python
│   ├── ingestion.py      # Carrega PDFs, gera chunks e indexa no ChromaDB
│   ├── retriever.py      # Pesquisa vetorial com filtros por diploma e tema
│   ├── rag_chain.py      # Pipeline RAG com histórico de conversa
│   └── prompts.py        # Templates de prompt especializados
├── assets/               # Imagens e logos
├── data/
│   └── raw/
│       ├── cp.pdf        # Código Penal (não incluído no repositório)
│       └── cpp.pdf       # Código de Processo Penal (não incluído no repositório)
├── app.py                # Interface Streamlit
├── config.py             # Configurações e constantes
├── requirements.txt      # Dependências Python
└── .env.example          # Exemplo de variáveis de ambiente
```

---

## Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| Framework RAG | LangChain 0.3 |
| LLM | OpenAI GPT-4o |
| Embeddings | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 |
| Base de dados vetorial | ChromaDB Cloud 1.0.9 |
| Observabilidade | LangSmith |
| Interface | Streamlit 1.44.1 |
| Deploy | Streamlit Community Cloud |

---

## Instalação Local

### Pré-requisitos
- Python 3.11
- Conta OpenAI com créditos disponíveis
- Conta ChromaDB Cloud
- PDFs do CP e CPP em `data/raw/`

### Passos
```bash
# 1. Clonar o repositório
git clone https://github.com/oJoaoSantos/chat-lei-pt.git
cd chat-lei-pt

# 2. Criar ambiente virtual
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

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

Cria um ficheiro `.env` na raiz do projeto com base no `.env.example`:
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

- *"Um pai abandona o seu filho de 3 anos no shopping. Qual o crime?"*
- *"Jovem de 21 anos promove bebida que provoca suicídio. Qual a moldura penal?"*
- *"Mulher de 56 anos agride física e verbalmente um rapaz de 32 anos."*
- *"Após prisão preventiva, pode o arguido pedir indemnização? Em que condições?"*
- *"Quando é aplicada a caução económica?"*

---

## Aviso Legal

Este assistente é uma ferramenta de apoio à pesquisa jurídica e **não substitui aconselhamento jurídico profissional**. As respostas baseiam-se nos diplomas indexados e devem ser sempre validadas com a legislação oficial em vigor.

---

## Desenvolvido com

- [LangChain](https://langchain.com)
- [ChromaDB](https://trychroma.com)
- [OpenAI](https://openai.com)
- [Streamlit](https://streamlit.io)
- [Hugging Face](https://huggingface.co) — modelos de embeddings
- [LangSmith](https://smith.langchain.com) — observabilidade e tracing