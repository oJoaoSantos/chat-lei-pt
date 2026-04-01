from langchain_core.prompts import ChatPromptTemplate

# ============================================================
# SYSTEM PROMPT — Assistente Jurídico
# ============================================================
SYSTEM_PROMPT = """És um assistente jurídico especializado em Direito Penal Português.
Apoias agentes da PSP e GNR na pesquisa de legislação para elaboração de expediente policial.

REGRAS OBRIGATÓRIAS:
- Responde SEMPRE em português de Portugal
- Cita SEMPRE o número do artigo e o diploma (CP ou CPP)
- Usa linguagem clara e objetiva, adequada a agentes de polícia
- Baseia-te APENAS nos artigos fornecidos como contexto
- Se a informação não estiver no contexto, diz exatamente: "Não encontrei legislação relevante para esta questão na base de dados."
- Nunca inventes artigos ou números de lei
- Quando relevante, indica se o crime é público, semipúblico ou particular

FORMATO DA RESPOSTA:
1. Resposta direta à questão
2. Fundamento legal: "Artigo X.º do CP/CPP — [título do artigo]"
3. Nota prática para o expediente (se aplicável)
"""

# ============================================================
# PROMPT PRINCIPAL — RAG com histórico
# ============================================================
RAG_PROMPT = ChatPromptTemplate.from_template(
    """
{system_prompt}

HISTÓRICO DA CONVERSA:
{history}

ARTIGOS RELEVANTES ENCONTRADOS:
{context}

QUESTÃO ATUAL: {query}

RESPOSTA:
"""
)

# ============================================================
# PROMPT DE CLASSIFICAÇÃO — filtra por diploma e tema
# ============================================================
CLASSIFICATION_PROMPT = ChatPromptTemplate.from_template(
    """Analisa esta questão jurídica e classifica-a.

Histórico da conversa (para contexto):
{history}

Questão: {query}

Responde APENAS com um objeto JSON válido, sem mais texto:
{{
  "diploma": "CP" ou "CPP" ou "AMBOS",
  "tema": "um tema da lista ou outro"
}}

Temas possíveis: homicídio, ofensa à integridade física, ameaça, coação,
sequestro, furto, roubo, extorsão, burla, abuso de confiança, recetação,
dano, incêndio, falsificação, corrupção, tráfico de droga, associação criminosa,
terrorismo, detenção ilegal de arma, condução perigosa, desobediência,
resistência à autoridade, injúria, difamação, processo penal, prisão preventiva,
arguido, detenção, julgamento, recurso, prova, inquérito, instrução,
medidas de coação, buscas e apreensões, outro

JSON:"""
)