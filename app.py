import streamlit as st
import sys
sys.path.append(".")

from src.rag_chain import inference, get_shared_vectorstore

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Assistente Jurídico — CP & CPP",
    page_icon="⚖️",
    layout="centered",
)

# ============================================================
# CSS PERSONALIZADO
# ============================================================
st.markdown("""
<style>
    .source-badge {
        display: inline-block;
        background-color: #1e3a5f;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        margin: 2px;
    }
    .source-container {
        margin-top: 4px;
        margin-bottom: 8px;
    }
    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.image("assets/logo.png")
st.divider()

# ============================================================
# INICIALIZAÇÃO DO ESTADO
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    with st.spinner("A carregar base de dados jurídica..."):
        st.session_state.vectorstore = get_shared_vectorstore()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚖️ Chat Lei PT")
    st.markdown("Assistente especializado em **Direito Penal Português**.")
    st.divider()

    st.markdown("**Diplomas disponíveis:**")
    st.markdown("- Código Penal (CP)")
    st.markdown("- Código de Processo Penal (CPP)")
    st.divider()

    st.markdown("**Exemplos de questões:**")
    examples = [
        "Um pai abandona o seu filho de 3 anos no shopping. Qual o crime?",
        "Jovem de 21 anos promove bebida que provoca suicídio. Qual a moldura penal?",
        "Mulher de 56 anos agride física e verbalmente um rapaz de 32 anos.",
        "Após prisão preventiva, pode o arguido pedir indemnização? Em que condições?",
        "Quando é aplicada a caução económica?",
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.pending_query = example

    st.divider()

    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Desenvolvido com LangChain + ChromaDB + OpenAI")

# ============================================================
# HISTÓRICO DE MENSAGENS
# ============================================================
for message in st.session_state.messages:
    with st.chat_message(
        message["role"],
        avatar="assets/avatar_bot.png" if message["role"] == "assistant" else "assets/avatar_agente.png"
    ):
        st.markdown(message["content"])

        # Mostra pills do histórico
        if (
            message["role"] == "assistant"
            and message.get("has_legal_context")
            and message.get("sources")
        ):
            sources_html = '<div class="source-container">'
            for source in message["sources"]:
                sources_html += f'<span class="source-badge">{source}</span>'
            sources_html += '</div>'
            st.markdown(sources_html, unsafe_allow_html=True)

# ============================================================
# INPUT DO UTILIZADOR
# ============================================================
if "pending_query" in st.session_state:
    query = st.session_state.pending_query
    del st.session_state.pending_query
else:
    query = None

user_input = st.chat_input("Coloque aqui a sua questão jurídica...")

if user_input:
    query = user_input

# ============================================================
# INFERÊNCIA
# ============================================================
if query:
    with st.chat_message("user", avatar="assets/avatar_agente.png"):
        st.markdown(query)

    st.session_state.messages.append({
        "role": "user",
        "content": query,
    })

    with st.chat_message("assistant", avatar="assets/avatar_bot.png"):
        with st.spinner("A pesquisar legislação..."):
            result = inference(
                query=query,
                chat_history=st.session_state.messages[:-1],
                vectorstore=st.session_state.vectorstore,
            )

        st.markdown(result["response"])

        # Pills logo abaixo do enquadramento legal — sem título duplicado
        if result["has_legal_context"] and result["sources"]:
            sources_html = '<div class="source-container">'
            for source in result["sources"]:
                sources_html += f'<span class="source-badge">{source}</span>'
            sources_html += '</div>'
            st.markdown(sources_html, unsafe_allow_html=True)

    # Guarda resposta com todos os campos necessários para o histórico
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "sources": result["sources"],
        "has_legal_context": result["has_legal_context"],
    })