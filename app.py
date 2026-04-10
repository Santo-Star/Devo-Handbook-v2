import streamlit as st
import os
import time
import json
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Page Configuration
st.set_page_config(
    page_title="Handbook Devoción AI",
    page_icon="☕",
    layout="centered"
)

# 2. Load Environment Variables (Local .env vs Streamlit Secrets)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key and "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]

if not api_key:
    st.error("⚠️ API Key no encontrada. Por favor, configúrala en el archivo .env o en st.secrets.")
    st.stop()

# 3. Constants
INDEX_PATH = "backend/faiss_index"
MODEL_NAME = "gemini-3-flash-preview"
EMBEDDING_MODEL = "models/gemini-embedding-2-preview"

# 4. Custom CSS for "Midnight Oasis" Theme
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #020617;
        color: #f8fafc;
    }
    
    /* Header Styling */
    .st-emotion-cache-1pxn98k { /* Top bar */
        background-color: #0f172a;
    }
    
    /* Chat Bubbles */
    .stChatMessage {
        border-radius: 20px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* User Message */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #1e293b !important;
        border-bottom-right-radius: 4px;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    /* Assistant Message */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1e1b4b !important;
        border-bottom-left-radius: 4px;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    
    /* Input Box */
    .stChatInputContainer {
        border-top: 1px solid rgba(245, 158, 11, 0.1);
        padding-top: 10px;
    }
    
    /* Premium Title Accent */
    .premium-title {
        font-family: 'Playfair Display', serif;
        color: #f59e0b;
        letter-spacing: 2px;
        font-size: 1.8rem;
        margin-bottom: 2px;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 20px;
    }

    /* Mascot Animation (Float) */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    .mascot {
        animation: float 4s ease-in-out infinite;
        border: 2px solid #f59e0b;
        border-radius: 50%;
    }
</style>
""", unsafe_allow_html=True)

# 5. Functions
@st.cache_resource
def load_rag_chain():
    if not os.path.exists(INDEX_PATH):
        st.error(f"Índice no encontrado en {INDEX_PATH}. Asegúrate de subir la carpeta 'backend/faiss_index'.")
        return None
    
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, google_api_key=api_key, max_retries=3)
    
    system_prompt = (
        "Eres un asistente inteligente bilingüe para Café Devoción. "
        "Basándote EXCLUSIVAMENTE en el manual (contexto), responde la pregunta. "
        "Si no está en el manual, di: 'Lo siento, esa información no está en mi manual de Café Devoción.' "
        "Mantén un tono profesional, amable y de barista premium. "
        "Responde en el idioma del usuario."
        "\n\nContexto:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 4}), 
        combine_docs_chain
    )

def get_pdf_download_link():
    pdf_path = "devohand.pdf"
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        b64 = base64.b64encode(pdf_bytes).decode()
        return f'<a href="data:application/pdf;base64,{b64}" download="Manual_Empleado_Devocion.pdf" class="download-btn" style="color: #94a3b8; text-decoration: none; font-size: 0.8rem; border: 1px solid #94a3b8; padding: 5px 10px; border-radius: 8px;">⬇️ Descargar Handbook</a>'
    return ""

# 6. Sidebar / Header
with st.container():
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        if os.path.exists("frontend/Devo3_original.gif"):
             st.image("frontend/Devo3_original.gif", width=80)
        else:
             st.write("☕")
    with col2:
        st.markdown('<div class="premium-title">DEVOCIÓN <span style="color:white">HANDBOOK</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Digital Barista Assistant v2.1</div>', unsafe_allow_html=True)

st.markdown(get_pdf_download_link(), unsafe_allow_html=True)
st.divider()

# 7. Chat Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input logic
if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        # Load Chain
        rag_chain = load_rag_chain()
        if rag_chain:
            try:
                # Optimized retry loop for 429 errors (simplified for Streamlit UI)
                max_retries = 3
                success = False
                for attempt in range(max_retries):
                    try:
                        # Stream the response
                        with st.spinner("Preparando tu café..."):
                            for chunk in rag_chain.stream({"input": prompt}):
                                if "answer" in chunk:
                                    full_response += chunk["answer"]
                                    placeholder.markdown(full_response + "▌")
                            placeholder.markdown(full_response)
                            success = True
                            break
                    except Exception as e:
                        error_str = str(e)
                        if ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str) and attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 5
                            placeholder.info(f"☕ *Estamos preparando tu respuesta (reintentando por límites de cuota en {wait_time}s)...*")
                            time.sleep(wait_time)
                            full_response = "" # Reset for retry
                            continue
                        raise e
                
                if success:
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error("No se pudo obtener una respuesta después de varios reintentos.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.error("Error al cargar el motor de búsqueda.")

# 8. Maintenance / Branding Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.7rem;'>"
    "Desarrollado para Café Devoción | Estándares de Servicio Premium"
    "</div>", 
    unsafe_allow_html=True
)
