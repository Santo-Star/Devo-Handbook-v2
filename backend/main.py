import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from fastapi.responses import StreamingResponse
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
INDEX_PATH = "backend/faiss_index"
MODEL_NAME = "gemini-3-flash-preview"

# Global variables
vectorstore = None
rag_chain = None

def load_rag_chain():
    global vectorstore, rag_chain
    if not os.path.exists(INDEX_PATH):
        print(f"Index not found at {INDEX_PATH}.")
        return False
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, max_retries=3)
    
    system_prompt = (
        "Eres un asistente inteligente para Café Devoción. "
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
    # k=4 finds the top 4 most relevant snippets, balancing context and speed.
    rag_chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 4}), 
        combine_docs_chain
    )
    return True

@app.on_event("startup")
async def startup():
    key = os.getenv("GOOGLE_API_KEY", "")
    masked_key = key[:6] + "..." + key[-4:] if len(key) > 10 else "NO ENCONTRADA"
    print(f"[*] Servidor iniciado | Clave API: {masked_key} | Modelo: {MODEL_NAME}")
    load_rag_chain()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    if rag_chain is None:
        if not load_rag_chain():
            raise HTTPException(status_code=500, detail="Index not ready.")
    
    # Anti-burst delay: Slow down requests slightly to stay under Free Tier quotas
    await asyncio.sleep(1)

    async def stream_generator():
        try:
            # Optimized retry loop for 429 errors with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async for chunk in rag_chain.astream({"input": request.message}):
                        if "answer" in chunk:
                            yield f"data: {json.dumps({'answer': chunk['answer']})}\n\n"
                    yield "data: [DONE]\n\n"
                    return # Success
                except Exception as e:
                    error_str = str(e)
                    if ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 15 # 15s, 30s, 45s
                        print(f"[*] Cuota excedida (429). Reintento {attempt + 1}/{max_retries} en {wait_time}s...")
                        # Send reset signal to frontend to clear partial text
                        yield f"data: {json.dumps({'reset': True, 'answer': '☕ *Estamos preparando tu respuesta (reintentando por límites de cuota)...*'})}\n\n"
                        await asyncio.sleep(wait_time)
                        continue
                    raise e
        except Exception as e:
            print(f"ERROR en streaming: {str(e)}")
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                error_msg = "Has agotado tu cuota gratuita momentáneamente. Por favor, espera un minuto o intenta con otra API Key."
            yield f"data: {json.dumps({'error': error_msg})}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

# Serving Frontend
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.get("/download")
async def download_pdf():
    pdf_path = "devohand.pdf"
    if os.path.exists(pdf_path):
        return FileResponse(pdf_path, media_type='application/pdf', filename="Manual_Empleado_Devocion.pdf")
    raise HTTPException(status_code=404, detail="Archivo no encontrado")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
