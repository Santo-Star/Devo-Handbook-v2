import os
import time
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def ingest_pdf(pdf_path: str, index_path: str):
    print(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print("Splitting documents into chunks...")
    # Smaller chunks for better context, but we will batch the API calls
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
    
    print(f"Creating vector store in small batches to respect FREE TIER limits...")
    # FREE TIER has a limit of 100 requests per minute.
    # To be safe, we use small batches and enough delay.
    batch_size = 20 
    vectorstore = None
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch)} chunks)...")
        
        retries = 3
        while retries > 0:
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
                break # Success
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"Rate limit hit. Waiting 60 seconds to reset quota... (Retries left: {retries-1})")
                    time.sleep(60)
                    retries -= 1
                else:
                    raise e
        
        if i + batch_size < len(chunks):
            print("Waiting 10 seconds between batches...")
            time.sleep(10)
    
    if vectorstore:
        print(f"Saving index to {index_path}...")
        vectorstore.save_local(index_path)
        print("Ingestion complete!")
    else:
        print("Error: No vector store created.")
        sys.exit(1)

if __name__ == "__main__":
    PDF_FILE = "devohand.pdf"
    INDEX_DIR = "backend/faiss_index"
    
    if os.path.exists(PDF_FILE):
        try:
            ingest_pdf(PDF_FILE, INDEX_DIR)
        except Exception as e:
            print(f"\n[!] Error fatal durante la ingesta: {e}")
            sys.exit(1)
    else:
        print(f"Error: {PDF_FILE} not found.")
        sys.exit(2)
