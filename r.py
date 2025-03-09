# -*- coding: utf-8 -*-
import chromadb
import logging
import subprocess
import sys
import time
import socket
import os
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DefaultEmbeddingFunction = None  # Disable Chroma's default ONNX

def system_check():
    """Verify and ensure Ollama service is running with 1-minute timeout"""
    try:
        logger.info("Starting system check...")
        start_time = time.time()
        ollama_started = False
        
        # Timeout-check loop
        while time.time() - start_time < 60:
            try:
                # Check port status
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2)
                    if s.connect_ex(('localhost', 11434)) == 0:
                        logger.info("Ollama service: ACTIVE")
                        ollama_started = True
                        break
                    
                # Start Ollama if not running
                logger.warning("Ollama not responding, attempting to start...")
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
                )
                time.sleep(5)  # Wait between checks
                
            except Exception as e:
                logger.warning("Connection check failed: %s", str(e))
                time.sleep(5)

        if not ollama_started:
            logger.error("Failed to start Ollama within 1 minute")
            sys.exit(1)
            
    except Exception as e:
        logger.error("System check failed: %s", str(e))
        sys.exit(1)

# Execute system check immediately after configuration
system_check()

# Import remaining components after successful system check
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import gradio as gr
import warnings
warnings.filterwarnings("ignore")

# CPU-optimized configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 30
LLM_TIMEOUT = 600  # 10 minutes


# Add to get_llm() function
def get_llm():
    return Ollama(
        model="llama3.2",
        temperature=0.1,
        num_ctx=2048,
        timeout=600
    )

def document_loader(file_path):
    """Load PDF with error handling"""
    try:
        logger.info("Loading document: %s", file_path)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        logger.info("Loaded %d pages", len(docs))
        return docs
    except Exception as e:
        logger.error("Document load error: %s", str(e))
        return "Error loading document: " + str(e)

def text_splitter(documents):
    """CPU-friendly text splitting"""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    ).split_documents(documents)


def vector_database(chunks):
    try:
        logger.info("Initializing vector database...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        if os.path.exists("./cpu_chroma_data"):
            logger.info("Reusing existing vector store")
            return Chroma(
                persist_directory="./cpu_chroma_data",
                embedding_function=embeddings
            )
            
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory="./cpu_chroma_data",
            client_settings=chromadb.config.Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
    except Exception as e:
        logger.error("Vector DB error: %s", str(e))
        raise

def retriever(file):
    """Document processing pipeline"""
    documents = document_loader(file)
    if isinstance(documents, str):
        return documents
    chunks = text_splitter(documents)
    return vector_database(chunks).as_retriever()

def retriever_qa(file, query):
    """QA pipeline with error handling"""
    try:
        llm = get_llm()
        retriever_obj = retriever(file)
        
        if isinstance(retriever_obj, str):
            return retriever_obj
            
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_obj
        )
        response = qa.invoke(query)
        return response.get("result", "No response generated.")
    except Exception as e:
        logger.error("QA error: %s", str(e))
        return "Processing error: " + str(e)

# Gradio interface
# ... [keep all your previous code except the Gradio launch part] ...

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# DocuMind (Your Best Reading Buddy)")
    with gr.Row():
        file_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
        query_input = gr.Textbox(label="Enter your question")
    output_text = gr.Textbox(label="Response")
    submit_btn = gr.Button("Submit")
    submit_btn.click(retriever_qa, inputs=[file_input, query_input], outputs=output_text)

# Run app
app.launch()