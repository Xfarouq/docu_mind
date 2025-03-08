from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import gradio as gr
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load LLM (Ollama)
def get_llm():
    return Ollama(model="llama3.2")

# Load PDF and extract text
def document_loader(file):
    try:
        loader = PyPDFLoader(file.name)
        return loader.load()
    except Exception as e:
        return f"Error loading document: {str(e)}"

# Split text into chunks
def text_splitter(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return splitter.split_documents(documents)

# Generate embeddings and store in ChromaDB
def vector_database(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma.from_documents(chunks, embeddings)

# Create retriever
def retriever(file):
    documents = document_loader(file)
    if isinstance(documents, str):  # Error handling
        return documents
    chunks = text_splitter(documents)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()

# Retrieve and generate response
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    if isinstance(retriever_obj, str):  # If error occurred
        return retriever_obj
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever_obj)
    response = qa.invoke(query)
    return response.get("result", "No response generated.")

# Gradio UI
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
