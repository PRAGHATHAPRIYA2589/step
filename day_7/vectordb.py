import os
import chromadb
from langchain.vectorstores import Chroma  # Ensure it's correctly imported
from langchain.embeddings.base import Embeddings  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

### 1️ CUSTOM EMBEDDINGS CLASS ###
class HuggingFaceEmbeddings(Embeddings):
    """Custom embedding model wrapper for LangChain"""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text):
        """Generate embeddings for a single query"""
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_documents(self, texts):
        """Generate embeddings for multiple documents"""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

### 2️ INITIALIZE EMBEDDINGS ###
def load_embeddings():
    """Initializes a Hugging Face embedding model."""
    return HuggingFaceEmbeddings()

### 3️ INITIALIZE CHROMADB ###
def initialize_chroma(persist_directory="./chroma_db"):
    """
    Initializes and returns a Chroma vector store.

    Args:
        persist_directory (str): Directory to store ChromaDB.

    Returns:
        Chroma vector store instance.
    """
    os.makedirs(persist_directory, exist_ok=True)
    
    # Load embeddings
    hf_embeddings = load_embeddings()

    # Initialize Chroma vector store
    vectordb = Chroma(embedding_function=hf_embeddings, persist_directory=persist_directory)

    # Ensure ChromaDB is initialized correctly
    if vectordb is None:
        print(" Error: ChromaDB Initialization Failed.")
    else:
        print(" ChromaDB Initialized Successfully!")

    return vectordb

### 4️ PROCESS PDF FILE FOR STORAGE ###
def process_pdf_for_rag(uploaded_file):
    """
    Processes the uploaded PDF file and prepares it for storage.

    Args:
        uploaded_file: file for RAG ingestion pipeline.

    Returns:
        List of processed document chunks.
    """
    temp_file_path = f"temp_{uploaded_file.name}"
    
    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        return splits
    
    finally:
        # Ensure the temp file is deleted
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

### 5️ STORE DOCUMENTS IN CHROMADB ###
def store_pdf_in_chroma(uploaded_file, vectorstore):
    """
    Stores the processed document chunks into ChromaDB.

    Args:
        uploaded_file: File for RAG ingestion pipeline.
        vectorstore: Instance of Chroma vector store.

    Returns:
        Success message if processing is complete.
    """
    splits = process_pdf_for_rag(uploaded_file)

    # Extract text from splits
    texts = [doc.page_content for doc in splits]

    # Embed and store in ChromaDB
    vectorstore.add_texts(texts)

    return f" File '{uploaded_file.name}' successfully processed and stored in vector database!"

### 6️ RETRIEVE DOCUMENTS FROM CHROMADB ###
def retrieve_from_chroma(query, vectorstore, top_k=5):
    """
    Retrieves the most relevant documents from the Chroma vector store.

    Args:
        query (str): The query string for searching the vector store.
        vectorstore: The Chroma vector store instance.
        top_k (int): Number of top relevant documents to retrieve.

    Returns:
        List of retrieved documents.
    """
    return vectorstore.similarity_search(query, k=top_k)

def retrieve_similar_documents(vectordb, query, top_k=5):
    """
    Retrieves similar documents from the vector database.
    
    Args:
        vectordb: Chroma instance.
        query: The input query for retrieval.
        top_k: Number of results to return.

    Returns:
        List of retrieved documents.
    """
    return vectordb.similarity_search(query, k=top_k)

### 7️ RETRIEVE RELEVANT TEXT FROM CHROMADB ###
def retrieve_relevant_text(query, vectordatabase, top_k=5):
    """
    Retrieves the most relevant text from ChromaDB.

    Args:
        query (str): Search query.
        vectordatabase: Chroma instance.
        top_k (int): Number of results to return.

    Returns:
        String of relevant retrieved text.
    """
    # Perform similarity search directly
    results = vectordatabase.similarity_search(query, k=top_k)

    # Extract relevant text
    relevant_texts = [doc.page_content for doc in results]

    return " ".join(relevant_texts) if relevant_texts else None