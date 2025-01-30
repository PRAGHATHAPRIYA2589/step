import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer  # type: ignore
from langchain.vectorstores import Chroma  # type: ignore
from langchain.embeddings.base import Embeddings  # type: ignore  # Import base class

class HuggingFaceEmbeddings(Embeddings):
    """Custom embedding model wrapper to work with LangChain"""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__()  # Ensure base class initialization
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text):
        """Embeds a single query text."""
        return self.model.encode(text).tolist()

    def embed_documents(self, texts):
        """Embeds multiple documents."""
        return self.model.encode(texts).tolist()

def load_embeddings():
    """
    Initializes a Hugging Face embedding model.
    """
    return HuggingFaceEmbeddings()

def load_or_initialize_vectordb(persist_directory="db"):
    """
    Initializes or loads the vector database.
    """
    os.makedirs(persist_directory, exist_ok=True)
    embeddings = load_embeddings()

    # Correctly pass embedding function
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings.embed_query)
    return vectordb

def process_pdf_for_rag(uploaded_file, vectordb):
    """
    Processes the uploaded PDF file for storage in ChromaDB.

    Args:
        uploaded_file: The file uploaded for RAG ingestion.
        vectordb: The ChromaDB instance where embeddings will be stored.

    Returns:
        Success message if processing is complete.
    """
    temp_file_path = f"temp_{uploaded_file.name}"
    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load Documents
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Extract text for embedding
        texts = [doc.page_content for doc in splits]

        # Store in ChromaDB
        vectordb.add_texts(texts)

        return "File successfully processed and stored in vector database!"
    
    finally:
        # Ensure the temp file is deleted
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
