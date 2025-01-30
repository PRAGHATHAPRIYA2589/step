import os
from dotenv import load_dotenv  
import streamlit as st
import vectordb
import chain  # For quiz generation
from home import home_page  
import utils
import pypdf

# Load environment variables
load_dotenv()

# Define pages
PAGES = {
    "Home": "home",
    "Generate Quiz": "generate_quiz",
    "RAG File Ingestion": "rag_ingestion"
}

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# Database Initialization
vectordatabase = vectordb.initialize_chroma()

#### RAG FILE INGESTION PAGE ####
def rag_ingestion_page():
    """
    Provides UI for uploading files and storing embeddings in ChromaDB.
    """
    st.title("📂 RAG File Ingestion")

    uploaded_file = st.file_uploader("Upload a PDF file for RAG processing:", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            # FIX: Use store_pdf_in_chroma instead of process_pdf_for_rag
            result_message = vectordb.store_pdf_in_chroma(uploaded_file, vectordatabase)
            st.success(result_message)

###  QUIZ GENERATOR PAGE ####
def generate_quiz_page():
    """
    Generates a quiz based on user-selected topics and difficulty levels.
    Allows enabling RAG to generate questions based on an uploaded book.
    """
    st.markdown("<div style='background-color: #4CAF50; padding: 20px; text-align: center; border-radius: 10px; color: white;'>"
                "<h1 style='font-family: Arial, sans-serif;'>Quiz Generator</h1></div>", unsafe_allow_html=True)

    with st.form("quiz_generator"):
        topic = st.selectbox("Select the Topic for the Quiz", ["English", "Maths", "Science", "History", "Technology"])
        difficulty = st.selectbox("Select Difficulty Level", ["Easy", "Medium", "Hard"])
        num_questions = st.number_input("Enter the Number of Questions", min_value=1, max_value=50, step=1)

        # Enable RAG for quiz generation
        use_rag = st.checkbox("Enable RAG (Use book data for quiz)")

        submitted = st.form_submit_button("Generate Quiz")

        if submitted:
            with st.spinner("Generating your quiz..."):
                if use_rag:
                    # Fetch relevant content from book embeddings
                    relevant_texts = vectordb.retrieve_from_chroma(topic, vectordatabase, top_k=5)  
                    
                    if relevant_texts:
                        extracted_text = " ".join([doc.page_content for doc in relevant_texts])
                        response = chain.generate_quiz_from_text(extracted_text, difficulty, num_questions)
                        st.success("📖 Quiz Generated Successfully from Book Data!")
                    else:
                        st.error("⚠ No relevant content found in the uploaded book. Try a different topic.")
                        return
                else:
                    response = chain.generate_quiz(topic, difficulty, num_questions)
                    st.success("✅ Quiz Generated Successfully!")

                st.info(response)

###  MAIN FUNCTION ####
def main():
    # Function to handle navigation
    def navigate_to(page):
        st.session_state.current_page = page  

    # Sidebar navigation
    st.sidebar.title("📌 Menu")
    selected_section = st.sidebar.radio("Navigate to:", list(PAGES.keys()))

    # Set session state
    st.session_state.current_page = PAGES[selected_section]

    # Render selected page
    if st.session_state.current_page == "home":
        home_page(navigate_to)
    elif st.session_state.current_page == "generate_quiz":
        generate_quiz_page()
    elif st.session_state.current_page == "rag_ingestion":
        rag_ingestion_page()

if __name__ == "__main__":
    main()
