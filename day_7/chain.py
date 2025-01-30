

from prompt import quiz_generator_prompt
from model import create_chat_groq
from utils import load_or_initialize_vectordb
from vectordb import retrieve_similar_documents

def generate_quiz(topic, difficulty, num_questions):
    """
    Generates a quiz using RAG and LangChain.
    """
    # Initialize vector database
    vectordb = load_or_initialize_vectordb()

    # Retrieve relevant documents
    query = f"Quiz on {topic} with {difficulty} difficulty"
    retrieved_docs = retrieve_similar_documents(vectordb, query)

    # Incorporate retrieved knowledge into the prompt
    retrieved_content = "\n".join([doc.page_content for doc in retrieved_docs])
    context = f"Context: {retrieved_content}"

    # Prepare prompt
    prompt_template = quiz_generator_prompt()
    llm = create_chat_groq()

    chain = prompt_template | llm
    response = chain.invoke({
        "topic": topic,
        "difficulty": difficulty,
        "num_questions": num_questions,
        "context": context  # Add context from retrieved documents
    })
    return response.content


# def generate_quiz_from_text(extracted_text, difficulty, num_questions):
#     """
#     Generates quiz questions from extracted text using AI.
#     """
#     llm = create_chat_groq()
    
#     prompt = f"Generate {num_questions} {difficulty}-level quiz questions from the following text:\n\n{extracted_text}"
    
#     response = llm.invoke(prompt)
    
#     return response.content


def generate_quiz_from_text(extracted_text, difficulty, num_questions):
    """
    Generates quiz questions with multiple-choice options from extracted text using AI.
    """
    llm = create_chat_groq()
    
    prompt = f"""
    Generate {num_questions} {difficulty}-level multiple-choice quiz questions from the following text.
    Each question should have four answer choices (A, B, C, D), and indicate the correct answer.
    
    Text:
    {extracted_text}

    Format:
    1. Question?
        A) Option 1
        B) Option 2
        C) Option 3
        D) Option 4
        Correct Answer: X
    """

    response = llm.invoke(prompt)
    
    return response.content