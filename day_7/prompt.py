from langchain_core.prompts import ChatPromptTemplate

def quiz_generator_prompt():
    """
    Generates a Prompt template for the quiz generator
    """
    system_msg = '''
        You are a dedicated quiz generator assistant. Your task is to create quizzes based on the topic, difficulty level, 
        number of questions provided, and retrieved context from the database. Follow these guidelines:
        - Use the retrieved context to enhance the questions and answers.
        - Only generate quiz questions relevant to the topic and difficulty level.
        - The output must contain exactly the number of questions requested, with each question clearly numbered.
        - Include multiple-choice options (A, B, C, D), with one correct answer.
    '''
    user_msg = "Context: {context}\nCreate a {num_questions}-question quiz on {topic} at {difficulty} difficulty level."

    return ChatPromptTemplate([
        ("system", system_msg),
        ("user", user_msg)
    ])