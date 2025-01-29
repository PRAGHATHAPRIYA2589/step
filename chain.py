from prompt import quiz_generator_prompt
from model import create_chat_groq

"""
Function to generate a quiz
Args:
    topic (str): The topic of the quiz
    difficulty (str): Difficulty level of the quiz
    num_questions (int): Number of questions
Returns:
    str: The generated quiz content
"""

def generate_quiz(topic, difficulty, num_questions):
    prompt_template = quiz_generator_prompt()
    llm = create_chat_groq()
    
    chain = prompt_template | llm
    response = chain.invoke({
        "topic": topic,
        "difficulty": difficulty,
        "num_questions": num_questions
    })
    return response.content