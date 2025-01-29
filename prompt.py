from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

def quiz_generator_prompt():
    """
    Generates a Prompt template for the quiz generator
    Returns:
        ChatPromptTemplate -> Configured ChatPromptTemplate instance
    """
    system_msg = '''
                You are a dedicated quiz generator assistant. Your task is to create quizzes based on the topic, difficulty level, and number of questions provided by the user. Follow these guidelines:
                1. Only generate quiz questions relevant to the specified topic, adhering to the selected difficulty level.
                2. The output must contain exactly the number of questions requested, with each question clearly numbered.
                3. Each question should include multiple-choice options (A, B, C, D), with one correct answer and no additional explanations.
                4. If the query is unrelated to quiz generation, respond with:
                "I am a quiz generator assistant, specialized in generating quizzes. Please ask me a quiz-related query."
                5. Always adhere to the user’s specifications of topic, difficulty, and the number of questions.
                '''
    user_msg = "Create a {num_questions}-question quiz on {topic} at {difficulty} difficulty level."
    
    prompt_template = ChatPromptTemplate([
        ("system", system_msg),
        ("user", user_msg)
    ])
    return prompt_template

def quiz_generator_prompt_from_hub(template="priya/quiz_generator"):
    """
    Generates Prompt template from the LangSmith prompt hub
    Returns:
        ChatPromptTemplate -> ChatPromptTemplate instance pulled from LangSmith Hub
    """
    prompt_template = hub.pull(template)
    return prompt_template
