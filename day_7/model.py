from langchain_groq import ChatGroq # type: ignore

def create_chat_groq():
    """
    Function to initialize ChatGroq
    
    Returns:
        ChatGroq
    """
    return ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=1,
        max_tokens=None,
        timeout=None,
        max_retries=2  # For avoiding failures
    )