from dotenv import load_dotenv
import chain
import streamlit as st

load_dotenv()


def quiz_generator_app():
     
    """
    Quiz Generator App
    """
    with st.form("quiz_generator"):
        topic = st.selectbox("Select the Topic for the Quiz", ["English", "Maths", "Science", "Social"])
        difficulty = st.selectbox("Select Difficulty Level", ["Easy", "Medium", "Hard"])
        num_questions = st.number_input("Enter the Number of Questions", min_value=1, max_value=50, step=1)
        submitted = st.form_submit_button("Submit")

        if submitted:
            response = chain.generate_quiz(topic, difficulty, num_questions)
            st.info(response)

quiz_generator_app()