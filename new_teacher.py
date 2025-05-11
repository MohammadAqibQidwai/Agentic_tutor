import os
import logging
from dotenv import load_dotenv
from textblob import TextBlob  # For sentiment analysis
import streamlit as st
from streamlit_chat import message  # pip install streamlit-chat
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import re
from typing import List

st.set_page_config(page_title="Agentic Tutoring Chatbot", layout="wide")
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)

class TutorState(BaseModel):
    user_input: str = ""
    interests: List[str] = Field(default_factory=list)
    subject: str = ""
    conversation_history: List[str] = Field(default_factory=list)
    response: str = ""

def format_latex(text: str) -> str:
    """Converts mathematical formulas in text to LaTeX format."""
    pattern = re.compile(r'([A-Za-z0-9]+ *= *[A-Za-z0-9/*+\-^() ]+)')
    return pattern.sub(r'$$\\(\1\\)$$', text)

def generate_answer_agent(state: TutorState) -> TutorState:
    extended_few_shot_examples = """
    Example: Gravity: A Detailed Explanation with Movie Context
    Question: What is gravity?
    Answer:
    Gravity is a fundamental force of nature that attracts objects with mass toward each other. It governs the motion of planets, stars, and galaxies. Key points include:
    - **Mass:** Greater mass results in a stronger gravitational pull.
    - **Distance:** The gravitational force decreases with increasing distance.
    - **Formula:** Newton’s Law of Universal Gravitation states that:
        F = G * (m1 * m2) / r^2
    """
    
    context = "\n".join(state.conversation_history)
    prompt = f"{extended_few_shot_examples}\nNow, provide a structured explanation about '{state.subject}'."
    response = llm(prompt)
    response_text = response.content if hasattr(response, "content") else response
    
    # Convert formulas to LaTeX
    state.response = format_latex(response_text)
    
    return state

def chatbot():
    st.title("📚 AI Tutoring Chatbot")
    state = TutorState()
    
    user_input = st.text_input("Ask me anything about an academic concept!")
    if user_input:
        state.user_input = user_input
        state.subject = user_input.strip()
        state.conversation_history.append("User: " + user_input)
        state = generate_answer_agent(state)
        
        # Display AI response properly formatted for LaTeX
        if "$$" in state.response:
            st.latex(state.response.replace("$$", ""))  # Render formulas correctly
        else:
            st.markdown(state.response)

if __name__ == "__main__":
    chatbot()