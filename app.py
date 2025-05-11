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
from sympy import symbols, Eq, latex

# --- Set Page Configuration for Wide Screen ---
st.set_page_config(page_title="Agentic Tutoring Chatbot", layout="wide")

# --- Load Environment Variables ---
load_dotenv()  # Loads variables from your .env file
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# --- Logging Setup ---
logging.basicConfig(
    filename='conversation.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# --- OpenAI & Model Setup ---
llm = ChatOpenAI(model="gpt-4o-mini")  # Using GPT-4o mini; adjust if needed

# --- Custom CSS for Improved UI and Background Image ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1527252432452-3e1b1e3c3a4f?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .main-header {
        font-size: 3em;
        font-weight: bold;
        color: #ffffff;
        text-shadow: 2px 2px 4px #000000;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Define the Conversation State using pydantic ---
class TutorState(BaseModel):
    user_input: str = ""
    interests: List[str] = Field(default_factory=list)
    detailed_interests: str = ""
    categories: List[str] = Field(default_factory=list)
    subject: str = ""
    selected_category: str = ""
    relevant_interest: str = ""  # The single most relevant interest for the subject
    conversation_history: List[str] = Field(default_factory=list)
    response: str = ""
    phase: str = "greeting"  # Starting phase

# --- Agent Functions for Each Phase ---

# Phase 1: Greeting
def initial_greeting_agent(state: TutorState) -> TutorState:
    prompt = (
        "Greeting: Hello! I'm your tutoring assistant. I can help you learn by connecting academic subjects to your "
        "favorite cartoons, movies, or books. To get started, please tell me about your favorite cartoons, movies, or books "
        "(e.g., SpongeBob, Avengers, Harry Potter)."
    )
    llm_response = llm(prompt)
    if hasattr(llm_response, "content"):
        llm_response = llm_response.content
    state.response = llm_response
    state.conversation_history.append("Bot: " + llm_response)
    state.phase = "gather_interests"
    return state

# Phase 2: Gather Interests (Require at least 6 interests)
def gather_interests_agent(state: TutorState) -> TutorState:
    if state.user_input:
        interests = [x.strip() for x in state.user_input.split(",") if x.strip()]
        if len(interests) < 6:
            state.response = "Please provide at least 6 interests (list 6 or more cartoons, movies, or books)."
        else:
            state.interests = interests
            state.conversation_history.append("User (Interests): " + state.user_input)
            state.phase = "followup_interests"
            state.response = ("Question: What do you like most about these shows or movies? "
                              "(For example, do you enjoy the humor, the action, or the characters?)")
    else:
        state.response = "Please tell me your favorite cartoons, movies, or books."
    return state

# Phase 3: Follow-up on Interests
def followup_interests_agent(state: TutorState) -> TutorState:
    if state.user_input:
        state.detailed_interests = state.user_input
        state.conversation_history.append("User (Details): " + state.user_input)
        state.phase = "categorize"
        state.response = "Thanks! Now I'll analyze your interests..."
    else:
        state.response = ("Question: What do you like most about these shows or movies? "
                          "(For example, humor, action, or characters?)")
    return state

# Phase 4: Categorize Interests
def categorize_agent(state: TutorState) -> TutorState:
    mapping = {
        "physics": ["Avengers", "Iron Man", "Batman"],
        "literature": ["Harry Potter", "Lord of the Rings"],
        "history": ["Troy", "Gladiator"]
    }
    categories = []
    for cat, keywords in mapping.items():
        for interest in state.interests:
            for keyword in keywords:
                if keyword.lower() in interest.lower():
                    categories.append(cat)
                    break
    state.categories = list(set(categories))
    state.response = (
        "Based on your interests, possible subject areas include: " +
        ", ".join(state.categories) +
        ". Now, please tell me what subject or concept you want to learn about, or ask a related question."
    )
    state.phase = "ask_topic"
    return state

# Phase 5: Ask Topic / Question
def learning_topic_agent(state: TutorState) -> TutorState:
    if state.user_input:
        state.subject = state.user_input.strip()
        state.conversation_history.append("User (Subject/Question): " + state.user_input)
        state = select_relevant_interest_agent(state)
        state.phase = "generate_answer"
    else:
        state.response = "Please specify the subject or concept you want to learn about, or ask a related question."
    return state

# New Agent: Select the Most Relevant Interest
def select_relevant_interest_agent(state: TutorState) -> TutorState:
    if not state.interests:
        state.relevant_interest = ""
        return state
    prompt = (
        f"Given the student's subject or question: '{state.user_input}', and the list of interests: "
        f"{', '.join(state.interests)}, indicate which single interest is most relevant to answer the query. "
        "Respond with only the interest."
    )
    logging.info("Select Relevant Interest Prompt:\n" + prompt)
    llm_response = llm(prompt)
    if hasattr(llm_response, "content"):
        llm_response = llm_response.content
    else:
        llm_response = str(llm_response)
    selected = llm_response.strip()
    available = [interest.lower() for interest in state.interests]
    if selected.lower() not in available:
        selected = state.interests[0]
    state.relevant_interest = selected
    return state
def format_latex(text: str) -> str:
    """Converts mathematical formulas in text to LaTeX format."""
    pattern = re.compile(r'([A-Za-z0-9]+ *= *[A-Za-z0-9/*+\-^() ]+)')
    text = pattern.sub(r'$$\\(\1\\)$$', text)

    # Convert formulas to proper LaTeX using sympy
    try:
        if "=" in text:
            left, right = text.split("=")
            equation = Eq(symbols(left.strip()), symbols(right.strip()))
            return f"$$ {latex(equation)} $$"
    except Exception:
        pass

    return text
# Phase 6: Generate Answer with Extended Few-Shot Examples and Structured Response
def generate_answer_agent(state: TutorState) -> TutorState:

    f"use LaTeX syntax for formulaes"
    # Extended few-shot examples to guide the LLM
    extended_few_shot_examples = """
Example: Gravity: A Detailed Explanation with Movie Context
Question: What is gravity?
Answer:
Gravity is a fundamental force of nature that attracts objects with mass toward each other. It governs the motion of planets, stars, and galaxies. Key points include:
- **Mass:** Greater mass results in a stronger gravitational pull.
- **Distance:** The gravitational force decreases with increasing distance.
- **Formula:** Newton’s Law of Universal Gravitation states that:
    F = G * (m1 * m2) / r^2
  where F is the gravitational force, G is the gravitational constant (6.674×10^-11 Nm²/kg²), and m1, m2 are the masses of two objects, with r being the distance between them.
Movies like *Gravity* and *Interstellar* showcase these principles by illustrating microgravity effects and gravitational time dilation, respectively.

Example: Newton's Laws of Motion: A Comprehensive Overview
Question: What are Newton's three laws of motion?
Answer:
1. **First Law (Inertia):** An object remains at rest or in uniform motion unless acted upon by an external force.
2. **Second Law (F = ma):** The acceleration of an object depends on the net force acting on it and its mass.
3. **Third Law (Action-Reaction):** Every action has an equal and opposite reaction.
These laws explain everyday phenomena, from vehicle acceleration to rocket propulsion.

Example: Algebra Fundamentals Explained
Question: What is algebra and why is it important?
Answer:
Algebra is the study of mathematical symbols and the rules for manipulating these symbols. It is essential for solving equations, modeling relationships, and understanding patterns. For instance, solving the equation 2x + 3 = 7 helps in developing logical thinking and problem-solving skills.
"""


    context = "\n".join(state.conversation_history)
    interest_context = f"using examples related to {state.relevant_interest}" if state.relevant_interest else ""
    
    prompt = (
       extended_few_shot_examples +
       "\n" +
       
        f"Please provide a structured, user-friendly explanation about '{state.subject}' {interest_context}.\n"
        f"Format the response like a detailed set of study notes, including:\n"
        f"- **Formal definition**\n"
        f"- **Contextual explanation (intuitive understanding)**\n"
        f"- **Historical background & origins**\n"
        f"- **Mathematical formulations (in LaTeX format if applicable)**\n"
        f"- **Real-world applications**\n"
        f"- **Key takeaways & summary tables**\n"
        f"Ensure that the explanation is highly detailed and well-structured, making it easy for a student to take notes.\n"
        f"\nConversation context:\n{context}\n"
        f"The student has these interests: " + ", ".join(state.interests) + "\n"
        f"Provide an in-depth and engaging response without follow-up questions."
    )
    
    
    logging.info("LLM Answer Prompt:\n" + prompt)
    
    llm_response = llm(prompt)
    if hasattr(llm_response, "content"):
        llm_response = llm_response.content
    else:
        llm_response = str(llm_response)
    
    state.response = llm_response
    state.conversation_history.append("Bot: " + llm_response)
    state.phase = "continue"
     # Convert formulas to LaTeX
    state.response = format_latex(llm_response)
    return state

# Phase 7: Continue / Follow-up
def continue_conversation_agent(state: TutorState) -> TutorState:
    state.response = (
        f"Would you like to ask a follow-up question about '{state.subject}'? If so, please type your question."
    )
    return state

def moderation_agent(state: TutorState) -> TutorState:
    banned_words = ["badword1", "badword2"]
    if any(bw in state.user_input.lower() for bw in banned_words) or any(bw in state.response.lower() for bw in banned_words):
        state.response = "The conversation contains inappropriate content. Please maintain a respectful language."
    return state

def sentiment_analysis_agent(state: TutorState) -> TutorState:
    blob = TextBlob(state.user_input)
    if blob.sentiment.polarity < -0.3:
        state.response = "I sense some frustration. Let's try to address your question clearly.\n" + state.response
    return state

def clean_response(text: str) -> str:
    cleaned_text = re.sub(r"\[.*?\]:", "", text)
    return cleaned_text.strip()

# --- Conversation Processing Function ---
def process_conversation():
    state = TutorState.parse_obj(st.session_state.state)
    if state.phase == "greeting":
        state = initial_greeting_agent(state)
    elif state.phase == "gather_interests":
        state = gather_interests_agent(state)
    elif state.phase == "followup_interests":
        state = followup_interests_agent(state)
    elif state.phase == "categorize":
        state = categorize_agent(state)
    elif state.phase == "ask_topic":
        state = learning_topic_agent(state)
    elif state.phase == "generate_answer":
        state = generate_answer_agent(state)
    elif state.phase == "continue":
        if state.user_input:
            state.conversation_history.append("User (Follow-up): " + state.user_input)
            state = generate_answer_agent(state)
        else:
            state = continue_conversation_agent(state)
    state = moderation_agent(state)
    state = sentiment_analysis_agent(state)
    state.user_input = ""  # Clear user input after processing.
    st.session_state.state = state.dict()

# --- Streamlit Session State Initialization ---
if "state" not in st.session_state:
    st.session_state.state = TutorState().dict()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List to store chat messages.

# --- Streamlit ChatGPT-like Interface with Improved UI ---
st.markdown("<div class='main-header'>Agentic Tutoring Chatbot</div>", unsafe_allow_html=True)
st.write("A ChatGPT‑like interface that guides you and uses previous context for tailored answers.")

# Display chat history using streamlit-chat with unique keys.
from streamlit_chat import message
for idx, chat in enumerate(st.session_state.chat_history):
    if chat["sender"] == "user":
        message(chat["message"], is_user=True, key=f"user_{idx}")
    else:
        message(chat["message"], key=f"bot_{idx}")

# --- User Input Section using a Form for Immediate Submission ---
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message:", key="input_text")
    submitted = st.form_submit_button("Send")

if submitted:
    state = TutorState.parse_obj(st.session_state.state)
    state.user_input = user_input
    state.conversation_history.append("User: " + user_input)
    st.session_state.state = state.dict()
    process_conversation()
    bot_response = clean_response(st.session_state.state["response"])
    st.session_state.chat_history.append({"sender": "user", "message": user_input})
    st.session_state.chat_history.append({"sender": "bot", "message": bot_response})
