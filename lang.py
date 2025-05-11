import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message  # pip install streamlit-chat
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from fpdf import FPDF

# --- Set Page Configuration and Load Environment Variables ---
st.set_page_config(page_title="Agentic Tutoring Chatbot", layout="wide")
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not set in environment variables.")
    st.stop()

# --- Create a LangChain Conversation Chain ---
llm = ChatOpenAI(api_key=api_key, model_name="gpt-4")
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Prompt instructs the chatbot to follow a tutoring flow:
template = """
You are an agentic tutoring chatbot. You help students connect academic subjects to their favorite cartoons, movies, or books.
Follow this conversation flow:
1. Greet the student and ask for at least 6 interests.
2. Ask what they like about those interests.
3. Suggest a subject area based on their interests.
4. Ask for the subject or question they want to learn about.
5. Provide a detailed, structured explanation including formal definitions, intuitive explanations, historical background, mathematical formulations (if applicable), real-world applications, and key takeaways.
If the student expresses frustration, acknowledge it and moderate any inappropriate language.

Conversation so far:
{history}
User: {input}
Chatbot:
"""
prompt_template = PromptTemplate(input_variables=["history", "input"], template=template)
conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt_template)

# --- Custom CSS for a Better Look ---
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
st.markdown("<div class='main-header'>Agentic Tutoring Chatbot</div>", unsafe_allow_html=True)
st.write("A streamlined tutoring chatbot using LangChain.")

# --- Chat History (stored in Streamlit session state) ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- PDF Generation Function ---
def create_pdf(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 6, text)
    return pdf.output(dest="S").encode("latin-1")

# --- Display Chat History ---
for idx, chat in enumerate(st.session_state.chat_history):
    if chat["sender"] == "user":
        message(chat["message"], is_user=True, key=f"user_{idx}")
    else:
        message(chat["message"], key=f"bot_{idx}")
        if chat.get("pdf", False):
            pdf_bytes = create_pdf(chat["message"])
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"bot_response_{idx}.pdf",
                mime="application/pdf"
            )

# --- User Input Section ---
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    st.session_state.chat_history.append({"sender": "user", "message": user_input})
    with st.spinner("Thinking..."):
        bot_reply = conversation.predict(input=user_input)
    # Optionally, if the reply is lengthy (e.g., detailed answer), flag it for PDF download.
    pdf_flag = len(bot_reply.split()) > 100
    st.session_state.chat_history.append({"sender": "bot", "message": bot_reply, "pdf": pdf_flag})

# --- Option to Restart the Conversation ---
if st.button("Start New Topic"):
    memory.clear()  # Reset conversation memory
    st.session_state.chat_history = []
    st.success("Conversation reset. Start a new topic.")
